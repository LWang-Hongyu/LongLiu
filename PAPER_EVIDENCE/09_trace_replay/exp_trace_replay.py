"""
exp_trace_replay: Lingjun 2023 trace 时段重放对照实验（论文 robustness 章节）。

对比 5 策略（Fair / CRUX / SP / D1 / v4）在真实 trace 到达模式下的表现。
与合成 workload 实验的关键区别：
- 到达时刻来自 trace 真实 gmt_job_submitted（最密集 6h 窗口等比压缩）
- 保留真实到达间隔分布（非 Poisson）
- 模型映射白名单 + 参数规模分层 SLO CI

用法：
    python experiments/exp_trace_replay.py --seeds 10
    python experiments/exp_trace_replay.py --seeds 2 --quick   # 快速验证
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import yaml

from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.policy.fair import Fair
from longliu_sim.policy.crux import CRUX
from longliu_sim.policy.srpt import SRPT
from longliu_sim.policy.dwrr import LongLiuDWRR, LongLiuAllocatorV4
from longliu_sim.trace.lingjun import LingjunTraceLoader

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_CONFIG = os.path.join(PROJECT_ROOT, "configs", "trace_replay.yaml")

# 锚点语义（与 config.yaml 一致）
SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0

# P-attn 判定容差（与 v3_batch2 一致）
V4_GUARANTEE_TOLERANCE = 0.02


def load_replay_config() -> dict:
    """读取 configs/trace_replay.yaml。"""
    with open(REPLAY_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    """读取 config.yaml 冻结参数。"""
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def get_policy(name: str, trace_file: str, frozen: dict):
    """构造 5 策略（与 exp_v3_batch2 相同的构造签名）。"""
    overhead = frozen["overhead_factor"]
    overlap = frozen["overlap_factor"]
    if name == "Fair":
        return Fair()
    elif name == "CRUX":
        return CRUX()
    elif name == "SP":
        return SRPT()
    elif name == "D1":
        return LongLiuDWRR(
            overhead_factor=overhead, overlap_factor=overlap,
            trace_file=trace_file,
        )
    elif name == "v4":
        return LongLiuAllocatorV4(
            overhead_factor=overhead, overlap_factor=overlap,
            trace_file=trace_file,
        )
    raise ValueError(f"Unknown policy: {name}")


def run_single(policy_name: str, seed: int, cfg: dict, frozen: dict,
               out_dir: str) -> dict:
    """运行单个 (policy, seed) 组合，返回统计。"""
    topo_cfg = cfg["topology"]
    trace_cfg = cfg["trace"]
    duration_ms = cfg["duration_ms"]
    premium_thr = cfg.get("premium_ci_threshold", 2.0)

    os.makedirs(out_dir, exist_ok=True)
    trace_file = os.path.join(out_dir, f"trace_{policy_name}_s{seed}.jsonl")

    topo = FatTreeTopology(
        k=topo_cfg["k"],
        host_bw_bps=topo_cfg["host_bw_bps"],
        spine_bw_bps=topo_cfg["spine_bw_bps"],
    )
    policy = get_policy(policy_name, trace_file, frozen)
    sim = Simulator(
        topo, policy, duration_ms=duration_ms, seed=seed,
        overhead_factor=frozen["overhead_factor"],
        overlap_factor=frozen["overlap_factor"],
    )

    loader = LingjunTraceLoader(
        zip_path=trace_cfg["zip_path"],
        min_gpus=trace_cfg.get("min_gpus", 2),
        max_gpus=trace_cfg["max_gpus"],
        duration_ms=duration_ms,
        seed=seed,
        target_bw_bps=topo_cfg["host_bw_bps"],
        overhead_factor=frozen["overhead_factor"],
        overlap_factor=frozen["overlap_factor"],
        window_hours=trace_cfg["window_hours"],
        window_start=trace_cfg.get("window_start"),
        max_jobs=trace_cfg.get("max_jobs"),
        num_hosts=trace_cfg["num_hosts"],
    )
    jobs = loader.load()
    if not jobs:
        policy.flush_trace()
        return {"error": "no_jobs"}

    for j in jobs:
        sim.submit(j)

    result = sim.run()
    if hasattr(policy, "flush_trace"):
        policy.flush_trace()

    stats = result.per_job_stats(host_bw_gbps=HOST_BW_GBPS)

    premium_jids = {
        jid for jid, job in sim.jobs.items() if job.slo_ci <= premium_thr
    }
    n_premium = len(premium_jids)
    n_premium_attn = sum(
        1 for jid in premium_jids if stats[jid]["sas"] >= 1.0 - V4_GUARANTEE_TOLERANCE
    )
    starv_premium = sum(
        1 for jid in premium_jids if stats[jid]["completed_iters"] == 0
    )
    all_sas = [s["sas"] for s in stats.values()]
    sas_mean = sum(all_sas) / len(all_sas) if all_sas else 0.0
    sas_sorted = sorted(all_sas)
    sas_median = (
        sas_sorted[len(sas_sorted) // 2] if sas_sorted else 0.0
    )
    sas_min = min(all_sas) if all_sas else 0.0
    meets_overall = sum(1 for s in stats.values() if s["meets_slo"]) / len(stats)

    meta = {
        "policy": policy_name,
        "seed": seed,
        "n_jobs": len(jobs),
        "n_premium": n_premium,
        "p_attn": round(n_premium_attn / n_premium, 4) if n_premium else 1.0,
        "sas_mean": round(sas_mean, 4),
        "sas_median": round(sas_median, 4),
        "sas_min": round(sas_min, 4),
        "slo_attainment": round(meets_overall, 4),
        "starv_premium": starv_premium,
        "total_iters": result.total_iterations(),
    }
    with open(os.path.join(out_dir, f"run_meta_{policy_name}_s{seed}.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Lingjun trace 时段重放对照实验")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick", action="store_true",
                        help="快速验证：仅跑 seed 0，检查 loader 与拓扑匹配")
    parser.add_argument("--output", default="outputs/trace_replay")
    args = parser.parse_args()

    cfg = load_replay_config()
    frozen = load_frozen()
    policies = cfg["policies"]
    seeds = [0] if args.quick else list(range(args.seeds))

    out_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("Lingjun 2023 Trace 时段重放对照实验")
    print(f"  frozen: overhead={frozen['overhead_factor']}, "
          f"overlap={frozen['overlap_factor']}, K={frozen['K']}")
    print(f"  trace: window={cfg['trace']['window_hours']}h, "
          f"max_gpus={cfg['trace']['max_gpus']}, num_hosts={cfg['trace']['num_hosts']}")
    print(f"  policies: {policies}, seeds: {seeds}")
    print("=" * 78)

    # 先验证 loader 可加载（避免 50 run 后才发现问题）
    probe = LingjunTraceLoader(
        zip_path=cfg["trace"]["zip_path"],
        min_gpus=cfg["trace"].get("min_gpus", 2),
        max_gpus=cfg["trace"]["max_gpus"],
        duration_ms=cfg["duration_ms"],
        seed=0,
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        overhead_factor=frozen["overhead_factor"],
        overlap_factor=frozen["overlap_factor"],
        window_hours=cfg["trace"]["window_hours"],
        max_jobs=cfg["trace"].get("max_jobs"),
        num_hosts=cfg["trace"]["num_hosts"],
    )
    probe_jobs = probe.load()
    if not probe_jobs:
        print("!! trace 过滤后无 job，请检查 max_gpus/白名单")
        return
    models = {}
    for j in probe_jobs:
        models[j.model] = models.get(j.model, 0) + 1
    print(f"  [probe] {len(probe_jobs)} jobs: {models}")
    if any(j.worker_hosts and max(j.worker_hosts) >= cfg["trace"]["num_hosts"]
           for j in probe_jobs):
        print("!! worker_hosts 越界，请检查 num_hosts")
        return

    results = {p: [] for p in policies}
    t0 = time.time()
    for pn in policies:
        for seed in seeds:
            r = run_single(pn, seed, cfg, frozen, out_dir)
            if "error" in r:
                print(f"  {pn} s{seed}: {r['error']}")
                continue
            results[pn].append(r)
            print(f"  {pn:<5} s{seed}: {r['n_jobs']} jobs, "
                  f"P-attn={r['p_attn']*100:.1f}%, sas={r['sas_mean']:.3f}, "
                  f"starvP={r['starv_premium']}")
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    # ---- 汇总 ----
    metric_names = ["n_jobs", "n_premium", "p_attn", "sas_mean",
                    "sas_median", "sas_min", "slo_attainment",
                    "starv_premium", "total_iters"]

    summary = {}
    for pn in policies:
        summary[pn] = {}
        if not results[pn]:
            continue
        for m in metric_names:
            vals = [r[m] for r in results[pn]]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 \
                if len(vals) > 1 else 0.0
            summary[pn][m] = mean
            summary[pn][f"{m}_std"] = std

    # 配对 t-test：相对 v4（主策略）
    if _HAS_SCIPY:
        for pn in policies:
            if pn == "v4" or not results[pn] or len(results[pn]) != len(results["v4"]):
                continue
            try:
                _, pv = scipy_stats.ttest_rel(
                    [r["sas_mean"] for r in results[pn]],
                    [r["sas_mean"] for r in results["v4"]],
                )
                summary[pn]["p_sas_vs_v4"] = pv
                _, pv_attn = scipy_stats.ttest_rel(
                    [r["p_attn"] for r in results[pn]],
                    [r["p_attn"] for r in results["v4"]],
                )
                summary[pn]["p_attn_vs_v4"] = pv_attn
            except Exception:
                pass

    csv_path = os.path.join(out_dir, "trace_replay_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Policy"] + metric_names +
                   ["p_sas_vs_v4", "p_attn_vs_v4"])
        for pn in policies:
            s = summary[pn]
            row = [pn] + [f"{s.get(m, 0):.4f}±{s.get(f'{m}_std', 0):.4f}"
                          if s.get(f"{m}_std", 0) else f"{s.get(m, 0):.4f}"
                          for m in metric_names]
            row.append(f"{s['p_sas_vs_v4']:.4e}" if "p_sas_vs_v4" in s else "")
            row.append(f"{s['p_attn_vs_v4']:.4e}" if "p_attn_vs_v4" in s else "")
            w.writerow(row)
    print(f"\n  CSV → {csv_path}")

    print("\n  摘要 (mean±std across seeds):")
    print(f"  {'Policy':<7} {'P-attn%':<12} {'sas_mean':<12} {'sas_min':<10} "
          f"{'starvP':<8} {'SLO-attain%':<14} {'total_iters':<12}")
    for pn in policies:
        s = summary[pn]
        print(f"  {pn:<7} {s.get('p_attn', 0)*100:<12.1f} {s.get('sas_mean', 0):<12.3f} "
              f"{s.get('sas_min', 0):<10.3f} {s.get('starv_premium', 0):<8.1f} "
              f"{s.get('slo_attainment', 0)*100:<14.1f} {s.get('total_iters', 0):<12.0f}")


if __name__ == "__main__":
    main()
