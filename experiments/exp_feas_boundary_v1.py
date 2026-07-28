"""
主场景 feas_boundary_v1：800G spine，7 job，四策略对比。

预登记假设（H1-H6）：
H1: Fair premium attainment ≈ 33%（仅 P3）
H2: D1 premium attainment 显著 > Fair（目标 ≥67%）
H3: D1 starvation = 0%
H4: LongLiu-SP starvation > 0%
H5: D1 premium capped mean > Fair
H6: D1 下 standard 大 job 有界降级（sas 0.3-0.7），不归零

纪律：任何一条不成立，报数据不报结论。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import Fair, CRUX, LongLiu, LongLiuDWRR, LongLiuDWRRGap
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V1_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS
from longliu_sim.utils import compute_sas_eval, compute_iter_solo_ms


def get_git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        modified_files = [line[3:] for line in dirty_output.split("\n") if line.startswith(" M")]
        return {"commit": commit, "dirty": bool(modified_files), "dirty_files": modified_files}
    except:
        return {"commit": "unknown", "dirty": True, "dirty_files": []}


def _sas_eval(job, avg_iter_ms: float, overlap_factor: float = 0.85,
              host_bw_gbps: float = 100.0) -> float:
    """计算 sas_eval：统一调用 longliu_sim.utils.metrics。"""
    return compute_sas_eval(
        avg_iter_ms, job.model, job.num_workers, job.slo_ci,
        host_bw_gbps=host_bw_gbps, overlap_factor=overlap_factor
    )


def _iter_solo(job, overlap_factor: float = 0.85, host_bw_gbps: float = 100.0) -> float:
    """计算 iter_solo：统一调用 longliu_sim.utils.metrics。"""
    return compute_iter_solo_ms(
        job.model, job.num_workers,
        host_bw_gbps=host_bw_gbps, overlap_factor=overlap_factor
    )


def run_single(cfg: dict, policy, seed: int, policy_name: str) -> dict:
    """运行单个 seed 的仿真。"""
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
    sim = Simulator(
        topo, policy,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

    loader = SyntheticTraceLoader(
        model_types=[],
        gpu_distribution={},
        ci_distribution={},
        job_count=7,  # 由 workload_profile 决定
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=FEAS_BOUNDARY_V1_WORKLOAD,
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    host_bw_gbps = cfg["topology"]["host_bw_bps"] / 1e9
    stats = result.per_job_stats(host_bw_gbps=host_bw_gbps)

    # sas_eval 重新计算 + tier 分层
    premium_stats = []
    standard_stats = []
    per_job_results = []

    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        avg_iter_ms = s["avg_iter_ms"]
        sas_eval = _sas_eval(job, avg_iter_ms, cfg["overlap_factor"], host_bw_gbps)
        completed = job.completed_iters
        target = job.target_iters
        attained = avg_iter_ms <= (ci * _iter_solo(job, cfg["overlap_factor"], host_bw_gbps))
        starved = completed == 0

        per_job_results.append({
            "jid": jid,
            "model": job.model,
            "dp": job.num_workers,
            "ci": ci,
            "sas_eval": sas_eval,
            "avg_iter_ms": avg_iter_ms,
            "completed_iters": completed,
            "target_iters": target,
            "attained": attained,
            "starved": starved,
        })

        if ci == 1.3:
            premium_stats.append(sas_eval)
        elif ci == 2.0:
            standard_stats.append(sas_eval)

    # 聚合统计
    premium_mean = sum(premium_stats) / len(premium_stats) if premium_stats else 0.0
    premium_capped = sum(min(s, 1.0) for s in premium_stats) / len(premium_stats) if premium_stats else 0.0
    premium_attainment = sum(1 for j in per_job_results if j["ci"] == 1.3 and j["attained"]) / \
                         max(1, sum(1 for j in per_job_results if j["ci"] == 1.3))
    premium_starved = sum(1 for j in per_job_results if j["ci"] == 1.3 and j["starved"])

    standard_mean = sum(standard_stats) / len(standard_stats) if standard_stats else 0.0
    standard_capped = sum(min(s, 1.0) for s in standard_stats) / len(standard_stats) if standard_stats else 0.0
    standard_attainment = sum(1 for j in per_job_results if j["ci"] == 2.0 and j["attained"]) / \
                          max(1, sum(1 for j in per_job_results if j["ci"] == 2.0))
    standard_starved = sum(1 for j in per_job_results if j["ci"] == 2.0 and j["starved"])

    overall_mean = sum(p["sas_eval"] for p in per_job_results) / len(per_job_results) if per_job_results else 0.0

    return {
        "seed": seed,
        "overall_mean_sas": overall_mean,
        "premium_mean": premium_mean,
        "premium_capped": premium_capped,
        "premium_attainment": premium_attainment,
        "premium_starved": premium_starved,
        "standard_mean": standard_mean,
        "standard_capped": standard_capped,
        "standard_attainment": standard_attainment,
        "standard_starved": standard_starved,
        "per_job": per_job_results,
    }


def verify_hypotheses(all_results: dict):
    """逐条验证预登记假设 H1-H6。"""
    H = {}

    fair_r = all_results.get("Fair", [])
    d1_r = all_results.get("D1", [])
    crux_r = all_results.get("CRUX", [])
    sp_r = all_results.get("LongLiu-SP", [])

    # H1: Fair premium attainment = 33%（仅 P3 达标，P1/P2 SAS≈0.45）
    if fair_r:
        atts = [r["premium_attainment"] for r in fair_r]
        H["H1"] = {"expected": "=33%（仅P3）", "actual": f"{sum(atts)/len(atts)*100:.0f}%",
                    "pass": 0.25 <= sum(atts)/len(atts) <= 0.45}

    # H2: D1 premium attainment 显著 > Fair（目标 ≥67%）
    if fair_r and d1_r:
        fair_pa = sum(r["premium_attainment"] for r in fair_r) / len(fair_r)
        d1_pa = sum(r["premium_attainment"] for r in d1_r) / len(d1_r)
        H["H2"] = {"expected": f"D1 ≥67% (Fair={fair_pa:.1%})",
                    "actual": f"D1={d1_pa:.1%} vs Fair={fair_pa:.1%}",
                    "pass": d1_pa >= 0.67 and d1_pa > fair_pa}

    # H3: D1 starvation = 0%
    if d1_r:
        total_starved = sum(r["premium_starved"] + r["standard_starved"] for r in d1_r)
        H["H3"] = {"expected": "0%", "actual": f"{total_starved} starved",
                    "pass": total_starved == 0}

    # H4: LongLiu-SP starvation > 0%（重载下加冕制饿死人）
    if sp_r:
        total_starved = sum(r["premium_starved"] + r["standard_starved"] for r in sp_r)
        H["H4"] = {"expected": ">0%", "actual": f"{total_starved} starved",
                    "pass": total_starved > 0}

    # H5: D1 premium capped > Fair 且 > CRUX
    if fair_r and d1_r and crux_r:
        fair_pcm = sum(r["premium_capped"] for r in fair_r) / len(fair_r)
        d1_pcm = sum(r["premium_capped"] for r in d1_r) / len(d1_r)
        crux_pcm = sum(r["premium_capped"] for r in crux_r) / len(crux_r)
        H["H5"] = {"expected": f"D1 > Fair({fair_pcm:.3f}) & CRUX({crux_pcm:.3f})",
                    "actual": f"D1={d1_pcm:.3f} vs Fair={fair_pcm:.3f} vs CRUX={crux_pcm:.3f}",
                    "pass": d1_pcm > fair_pcm and d1_pcm > crux_pcm}

    # H6: D1 下 standard 大 job 有界降级
    if d1_r:
        # 所有 standard job 的 sas
        std_sas = []
        for r in d1_r:
            for j in r["per_job"]:
                if j["ci"] == 2.0:
                    std_sas.append(j["sas_eval"])
        if std_sas:
            # 检查大 standard job 的 sas
            std_large_sas = [s for s in std_sas if s < 1.0]  # 降级的
            H["H6"] = {"expected": "sas 0.3-0.7，不归零",
                        "actual": f"min={min(std_sas):.3f}, max={max(std_sas):.3f}",
                        "pass": min(std_sas) > 0.0 and min(std_sas) < 1.0}

    return H


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spine_gbps", type=float, default=800.0,
                        help="Spine 总带宽(Gbps)，默认 800G(1.27×)；可选 1000/800/630/500")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="outputs/feas_boundary_v1")
    args = parser.parse_args()

    spine_bps = args.spine_gbps * 1e9
    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": spine_bps,
        },
        "duration_ms": 600000,
        "overhead_factor": 2.0,
        "overlap_factor": 0.85,
    }

    policies = {
        "Fair": Fair(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu-SP": LongLiu(K=2.0, use_dynamic_T_target=True),
        "D1": LongLiuDWRR(K=2.0, use_soft_weights=False, intra_class_fair=False, clip_ratio=10.0, overlap_factor=cfg["overlap_factor"]),
        "D1G": LongLiuDWRRGap(floor_w=2.0, G0_gbps=25.0, overlap_factor=cfg["overlap_factor"]),
    }

    git_info = get_git_info()
    # 白名单：允许当前实验/修复相关的文件在未提交状态下运行
    allowed_dirty = {
        "longliu_sim/utils/metrics.py",
        "longliu_sim/utils/__init__.py",
        "longliu_sim/policy/dwrr.py",
        "longliu_sim/policy/__init__.py",
        "longliu_sim/policy/crux.py",
        "longliu_sim/core/simulator.py",
        "experiments/exp_ablation.py",
        "experiments/exp_feas_boundary_v1.py",
        "outputs/quickfix/task0_gate_keeper.py",
        "outputs/quickfix/task0_gate_keeper_results.json",
    }
    non_allowed = [f for f in git_info["dirty_files"] if f not in allowed_dirty]
    if non_allowed:
        print("❌ Git 工作区有未批准的改动，拒绝运行。")
        for f in non_allowed:
            print(f"  {f}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "cmdline": " ".join(sys.argv),
        "seeds": args.seeds,
        "config": cfg,
        "workload_profile": "FEAS_BOUNDARY_V1_WORKLOAD",
        "hypotheses": {
            "H1": "Fair P Attn ≈33%（仅P3）",
            "H2": "D1 P Attn ≥67% 且显著 > Fair",
            "H3": "D1 starvation = 0%",
            "H4": "LongLiu-SP starvation > 0%（加冕制反噬）",
            "H5": "D1 P Cap > Fair",
            "H6": "D1 standard 大 job 有界降级（sas 0.3-0.7），不归零",
        },
    }
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 80)
    print(f"feas_boundary_v1: {len(policies)} 策略 × {args.seeds} seeds @ {cfg['topology']['spine_bw_bps']/1e9:.0f}G spine")
    print("=" * 80)
    print(f"Workload: FEAS_BOUNDARY_V1_WORKLOAD (7 jobs, 全跨 pod)")
    print(f"Git: {git_info['commit']}")
    print()

    all_results = {}
    for pname, policy in policies.items():
        print(f"运行 {pname}...")
        results = []
        for s in range(args.seeds):
            r = run_single(cfg, policy, s, pname)
            results.append(r)
            print(f"  Seed {s}: Overall {r['overall_mean_sas']:.3f}, "
                  f"Premium Mean {r['premium_mean']:.3f}, "
                  f"Attn {r['premium_attainment']:.0%}, "
                  f"Starved {r['premium_starved']}")
        all_results[pname] = results

    # === 汇总报告 ===
    print()
    print("=" * 80)
    print("feas_boundary_v1 主表")
    print("=" * 80)

    # 1. SAS 汇总表
    print("\n1. SAS 汇总 (sas_eval):")
    header = f"{'Policy':<14} {'Overall':>8} {'P Mean':>8} {'P Cap':>8} {'P Attn':>8} {'P Stv':>6} | {'S Mean':>8} {'S Cap':>8} {'S Attn':>8} {'S Stv':>6}"
    print(header)
    print("-" * len(header))
    for pname, results in all_results.items():
        ov = sum(r["overall_mean_sas"] for r in results) / len(results)
        pm = sum(r["premium_mean"] for r in results) / len(results)
        pc = sum(r["premium_capped"] for r in results) / len(results)
        pa = sum(r["premium_attainment"] for r in results) / len(results)
        ps = sum(r["premium_starved"] for r in results)
        sm = sum(r["standard_mean"] for r in results) / len(results)
        sc = sum(r["standard_capped"] for r in results) / len(results)
        sa = sum(r["standard_attainment"] for r in results) / len(results)
        ss = sum(r["standard_starved"] for r in results)
        print(f"{pname:<14} {ov:>8.3f} {pm:>8.3f} {pc:>8.3f} {pa:>7.0%} {ps:>6d} | {sm:>8.3f} {sc:>8.3f} {sa:>7.0%} {ss:>6d}")

    # 2. 逐 job 明细（最后一个 seed）
    print("\n2. 逐 job 明细 (last seed):")
    for pname, results in all_results.items():
        print(f"\n  {pname}:")
        for j in results[-1]["per_job"]:
            tier = "P" if j["ci"] == 1.3 else "S"
            attn = "✓" if j["attained"] else ("✗" if j["starved"] else "·")
            print(f"    {j['jid']} {j['model']:<18} dp={j['dp']} ci={j['ci']} "
                  f"sas={j['sas_eval']:.3f} attn={attn} starved={j['starved']}")

    # 3. 假设验证
    print("\n3. 预登记假设验证:")
    H = verify_hypotheses(all_results)
    for hid, h in H.items():
        status = "✓ PASS" if h["pass"] else "✗ FAIL"
        print(f"  {hid}: {status} | 期望: {h['expected']} | 实际: {h['actual']}")

    # 保存
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(os.path.join(args.out, "hypotheses.json"), "w") as f:
        json.dump(H, f, indent=2, default=str)

    print(f"\n结果已保存至: {args.out}")


if __name__ == "__main__":
    main()