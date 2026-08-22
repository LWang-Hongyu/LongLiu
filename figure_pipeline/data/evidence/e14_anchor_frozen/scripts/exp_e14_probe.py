"""
exp_e14_probe: 锚点冻结与主动探测实验（论文 E14 章节）。

验证长期高负载下锚点冻结导致性能下降，单次主动探测可恢复。

实验设计：
- 构造长期满载场景（95%+ 负载），注入新作业
- 测试 T_target_ema 冻结在链路容量 B 时的系统表现
- 激活"主动探测"机制（临时提升到 P6 采样一次）
- 展示探测后锚点修正带来的优先级调整

用法：
    python experiments/exp_e14_probe.py --seeds 5
    python experiments/exp_e14_probe.py --seeds 2 --quick
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
import numpy as np

from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.policy.longliu import LongLiu
from longliu_sim.job import Job
from longliu_sim.trace.synthetic import SyntheticTraceLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E14_CONFIG = os.path.join(PROJECT_ROOT, "configs", "e14_probe.yaml")

SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02


def load_e14_config() -> dict:
    with open(E14_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def create_high_load_workload(n_jobs: int, seed: int) -> list:
    """创建高负载 workload（95%+ 利用率）。

    策略：所有 job 都是 premium（tight SLO），制造资源竞争。
    """
    import random
    rng = random.Random(seed)

    workload = []
    # 全部使用大模型 + tight SLO，制造持续拥塞
    models = ["LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"]
    for i in range(n_jobs):
        model = rng.choice(models)
        dp = rng.choice([4, 8])
        ci = rng.choice([1.5, 2.0])  # 全部 tight SLO
        workload.append((model, dp, ci))

    return workload


def run_single(probe_enabled: bool, workload, spine_bw: float,
               seed: int, frozen: dict,
               probe_frozen_threshold: int = 10,
               probe_duration: int = 0,
               ema_passive: bool = False,
               ema_weights: list = None) -> dict:
    """运行单次实验，返回 run_meta dict。"""
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"probe{int(probe_enabled)}_passive{int(ema_passive)}_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/e14_probe/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)

    # 使用 LongLiu 控制环策略（不是 v4 闭式解）
    policy = LongLiu(
        window_size=20,  # 使用滑窗
        use_dynamic_T_target=True,
        ema_passive=ema_passive,  # 被动弱更新（方案1+2）
        ema_weights=ema_weights,  # 信任权重表（level 0-6），None 用默认
    )

    # 如果启用主动探测，设置探测参数
    if probe_enabled:
        policy.probe_enabled = True
        policy.probe_frozen_threshold = probe_frozen_threshold
        policy.probe_duration = probe_duration

    sim = Simulator(topo, policy, duration_ms=600000, seed=seed,
                    overhead_factor=frozen["overhead_factor"],
                    overlap_factor=frozen["overlap_factor"])

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=n_jobs, duration_ms=600000, seed=seed,
        overhead_factor=frozen["overhead_factor"], target_bw_bps=100e9, num_hosts=16,
        workload_profile=list(workload),
    )
    jobs = loader.load()
    for i, j in enumerate(jobs):
        j.jid = f"J{i}"
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    if hasattr(policy, 'flush_trace'):
        policy.flush_trace()

    stats = result.per_job_stats(host_bw_gbps=HOST_BW_GBPS)

    n_premium = len(premium_jids)
    n_premium_attn = 0
    p_cap_total = 0.0
    s_sas_values = []

    for jid, s in stats.items():
        sas = s["sas"]
        if jid in premium_jids:
            if sas >= 1.0 - V4_TOLERANCE:
                n_premium_attn += 1
            p_cap_total += min(sas, 1.0)
        else:
            s_sas_values.append(min(sas, 1.0))

    p_attn = n_premium_attn / n_premium if n_premium > 0 else 1.0
    p_cap = p_cap_total / n_premium if n_premium > 0 else 1.0
    s_cont_cap = np.mean(s_sas_values) if s_sas_values else 0.0

    starv = sum(1 for jid in premium_jids
                if result.jobs[jid].completed_iters == 0)

    # 统计 T_target_ema 冻结情况
    n_frozen_jobs = 0
    for jid in premium_jids:
        job = result.jobs[jid]
        if job.T_target_ema is not None:
            # 检查 EMA 是否冻结在异常值
            expected_comm_ms = job.comm_solo_ms * frozen["overhead_factor"]
            if job.T_target_ema > expected_comm_ms * 2.0:
                n_frozen_jobs += 1

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "probe_enabled": probe_enabled, "spine_bw": int(spine_bw),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_premium": n_premium,
        "n_standard": len(stats) - n_premium,
        "p_attn": round(p_attn, 4),
        "p_cap": round(p_cap, 4),
        "s_cont_cap": round(s_cont_cap, 4),
        "starv": starv,
        "n_frozen_jobs": n_frozen_jobs,
        "total_iters": result.total_iterations(),
    }
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def main():
    parser = argparse.ArgumentParser(description="E14 Anchor freeze + probe experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick validation (2 seeds)")
    args = parser.parse_args()

    cfg = load_e14_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    n_seeds = 2 if args.quick else args.seeds
    seeds = list(range(n_seeds))

    print("=" * 80)
    print("E14: Anchor Freeze + Active Probe Experiment")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
    print(f"Seeds: {seeds}")
    print(f"Probe enabled: {cfg['probe_enabled']}")
    print()

    all_runs = []
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9  # 800 Gbps

    # 创建高负载 workload
    workload = create_high_load_workload(cfg["n_jobs"], seed=42)

    for probe_enabled in cfg["probe_enabled"]:
        print(f"--- Probe Enabled = {probe_enabled} ---")
        seed_vals = []
        for seed in seeds:
            label = f"probe={probe_enabled} s{seed}"
            print(f"[{label}] ", end="", flush=True)
            try:
                r = run_single(probe_enabled, workload, spine_bw, seed, frozen)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            all_runs.append(r)
            seed_vals.append(r["p_attn"])
            print(f"P-attn={r['p_attn']*100:5.1f}%  Frozen jobs={r['n_frozen_jobs']}  starv={r['starv']}")

        if len(seed_vals) == len(seeds):
            mean_p = np.mean(seed_vals) * 100
            std_p = np.std(seed_vals) * 100
            print(f"  → mean±std = {mean_p:.1f}±{std_p:.1f}%")
        print()

    # 聚合
    groups = {}
    for r in all_runs:
        k = r["probe_enabled"]
        groups.setdefault(k, []).append(r)

    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"\n{'Probe':>5s} | {'P-attn':>9s} | {'P-cap':>9s} | {'S-cap':>9s} | {'Frozen':>6s} | {'Starv':>5s}")
    print("-" * 60)
    for probe_enabled in cfg["probe_enabled"]:
        if probe_enabled not in groups:
            continue
        runs = groups[probe_enabled]
        p_attns = [r["p_attn"] for r in runs]
        p_caps = [r["p_cap"] for r in runs]
        s_caps = [r["s_cont_cap"] for r in runs]
        frozen_jobs = [r["n_frozen_jobs"] for r in runs]
        p_attn_mean = np.mean(p_attns) * 100
        p_attn_std = np.std(p_attns) * 100
        p_cap_mean = np.mean(p_caps)
        s_cap_mean = np.mean(s_caps)
        frozen_mean = np.mean(frozen_jobs)
        starv = max(r["starv"] for r in runs)
        print(f"{str(probe_enabled):>5s} | {p_attn_mean:5.1f}±{p_attn_std:3.1f}% | "
              f"{p_cap_mean:.3f} | {s_cap_mean:.3f} | {frozen_mean:>6.1f} | {starv:>5d}")

    # 保存 summary CSV
    os.makedirs("outputs/e14_probe", exist_ok=True)
    with open("outputs/e14_probe/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe_enabled", "n_seeds", "seeds",
                         "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean",
                         "n_frozen_jobs_mean"])
        for probe_enabled in cfg["probe_enabled"]:
            if probe_enabled not in groups:
                continue
            runs = groups[probe_enabled]
            p_attns = [r["p_attn"] for r in runs]
            p_caps = [r["p_cap"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            frozen_jobs = [r["n_frozen_jobs"] for r in runs]
            writer.writerow([probe_enabled, len(runs),
                             f"[{', '.join(str(r['seed']) for r in runs)}]",
                             round(np.mean(p_attns), 4),
                             round(np.std(p_attns), 4),
                             round(np.mean(p_caps), 4),
                             round(np.mean(s_caps), 4),
                             round(np.mean(frozen_jobs), 2)])

    print(f"\nSummary saved to outputs/e14_probe/summary.csv")
    print("*** E14 COMPLETE ***")


if __name__ == "__main__":
    main()
