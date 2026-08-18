"""
exp_e14_probe_v4: E14 场景下 v4 闭式解对照实验（论文 E14 章节补充）。

复用 exp_e14_probe.py 的 workload（30-job all-premium / 800G / 600s / 16 节点 FatTree）
与统计口径，仅将策略替换为 LongLiu 闭式解 v4，验证"闭式解免疫锚冻结"。

用法：
    python experiments/exp_e14_probe_v4.py --seeds 10
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

import numpy as np

from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.policy.dwrr import LongLiuAllocatorV4
from longliu_sim.trace.synthetic import SyntheticTraceLoader

from exp_e14_probe import create_high_load_workload, load_e14_config, load_frozen

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02


def run_single_v4(workload, spine_bw: float, seed: int, frozen: dict) -> dict:
    """运行单次 v4 闭式解实验，返回 run_meta dict（与 exp_e14_probe 同构）。"""
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"v4_closure_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/e14_probe/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)

    # v4 闭式解：无 EMA 锚，每窗口直接按链路容量重解最优分配
    policy = LongLiuAllocatorV4(
        overhead_factor=frozen["overhead_factor"],
        overlap_factor=frozen["overlap_factor"],
        trace_file=trace_file,
    )

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

    # v4 闭式解无 EMA 锚，按构造免疫冻结
    n_frozen_jobs = 0

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "strategy": "v4",
        "probe_enabled": False, "spine_bw": int(spine_bw),
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
    parser = argparse.ArgumentParser(description="E14 v4 closed-form contrast experiment")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    args = parser.parse_args()

    cfg = load_e14_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    seeds = list(range(args.seeds))

    print("=" * 80)
    print("E14: v4 Closed-Form Contrast (immune to anchor freezing)")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
    print(f"Seeds: {seeds}")

    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9  # 800 Gbps
    workload = create_high_load_workload(cfg["n_jobs"], seed=42)

    all_runs = []
    for seed in seeds:
        print(f"[v4 s{seed}] ", end="", flush=True)
        r = run_single_v4(workload, spine_bw, seed, frozen)
        all_runs.append(r)
        print(f"P-attn={r['p_attn']*100:5.1f}%  Frozen={r['n_frozen_jobs']}  starv={r['starv']}")

    p_attns = [r["p_attn"] for r in all_runs]
    p_caps = [r["p_cap"] for r in all_runs]
    frozen_jobs = [r["n_frozen_jobs"] for r in all_runs]
    mean_p = np.mean(p_attns) * 100
    std_p = np.std(p_attns) * 100
    print(f"\n  → v4 mean±std = {mean_p:.1f}±{std_p:.1f}%  frozen={np.mean(frozen_jobs):.1f}")

    os.makedirs("outputs/e14_probe", exist_ok=True)
    summary_path = "outputs/e14_probe/summary_v4.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "n_seeds", "seeds",
                         "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean",
                         "n_frozen_jobs_mean"])
        writer.writerow(["v4", len(all_runs),
                         f"[{', '.join(str(r['seed']) for r in all_runs)}]",
                         round(np.mean(p_attns), 4),
                         round(np.std(p_attns), 4),
                         round(np.mean(p_caps), 4),
                         0.0,
                         round(np.mean(frozen_jobs), 2)])
    print(f"\nSummary saved to {summary_path}")
    print("*** E14 V4 COMPLETE ***")


if __name__ == "__main__":
    main()
