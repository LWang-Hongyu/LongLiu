"""
exp_e16_beta: β（standard 降级界限）敏感性实验。

β ∈ {0.3, 0.5, 0.7, 1.0}，E1 工作负载（14 作业，8P/6S），500/630 Gbps。
指标：P-attn、S-cont（standard-tier 平均 SAS）、total utilization、starved count。

用法：
    python experiments/exp_e16_beta.py --seeds 5
    python experiments/exp_e16_beta.py --seeds 2 --quick
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

from longliu_sim.network import FatTreeTopology
from longliu_sim.policy.dwrr import LongLiuAllocatorV4
from longliu_sim.trace.synthetic import (
    SyntheticTraceLoader,
    FEAS_BOUNDARY_V3_WORKLOAD,
)
from experiments._spine_probe import BwProbeSimulator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E16_CONFIG = os.path.join(PROJECT_ROOT, "configs", "e16_beta.yaml")

SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02
DURATION_MS = 600000.0


def load_config() -> dict:
    with open(E16_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def run_single(scene: str, workload, spine_bw: float, beta: float,
               seed: int, frozen: dict) -> dict:
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"{scene}_b{beta}_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/e16_beta/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)
    policy = LongLiuAllocatorV4(
        overhead_factor=frozen["overhead_factor"],
        overlap_factor=frozen["overlap_factor"],
        trace_file=f"{out_dir}/trace.jsonl",
    )
    policy.BETA = beta  # 实例属性覆盖类属性，无需改动 longliu_sim

    sim = BwProbeSimulator(topo, policy, duration_ms=DURATION_MS, seed=seed,
                           overhead_factor=frozen["overhead_factor"],
                           overlap_factor=frozen["overlap_factor"])

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=n_jobs, duration_ms=DURATION_MS, seed=seed,
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
    n_starv = 0

    for jid, s in stats.items():
        sas = s["sas"]
        if jid in premium_jids:
            if sas >= 1.0 - V4_TOLERANCE:
                n_premium_attn += 1
            p_cap_total += min(sas, 1.0)
            if s["completed_iters"] == 0:
                n_starv += 1
        else:
            s_sas_values.append(min(sas, 1.0))

    p_attn = n_premium_attn / n_premium if n_premium > 0 else 1.0
    p_cap = p_cap_total / n_premium if n_premium > 0 else 1.0
    s_cont_cap = np.mean(s_sas_values) if s_sas_values else 0.0

    # 总链路利用率（per-spine 物理 cap，v4 超分配不会使单链路利用率 >100%）
    total_util = sim.time_avg_spine_util(DURATION_MS)

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "scene": scene, "spine_bw": int(spine_bw),
        "beta": beta,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_premium": n_premium,
        "n_standard": len(stats) - n_premium,
        "p_attn": round(p_attn, 4),
        "p_cap": round(p_cap, 4),
        "s_cont_cap": round(s_cont_cap, 4),
        "total_util": round(total_util, 4),
        "starv": n_starv,
        "total_iters": result.total_iterations(),
    }
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def main():
    parser = argparse.ArgumentParser(description="E16 beta sensitivity experiment")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    n_seeds = 2 if args.quick else args.seeds
    seeds = list(range(n_seeds))

    print("=" * 80)
    print("E16: Beta Sensitivity (standard floor)")
    print("=" * 80)
    print(f"Seeds: {seeds}  Beta: {cfg['beta_values']}")
    print()

    all_runs = []
    for beta in cfg["beta_values"]:
        for spine_bw in cfg["spine_bw_gbps"]:
            seed_vals = []
            for seed in seeds:
                label = f"@{spine_bw}G beta={beta} s{seed}"
                print(f"[{label}] ", end="", flush=True)
                try:
                    r = run_single("E1", FEAS_BOUNDARY_V3_WORKLOAD, spine_bw,
                                   beta, seed, frozen)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
                all_runs.append(r)
                seed_vals.append(r["p_attn"])
                print(f"P-attn={r['p_attn']*100:5.1f}%  S-cont={r['s_cont_cap']:.3f}  "
                      f"util={r['total_util']*100:4.1f}%  starv={r['starv']}")
            if len(seed_vals) == len(seeds):
                print(f"  → mean±std = {np.mean(seed_vals)*100:.1f}±{np.std(seed_vals)*100:.1f}%")
        print()

    groups = {}
    for r in all_runs:
        k = (r["spine_bw"], r["beta"])
        groups.setdefault(k, []).append(r)

    os.makedirs("outputs/e16_beta", exist_ok=True)
    with open("outputs/e16_beta/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["spine_bw", "beta", "n_seeds",
                         "p_attn_mean", "p_attn_std",
                         "s_cont_cap_mean", "total_util_mean", "starv_max"])
        for k, runs in sorted(groups.items()):
            bw, beta = k
            p_attns = [r["p_attn"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            utils = [r["total_util"] for r in runs]
            starv = max(r["starv"] for r in runs)
            writer.writerow([bw, beta, len(runs),
                             round(np.mean(p_attns), 4),
                             round(np.std(p_attns), 4),
                             round(np.mean(s_caps), 4),
                             round(np.mean(utils), 4),
                             starv])

    print("Summary saved to outputs/e16_beta/summary.csv")
    print("*** E16 COMPLETE ***")


if __name__ == "__main__":
    main()
