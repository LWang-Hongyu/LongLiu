"""
exp_e11_overlap: Overlap 敏感性实验（论文 E11 章节）。

验证 LongLiu 基于串行模型推导的带宽分配在不同重叠率下的鲁棒性。
overlap_factor ∈ {0.0, 0.3, 0.5, 0.85, 1.0}，在过渡区带宽（500/630 Gbps）测试。

用法：
    python experiments/exp_e11_overlap.py --seeds 5
    python experiments/exp_e11_overlap.py --seeds 2 --quick   # 快速验证
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
from longliu_sim.policy.fair import Fair
from longliu_sim.policy.crux import CRUX
from longliu_sim.policy.srpt import SRPT
from longliu_sim.policy.dwrr import LongLiuDWRR, LongLiuAllocatorV4
from longliu_sim.trace.synthetic import (
    SyntheticTraceLoader,
    FEAS_BOUNDARY_V3_WORKLOAD,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E11_CONFIG = os.path.join(PROJECT_ROOT, "configs", "e11_overlap.yaml")

SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02


def load_e11_config() -> dict:
    with open(E11_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def get_policy(name: str, trace_file: str, overhead: float, overlap: float):
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


def run_single(scene: str, workload, spine_bw: float, overlap: float,
               policy_name: str, seed: int, frozen: dict) -> dict:
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"{scene}_{policy_name}_{int(spine_bw)}g_ov{overlap}_s{seed}"
    out_dir = f"outputs/e11_overlap/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)
    policy = get_policy(policy_name, trace_file, frozen["overhead_factor"], overlap)
    sim = Simulator(topo, policy, duration_ms=600000, seed=seed,
                    overhead_factor=frozen["overhead_factor"],
                    overlap_factor=overlap)

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
    n_starv = 0

    for jid, s in stats.items():
        sas = s["sas"]
        if jid in premium_jids:
            if sas >= 1.0 - V4_TOLERANCE:
                n_premium_attn += 1
            p_cap_total += min(sas, 1.0)
            if result.jobs[jid].completed_iters == 0:
                n_starv += 1
        else:
            s_sas_values.append(min(sas, 1.0))

    p_attn = n_premium_attn / n_premium if n_premium > 0 else 1.0
    p_cap = p_cap_total / n_premium if n_premium > 0 else 1.0
    s_cont_cap = np.mean(s_sas_values) if s_sas_values else 0.0

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "scene": scene, "spine_bw": int(spine_bw),
        "overlap_factor": overlap,
        "policy": policy_name, "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_premium": n_premium,
        "n_standard": len(stats) - n_premium,
        "p_attn": round(p_attn, 4),
        "p_cap": round(p_cap, 4),
        "s_cont_cap": round(s_cont_cap, 4),
        "starv": n_starv,
        "total_iters": result.total_iterations(),
    }
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def main():
    parser = argparse.ArgumentParser(description="E11 Overlap sensitivity experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick validation (2 seeds)")
    args = parser.parse_args()

    cfg = load_e11_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    n_seeds = 2 if args.quick else args.seeds
    seeds = list(range(n_seeds))

    print("=" * 80)
    print("E11: Overlap Sensitivity Analysis")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
    print(f"Seeds: {seeds}")
    print(f"Overlap factors: {cfg['overlap_factors']}")
    print()

    all_runs = []

    for overlap in cfg["overlap_factors"]:
        print(f"--- Overlap = {overlap} ---")
        for spine_bw in cfg["spine_bw_gbps"]:
            for pn in cfg["policies"]:
                seed_vals = []
                for seed in seeds:
                    label = f"@{spine_bw}G ov={overlap} {pn} s{seed}"
                    print(f"[{label}] ", end="", flush=True)
                    try:
                        r = run_single("E1", FEAS_BOUNDARY_V3_WORKLOAD, spine_bw,
                                       overlap, pn, seed, frozen)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue
                    all_runs.append(r)
                    seed_vals.append(r["p_attn"])
                    print(f"P-attn={r['p_attn']*100:5.1f}%  starv={r['starv']}")

                if len(seed_vals) == len(seeds):
                    mean_p = np.mean(seed_vals) * 100
                    std_p = np.std(seed_vals) * 100
                    print(f"  → mean±std = {mean_p:.1f}±{std_p:.1f}%")
        print()

    # 聚合
    groups = {}
    for r in all_runs:
        k = (r["spine_bw"], r["overlap_factor"], r["policy"])
        groups.setdefault(k, []).append(r)

    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"\n{'BW':>4s} | {'Overlap':>7s} | {'Policy':>4s} | {'P-attn':>9s} | {'Starv':>5s}")
    print("-" * 50)
    for bw in cfg["spine_bw_gbps"]:
        for ov in cfg["overlap_factors"]:
            for pn in cfg["policies"]:
                k = (bw, ov, pn)
                if k not in groups:
                    continue
                runs = groups[k]
                p_attns = [r["p_attn"] for r in runs]
                mean_p = np.mean(p_attns) * 100
                std_p = np.std(p_attns) * 100
                starv = max(r["starv"] for r in runs)
                print(f"{bw:>4d} | {ov:>7.2f} | {pn:>4s} | {mean_p:5.1f}±{std_p:3.1f}% | {starv:>5d}")

    # 保存 summary CSV
    os.makedirs("outputs/e11_overlap", exist_ok=True)
    with open("outputs/e11_overlap/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["spine_bw", "overlap_factor", "policy", "n_seeds",
                         "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean"])
        for k, runs in sorted(groups.items()):
            bw, ov, pol = k
            p_attns = [r["p_attn"] for r in runs]
            p_caps = [r["p_cap"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            writer.writerow([bw, ov, pol, len(runs),
                             round(np.mean(p_attns), 4),
                             round(np.std(p_attns), 4),
                             round(np.mean(p_caps), 4),
                             round(np.mean(s_caps), 4)])

    print(f"\nSummary saved to outputs/e11_overlap/summary.csv")
    print("*** E11 COMPLETE ***")


if __name__ == "__main__":
    main()
