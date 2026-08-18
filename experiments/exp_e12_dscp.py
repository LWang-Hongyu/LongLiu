"""
exp_e12_dscp: DSCP 量化误差实验（论文 E12 章节）。

对比 v4 (闭式解) 与 D1 (DWRR + DSCP 量化) 在不同作业规模下的表现。
验证闭式解相对于量化分配的优越性，特别是在作业数量增加时。

用法：
    python experiments/exp_e12_dscp.py --seeds 5
    python experiments/exp_e12_dscp.py --seeds 2 --quick   # 快速验证
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
from longliu_sim.policy.dwrr import LongLiuAllocatorV4, LongLiuDWRR
from longliu_sim.trace.synthetic import (
    SyntheticTraceLoader,
    FEAS_BOUNDARY_V3_WORKLOAD,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E12_CONFIG = os.path.join(PROJECT_ROOT, "configs", "e12_dscp.yaml")

SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02


def load_e12_config() -> dict:
    with open(E12_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def get_policy(name: str, trace_file: str, frozen: dict):
    overhead = frozen["overhead_factor"]
    overlap = frozen["overlap_factor"]
    if name == "v4":
        return LongLiuAllocatorV4(
            overhead_factor=overhead, overlap_factor=overlap,
            trace_file=trace_file,
        )
    elif name == "D1":
        return LongLiuDWRR(
            overhead_factor=overhead, overlap_factor=overlap,
            trace_file=trace_file,
        )
    raise ValueError(f"Unknown policy: {name}")


def extend_workload(base_workload, target_n_jobs: int, seed: int):
    """扩展 workload 到 target_n_jobs 个 job。

    策略：循环复制基础 workload，调整 ci 分布保持一致性。
    """
    import random
    rng = random.Random(seed)

    if target_n_jobs <= len(base_workload):
        return list(base_workload)[:target_n_jobs]

    extended = list(base_workload)
    while len(extended) < target_n_jobs:
        # 随机选择一个基础 job 并复制
        base_job = rng.choice(base_workload)
        extended.append(base_job)

    return extended[:target_n_jobs]


def run_single(n_jobs: int, workload, spine_bw: float,
               policy_name: str, seed: int, frozen: dict) -> dict:
    """运行单次实验，返回 run_meta dict。"""
    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"n{n_jobs}_{policy_name}_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/e12_dscp/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)
    policy = get_policy(policy_name, trace_file, frozen)
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

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "n_jobs": n_jobs, "spine_bw": int(spine_bw),
        "policy": policy_name, "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_premium": n_premium,
        "n_standard": len(stats) - n_premium,
        "p_attn": round(p_attn, 4),
        "p_cap": round(p_cap, 4),
        "s_cont_cap": round(s_cont_cap, 4),
        "starv": starv,
        "total_iters": result.total_iterations(),
    }
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def main():
    parser = argparse.ArgumentParser(description="E12 DSCP quantization experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick validation (2 seeds)")
    args = parser.parse_args()

    cfg = load_e12_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    n_seeds = 2 if args.quick else args.seeds
    seeds = list(range(n_seeds))

    print("=" * 80)
    print("E12: DSCP Quantization Error Analysis")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
    print(f"Seeds: {seeds}")
    print(f"Job counts: {cfg['n_jobs_list']}")
    print()

    all_runs = []
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9  # 800 Gbps

    for n_jobs in cfg["n_jobs_list"]:
        print(f"--- n_jobs = {n_jobs} ---")
        # 扩展 workload
        workload = extend_workload(FEAS_BOUNDARY_V3_WORKLOAD, n_jobs, seed=42)

        for pn in cfg["policies"]:
            seed_vals = []
            for seed in seeds:
                label = f"n={n_jobs} {pn} s{seed}"
                print(f"[{label}] ", end="", flush=True)
                try:
                    r = run_single(n_jobs, workload, spine_bw, pn, seed, frozen)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
                all_runs.append(r)
                seed_vals.append(r["p_attn"])
                print(f"P-attn={r['p_attn']*100:5.1f}%  P-cap={r['p_cap']:.3f}  starv={r['starv']}")

            if len(seed_vals) == len(seeds):
                mean_p = np.mean(seed_vals) * 100
                std_p = np.std(seed_vals) * 100
                print(f"  → mean±std = {mean_p:.1f}±{std_p:.1f}%")
        print()

    # 聚合
    groups = {}
    for r in all_runs:
        k = (r["n_jobs"], r["policy"])
        groups.setdefault(k, []).append(r)

    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"\n{'N_jobs':>6s} | {'Policy':>18s} | {'P-attn':>9s} | {'P-cap':>9s} | {'S-cap':>9s} | {'Starv':>5s}")
    print("-" * 80)
    for n_jobs in cfg["n_jobs_list"]:
        for pn in cfg["policies"]:
            k = (n_jobs, pn)
            if k not in groups:
                continue
            runs = groups[k]
            p_attns = [r["p_attn"] for r in runs]
            p_caps = [r["p_cap"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            p_attn_mean = np.mean(p_attns) * 100
            p_attn_std = np.std(p_attns) * 100
            p_cap_mean = np.mean(p_caps)
            s_cap_mean = np.mean(s_caps)
            starv = max(r["starv"] for r in runs)
            print(f"{n_jobs:>6d} | {pn:>18s} | {p_attn_mean:5.1f}±{p_attn_std:3.1f}% | "
                  f"{p_cap_mean:.3f} | {s_cap_mean:.3f} | {starv:>5d}")

    # 保存 summary CSV
    os.makedirs("outputs/e12_dscp", exist_ok=True)
    with open("outputs/e12_dscp/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_jobs", "policy", "n_seeds", "seeds",
                         "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean"])
        for k, runs in sorted(groups.items()):
            n_jobs, pol = k
            p_attns = [r["p_attn"] for r in runs]
            p_caps = [r["p_cap"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            writer.writerow([n_jobs, pol, len(runs),
                             f"[{', '.join(str(r['seed']) for r in runs)}]",
                             round(np.mean(p_attns), 4),
                             round(np.std(p_attns), 4),
                             round(np.mean(p_caps), 4),
                             round(np.mean(s_caps), 4)])

    print(f"\nSummary saved to outputs/e12_dscp/summary.csv")
    print("*** E12 COMPLETE ***")


if __name__ == "__main__":
    main()
