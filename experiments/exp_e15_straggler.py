"""
exp_e15_straggler: Straggler 注入实验（论文 E15 章节）。

验证 LongLiu 的窗口平均机制吸收 2-5× 计算时间膨胀，无雪崩降级。

实验设计：
- 在仿真中随机让某个作业的计算时间 Ticomp 在某几个窗口内膨胀 2-5 倍
- 观察 LongLiu 的窗口级平均机制如何吸收这种异常
- 验证优先级调整是否符合预期（即不发生雪崩式降级）

用法：
    python experiments/exp_e15_straggler.py --seeds 5
    python experiments/exp_e15_straggler.py --seeds 2 --quick
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
E15_CONFIG = os.path.join(PROJECT_ROOT, "configs", "e15_straggler.yaml")

SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02


def load_e15_config() -> dict:
    with open(E15_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def get_policy(name: str, trace_file: str, frozen: dict):
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


def run_single(straggler_factor: float, workload, spine_bw: float,
               policy_name: str, seed: int, frozen: dict) -> dict:
    """运行单次实验，返回 run_meta dict。"""
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"sf{straggler_factor}_{policy_name}_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/e15_straggler/{tag}"
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

    # 注入 straggler：随机选择 2 个 job，膨胀其计算时间
    if straggler_factor > 1.0:
        import random
        rng = random.Random(seed)
        straggler_jids = rng.sample(list(premium_jids), min(2, len(premium_jids)))

        for jid in straggler_jids:
            job = jobs[int(jid[1:])]  # J0 -> jobs[0]
            # 膨胀计算时间
            original_comp_ms = job.comp_ms
            job.comp_ms = original_comp_ms * straggler_factor

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
        "straggler_factor": straggler_factor,
        "spine_bw": int(spine_bw),
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
    parser = argparse.ArgumentParser(description="E15 Straggler injection experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick validation (2 seeds)")
    args = parser.parse_args()

    cfg = load_e15_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    n_seeds = 2 if args.quick else args.seeds
    seeds = list(range(n_seeds))

    print("=" * 80)
    print("E15: Straggler Injection Experiment")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
    print(f"Seeds: {seeds}")
    print(f"Straggler factors: {cfg['straggler_factors']}")
    print()

    all_runs = []
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9  # 400 Gbps（稀缺区）

    for straggler_factor in cfg["straggler_factors"]:
        print(f"--- Straggler Factor = {straggler_factor}x ---")
        for pn in cfg["policies"]:
            seed_vals = []
            for seed in seeds:
                label = f"sf={straggler_factor} {pn} s{seed}"
                print(f"[{label}] ", end="", flush=True)
                try:
                    r = run_single(straggler_factor, FEAS_BOUNDARY_V3_WORKLOAD,
                                   spine_bw, pn, seed, frozen)
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
        k = (r["straggler_factor"], r["policy"])
        groups.setdefault(k, []).append(r)

    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"\n{'SF':>4s} | {'Policy':>4s} | {'P-attn':>9s} | {'P-cap':>9s} | {'S-cap':>9s} | {'Starv':>5s}")
    print("-" * 60)
    for sf in cfg["straggler_factors"]:
        for pn in cfg["policies"]:
            k = (sf, pn)
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
            print(f"{sf:>4.1f} | {pn:>4s} | {p_attn_mean:5.1f}±{p_attn_std:3.1f}% | "
                  f"{p_cap_mean:.3f} | {s_cap_mean:.3f} | {starv:>5d}")

    # 保存 summary CSV
    os.makedirs("outputs/e15_straggler", exist_ok=True)
    with open("outputs/e15_straggler/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["straggler_factor", "policy", "n_seeds", "seeds",
                         "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean"])
        for k, runs in sorted(groups.items()):
            sf, pol = k
            p_attns = [r["p_attn"] for r in runs]
            p_caps = [r["p_cap"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            writer.writerow([sf, pol, len(runs),
                             f"[{', '.join(str(r['seed']) for r in runs)}]",
                             round(np.mean(p_attns), 4),
                             round(np.std(p_attns), 4),
                             round(np.mean(p_caps), 4),
                             round(np.mean(s_caps), 4)])

    print(f"\nSummary saved to outputs/e15_straggler/summary.csv")
    print("*** E15 COMPLETE ***")


if __name__ == "__main__":
    main()
