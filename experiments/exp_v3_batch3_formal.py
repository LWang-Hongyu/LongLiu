"""
exp_v3_batch3_formal: feas_boundary_v3 3 seeds 正式批（矩阵 v2.1）

配置点：E1×6 + E2'×4 + E2-pro×2 = 12 点
策略：5 (Fair/CRUX/SP/D1/v4)
种子：3 (0/1/2)
= 180 run

判定（矩阵 v2.1）：
  v4保障：三场景 @800G mean P-attn ≥ 0.98
  P1a: E2' @630G CRUX mean P-attn < v4 mean by ≥ 10pp
  P1b: E2' @500G CRUX P-cap ≪ v4 P-cap
  P2: E1 @400/500G v4 mean P-attn ≥ D1 mean
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy.fair import Fair
from longliu_sim.policy.crux import CRUX
from longliu_sim.policy.srpt import SRPT
from longliu_sim.policy.dwrr import LongLiuDWRR, LongLiuAllocatorV4
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import (
    FEAS_BOUNDARY_V3_WORKLOAD,
    FEAS_BOUNDARY_V3_PRIME_WORKLOAD,
    FEAS_BOUNDARY_V3_PRO_WORKLOAD,
)
from longliu_sim.utils.config import load_config

_cfg = load_config()
SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
OVERLAP = _cfg["frozen"]["overlap_factor"]
OVERHEAD = _cfg["frozen"]["overhead_factor"]
K = _cfg["frozen"]["K"]

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")) as f:
    CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

V4_TOLERANCE = 0.02
P1A_GAP = 0.10  # 10pp
P1B_V4_PCAP = 0.60  # v4 expected P-cap at 500G
P1B_CRUX_PCAP_MAX = 0.35  # CRUX expected max P-cap

POLICIES = ["Fair", "CRUX", "SP", "D1", "v4"]
SEEDS = [0, 1, 2]

SCENARIOS = [
    ("E1", FEAS_BOUNDARY_V3_WORKLOAD, [400, 500, 630, 800, 1000, 1200]),
    ("E2'", FEAS_BOUNDARY_V3_PRIME_WORKLOAD, [400, 500, 630, 800]),
    ("E2-pro", FEAS_BOUNDARY_V3_PRO_WORKLOAD, [630, 800]),
]


def get_policy(name: str, trace_file: str):
    if name == "Fair":
        return Fair()
    elif name == "CRUX":
        return CRUX()
    elif name == "SP":
        return SRPT()
    elif name == "D1":
        return LongLiuDWRR(K=K, overlap_factor=OVERLAP,
                           overhead_factor=OVERHEAD, trace_file=trace_file)
    elif name == "v4":
        return LongLiuAllocatorV4(overhead_factor=OVERHEAD,
                                  overlap_factor=OVERLAP, trace_file=trace_file)
    raise ValueError(name)


def run_single(scene: str, workload, spine_bw: float,
               policy_name: str, seed: int) -> dict:
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"{scene}_{policy_name}_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/v3_batch3_formal/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)
    policy = get_policy(policy_name, trace_file)
    sim = Simulator(topo, policy, duration_ms=600000, seed=seed,
                    overhead_factor=OVERHEAD, overlap_factor=OVERLAP)

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=n_jobs, duration_ms=600000, seed=seed,
        overhead_factor=OVERHEAD, target_bw_bps=100e9, num_hosts=16,
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
    # P-cap: proportion of premium bandwidth demand satisfied
    # (approximate: what fraction of premium attain_bw is achieved on average)
    p_cap_total = 0.0
    s_sas_values = []
    sas_per_premium = []

    for jid, s in stats.items():
        sas = s["sas"]
        if jid in premium_jids:
            if sas >= 1.0 - V4_TOLERANCE:
                n_premium_attn += 1
            p_cap_total += min(sas, 1.0)
            sas_per_premium.append(min(sas, 1.0))
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
        "scene": scene, "spine_bw": int(spine_bw),
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


# ---------- aggregation ----------

ResultKey = Tuple[str, int, str]  # (scene, spine_bw, policy)

def aggregate(all_runs: List[dict]) -> Dict[ResultKey, dict]:
    groups = defaultdict(list)
    for r in all_runs:
        k = (r["scene"], r["spine_bw"], r["policy"])
        groups[k].append(r)
    agg = {}
    for k, runs in groups.items():
        p_attns = [r["p_attn"] for r in runs]
        p_caps = [r["p_cap"] for r in runs]
        s_caps = [r["s_cont_cap"] for r in runs]
        agg[k] = {
            "scene": k[0], "spine_bw": k[1], "policy": k[2],
            "n_seeds": len(runs),
            "p_attn_mean": round(np.mean(p_attns), 4),
            "p_attn_std": round(np.std(p_attns), 4),
            "p_cap_mean": round(np.mean(p_caps), 4),
            "p_cap_std": round(np.std(p_caps), 4),
            "s_cont_cap_mean": round(np.mean(s_caps), 4),
            "s_cont_cap_std": round(np.std(s_caps), 4),
            "p_attn_each": [round(v, 4) for v in p_attns],
            "p_cap_each": [round(v, 4) for v in p_caps],
            "starv": max(r["starv"] for r in runs),
        }
    return agg


def verify(agg: Dict[ResultKey, dict]) -> list[str]:
    failures = []

    # v4 guarantee: @800G P-attn ≥ 0.98
    for scene in ["E1", "E2'", "E2-pro"]:
        k = (scene, 800, "v4")
        if k not in agg:
            failures.append(f"[v4保障] {scene} @800G: no data")
            continue
        if agg[k]["p_attn_mean"] < 1.0 - V4_TOLERANCE:
            failures.append(
                f"[v4保障 FAIL] {scene} @800G: "
                f"mean P-attn={agg[k]['p_attn_mean']*100:.1f}% < 98%"
            )

    # P1a: E2' @630G CRUX P-attn < v4 by ≥10pp
    k_v4 = ("E2'", 630, "v4")
    k_crux = ("E2'", 630, "CRUX")
    if k_v4 in agg and k_crux in agg:
        v4_a = agg[k_v4]["p_attn_mean"]
        crux_a = agg[k_crux]["p_attn_mean"]
        diff = v4_a - crux_a
        if diff < P1A_GAP:
            failures.append(
                f"[P1a FAIL] E2' @630G: CRUX mean P-attn={crux_a*100:.1f}% "
                f"vs v4={v4_a*100:.1f}% (diff={diff*100:.1f}pp < 10pp)"
            )
    else:
        failures.append("[P1a] E2' @630G: missing data")

    # P1b: E2' @500G CRUX P-cap ≪ v4
    k_v4_500 = ("E2'", 500, "v4")
    k_crux_500 = ("E2'", 500, "CRUX")
    if k_v4_500 in agg and k_crux_500 in agg:
        v4_pcap = agg[k_v4_500]["p_cap_mean"]
        crux_pcap = agg[k_crux_500]["p_cap_mean"]
        if not (crux_pcap <= P1B_CRUX_PCAP_MAX):
            failures.append(
                f"[P1b FAIL] E2' @500G: CRUX mean P-cap={crux_pcap:.3f} "
                f"> {P1B_CRUX_PCAP_MAX} (expected ≤0.35)"
            )
        if not (v4_pcap >= P1B_V4_PCAP - 0.05):
            failures.append(
                f"[P1b FAIL] E2' @500G: v4 mean P-cap={v4_pcap:.3f} "
                f"< {P1B_V4_PCAP - 0.05} (expected ~0.60)"
            )
    else:
        failures.append("[P1b] E2' @500G: missing data")

    # P2: E1 @400/500G v4 mean ≥ D1 mean
    for bw in [400, 500]:
        k_v4_p2 = ("E1", bw, "v4")
        k_d1_p2 = ("E1", bw, "D1")
        if k_v4_p2 in agg and k_d1_p2 in agg:
            v4_a = agg[k_v4_p2]["p_attn_mean"]
            d1_a = agg[k_d1_p2]["p_attn_mean"]
            if v4_a < d1_a - 0.01:
                failures.append(
                    f"[P2 FAIL] E1 @{bw}G: v4={v4_a*100:.1f}% < "
                    f"D1={d1_a*100:.1f}%"
                )
        else:
            failures.append(f"[P2] E1 @{bw}G: missing data")

    return failures


def main():
    print("=" * 80)
    print("feas_boundary_v3 Batch 3 — 3 seeds Formal (Matrix v2.1)")
    print("=" * 80)
    n_configs = sum(len(pts) for _, _, pts in SCENARIOS)
    n_total = n_configs * len(POLICIES) * len(SEEDS)
    print(f"{n_configs} configs × {len(POLICIES)} policies × {len(SEEDS)} seeds = {n_total} runs")
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
    print()

    all_runs = []

    for scene_name, workload, spine_pts in SCENARIOS:
        for spine_bw in spine_pts:
            for pn in POLICIES:
                seed_vals = []
                for seed in SEEDS:
                    label = f"{scene_name} @{spine_bw}G {pn} s{seed}"
                    print(f"[{label}] ", end="", flush=True)
                    try:
                        r = run_single(scene_name, workload, spine_bw, pn, seed)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue
                    all_runs.append(r)
                    seed_vals.append(r["p_attn"])
                    print(f"P-attn={r['p_attn']*100:5.1f}%  "
                          f"P-cap={r['p_cap']:.3f}  starv={r['starv']}")

                # after 3 seeds, print mean
                if len(seed_vals) == len(SEEDS):
                    mean_p = np.mean(seed_vals) * 100
                    std_p = np.std(seed_vals) * 100
                    flag = " *** HIGH VAR ***" if std_p > 5.0 else ""
                    print(f"  → mean±std = {mean_p:.1f}±{std_p:.1f}%{flag}")

    # Aggregate
    agg = aggregate(all_runs)

    # ---- Output ----
    print()
    print("=" * 80)
    print("3-Seed Summary Table")
    print("=" * 80)

    for scene_name, _, spine_pts in SCENARIOS:
        print(f"\n--- {scene_name} ---")
        header = f"{'Spine':>6s} | {'Policy':>4s} | {'P-attn':>9s} | {'P-cap':>9s} | {'S-cap':>9s} | {'Starv':>5s}"
        print(header)
        print("-" * len(header))
        for spine_bw in spine_pts:
            for pn in POLICIES:
                k = (scene_name, spine_bw, pn)
                if k not in agg:
                    continue
                a = agg[k]
                p_attn_str = f"{a['p_attn_mean']*100:5.1f}±{a['p_attn_std']*100:3.1f}%"
                p_cap_str = f"{a['p_cap_mean']:.3f}±{a['p_cap_std']:.3f}"
                s_cap_str = f"{a['s_cont_cap_mean']:.3f}±{a['s_cont_cap_std']:.3f}"
                print(f"{spine_bw:>6d} | {pn:>4s} | {p_attn_str:>9s} | "
                      f"{p_cap_str:>9s} | {s_cap_str:>9s} | {a['starv']:>5d}")

    # Verification
    print()
    print("=" * 80)
    print("Verification (Matrix v2.1)")
    print("=" * 80)
    failures = verify(agg)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print("\n*** BATCH FAILED — stopping ***")
        sys.exit(1)
    else:
        print("  ✓ All checks passed (v4 guarantee + P1a + P1b + P2)")

    # Save
    out_path = "outputs/v3_batch3_formal/summary.json"
    os.makedirs("outputs/v3_batch3_formal", exist_ok=True)
    serializable = {}
    for k, v in agg.items():
        key_str = f"{k[0]}_{k[1]}_{k[2]}"
        serializable[key_str] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSummary saved to {out_path}")
    print("*** BATCH PASSED ***")


if __name__ == "__main__":
    main()
