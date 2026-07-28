#!/usr/bin/env python3
"""D1G G0/floor_w 参数扫描 @1.21× load。

扫描范围：
  - G0: 5, 10, 25, 50 Gbps
  - floor_w: 1, 2

输出：outputs/d1g_scan/
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import LongLiuDWRRGap
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V1_WORKLOAD
from longliu_sim.utils import compute_sas_eval


HOST_BW_GBPS = 100.0


def run_single(cfg, policy, seed, pname):
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
    sim = Simulator(topo, policy, duration_ms=cfg["duration_ms"],
                    seed=seed, overhead_factor=cfg["overhead_factor"],
                    overlap_factor=cfg["overlap_factor"])
    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=7, duration_ms=cfg["duration_ms"], seed=seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16, workload_profile=FEAS_BOUNDARY_V1_WORKLOAD)
    for j in loader.load():
        sim.submit(j)
    result = sim.run()

    stats = result.per_job_stats(host_bw_gbps=HOST_BW_GBPS)
    per_job = []
    premium_sas = []
    standard_sas = []
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        avg_iter_ms = s["avg_iter_ms"]
        sas_eval = compute_sas_eval(
            avg_iter_ms, job.model, job.num_workers, ci,
            host_bw_gbps=HOST_BW_GBPS, overlap_factor=cfg["overlap_factor"])
        attained = sas_eval >= 1.0
        completed = job.completed_iters
        per_job.append({"jid": jid, "model": job.model, "dp": job.num_workers,
                        "ci": ci, "sas_eval": sas_eval, "attained": attained,
                        "starved": completed == 0})
        if ci == 1.3:
            premium_sas.append(sas_eval)
        elif ci == 2.0:
            standard_sas.append(sas_eval)

    p_mean = mean(premium_sas) if premium_sas else 0.0
    p_cap = mean(min(s, 1.0) for s in premium_sas) if premium_sas else 0.0
    p_attn = sum(1 for j in per_job if j["ci"] == 1.3 and j["attained"]) / \
             max(1, sum(1 for j in per_job if j["ci"] == 1.3))
    p_stv = sum(1 for j in per_job if j["ci"] == 1.3 and j["starved"])
    s_mean = mean(standard_sas) if standard_sas else 0.0
    s_cap = mean(min(s, 1.0) for s in standard_sas) if standard_sas else 0.0
    s_attn = sum(1 for j in per_job if j["ci"] == 2.0 and j["attained"]) / \
             max(1, sum(1 for j in per_job if j["ci"] == 2.0))
    s_stv = sum(1 for j in per_job if j["ci"] == 2.0 and j["starved"])
    overall_mean = mean(premium_sas + standard_sas) if (premium_sas + standard_sas) else 0.0
    return {
        "premium_mean": p_mean, "premium_capped": p_cap,
        "premium_attainment": p_attn, "premium_starved": p_stv,
        "standard_mean": s_mean, "standard_capped": s_cap,
        "standard_attainment": s_attn, "standard_starved": s_stv,
        "overall_mean_sas": overall_mean,
        "per_job": per_job,
    }


def main():
    out_dir = "outputs/d1g_scan"
    os.makedirs(out_dir, exist_ok=True)

    cfg = {
        "topology": {"type": "fatree", "k": 4,
                     "host_bw_bps": 100e9, "spine_bw_bps": 800e9},
        "duration_ms": 600000, "overhead_factor": 2.0, "overlap_factor": 0.85,
    }

    scan_params = []
    for G0 in [5, 10, 25, 50]:
        for fw in [1, 2]:
            scan_params.append((G0, fw))

    print(f"D1G 参数扫描: {len(scan_params)} 组合 × 3 seeds @ 800G spine")
    print(f"{'G0':>4} {'floor':>5} | {'P Mean':>7} {'P Cap':>6} {'P Attn':>6} {'S Cap':>6}")
    print("-" * 55)

    all_results = {}
    for G0, fw in scan_params:
        p_means, p_caps, p_attns, s_caps = [], [], [], []
        pname = f"D1G_G{G0}_F{fw}"
        for s in range(3):
            policy = LongLiuDWRRGap(
                floor_w=float(fw), G0_gbps=float(G0),
                overlap_factor=cfg["overlap_factor"])
            r = run_single(cfg, policy, s, pname)
            p_means.append(r["premium_mean"])
            p_caps.append(r["premium_capped"])
            p_attns.append(r["premium_attainment"])
            s_caps.append(r["standard_capped"])
        all_results[pname] = {
            "G0": G0, "floor_w": fw,
            "p_mean": mean(p_means), "p_cap": mean(p_caps),
            "p_attn": mean(p_attns), "s_cap": mean(s_caps),
        }
        print(f"{G0:>4} {fw:>5} | {mean(p_means):.3f}  {mean(p_caps):.3f}  "
              f"{mean(p_attns):.0%}  {mean(s_caps):.3f}")

    # Save results
    out_path = os.path.join(out_dir, "scan_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print best config
    best = max(all_results.values(), key=lambda x: x["p_cap"])
    print(f"\nBest config: G0={best['G0']}, floor_w={best['floor_w']} → P-cap={best['p_cap']:.3f}")

    # Comparison table
    print("\n=== D1G 参数扫描汇总 ===")
    print(f"{'G0':>4} {'floor':>5} | {'P Mean':>7} {'P Cap':>6} {'P Attn':>6} {'S Cap':>6}")
    print("-" * 55)
    for G0, fw in scan_params:
        pname = f"D1G_G{G0}_F{fw}"
        r = all_results[pname]
        print(f"{G0:>4} {fw:>5} | {r['p_mean']:.3f}  {r['p_cap']:.3f}  "
              f"{r['p_attn']:.0%}  {r['s_cap']:.3f}")
    print()
    print(f"Best: D1G G0={best['G0']} floor_w={best['floor_w']} (P-cap={best['p_cap']:.3f})")
    print(f"Baseline D1: P-cap=0.585 (from feas_boundary_v1_800g_d1g_3seeds)")
    print(f"Baseline CRUX: P-cap=0.605")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
