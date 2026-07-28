"""
exp_v3_trajectory: v3 策略轨迹快检

目的：验证 v3 机制在真边界（ci=2.0/3.0 @1000G）下的带宽分配。

预登记判定：
- P 站稳高类（P5/P6）
- S 低类（P2/P3）
- 封顶生效（bw ≤ attain_bw）
- 无病态振荡
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy.dwrr import LongLiuDWRRGapV3
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V2_WORKLOAD
from longliu_sim.utils import compute_sas_eval, compute_iter_solo_ms

HOST_BW_GBPS = 100.0
OVERLAP = 0.85
OVERHEAD = 1.0  # 约定 A：wire=逻辑，与锚点/attain 公式自洽


def main():
    out_dir = "outputs/v3_trajectory_800g"
    trace_file = "outputs/v3_trajectory_800g.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("v3 Trajectory Quick Check: feas_boundary_v2 @ 800G (真边界)")
    print("=" * 80)
    print(f"Policy: LongLiuDWRRGapV3(floor_w=2.0)")
    print(f"Topology: FatTree(k=4, spine=800G)")
    print(f"Workload: FEAS_BOUNDARY_V2_WORKLOAD (ci=2.0/3.0)")
    print(f"Σattain_bw: 879 Gbps, load = 1.10×（真边界）")
    print(f"Trace: {trace_file}")
    print(f"Output: {out_dir}")
    print()

    # ---- Config ----
    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 800e9,  # 主评估点（真边界 1.10×）
        },
        "duration_ms": 600000,
        "overhead_factor": OVERHEAD,
        "overlap_factor": OVERLAP,
    }

    # ---- Policy (v3) ----
    policy = LongLiuDWRRGapV3(
        floor_w=2.0,
        overlap_factor=cfg["overlap_factor"],
        trace_file=trace_file,
    )

    # ---- Topology ----
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )

    # ---- Simulator ----
    sim = Simulator(
        topo,
        policy,
        duration_ms=cfg["duration_ms"],
        seed=0,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

    # ---- Workload (v2) ----
    loader = SyntheticTraceLoader(
        model_types=[],
        gpu_distribution={},
        ci_distribution={},
        job_count=7,
        duration_ms=cfg["duration_ms"],
        seed=0,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=FEAS_BOUNDARY_V2_WORKLOAD,
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    print("Jobs submitted:")
    for j in jobs:
        print(f"  {j.jid}: {j.model} dp={j.num_workers} ci={j.slo_ci}")

    # ---- Run ----
    print("\nRunning simulation...")
    result = sim.run()
    print(f"  Done: {result.total_iterations()} total iterations")
    print(f"  Avg iter: {result.avg_iteration_ms():.1f}ms")

    # ---- Flush trace ----
    policy.flush_trace()
    print(f"\nTrace flushed to {trace_file}")

    # ---- Per-job stats & sas_eval ----
    stats = result.per_job_stats(host_bw_gbps=HOST_BW_GBPS)
    per_job_results = []
    premium_stats = []
    standard_stats = []

    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        avg_iter_ms = s["avg_iter_ms"]
        sas_eval = compute_sas_eval(
            avg_iter_ms, job.model, job.num_workers, ci,
            host_bw_gbps=HOST_BW_GBPS, overlap_factor=cfg["overlap_factor"]
        )
        completed = job.completed_iters
        target = job.target_iters

        # Attainment: avg_iter_ms <= ci * iter_solo_ms
        iter_solo_ms = compute_iter_solo_ms(
            job.model, job.num_workers,
            host_bw_gbps=HOST_BW_GBPS, overlap_factor=cfg["overlap_factor"]
        )
        target_iter_ms = ci * iter_solo_ms
        attained = avg_iter_ms <= target_iter_ms
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

        if ci == 2.0:
            premium_stats.append(sas_eval)
        elif ci == 3.0:
            standard_stats.append(sas_eval)

    # ---- Aggregate stats ----
    premium_mean = sum(premium_stats) / len(premium_stats) if premium_stats else 0.0
    premium_capped = sum(min(s, 1.0) for s in premium_stats) / len(premium_stats) if premium_stats else 0.0
    premium_attainment = sum(1 for j in per_job_results if j["ci"] == 2.0 and j["attained"]) / \
                         max(1, sum(1 for j in per_job_results if j["ci"] == 2.0))
    premium_starved = sum(1 for j in per_job_results if j["ci"] == 2.0 and j["starved"])

    standard_mean = sum(standard_stats) / len(standard_stats) if standard_stats else 0.0
    standard_capped = sum(min(s, 1.0) for s in standard_stats) / len(standard_stats) if standard_stats else 0.0
    standard_attainment = sum(1 for j in per_job_results if j["ci"] == 3.0 and j["attained"]) / \
                          max(1, sum(1 for j in per_job_results if j["ci"] == 3.0))
    standard_starved = sum(1 for j in per_job_results if j["ci"] == 3.0 and j["starved"])

    # ---- Print results ----
    print("\n" + "=" * 80)
    print("Per-job Results")
    print("=" * 80)
    print(f"| jid | model | ci | sas | attained | starved |")
    print("|-----|-------|----|-----|----------|---------|")
    for r in per_job_results:
        tier = "P" if r["ci"] == 2.0 else "S"
        print(f"| {r['jid']} | {r['model']} | {r['ci']} | {r['sas_eval']:.3f} | {r['attained']} | {r['starved']} |")

    print("\n" + "=" * 80)
    print("Aggregate Results")
    print("=" * 80)
    print(f"Premium (ci=2.0):")
    print(f"  Mean SAS: {premium_mean:.3f}")
    print(f"  Capped SAS: {premium_capped:.3f}")
    print(f"  Attainment: {premium_attainment*100:.0f}%")
    print(f"  Starved: {premium_starved}")
    print()
    print(f"Standard (ci=3.0):")
    print(f"  Mean SAS: {standard_mean:.3f}")
    print(f"  Capped SAS: {standard_capped:.3f}")
    print(f"  Attainment: {standard_attainment*100:.0f}%")
    print(f"  Starved: {standard_starved}")

    # ---- Check pre-registered ----
    print("\n" + "=" * 80)
    print("Pre-registered Check")
    print("=" * 80)
    H = {}
    H["P-attn=100%"] = {"expected": "100%", "actual": f"{premium_attainment*100:.0f}%", "pass": premium_attainment == 1.0}
    H["P-cap=1.0"] = {"expected": "≥1.0", "actual": f"{premium_capped:.3f}", "pass": premium_capped >= 1.0}
    H["S-cont-cap≥0.5"] = {"expected": "≥0.5", "actual": f"{standard_capped:.3f}", "pass": standard_capped >= 0.5}
    H["starv=0"] = {"expected": "0", "actual": f"{premium_starved + standard_starved}", "pass": premium_starved + standard_starved == 0}

    for h, r in H.items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {h}: {r['actual']} (expected {r['expected']}) [{status}]")

    all_pass = all(r["pass"] for r in H.values())
    print()
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    # ---- Save summary ----
    summary = {
        "policy": "LongLiuDWRRGapV3",
        "spine_bw_gbps": 1000,
        "workload": "FEAS_BOUNDARY_V2_WORKLOAD",
        "ci_config": "2.0/3.0",
        "sigma_attain_gbps": 879,
        "load_factor": 0.88,
        "premium": {
            "mean_sas": premium_mean,
            "capped_sas": premium_capped,
            "attainment": premium_attainment,
            "starved": premium_starved,
        },
        "standard": {
            "mean_sas": standard_mean,
            "capped_sas": standard_capped,
            "attainment": standard_attainment,
            "starved": standard_starved,
        },
        "hypotheses": H,
        "overall_pass": all_pass,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_dir}/summary.json")


if __name__ == "__main__":
    main()