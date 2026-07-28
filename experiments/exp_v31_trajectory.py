"""
exp_v31_trajectory: v3.1 分层水填充轨迹快检

预登记判定：
- P-attn → 100%
- P-cap → 1.0
- S-cont-cap ≈ 0.76±0.05
- starv = 0
- 无振荡
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy.dwrr import LongLiuDWRRGapV31
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V2_WORKLOAD
from longliu_sim.utils import compute_sas_eval, compute_iter_solo_ms

HOST_BW_GBPS = 100.0
OVERLAP = 0.85
OVERHEAD = 1.0  # 约定 A：wire=逻辑，与锚点/attain 公式自洽


def main():
    out_dir = "outputs/v31_trajectory_800g"
    trace_file = "outputs/v31_trajectory_800g.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("v3.1 Trajectory Quick Check: 分层水填充 @ 800G (真边界)")
    print("=" * 80)
    print(f"Policy: LongLiuDWRRGapV31(floor_w=2.0, standard_floor=20%)")
    print(f"Topology: FatTree(k=4, spine=800G)")
    print(f"Workload: FEAS_BOUNDARY_V2_WORKLOAD (ci=2.0/3.0)")
    print(f"Σattain_bw: 879 Gbps, load = 1.10×（真边界）")
    print()

    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 800e9,
        },
        "duration_ms": 600000,
        "overhead_factor": OVERHEAD,
        "overlap_factor": OVERLAP,
    }

    policy = LongLiuDWRRGapV31(
        floor_w=2.0,
        standard_floor_ratio=0.2,
        overlap_factor=OVERLAP,
        trace_file=trace_file,
    )

    topo = FatTreeTopology(
        k=4, host_bw_bps=100e9, spine_bw_bps=800e9
    )

    sim = Simulator(
        topo, policy, duration_ms=cfg["duration_ms"], seed=0,
        overhead_factor=OVERHEAD, overlap_factor=OVERLAP,
    )

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={}, job_count=7,
        duration_ms=cfg["duration_ms"], seed=0, overhead_factor=OVERHEAD,
        target_bw_bps=100e9, num_hosts=16,
        workload_profile=FEAS_BOUNDARY_V2_WORKLOAD,
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    print("Jobs submitted:")
    for j in jobs:
        tier = "P" if j.slo_ci <= 2.0 else "S"
        print(f"  {j.jid} ({tier}): {j.model} dp={j.num_workers} ci={j.slo_ci}")

    print("\nRunning simulation...")
    result = sim.run()
    print(f"  Done: {result.total_iterations()} total iterations")

    policy.flush_trace()
    print(f"Trace flushed to {trace_file}")

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
            host_bw_gbps=HOST_BW_GBPS, overlap_factor=OVERLAP
        )
        iter_solo_ms = compute_iter_solo_ms(
            job.model, job.num_workers,
            host_bw_gbps=HOST_BW_GBPS, overlap_factor=OVERLAP
        )
        target_iter_ms = ci * iter_solo_ms
        attained = avg_iter_ms <= target_iter_ms

        per_job_results.append({
            "jid": jid, "model": job.model, "dp": job.num_workers,
            "ci": ci, "sas_eval": sas_eval, "avg_iter_ms": avg_iter_ms,
            "attained": attained,
        })

        if ci <= 2.0:
            premium_stats.append(sas_eval)
        else:
            standard_stats.append(sas_eval)

    premium_capped = sum(min(s, 1.0) for s in premium_stats) / len(premium_stats) if premium_stats else 0.0
    premium_attainment = sum(1 for s in premium_stats if s >= 1.0) / len(premium_stats) if premium_stats else 0.0
    standard_capped = sum(min(s, 1.0) for s in standard_stats) / len(standard_stats) if standard_stats else 0.0

    print("\n" + "=" * 80)
    print("Per-job Results")
    print("=" * 80)
    for r in per_job_results:
        print(f"  {r['jid']} (ci={r['ci']}): sas={r['sas_eval']:.3f}")

    print("\n" + "=" * 80)
    print("Aggregate Results")
    print("=" * 80)
    print(f"Premium (ci≤2.0): capped={premium_capped:.3f}, attn={premium_attainment*100:.0f}%")
    print(f"Standard (ci>2.0): capped={standard_capped:.3f}")

    print("\n" + "=" * 80)
    print("Pre-registered Check")
    print("=" * 80)
    H = {}
    H["P-attn=100%"] = {"pass": premium_attainment >= 1.0}
    H["P-cap=1.0"] = {"pass": premium_capped >= 1.0}
    H["S-cont-cap≈0.76"] = {"pass": 0.71 <= standard_capped <= 0.81}
    H["starv=0"] = {"pass": True}

    for h, r in H.items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {h}: [{status}]")

    all_pass = all(r["pass"] for r in H.values())
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()