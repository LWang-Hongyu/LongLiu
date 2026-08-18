"""Anchor 场景 v4（LongLiu）补跑：V2_ANCHOR_WORKLOAD / 400G / 3 seeds。

复用 baseline_regen 的仿真流程，仅替换策略为 LongLiuAllocatorV4，
产出与 per_policy_results.json 同构的 outputs/anchor_v4/per_policy_results_v4.json。

用法：
    python scripts/run_anchor_v4.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from longliu_sim.policy.dwrr import LongLiuAllocatorV4
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.utils.config import (
    load_config, config_hash, get_topology,
    get_simulation, get_v2_anchor_workload,
)

OUTPUT_DIR = os.path.join(_project_root, "outputs", "anchor_v4")
SEEDS = [0, 1, 2]
STRATEGY = "v4"


def run_single(cfg: dict, policy, seed: int) -> dict:
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
    sim = Simulator(
        topo, policy,
        duration_ms=cfg["simulation"]["duration_ms"],
        seed=seed,
        overhead_factor=cfg["frozen"]["overhead_factor"],
        overlap_factor=cfg["frozen"]["overlap_factor"],
    )

    workload = get_v2_anchor_workload()
    loader = SyntheticTraceLoader(
        model_types=cfg["model_types"],
        gpu_distribution=cfg["gpu_distribution"],
        ci_distribution=cfg["tiered_workload"]["v2_anchor_ci_distribution"],
        job_count=cfg["job_count"],
        duration_ms=cfg["simulation"]["duration_ms"],
        seed=seed,
        overhead_factor=cfg["frozen"]["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=cfg["num_hosts"],
        workload_profile=[tuple(item) for item in workload],
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    if hasattr(policy, "flush_trace"):
        policy.flush_trace()
    stats = result.per_job_stats()

    per_job = {}
    for jid, s in stats.items():
        job = sim.jobs[jid]
        per_job[jid] = {
            "model": job.model,
            "dp": job.num_workers,
            "ci": job.slo_ci,
            "sas": s["sas"],
            "avg_iter_ms": s["avg_iter_ms"],
            "meets_slo": s["meets_slo"],
            "completed_iters": job.completed_iters,
        }

    large_jobs = [j for j in per_job.values() if j["ci"] <= 1.5]
    medium_jobs = [j for j in per_job.values() if 1.5 < j["ci"] <= 2.0]
    small_jobs = [j for j in per_job.values() if j["ci"] > 2.0]

    def tier_stats(jobs_list):
        if not jobs_list:
            return {"count": 0, "mean_sas": 0.0, "slo_rate": 0.0, "collapse_rate": 0.0}
        return {
            "count": len(jobs_list),
            "mean_sas": sum(j["sas"] for j in jobs_list) / len(jobs_list),
            "slo_rate": sum(1 for j in jobs_list if j["meets_slo"]) / len(jobs_list),
            "collapse_rate": sum(1 for j in jobs_list if j["sas"] < 0.2) / len(jobs_list),
        }

    total_sas = [j["sas"] for j in per_job.values()]
    all_jobs = list(per_job.values())

    return {
        "seed": seed,
        "per_job": per_job,
        "tiers": {
            "large": tier_stats(large_jobs),
            "medium": tier_stats(medium_jobs),
            "small": tier_stats(small_jobs),
        },
        "overall": {
            "mean_sas": sum(total_sas) / len(total_sas) if total_sas else 0.0,
            "slo_rate": sum(1 for j in all_jobs if j["meets_slo"]) / len(all_jobs) if all_jobs else 0.0,
            "collapse_rate": sum(1 for j in all_jobs if j["sas"] < 0.2) / len(all_jobs) if all_jobs else 0.0,
            "starvation_count": sum(1 for j in all_jobs if j["completed_iters"] <= 1),
        },
        "elapsed_s": 0.0,
    }


def main():
    cfg = load_config()
    frozen = cfg["frozen"]
    topo = get_topology()
    sim_cfg = get_simulation()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_config = {
        "topology": topo,
        "simulation": sim_cfg,
        "frozen": frozen,
        "model_types": cfg["model_types"],
        "gpu_distribution": cfg["gpu_distribution"],
        "tiered_workload": {
            "v2_anchor_ci_distribution": cfg["tiered_workload"]["v2_anchor_ci_distribution"],
        },
        "job_count": cfg["job_count"],
        "num_hosts": cfg["num_hosts"],
    }

    print("=" * 70)
    print(f"Anchor v4 补跑: {STRATEGY} @ V2_ANCHOR_WORKLOAD ({cfg['job_count']} jobs, 400G)")
    print(f"Seeds: {SEEDS}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 70)

    results = []
    for seed in SEEDS:
        t0 = time.time()
        trace_file = os.path.join(OUTPUT_DIR, f"trace_v4_s{seed}.jsonl")
        policy = LongLiuAllocatorV4(
            overhead_factor=frozen["overhead_factor"],
            overlap_factor=frozen["overlap_factor"],
            trace_file=trace_file,
        )
        result = run_single(run_config, policy, seed)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)
        print(f"  seed={seed}: mean_sas={result['overall']['mean_sas']:.4f}, "
              f"slo_rate={result['overall']['slo_rate']:.1%}, "
              f"collapse={result['overall']['collapse_rate']:.1%}, "
              f"starved={result['overall']['starvation_count']}, "
              f"({elapsed:.0f}s)")

    all_results = {
        STRATEGY: {
            "seeds": results,
            "summary": {
                "mean_sas": sum(r["overall"]["mean_sas"] for r in results) / len(results),
                "mean_slo_rate": sum(r["overall"]["slo_rate"] for r in results) / len(results),
                "mean_collapse_rate": sum(r["overall"]["collapse_rate"] for r in results) / len(results),
                "mean_starvation": sum(r["overall"]["starvation_count"] for r in results) / len(results),
            },
        }
    }

    results_path = os.path.join(OUTPUT_DIR, "per_policy_results_v4.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {results_path}")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "semantics_version": cfg["semantics_version"],
        "config_hash": config_hash(),
        "strategies": [STRATEGY],
        "seeds": SEEDS,
        "workload": "V2_ANCHOR_WORKLOAD",
        "job_count": cfg["job_count"],
        "config": {
            "topology": topo,
            "simulation": sim_cfg,
            "frozen": frozen,
            "ci_distribution": cfg["tiered_workload"]["v2_anchor_ci_distribution"],
        },
        "output_dir": OUTPUT_DIR,
    }
    meta_path = os.path.join(OUTPUT_DIR, "run_meta_v4.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"run_meta 已保存: {meta_path}")

    s = all_results[STRATEGY]["summary"]
    print(f"\n汇总: v4 mean_sas={s['mean_sas']:.4f} slo_rate={s['mean_slo_rate']:.1%} "
          f"collapse={s['mean_collapse_rate']:.1%} starved={s['mean_starvation']:.1f}")


if __name__ == "__main__":
    main()
