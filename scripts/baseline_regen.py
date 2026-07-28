"""基线重生成：Fair/CRUX/SP/D1 @ V2_ANCHOR_WORKLOAD，3 seeds。

产出 outputs/anchor_regen_v1/ 下的 run_meta.json + per_policy_results.json。
运行完成后 git tag anchor-regen-v1。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from longliu_sim.policy import Fair, CRUX, LongLiu
from longliu_sim.policy.dwrr import LongLiuDWRR
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.utils.config import (
    load_config, config_hash, get_topology,
    get_simulation, get_v2_anchor_workload,
)


OUTPUT_DIR = os.path.join(_project_root, "outputs", "anchor_regen_v1")
SEEDS = [0, 1, 2]
STRATEGIES = ["Fair", "CRUX", "SP", "D1"]


def build_policy(name: str, overlap_factor: float):
    """构建策略实例。"""
    if name == "Fair":
        return Fair()
    elif name == "CRUX":
        return CRUX(alpha=1.0, eps=1e-6, profile_iters=3)
    elif name == "SP":
        return LongLiu(K=2.0, use_dynamic_T_target=True)
    elif name == "D1":
        return LongLiuDWRR(
            K=2.0, use_soft_weights=False, intra_class_fair=False,
            clip_ratio=10.0, overlap_factor=overlap_factor,
            overhead_factor=frozen["overhead_factor"],
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")


def run_single(cfg: dict, policy, seed: int) -> dict:
    """运行单个 seed 的仿真。"""
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
    stats = result.per_job_stats()

    # 收集 per-job 数据
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

    # 分层统计
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

    # attention / capacity metrics
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

    # 构建统一的仿真配置包
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

    all_results = {}

    print("=" * 70)
    print("基线重生成：Fair / CRUX / SP / D1")
    print(f"Workload: V2_ANCHOR_WORKLOAD ({cfg['job_count']} jobs, ci=1.5/2.0/3.0)")
    print(f"Seeds: {SEEDS}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 70)

    for strategy_name in STRATEGIES:
        print(f"\n--- {strategy_name} ---")
        strategy_results = []

        for seed in SEEDS:
            t0 = time.time()
            policy = build_policy(strategy_name, frozen["overlap_factor"])
            result = run_single(run_config, policy, seed)
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)
            strategy_results.append(result)

            print(f"  seed={seed}: mean_sas={result['overall']['mean_sas']:.4f}, "
                  f"slo_rate={result['overall']['slo_rate']:.1%}, "
                  f"collapse={result['overall']['collapse_rate']:.1%}, "
                  f"starved={result['overall']['starvation_count']}, "
                  f"({elapsed:.0f}s)")

        all_results[strategy_name] = {
            "seeds": strategy_results,
            "summary": {
                "mean_sas": sum(r["overall"]["mean_sas"] for r in strategy_results) / len(strategy_results),
                "mean_slo_rate": sum(r["overall"]["slo_rate"] for r in strategy_results) / len(strategy_results),
                "mean_collapse_rate": sum(r["overall"]["collapse_rate"] for r in strategy_results) / len(strategy_results),
                "mean_starvation": sum(r["overall"]["starvation_count"] for r in strategy_results) / len(strategy_results),
            },
        }

    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, "per_policy_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {results_path}")

    # 保存 run_meta
    meta = {
        "timestamp": datetime.now().isoformat(),
        "semantics_version": cfg["semantics_version"],
        "config_hash": config_hash(),
        "strategies": STRATEGIES,
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
    meta_path = os.path.join(OUTPUT_DIR, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"run_meta 已保存: {meta_path}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("汇总表（3-seed mean）")
    print("=" * 70)
    print(f"{'Strategy':<8} {'Mean SAS':>10} {'SLO Rate':>10} {'Collapse':>10} {'Starved':>8}")
    print("-" * 50)
    for name in STRATEGIES:
        s = all_results[name]["summary"]
        print(f"{name:<8} {s['mean_sas']:>10.4f} {s['mean_slo_rate']:>9.1%} "
              f"{s['mean_collapse_rate']:>9.1%} {s['mean_starvation']:>8.1f}")

    print(f"\n下一步: git tag anchor-regen-v1")
    print(f"  cd {_project_root}")
    print(f"  git tag -a anchor-regen-v1 -m 'baseline regeneration: Fair/CRUX/SP/D1 @ V2_ANCHOR_WORKLOAD, 3 seeds'")


if __name__ == "__main__":
    main()
