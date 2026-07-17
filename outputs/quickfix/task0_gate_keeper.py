"""
门禁：验证基线复现 final_v3

用 final_v3 完全相同的配置跑纯 SP 基线，验证：
- Large 崩溃率 ≈ 17%（容差±噪声）
- Large Mean SAS ≈ 1.09

关键配置：
- spine_bw_bps = 400 Gbps（非 210 Gbps）
- duration_ms = 600000 ms（非 300000 ms）
- K = 2.0
- overhead_factor = 1.3
- overlap_factor = 0.85
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from longliu_sim.policy import LongLiu
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD


def run_single_seed(cfg: dict, policy, seed: int) -> dict:
    """运行单个 seed 的仿真。"""
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
    sim = Simulator(
        topo,
        policy,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

    # 生成 jobs（与 exp_ablation.py 一致）
    loader = SyntheticTraceLoader(
        model_types=[
            "ResNet-18",
            "ResNet-50-fp16",
            "BERT-Base",
            "BERT-Large-fp16",
            "ViT-Base",
            "ViT-Large",
            "LLaMA-2-1B",
            "LLaMA-2-7B",
            "T5-1B",
        ],
        gpu_distribution={1: 0.2, 2: 0.2, 4: 0.3, 8: 0.3},
        ci_distribution={1.5: 0.3, 2.0: 0.35, 3.0: 0.35},
        job_count=24,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=DEFAULT_TIERED_WORKLOAD,
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    stats = result.per_job_stats()

    # 分层统计
    tier_results = {"large": [], "medium": [], "small": []}
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        tier = "large" if ci == 1.5 else ("medium" if ci == 2.0 else "small")
        tier_results[tier].append(
            {
                "jid": jid,
                "model": job.model,
                "dp": job.num_workers,
                "ci": ci,
                "sas": s["sas"],
                "avg_iter_ms": s["avg_iter_ms"],
                "meets_slo": s["meets_slo"],
            }
        )

    # 计算分层崩溃率
    tier_collapse = {}
    for tier, jobs_list in tier_results.items():
        if not jobs_list:
            continue
        collapse_count = sum(1 for j in jobs_list if j["sas"] < 0.2)
        collapse_rate = collapse_count / len(jobs_list) if jobs_list else 0.0
        mean_sas = (
            sum(j["sas"] for j in jobs_list) / len(jobs_list) if jobs_list else 0.0
        )
        slo_rate = (
            sum(1 for j in jobs_list if j["meets_slo"]) / len(jobs_list)
            if jobs_list
            else 0.0
        )
        tier_collapse[tier] = {
            "count": len(jobs_list),
            "collapse_count": collapse_count,
            "collapse_rate": collapse_rate,
            "mean_sas": mean_sas,
            "slo_rate": slo_rate,
        }

    return {
        "seed": seed,
        "tier_collapse": tier_collapse,
        "overall_collapse": (
            sum(1 for s in stats.values() if s["sas"] < 0.2) / len(stats)
            if stats
            else 0.0
        ),
        "overall_mean_sas": (
            sum(s["sas"] for s in stats.values()) / len(stats) if stats else 0.0
        ),
        "overall_slo_rate": (
            sum(1 for s in stats.values() if s["meets_slo"]) / len(stats)
            if stats
            else 0.0
        ),
    }


def main():
    # final_v3 完全相同的配置
    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 400e9,  # 关键：400 Gbps（非 210 Gbps）
        },
        "duration_ms": 600000,  # 关键：10 分钟（非 5 分钟）
        "overhead_factor": 1.3,
        "overlap_factor": 0.85,
    }

    seeds = list(range(10))  # 先跑 10 seeds 快速验证

    policy = LongLiu(K=2.0, use_dynamic_T_target=True)

    print("=" * 80)
    print("门禁：验证基线复现 final_v3")
    print("=" * 80)
    print(f"配置: k={cfg['topology']['k']}, duration={cfg['duration_ms']}ms")
    print(f"       spine_bw={cfg['topology']['spine_bw_bps']/1e9:.0f}Gbps")
    print(f"       overhead={cfg['overhead_factor']}, overlap={cfg['overlap_factor']}")
    print(f"Seeds: {len(seeds)}")
    print()

    print("运行纯 SP 基线...")
    results = []
    for seed in seeds:
        r = run_single_seed(cfg, policy, seed)
        results.append(r)
        print(
            f"  Seed {seed}: Overall collapse {r['overall_collapse']:.1%}, Mean SAS {r['overall_mean_sas']:.3f}"
        )

    # 汇总结果
    large_collapse_rates = [
        r["tier_collapse"].get("large", {}).get("collapse_rate", 0.0) for r in results
    ]
    large_mean_sas = [
        r["tier_collapse"].get("large", {}).get("mean_sas", 0.0) for r in results
    ]

    mean_large_collapse = (
        sum(large_collapse_rates) / len(large_collapse_rates)
        if large_collapse_rates
        else 0.0
    )
    mean_large_sas = (
        sum(large_mean_sas) / len(large_mean_sas) if large_mean_sas else 0.0
    )

    print()
    print("=" * 80)
    print("验证结果")
    print("=" * 80)
    print(f"Large 崩溃率（SAS<0.2）: {mean_large_collapse:.1%}")
    print(f"  final_v3 目标: ~17%（容差±噪声）")
    if abs(mean_large_collapse - 0.17) < 0.05:
        print("  ✅ 复现成功")
    else:
        print(
            f"  ❌ 复现失败：偏差 {abs(mean_large_collapse - 0.17):.1%} > 5%"
        )

    print()
    print(f"Large Mean SAS: {mean_large_sas:.3f}")
    print(f"  final_v3 目标: ~1.09（容差±噪声）")
    if abs(mean_large_sas - 1.09) < 0.2:
        print("  ✅ 复现成功")
    else:
        print(f"  ❌ 复现失败：偏差 {abs(mean_large_sas - 1.09):.3f}")

    # 输出配置对比
    print()
    print("=" * 80)
    print("配置对比")
    print("=" * 80)
    print("final_v3 配置:")
    print(f"  spine_bw_bps: {400e9 / 1e9:.0f} Gbps")
    print(f"  duration_ms: {600000} ms")
    print()
    print("任务3 错误配置:")
    print(f"  spine_bw_bps: {210e9 / 1e9:.0f} Gbps（错误：应该是 400 Gbps）")
    print(f"  duration_ms: {300000} ms（错误：应该是 600000 ms）")
    print()
    print("关键差异:")
    print(f"  Spine 带宽减半 → 竞争强度翻倍 → 崩溃率从 17% 飙升到 69.6%")

    # 保存结果
    output = {
        "config": cfg,
        "seeds": seeds,
        "results": results,
        "summary": {
            "large_collapse_rate": mean_large_collapse,
            "large_mean_sas": mean_large_sas,
        },
    }

    output_path = args.out if args.out else os.path.join(
        os.path.dirname(__file__), "task0_gate_keeper_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()pen(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()