"""
任务3: 低优先级地板消融实验

对比：
- 基线：纯 SP（LongLiu 默认）
- 消融：纯 SP + P0/P1/P2 各 5% 保底（low_priority_floor=True）

实验设计：
- 20 seeds，CRN（Common Random Numbers）
- paired t-test
- 报告效应量（Cohen's d）和 95% CI
- 分层报告崩溃率（Large/Medium 的 SAS<0.2 比例）

预期：
- Medium 崩溃显著下降
- Large 崩溃持平或略升
- 若 Large 大幅改善，标记实现可疑
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from longliu_sim.policy import LongLiu
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def run_single_seed(cfg: dict, policy, seed: int) -> dict:
    """运行单个 seed 的仿真。"""
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
    sim = Simulator(topo, policy, duration_ms=cfg["duration_ms"], seed=seed)

    # 生成 jobs（与 exp_ablation.py 一致）
    loader = SyntheticTraceLoader(
        model_types=[
            "ResNet-18", "ResNet-50-fp16", "BERT-Base", "BERT-Large-fp16",
            "ViT-Base", "ViT-Large", "LLaMA-2-1B", "LLaMA-2-7B", "T5-1B",
        ],
        gpu_distribution={1: 0.2, 2: 0.2, 4: 0.3, 8: 0.3},
        ci_distribution={1.5: 0.3, 2.0: 0.35, 3.0: 0.35},
        job_count=24,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=1.3,
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
        tier_results[tier].append({
            "jid": jid,
            "model": job.model,
            "dp": job.num_workers,
            "ci": ci,
            "sas": s["sas"],
            "avg_iter_ms": s["avg_iter_ms"],
            "meets_slo": s["meets_slo"],
        })

    # 计算分层崩溃率
    tier_collapse = {}
    for tier, jobs in tier_results.items():
        if not jobs:
            continue
        collapse_count = sum(1 for j in jobs if j["sas"] < 0.2)
        collapse_rate = collapse_count / len(jobs) if jobs else 0.0
        mean_sas = sum(j["sas"] for j in jobs) / len(jobs) if jobs else 0.0
        slo_rate = sum(1 for j in jobs if j["meets_slo"]) / len(jobs) if jobs else 0.0
        tier_collapse[tier] = {
            "count": len(jobs),
            "collapse_count": collapse_count,
            "collapse_rate": collapse_rate,
            "mean_sas": mean_sas,
            "slo_rate": slo_rate,
        }

    return {
        "seed": seed,
        "tier_collapse": tier_collapse,
        "overall_collapse": sum(1 for s in stats.values() if s["sas"] < 0.2) / len(stats),
        "overall_mean_sas": sum(s["sas"] for s in stats.values()) / len(stats),
        "overall_slo_rate": sum(1 for s in stats.values() if s["meets_slo"]) / len(stats),
    }


def compute_effect_size(baseline_vals: list, treatment_vals: list) -> dict:
    """计算效应量和置信区间。"""
    n = len(baseline_vals)
    if n < 2:
        return {"cohens_d": 0.0, "ci_95": (0.0, 0.0)}

    # Paired differences
    diffs = [t - b for b, t in zip(baseline_vals, treatment_vals)]
    mean_diff = sum(diffs) / n
    std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)) ** 0.5 if n > 1 else 0.0

    # Cohen's d (paired)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # 95% CI for mean difference
    if std_diff > 0 and n > 1:
        se = std_diff / (n ** 0.5)
        t_crit = scipy_stats.t.ppf(0.975, n - 1) if _HAS_SCIPY else 2.093  # 近似
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
    else:
        ci_low = ci_high = mean_diff

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_d": cohens_d,
        "ci_95": (ci_low, ci_high),
    }


def main():
    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 210e9,  # final_v3 配置
        },
        "duration_ms": 300000,  # 5 分钟
    }

    seeds = list(range(20))

    # 基线：纯 SP
    baseline_policy = LongLiu(K=2.0, use_dynamic_T_target=True)

    # 消融：纯 SP + P0/P1/P2 各 5% 保底
    treatment_policy = LongLiu(
        K=2.0,
        use_dynamic_T_target=True,
        low_priority_floor=True,
    )

    print("=" * 80)
    print("任务3: 低优先级地板消融实验")
    print("=" * 80)
    print(f"配置: k={cfg['topology']['k']}, duration={cfg['duration_ms']}ms")
    print(f"Seeds: {len(seeds)}")
    print()

    baseline_results = []
    treatment_results = []

    print("运行基线（纯 SP）...")
    for seed in seeds:
        r = run_single_seed(cfg, baseline_policy, seed)
        baseline_results.append(r)
        print(f"  Seed {seed}: Overall collapse {r['overall_collapse']:.1%}, Mean SAS {r['overall_mean_sas']:.3f}")

    print()
    print("运行消融（纯 SP + P0/P1/P2 保底）...")
    for seed in seeds:
        r = run_single_seed(cfg, treatment_policy, seed)
        treatment_results.append(r)
        print(f"  Seed {seed}: Overall collapse {r['overall_collapse']:.1%}, Mean SAS {r['overall_mean_sas']:.3f}")

    # 配对 t-test
    baseline_sas = [r["overall_mean_sas"] for r in baseline_results]
    treatment_sas = [r["overall_mean_sas"] for r in treatment_results]

    if _HAS_SCIPY:
        t_stat, p_value = scipy_stats.ttest_rel(baseline_sas, treatment_sas)
    else:
        # 手动计算 t-statistic
        diffs = [t - b for b, t in zip(baseline_sas, treatment_sas)]
        mean_diff = sum(diffs) / len(diffs)
        std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
        se = std_diff / (len(diffs) ** 0.5)
        t_stat = mean_diff / se if se > 0 else 0.0
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), len(diffs) - 1)) if _HAS_SCIPY else None

    # 效应量
    effect = compute_effect_size(baseline_sas, treatment_sas)

    # 分层统计
    tiers = ["large", "medium"]
    tier_stats = {}
    for tier in tiers:
        baseline_collapse = [r["tier_collapse"].get(tier, {}).get("collapse_rate", 0.0) for r in baseline_results]
        treatment_collapse = [r["tier_collapse"].get(tier, {}).get("collapse_rate", 0.0) for r in treatment_results]

        baseline_mean = sum(baseline_collapse) / len(baseline_collapse) if baseline_collapse else 0.0
        treatment_mean = sum(treatment_collapse) / len(treatment_collapse) if treatment_collapse else 0.0

        tier_effect = compute_effect_size(baseline_collapse, treatment_collapse)

        if _HAS_SCIPY:
            tier_t, tier_p = scipy_stats.ttest_rel(baseline_collapse, treatment_collapse)
        else:
            tier_t, tier_p = 0.0, None

        tier_stats[tier] = {
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "diff": treatment_mean - baseline_mean,
            "effect": tier_effect,
            "t_stat": tier_t,
            "p_value": tier_p,
        }

    # 额外诊断：检查各 tier 的 mean SAS 和 SLO rate
    print()
    print("表3: 分层详细指标（诊断）")
    print("-" * 80)
    print(f"{'Tier':<10} {'Baseline SAS':>15} {'Treatment SAS':>15} {'Baseline SLO%':>15} {'Treatment SLO%':>15}")
    print("-" * 80)
    for tier in tiers:
        baseline_sas_list = [r["tier_collapse"].get(tier, {}).get("mean_sas", 0.0) for r in baseline_results]
        treatment_sas_list = [r["tier_collapse"].get(tier, {}).get("mean_sas", 0.0) for r in treatment_results]
        baseline_slo_list = [r["tier_collapse"].get(tier, {}).get("slo_rate", 0.0) for r in baseline_results]
        treatment_slo_list = [r["tier_collapse"].get(tier, {}).get("slo_rate", 0.0) for r in treatment_results]

        baseline_sas_mean = sum(baseline_sas_list) / len(baseline_sas_list) if baseline_sas_list else 0.0
        treatment_sas_mean = sum(treatment_sas_list) / len(treatment_sas_list) if treatment_sas_list else 0.0
        baseline_slo_mean = sum(baseline_slo_list) / len(baseline_slo_list) if baseline_slo_list else 0.0
        treatment_slo_mean = sum(treatment_slo_list) / len(treatment_slo_list) if treatment_slo_list else 0.0

        print(f"{tier:<10} {baseline_sas_mean:>15.3f} {treatment_sas_mean:>15.3f} {baseline_slo_mean:>15.1%} {treatment_slo_mean:>15.1%}")
    print()

    # 输出结果
    print()
    print("=" * 80)
    print("实验结果")
    print("=" * 80)
    print()
    print("表1: Overall 指标")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':>12} {'Treatment':>12} {'Diff':>10}")
    print("-" * 80)
    print(f"{'Mean SAS':<30} {sum(baseline_sas)/len(baseline_sas):>12.3f} {sum(treatment_sas)/len(treatment_sas):>12.3f} {effect['mean_diff']:>10.3f}")
    print(f"{'Overall Collapse Rate':<30} {sum(r['overall_collapse'] for r in baseline_results)/len(baseline_results):>12.1%} {sum(r['overall_collapse'] for r in treatment_results)/len(treatment_results):>12.1%} {effect['mean_diff']:>10.3f}")
    print()
    p_val_str1 = f"{p_value:.4f}" if p_value else "N/A"
    print(f"配对 t-test: t = {t_stat:.3f}, p = {p_val_str1}")
    print(f"效应量 Cohen's d = {effect['cohens_d']:.3f}")
    print(f"95% CI: [{effect['ci_95'][0]:.3f}, {effect['ci_95'][1]:.3f}]")
    print()

    print("表2: 分层崩溃率（SAS < 0.2）")
    print("-" * 80)
    cohens_d_label = "Cohen's d"
    print(f"{'Tier':<15} {'Baseline':>12} {'Treatment':>12} {'Diff':>10} {cohens_d_label:>12} {'p-value':>10}")
    print("-" * 80)
    for tier in tiers:
        s = tier_stats[tier]
        p_val_str = f"{s['p_value']:.4f}" if s['p_value'] else "N/A"
        print(f"{tier:<15} {s['baseline_mean']:>12.1%} {s['treatment_mean']:>12.1%} {s['diff']:>10.1%} {s['effect']['cohens_d']:>12.3f} {p_val_str:>10}")
    print()

    # 检查预期
    print("=" * 80)
    print("预期验证")
    print("=" * 80)
    large_diff = tier_stats["large"]["diff"]
    medium_diff = tier_stats["medium"]["diff"]

    print(f"✓ Medium 崩溃率变化: {medium_diff:+.1%}（预期：显著下降）")
    if medium_diff < -0.05:
        print("  ✅ 符合预期：Medium 崩溃显著下降")
    elif medium_diff < 0:
        print("  ⚠️  部分符合：Medium 崩溃下降但未达显著水平")
    else:
        print("  ❌ 不符合预期：Medium 崩溃未下降或上升")

    print(f"✓ Large 崩溃率变化: {large_diff:+.1%}（预期：持平或略升）")
    if large_diff <= 0.02:
        print("  ✅ 符合预期：Large 崩溃持平或略升")
    elif large_diff > 0.10:
        print("  ⚠️  实现可疑：Large 崩溃大幅改善，需检查代码")
    else:
        print("  ℹ️  Large 崩溃略有变化，但未触发可疑阈值")

    print()

    # 保存结果
    output = {
        "config": cfg,
        "seeds": seeds,
        "baseline_results": baseline_results,
        "treatment_results": treatment_results,
        "overall_effect": effect,
        "tier_stats": tier_stats,
        "t_stat": t_stat,
        "p_value": p_value,
    }

    output_path = os.path.join(os.path.dirname(__file__), "task3_floor_ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    main()