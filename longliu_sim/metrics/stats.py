"""
输出指标统计模块。

提供 `compute_stats()` 函数，从 SimulationResult 计算端到端指标：
- Total iterations / Avg iteration time
- SLO attainment per ci tier (tight ci<1.5, medium 1.5≤ci<2.5, loose ci≥2.5)
- JCT (job completion time) ratio
- Fairness index (Jain's index over per-JCT ratio)
"""

from __future__ import annotations

import math
from typing import Any

from longliu_sim.core import SimulationResult, IterationRecord


def compute_stats(result: SimulationResult) -> dict[str, Any]:
    """从 SimulationResult 计算完整指标字典。

    返回包含以下键的字典：
        total_iters         总迭代次数（int）
        avg_iter_ms         平均迭代时间（float）
        slo_tight           紧 SLO (ci<1.5) attainment（float, 0~1）
        slo_medium          中 SLO (1.5≤ci<2.5) attainment（float, 0~1）
        slo_loose           松 SLO (ci≥2.5) attainment（float, 0~1）
        slo_overall         整体 attainment（float, 0~1）
        jct_ratio_avg       平均 JCT ratio（float）
        jct_ratio_p95       P95 JCT ratio（float）
        fairness_index      Jain's fairness index over JCT ratio（float, 0~1）
    """
    records = result.records
    jobs = result.jobs

    total_iters = len(records)
    avg_iter_ms = sum(r.iter_ms for r in records) / max(len(records), 1)

    # ── SLO by ci tier ──
    tight_ok = tight_total = 0
    med_ok = med_total = 0
    loose_ok = loose_total = 0

    for j in jobs.values():
        ok = j.completed_iters >= j.target_iters
        if j.slo_ci < 1.5:
            tight_total += 1
            tight_ok += ok
        elif j.slo_ci < 2.5:
            med_total += 1
            med_ok += ok
        else:
            loose_total += 1
            loose_ok += ok

    def _safe(v, d):
        return v / d if d > 0 else 0.0

    slo_tight = _safe(tight_ok, tight_total)
    slo_medium = _safe(med_ok, med_total)
    slo_loose = _safe(loose_ok, loose_total)
    slo_overall = _safe(
        tight_ok + med_ok + loose_ok,
        tight_total + med_total + loose_total,
    )

    # ── JCT ratio（迭代完成时间 / 目标迭代时间）─
    # 目标迭代时间 = target_iters * iter_interval_ms
    # JCT = last iter end time - start_time_ms
    jct_ratios: list[float] = []
    for j in jobs.values():
        if j.target_iters <= 0:
            continue
        target_jct = j.target_iters * j.iter_interval_ms
        completed_records = [r for r in records if r.jid == j.jid]
        if not completed_records:
            continue
        actual_jct = max(r.end_ms for r in completed_records) - j.start_time_ms
        jct_ratios.append(actual_jct / max(target_jct, 1e-6))

    jct_ratio_avg = sum(jct_ratios) / max(len(jct_ratios), 1)
    jct_ratios_sorted = sorted(jct_ratios)
    idx_95 = int(0.95 * len(jct_ratios_sorted))
    jct_ratio_p95 = jct_ratios_sorted[idx_95] if jct_ratios_sorted else 0.0

    # ── Jain's fairness index（JCT ratio 越接近越好）─
    if jct_ratios:
        sum_ = sum(jct_ratios)
        sum_sq = sum(x * x for x in jct_ratios)
        n = len(jct_ratios)
        fairness_index = sum_ * sum_ / (n * sum_sq) if sum_sq > 0 else 1.0
    else:
        fairness_index = 1.0

    return {
        "total_iters": total_iters,
        "avg_iter_ms": avg_iter_ms,
        "slo_tight": slo_tight,
        "slo_medium": slo_medium,
        "slo_loose": slo_loose,
        "slo_overall": slo_overall,
        "jct_ratio_avg": jct_ratio_avg,
        "jct_ratio_p95": jct_ratio_p95,
        "fairness_index": fairness_index,
        "n_jobs": len(jobs),
        "n_records": len(records),
    }


def format_stats_table(stats: dict[str, Any]) -> str:
    """将 stats 字典格式化为可打印的表格字符串。"""
    lines = [
        f"{'Metric':<25} {'Value':<12}",
        "-" * 37,
        f"{'Total iters':<25} {stats['total_iters']:<12}",
        f"{'Avg iter (ms)':<25} {stats['avg_iter_ms']:<12.2f}",
        f"{'SLO tight':<25} {stats['slo_tight']:<12.2%}",
        f"{'SLO medium':<25} {stats['slo_medium']:<12.2%}",
        f"{'SLO loose':<25} {stats['slo_loose']:<12.2%}",
        f"{'SLO overall':<25} {stats['slo_overall']:<12.2%}",
        f"{'Avg JCT ratio':<25} {stats['jct_ratio_avg']:<12.4f}",
        f"{'P95 JCT ratio':<25} {stats['jct_ratio_p95']:<12.4f}",
        f"{'Fairness index':<25} {stats['fairness_index']:<12.4f}",
        f"{'Jobs':<25} {stats['n_jobs']:<12}",
        f"{'Records':<25} {stats['n_records']:<12}",
    ]
    return "\n".join(lines)