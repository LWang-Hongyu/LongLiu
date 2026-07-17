"""
绘图模块。

提供常用论文级可视化函数：
- plot_cdf: 迭代时间 CDF
- plot_slo_bar: 各策略 SLO attainment 柱状图
- plot_convergence: 累计迭代数随时间变化曲线
- plot_timeline: 作业调度时间线（甘特图风格）
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非交互后端，适用于 headless 环境
import matplotlib.pyplot as plt
import numpy as np

from longliu_sim.core import SimulationResult


def _save_or_show(path: Optional[str] = None, dpi: int = 150):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_cdf(
    result: SimulationResult,
    title: str = "CDF of Iteration Time",
    path: Optional[str] = None,
):
    """绘制所有完成的迭代的耗时 CDF。

    Args:
        result: 仿真结果
        title: 图标题
        path: 保存路径（可选；不传则交互显示）
    """
    records = result.records
    if not records:
        print("[plot_cdf] No records to plot.")
        return

    iter_ms = [r.iter_ms for r in records]
    sorted_ = np.sort(iter_ms)
    cdf = np.arange(1, len(sorted_) + 1) / len(sorted_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_, cdf, linewidth=1.5)
    ax.set_xlabel("Iteration Time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    _save_or_show(path)


def plot_slo_bar(
    stats_by_policy: dict[str, dict],
    title: str = "SLO Attainment by Policy",
    path: Optional[str] = None,
):
    """绘制多策略 SLO attainment 柱状图（整体 + 三 tier）。

    Args:
        stats_by_policy: {policy_name: compute_stats_output}
        title: 图标题
        path: 保存路径
    """
    policies = list(stats_by_policy.keys())
    metrics = ["slo_tight", "slo_medium", "slo_loose", "slo_overall"]
    labels = ["Tight", "Medium", "Loose", "Overall"]

    x = np.arange(len(policies))
    width = 0.18
    multiplier = 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [stats_by_policy[p][metric] * 100 for p in policies]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("SLO Attainment (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    _save_or_show(path)


def plot_convergence(
    result: SimulationResult,
    title: str = "Iteration Convergence Over Time",
    path: Optional[str] = None,
    max_time_ms: Optional[float] = None,
):
    """绘制所有 job 累计迭代数随时间的变化曲线。

    Args:
        result: 仿真结果
        title: 图标题
        path: 保存路径
        max_time_ms: x 轴上限（可选）
    """
    records = result.records
    if not records:
        print("[plot_convergence] No records.")
        return

    # 按 job 分组
    by_job: dict[str, list[tuple[float, int]]] = {}
    for r in sorted(records, key=lambda x: x.end_ms):
        by_job.setdefault(r.jid, []).append((r.end_ms, len(by_job[r.jid]) + 1))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for jid, pts in by_job.items():
        times, iters = zip(*pts)
        ax.plot(times, iters, linewidth=1, alpha=0.6, label=jid)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Completed Iterations")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if max_time_ms:
        ax.set_xlim(0, max_time_ms)
    if len(by_job) <= 10:
        ax.legend(fontsize=7)

    _save_or_show(path)


def plot_timeline(
    result: SimulationResult,
    title: str = "Job Scheduling Timeline",
    path: Optional[str] = None,
):
    """绘制作业调度时间线（甘特图风格）。

    Args:
        result: 仿真结果
        title: 图标题
        path: 保存路径
    """
    records = result.records
    if not records:
        print("[plot_timeline] No records.")
        return

    jids = sorted(set(r.jid for r in records))
    jid_to_idx = {j: i for i, j in enumerate(jids)}

    fig, ax = plt.subplots(figsize=(10, max(4, len(jids) * 0.4)))

    for r in records:
        y = jid_to_idx[r.jid]
        ax.barh(y, r.iter_ms, left=r.start_ms, height=0.6,
                alpha=0.7, label=r.jid if y == 0 else "")

    ax.set_yticks(range(len(jids)))
    ax.set_yticklabels(jids)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    _save_or_show(path)