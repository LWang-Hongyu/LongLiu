#!/usr/bin/env python3
"""
生成 SAS 指标的 CDF 图表和分层箱线图。

展示 CRUX 的两极分化 vs LongLiu 的均匀分布。
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 使用非交互式 backend
plt.switch_backend('Agg')


def load_per_job_data(path: str) -> dict:
    """加载 per_job.json 数据。"""
    with open(path) as f:
        return json.load(f)


def plot_sas_cdf(data: dict, out_dir: str):
    """图 1: SAS CDF 按策略。"""
    fig, ax = plt.subplots(figsize=(6, 4))

    policies = ['Fair', 'SRPT', 'CRUX', 'LongLiu']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for policy, color in zip(policies, colors):
        jobs = data[policy]
        sas_values = sorted([j['sas'] for j in jobs])
        cdf = np.arange(1, len(sas_values) + 1) / len(sas_values)
        ax.plot(sas_values, cdf, label=policy, linewidth=2, color=color,
                marker='o', markersize=3, markevery=5)

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='SLO Boundary')
    ax.set_xlabel('SLO Achievement Score (SAS)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_sas_cdf.pdf')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  CDF chart → {out_path}")


def plot_sas_tier_boxplot(data: dict, out_dir: str):
    """图 2: 按 ci 分层的箱线图。"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    policies = ['Fair', 'SRPT', 'CRUX', 'LongLiu']
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']

    for idx, ci in enumerate([1.5, 2.0, 3.0]):
        ax = axes[idx]
        tier_data = []
        for policy in policies:
            jobs = data[policy]
            tier_sas = [j['sas'] for j in jobs if j['ci'] == ci]
            tier_data.append(tier_sas if tier_sas else [0])

        positions = [1, 2, 3, 4]
        bp = ax.boxplot(tier_data, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticklabels(policies, rotation=15)
        ax.set_title(f'$c_i = {ci}$')
        ax.set_ylabel('SAS' if idx == 0 else '')
        ax.set_ylim(0, 3.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_sas_tier_boxplot.pdf')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Tier boxplot → {out_path}")


def plot_sas_heatmap(data: dict, out_dir: str):
    """图 3: 按模型类型的 SAS 分布（热力图）。"""
    # 收集所有模型
    models = set()
    for policy in data:
        for job in data[policy]:
            models.add(job['model'])
    models = sorted(models)

    policies = ['Fair', 'SRPT', 'CRUX', 'LongLiu']

    # 构建矩阵
    matrix = np.zeros((len(models), len(policies)))
    for i, model in enumerate(models):
        for j, policy in enumerate(policies):
            jobs = data[policy]
            model_sas = [job['sas'] for job in jobs if job['model'] == model]
            matrix[i, j] = np.mean(model_sas) if model_sas else 0

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2.0)

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)

    for i in range(len(models)):
        for j in range(len(policies)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    plt.colorbar(im, ax=ax, label='Mean SAS')
    plt.title('SAS by Model and Policy')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_sas_heatmap.pdf')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Heatmap → {out_path}")


def main():
    import sys
    base_dir = Path(__file__).parent.parent
    # 优先使用 10 seeds 数据，否则用 1 seed
    candidates = [
        base_dir / 'outputs' / 'weighted_bw_10seeds' / 'per_job.json',
        base_dir / 'outputs' / 'table3_perjob_test' / 'per_job.json',
    ]
    per_job_path = None
    for c in candidates:
        if c.exists():
            per_job_path = c
            break
    if per_job_path is None:
        print("ERROR: No per_job.json found. Run exp_ablation.py first.")
        sys.exit(1)
    out_dir = base_dir / 'outputs' / 'figures'

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading per-job data from: {per_job_path}")
    data = load_per_job_data(str(per_job_path))

    print("\nGenerating charts...")
    plot_sas_cdf(data, str(out_dir))
    plot_sas_tier_boxplot(data, str(out_dir))
    plot_sas_heatmap(data, str(out_dir))

    print("\nDone!")


if __name__ == "__main__":
    main()