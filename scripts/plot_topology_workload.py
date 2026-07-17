#!/usr/bin/env python3
"""
可视化 Fat-Tree 拓扑和任务分布。

输出:
1. 拓扑结构图（spine-tor-host 层次）
2. 任务分布图（job 在 host 上的分配）
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_workload_data():
    """从 DEFAULT_TIERED_WORKLOAD 加载任务数据。"""
    # 硬编码的任务分布（来自 synthetic.py）
    workload = [
        # 大模型 (ci=1.5), 12 个
        ("J1", "LLaMA-2-13B", 8, 1.5),
        ("J2", "LLaMA-2-13B", 8, 1.5),
        ("J3", "LLaMA-2-7B", 8, 1.5),
        ("J4", "LLaMA-2-7B", 8, 1.5),
        ("J5", "LLaMA-2-7B", 8, 1.5),
        ("J6", "T5-11B", 8, 1.5),
        ("J7", "T5-11B", 8, 1.5),
        ("J8", "T5-11B", 8, 1.5),
        ("J9", "LLaMA-2-7B", 8, 1.5),
        ("J10", "LLaMA-2-7B", 8, 1.5),
        ("J11", "LLaMA-2-7B", 8, 1.5),
        ("J12", "LLaMA-2-7B", 8, 1.5),
        # 中模型 (ci=2.0), 8 个
        ("J13", "BERT-Large", 2, 2.0),
        ("J14", "BERT-Large", 2, 2.0),
        ("J15", "BERT-Large", 2, 2.0),
        ("J16", "BERT-Large", 2, 2.0),
        ("J17", "T5-base", 4, 2.0),
        ("J18", "T5-base", 4, 2.0),
        ("J19", "T5-base", 4, 2.0),
        ("J20", "T5-base", 4, 2.0),
        # 小模型 (ci=3.0), 4 个
        ("J21", "ResNet-18", 1, 3.0),
        ("J22", "ResNet-18", 1, 3.0),
        ("J23", "GPT-2-small", 1, 3.0),
        ("J24", "GPT-2-small", 1, 3.0),
    ]
    return workload


def plot_fattree_topology(out_dir: Path):
    """绘制 Fat-Tree 拓扑结构。"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 层次布局
    # Spine: y = 3, x = 0, 1, 2, 3
    # TOR: y = 2, x = 0-7
    # Host: y = 1, x = 0-15

    # Spine switches
    spine_x = np.arange(4) * 4 + 1.5  # 间隔开
    spine_y = 3.0
    ax.scatter(spine_x, [spine_y] * 4, s=400, c='red', marker='s',
               label='Spine Switch', zorder=3, edgecolors='black', linewidth=2)

    # TOR switches
    tor_x = np.arange(8) * 2 + 0.5
    tor_y = 2.0
    ax.scatter(tor_x, [tor_y] * 8, s=300, c='orange', marker='s',
               label='TOR Switch', zorder=3, edgecolors='black', linewidth=1.5)

    # Hosts
    host_x = np.arange(16) + 0.5
    host_y = 1.0
    ax.scatter(host_x, [host_y] * 16, s=200, c='lightblue', marker='o',
               label='Host', zorder=3, edgecolors='black', linewidth=1)

    # 连接: Host -> TOR (每个 TOR 挂 2 hosts)
    for i in range(8):
        ax.plot([tor_x[i], host_x[2*i]], [tor_y, host_y], 'k-', linewidth=1.5, alpha=0.6)
        ax.plot([tor_x[i], host_x[2*i+1]], [tor_y, host_y], 'k-', linewidth=1.5, alpha=0.6)

    # 连接: TOR -> Spine (ECMP, 每个 TOR 连接所有 4 spine)
    for i in range(8):
        for j in range(4):
            # 用不同颜色表示 ECMP 路径
            color = 'blue' if i % 2 == 0 else 'green'
            ax.plot([tor_x[i], spine_x[j]], [tor_y, spine_y], color=color,
                   linewidth=1, alpha=0.3, linestyle='--')

    # 标注
    for i in range(4):
        ax.text(spine_x[i], spine_y + 0.15, f'S{j}', ha='center', fontsize=10, weight='bold')
    for i in range(8):
        ax.text(tor_x[i], tor_y + 0.1, f'TOR{i}', ha='center', fontsize=8)
    for i in range(16):
        ax.text(host_x[i], host_y - 0.15, f'H{i}', ha='center', fontsize=7)

    # 带宽标注
    ax.annotate('', xy=(spine_x[0], spine_y), xytext=(spine_x[1], spine_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text((spine_x[0]+spine_x[1])/2, spine_y + 0.25, '210 Gbps', ha='center',
            fontsize=9, color='red', weight='bold')

    ax.annotate('', xy=(tor_x[0], tor_y), xytext=(host_x[0], host_y),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(tor_x[0] - 0.3, (tor_y + host_y) / 2, '25 Gbps', ha='center',
            fontsize=9, color='blue', weight='bold', rotation=90)

    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Fat-Tree Topology (k=4, 16 hosts)\nECMP: 2 spine links per TOR',
                fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / 'topology_fattree.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'topology_fattree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Topology plot → {out_dir / 'topology_fattree.pdf'}")


def plot_job_distribution(workload: list, out_dir: Path):
    """绘制 Job 在 Host 上的分布。"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 分配 host 给每个 job
    # 简化假设：按顺序分配
    host_assignment = defaultdict(list)  # host_id -> list of (job_id, tier)
    host_idx = 0

    for jid, model, dp, ci in workload:
        tier = 'Tight' if ci == 1.5 else ('Medium' if ci == 2.0 else 'Loose')
        for _ in range(dp):
            host_assignment[host_idx % 16].append((jid, tier, model))
            host_idx += 1

    # 绘制 host 柱状图
    colors = {'Tight': '#e74c3c', 'Medium': '#f39c12', 'Loose': '#27ae60'}

    for h in range(16):
        jobs_on_host = host_assignment.get(h, [])
        y_offset = 0
        for jid, tier, model in jobs_on_host:
            ax.barh(h, 1, left=y_offset, height=0.8, color=colors[tier],
                   edgecolor='black', linewidth=0.5, alpha=0.7)
            # 只在第一个块上标注 jid
            if y_offset == 0:
                ax.text(y_offset + 0.5, h, jid, ha='center', va='center',
                       fontsize=7, color='white', weight='bold')
            y_offset += 1

    # 图例
    legend_patches = [
        mpatches.Patch(color='#e74c3c', label=f'Tight (ci=1.5): 12 jobs, dp≥4'),
        mpatches.Patch(color='#f39c12', label=f'Medium (ci=2.0): 8 jobs'),
        mpatches.Patch(color='#27ae60', label=f'Loose (ci=3.0): 4 jobs, dp=1'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    ax.set_xlabel('GPU slots occupied', fontsize=11)
    ax.set_ylabel('Host ID', fontsize=11)
    ax.set_yticks(range(16))
    ax.set_yticklabels([f'H{i}' for i in range(16)])
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 16.5)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_title('Job Distribution across Hosts (24 concurrent DDP jobs)\n'
                'Color = SLO tier, Bar width = GPUs used',
                fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / 'job_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'job_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Job distribution plot → {out_dir / 'job_distribution.pdf'}")


def plot_job_summary(workload: list, out_dir: Path):
    """绘制 Job 汇总表（模型、DP、SLO tier）。"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 按-tier 排序
    sorted_jobs = sorted(workload, key=lambda x: -x[3])

    # 绘制表格
    cell_text = []
    cell_colors = []
    tier_colors = {'Tight': '#ffcccc', 'Medium': '#fff3cd', 'Loose': '#d4edda'}

    for jid, model, dp, ci in sorted_jobs:
        tier = 'Tight' if ci == 1.5 else ('Medium' if ci == 2.0 else 'Loose')
        comm_mb = {  # 通信量估算
            "LLaMA-2-13B": 650, "LLaMA-2-7B": 350, "T5-11B": 550,
            "BERT-Large": 34, "T5-base": 110,
            "ResNet-18": 0, "GPT-2-small": 25
        }.get(model, 100)
        cell_text.append([jid, model, str(dp), f'{comm_mb} MB', f'{ci}', tier])
        cell_colors.append(['white', 'white', 'white', 'white', 'white', tier_colors[tier]])

    table = ax.table(cellText=cell_text,
                     colLabels=['Job ID', 'Model', 'DP', 'Comm/iter', 'ci', 'Tier'],
                     cellLoc='center',
                     loc='center',
                     cellColours=cell_colors,
                     colColours=['#f0f0f0']*6)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # 样式
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')

    ax.axis('off')
    ax.set_title('Workload Summary: 24 DDP Training Jobs\n'
                'Sorted by SLO tier (Tight → Loose)',
                fontsize=12, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(out_dir / 'job_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'job_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Job summary table → {out_dir / 'job_summary.pdf'}")


def plot_network_contention(out_dir: Path):
    """绘制网络竞争示意图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：链路竞争热点
    ax1 = axes[0]

    # Spine links
    spine_y = 3.0
    spine_x = [1.5, 5.5, 9.5, 13.5]

    # TOR nodes
    tor_y = 2.0
    tor_x = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]

    # Host nodes
    host_y = 1.0
    hosts = []
    for i in range(8):
        hosts.append((tor_x[i] - 0.3, host_y))
        hosts.append((tor_x[i] + 0.3, host_y))

    # 绘制竞争强度的链路
    # 假设：大模型流量集中在少数 spine link
    contention_links = [
        (0, 0, 0.9),  # spine0 - tor0, 高竞争
        (0, 1, 0.8),
        (1, 2, 0.7),
        (2, 4, 0.6),
        (3, 6, 0.5),
    ]

    for spine_idx, tor_idx, intensity in contention_links:
        ax1.plot([spine_x[spine_idx], tor_x[tor_idx]],
                [spine_y, tor_y],
                color='red', linewidth=3*intensity, alpha=0.7)

    # 绘制低竞争链路
    for spine_idx in range(4):
        for tor_idx in range(8):
            if (spine_idx, tor_idx) not in [(s, t) for s, t, _ in contention_links]:
                ax1.plot([spine_x[spine_idx], tor_x[tor_idx]],
                        [spine_y, tor_y],
                        color='gray', linewidth=1, alpha=0.3, linestyle='--')

    # 绘制节点
    ax1.scatter(spine_x, [spine_y]*4, s=400, c='red', marker='s', zorder=3)
    ax1.scatter(tor_x, [tor_y]*8, s=250, c='orange', marker='s', zorder=3)
    for hx, hy in hosts:
        ax1.scatter(hx, hy, s=150, c='lightblue', marker='o', zorder=3)

    ax1.set_xlim(-0.5, 15.5)
    ax1.set_ylim(0.5, 3.5)
    ax1.axis('off')
    ax1.set_title('Network Contention Hotspots\n(Red = High congestion)',
                  fontsize=11, weight='bold')

    # 右图：带宽分配示意
    ax2 = axes[1]

    # 饼图：带宽分配给不同 tier
    labels = ['Tight (12 jobs)', 'Medium (8 jobs)', 'Loose (4 jobs)']
    sizes = [60, 30, 10]  # 假设的带宽分配比例
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    explode = (0.05, 0, 0)

    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels,
                                        colors=colors, autopct='%1.0f%%',
                                        startangle=90, pctdistance=0.6)
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_weight('bold')

    ax2.set_title('Bandwidth Allocation by SLO Tier\n(LongLiu dynamic scheduling)',
                  fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / 'network_contention.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'network_contention.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Network contention plot → {out_dir / 'network_contention.pdf'}")


def main():
    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / 'outputs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading workload data...")
    workload = load_workload_data()

    print("\nGenerating plots...")
    plot_fattree_topology(out_dir)
    plot_job_distribution(workload, out_dir)
    plot_job_summary(workload, out_dir)
    plot_network_contention(out_dir)

    print("\nDone! Output directory:", out_dir)


if __name__ == "__main__":
    main()