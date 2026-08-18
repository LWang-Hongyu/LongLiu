"""
绘制 E14 锚点冻结设计案例图（v2 叙事）。
对比 baseline / naive probe / passive recalibration 的 P-attn。
完全匹配 E 系列主图风格。
"""

import matplotlib.pyplot as plt


# ---- 统一路径（figure_pipeline 根，相对于本脚本位置）----
import os
from pathlib import Path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(_THIS_DIR)        # figure_pipeline/
DATA_DIR = os.path.join(PIPE_DIR, "data")
FIG_DIR  = os.path.join(PIPE_DIR, "figs")
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
})


def plot_e14_anchor():
    # 数据来源：
    #   baseline   : outputs/e14_probe/summary_passive_10seeds.csv (10 seeds)
    #   naive probe: experiments/_quick_scan_e14_passive.py         (2 seeds, best config)
    #   passive    : outputs/e14_probe/summary_passive_10seeds.csv (10 seeds)
    labels = ["Baseline\n(control loop)", "Naive probe", "Passive\nrecalibration"]
    p_attns = [53.3, 48.3, 55.3]
    p_stds = [11.5, 2.3, 8.6]
    colors = ['#808080', '#D2691E', '#1f77b4']

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))

    ax.bar(x, p_attns, 0.55, color=colors,
           edgecolor='black', linewidth=0.8,
           yerr=p_stds, capsize=5, ecolor='black')

    ax.set_ylabel("P-attn (%)", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 75)

    plt.tight_layout()
    plt.savefig(str(Path(FIG_DIR, "fig_e14_anchor.png")), dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e14_anchor.pdf")), bbox_inches='tight')
    print("OK fig_e14_anchor")


if __name__ == "__main__":
    plot_e14_anchor()
