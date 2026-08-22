"""
绘制 E13 窗口大小敏感性实验结果图。
完全匹配主图风格。
"""

import csv


# ---- 统一路径（figure_pipeline 根，相对于本脚本位置）----
import os
from pathlib import Path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(_THIS_DIR)        # figure_pipeline/
DATA_DIR = os.path.join(PIPE_DIR, "data")
FIG_DIR  = os.path.join(PIPE_DIR, "figs")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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


def load_summary():
    csv_path = Path(DATA_DIR, "e13_window", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e13_window():
    data = load_summary()
    window_sizes = sorted(set(int(r["window_size"]) for r in data))

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(window_sizes))
    width = 0.5

    p_attns, p_stds = [], []
    for w in window_sizes:
        match = [r for r in data if int(r["window_size"]) == w]
        if match:
            p_attns.append(float(match[0]["p_attn_mean"]) * 100)
            p_stds.append(float(match[0]["p_attn_std"]) * 100)
        else:
            p_attns.append(0)
            p_stds.append(0)

    ax.bar(x, p_attns, width,
           color='#1f77b4', edgecolor='black', linewidth=0.8,
           yerr=p_stds, capsize=4, ecolor='black')

    ax.set_xlabel("Window Size $W$", fontsize=22)
    ax.set_ylabel("P-attn (%)", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in window_sizes])
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(str(Path(FIG_DIR, "fig_e13_window.png")), dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e13_window.pdf")), bbox_inches='tight')
    print("OK fig_e13_window")


if __name__ == "__main__":
    plot_e13_window()
