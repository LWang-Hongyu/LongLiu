"""
绘制 E14 锚点冻结与主动探测实验结果图。
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
    csv_path = Path(DATA_DIR, "e14_probe", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e14_probe():
    data = load_summary()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(2)
    width = 0.5

    labels = ["No Probe", "With Probe"]
    p_attns, p_stds = [], []
    for row in sorted(data, key=lambda r: r["probe_enabled"]):
        p_attns.append(float(row["p_attn_mean"]) * 100)
        p_stds.append(float(row["p_attn_std"]) * 100)

    ax.bar(x, p_attns, width,
           color=['#D2691E', '#1f77b4'],
           edgecolor='black', linewidth=0.8,
           yerr=p_stds, capsize=4, ecolor='black')

    ax.set_xlabel("Active Probing", fontsize=22)
    ax.set_ylabel("P-attn (%)", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(str(Path(FIG_DIR, "fig_e14_probe.png")), dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e14_probe.pdf")), bbox_inches='tight')
    print("OK fig_e14_probe")


if __name__ == "__main__":
    plot_e14_probe()
