"""
绘制 E12 DSCP 量化误差实验结果图。
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

COLOR_MAP = {'v4': '#1f77b4', 'D1': '#2E8B57'}
LABEL_MAP = {'v4': 'LongLiu', 'D1': 'LongLiu-DWRR'}


def load_summary():
    csv_path = Path(DATA_DIR, "e12_dscp", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e12_dscp():
    data = load_summary()
    n_jobs_list = sorted(set(int(r["n_jobs"]) for r in data))
    policies = ["v4", "D1"]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(n_jobs_list))
    width = 0.35

    for i, policy in enumerate(policies):
        p_attns, p_stds = [], []
        for nj in n_jobs_list:
            match = [r for r in data if int(r["n_jobs"]) == nj and r["policy"] == policy]
            if match:
                p_attns.append(float(match[0]["p_attn_mean"]) * 100)
                p_stds.append(float(match[0]["p_attn_std"]) * 100)
            else:
                p_attns.append(0)
                p_stds.append(0)

        ax.bar(x + i * width, p_attns, width,
               label=LABEL_MAP[policy], color=COLOR_MAP[policy],
               edgecolor='black', linewidth=0.8,
               yerr=p_stds, capsize=4, ecolor='black')

    ax.set_xlabel("Number of Jobs", fontsize=22)
    ax.set_ylabel("P-attn (%)", fontsize=22)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(nj) for nj in n_jobs_list])
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(str(Path(FIG_DIR, "fig_e12_dscp.png")), dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e12_dscp.pdf")), bbox_inches='tight')
    print("OK fig_e12_dscp")


if __name__ == "__main__":
    plot_e12_dscp()
