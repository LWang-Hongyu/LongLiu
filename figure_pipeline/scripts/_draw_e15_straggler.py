"""
绘制 E15 straggler 注入实验结果图。
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
    'font.size': 26,
    'axes.labelsize': 28.6,
    'axes.titlesize': 31.2,
    'xtick.labelsize': 23.4,
    'ytick.labelsize': 23.4,
    'legend.fontsize': 20.8,
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
})

COLOR_MAP = {
    'Fair': '#808080', 'CRUX': '#D2691E', 'SP': '#DAA520',
    'D1': '#2E8B57', 'v4': '#1f77b4',
}
LABEL_MAP = {
    'Fair': 'Fair', 'CRUX': 'CRUX', 'SP': 'SP',
    'D1': 'LongLiu-DWRR', 'v4': 'LongLiu',
}


def load_summary():
    csv_path = Path(DATA_DIR, "e15_straggler", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e15_straggler():
    data = load_summary()
    straggler_factors = sorted(set(float(r["straggler_factor"]) for r in data))
    policies = ["Fair", "CRUX", "SP", "D1", "v4"]

    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(straggler_factors))
    width = 0.15

    for i, policy in enumerate(policies):
        p_attns, p_stds = [], []
        for sf in straggler_factors:
            match = [r for r in data
                     if float(r["straggler_factor"]) == sf and r["policy"] == policy]
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

    ax.set_xlabel("Straggler Factor", fontsize=28.6)
    ax.set_ylabel("P-attn (%)", fontsize=28.6)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"{sf:.1f}$\\times$" for sf in straggler_factors])
    ax.set_ylim(0, 100)

    fig.subplots_adjust(top=0.80, bottom=0.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False)
    plt.savefig(str(Path(FIG_DIR, "fig_e15_straggler.png")), dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e15_straggler.pdf")), bbox_inches='tight')
    print("OK fig_e15_straggler")


if __name__ == "__main__":
    plot_e15_straggler()
