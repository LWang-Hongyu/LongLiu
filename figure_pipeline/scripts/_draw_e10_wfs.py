"""
绘制 E10 WFS 基线对照实验结果图。
完全匹配 fig2_e1_ladder / fig6_trace_compare 风格。
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
    'Fair': '#808080',
    'WFS': '#A0522D',
    'CRUX': '#D2691E',
    'SP': '#DAA520',
    'D1': '#2E8B57',
    'v4': '#1f77b4',
}
LABEL_MAP = {
    'Fair': 'Fair', 'WFS': 'WFS', 'CRUX': 'CRUX',
    'SP': 'SP', 'D1': 'LongLiu-DWRR', 'v4': 'LongLiu',
}


def load_summary():
    csv_path = Path(DATA_DIR, "e10_wfs", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e10_wfs():
    data = load_summary()
    e1_data = [r for r in data if r["scene"] == "E1"]
    spine_bws = sorted(set(int(r["spine_bw"]) for r in e1_data))
    policies = ["Fair", "WFS", "CRUX", "SP", "D1", "v4"]

    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(spine_bws))
    width = 0.13

    for i, policy in enumerate(policies):
        p_attns, p_stds = [], []
        for bw in spine_bws:
            match = [r for r in e1_data if int(r["spine_bw"]) == bw and r["policy"] == policy]
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

    ax.set_xlabel("Spine Bandwidth (Gbps)", fontsize=28.6)
    ax.set_ylabel("P-attn (%)", fontsize=28.6)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([str(bw) for bw in spine_bws])
    ax.set_ylim(0, 115)

    fig.subplots_adjust(top=0.80, bottom=0.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False)
    plt.savefig(str(Path(FIG_DIR, "fig_e10_wfs.png")), dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e10_wfs.pdf")), bbox_inches='tight')
    print("OK fig_e10_wfs")


if __name__ == "__main__":
    plot_e10_wfs()
