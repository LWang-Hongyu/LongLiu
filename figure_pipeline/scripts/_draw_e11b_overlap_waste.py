"""
绘制 E11b 分配精度（Allocation Precision）图（fig_e11b_overlap_waste）。
完全匹配 E10-E15 系列风格（分组柱状、扁平、大字号、图例框外上方）。

x 轴: Overlap factor ρ ∈ {0, 0.3, 0.5, 0.85, 1.0}
y 轴: Allocation Precision = Σ min(a_i, b_i^att) / Σ a_i
双 panel: 500 Gbps / 630 Gbps，LongLiu vs WFS。
"""

import csv


# ---- 统一路径（figure_pipeline 根，相对于本脚本位置）----
import os
import sys
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
    'axes.titlesize': 24,
    'xtick.labelsize': 23.4,
    'ytick.labelsize': 23.4,
    'legend.fontsize': 20.8,
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
})

COLOR_MAP = {
    'v4': '#1f77b4',   # LongLiu 蓝
    'WFS': '#A0522D',  # WFS 棕（与 E10 一致）
}
LABEL_MAP = {
    'v4': 'LongLiu',
    'WFS': 'WFS',
}


def load_summary():
    csv_path = Path(DATA_DIR, "e11b_overlap_waste", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e11b_precision():
    # 输出目录：优先命令行参数，否则系统临时目录（figs 目录可能被占用/沙箱限制）
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", FIG_DIR)
    data = load_summary()
    spine_bws = sorted(set(int(r["spine_bw"]) for r in data))
    overlaps = sorted(set(float(r["overlap_factor"]) for r in data))
    policies = ["v4", "WFS"]

    fig, axes = plt.subplots(1, len(spine_bws), figsize=(13, 4.2), sharey=True)
    if len(spine_bws) == 1:
        axes = [axes]

    x = np.arange(len(overlaps))
    width = 0.3

    for ax_idx, bw in enumerate(spine_bws):
        ax = axes[ax_idx]
        bw_data = [r for r in data if int(r["spine_bw"]) == bw]

        for i, policy in enumerate(policies):
            precs, prec_stds = [], []
            for ov in overlaps:
                match = [r for r in bw_data
                         if float(r["overlap_factor"]) == ov and r["policy"] == policy]
                if match:
                    precs.append(float(match[0]["allocation_precision_mean"]))
                    prec_stds.append(float(match[0]["allocation_precision_std"]))
                else:
                    precs.append(0.0)
                    prec_stds.append(0.0)
            ax.bar(x + i * width, precs, width,
                   label=LABEL_MAP[policy], color=COLOR_MAP[policy],
                   edgecolor='black', linewidth=0.8,
                   yerr=prec_stds, capsize=4, ecolor='black')

        ax.set_xlabel(r"Overlap factor $\rho$", fontsize=28.6)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{ov:g}" for ov in overlaps])
        ax.set_title(f"{bw} Gbps", fontsize=24)
        ax.set_ylim(0, 1.15)

    axes[0].set_ylabel("Allocation Precision", fontsize=28.6)

    fig.subplots_adjust(top=0.80, bottom=0.12)
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="lower center",
                    bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False,
                    fontsize=20.8)
    plt.savefig(str(Path(out_dir, "fig_e11b_overlap_waste.pdf")), bbox_inches='tight')
    try:
        plt.savefig(str(Path(out_dir, "fig_e11b_overlap_waste.png")), dpi=300, bbox_inches='tight')
    except PermissionError:
        print("WARN: PNG locked (skipped), PDF saved")
    print("OK fig_e11b_overlap_waste")


if __name__ == "__main__":
    plot_e11b_precision()
