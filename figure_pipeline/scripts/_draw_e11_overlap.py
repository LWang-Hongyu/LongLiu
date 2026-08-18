"""
绘制 E11 overlap 因子敏感性实验结果图。
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
    csv_path = Path(DATA_DIR, "e11_overlap", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e11_overlap():
    data = load_summary()
    spine_bws = sorted(set(int(r["spine_bw"]) for r in data))
    overlap_factors = sorted(set(float(r["overlap_factor"]) for r in data))
    policies = ["Fair", "CRUX", "SP", "D1", "v4"]

    fig, axes = plt.subplots(1, len(spine_bws), figsize=(15, 5), sharey=True)
    if len(spine_bws) == 1:
        axes = [axes]

    for ax_idx, bw in enumerate(spine_bws):
        ax = axes[ax_idx]
        bw_data = [r for r in data if int(r["spine_bw"]) == bw]
        x = np.arange(len(overlap_factors))
        width = 0.15

        for i, policy in enumerate(policies):
            p_attns, p_stds = [], []
            for ov in overlap_factors:
                match = [r for r in bw_data
                         if float(r["overlap_factor"]) == ov and r["policy"] == policy]
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

        ax.set_xlabel(r"Overlap factor $\rho$", fontsize=28.6)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f"{ov:.2g}" for ov in overlap_factors])
        ax.set_title(f"{bw} Gbps", fontsize=31.2)
        ax.set_ylim(0, 115)

    axes[0].set_ylabel("P-attn (%)", fontsize=28.6)

    # 收集所有 handles/labels，放到图框上方、上框线之外
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=5, frameon=False, fontsize=20.8)
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.09, right=0.97, wspace=0.12)
    plt.savefig(str(Path(FIG_DIR, "fig_e11_overlap.png")), dpi=300, bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e11_overlap.pdf")), bbox_inches='tight')
    print("OK fig_e11_overlap")


if __name__ == "__main__":
    plot_e11_overlap()
