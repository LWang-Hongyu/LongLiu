"""
绘制 E11 overlap 因子敏感性实验结果图。
完全匹配主图风格。
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
    'Fair': '#999999', 'SRPT': '#E69F00', 'CRUX': '#D55E00',
    'CASSINI': '#CC79A7', 'DF': '#009E73', 'LL-S': '#A65628',
    'LongLiu': '#0072B2',
}
LABEL_MAP = {
    'Fair': 'Fair', 'SRPT': 'SRPT', 'CRUX': 'CRUX',
    'CASSINI': 'CASSINI', 'DF': 'DF', 'LL-S': 'LL-S',
    'LongLiu': 'LongLiu',
}
# LongLiu 及其变体（LongLiu/LL-S/DF）在最前，Fair 最后
POLICIES = ["LongLiu", "LL-S", "DF", "SRPT", "CRUX", "CASSINI", "Fair"]


def load_summary():
    csv_path = Path(DATA_DIR, "e11_overlap", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e11_overlap():
    # 输出目录：优先命令行参数，否则系统临时目录（figs 目录可能被占用/沙箱限制）
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", FIG_DIR)
    data = load_summary()
    spine_bws = sorted(set(int(r["spine_bw"]) for r in data))
    overlap_factors = sorted(set(float(r["overlap_factor"]) for r in data))
    policies = [p for p in POLICIES if any(r["policy"] == p for r in data)]

    fig, axes = plt.subplots(len(spine_bws), 1, figsize=(10, 5), sharex=True)
    if len(spine_bws) == 1:
        axes = [axes]

    for ax_idx, bw in enumerate(spine_bws):
        ax = axes[ax_idx]
        bw_data = [r for r in data if int(r["spine_bw"]) == bw]
        x = np.arange(len(overlap_factors))
        width = 0.12

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

        ax.set_xticks(x + (len(policies) - 1) * width / 2)
        ax.set_ylabel("P-attn (%)", fontsize=28.6)
        ax.set_title(f"({chr(97 + ax_idx)}) {bw} Gbps", fontsize=24, pad=12)
        ax.set_ylim(0, 115)

    axes[-1].set_xlabel(r"Overlap factor $\rho$", fontsize=28.6)
    axes[-1].set_xticklabels([f"{ov:.2g}" for ov in overlap_factors])

    # 收集所有 handles/labels，放到图框上方、上框线之外（尽量一行）
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.17),
               ncol=4, frameon=False, fontsize=20.8)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.09, right=0.97, hspace=0.5)
    plt.savefig(str(Path(out_dir, "fig_e11_overlap.pdf")), bbox_inches='tight')
    try:
        plt.savefig(str(Path(out_dir, "fig_e11_overlap.png")), dpi=300, bbox_inches='tight')
    except PermissionError:
        print("WARN: PNG locked (skipped), PDF saved")
    print("OK fig_e11_overlap")


if __name__ == "__main__":
    plot_e11_overlap()
