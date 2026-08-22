"""
绘制 E16 β 敏感性实验结果图（fig_e16_beta）。

单面板（无子图）:
- x 轴: β ∈ {0.3, 0.5, 0.7, 1.0}
- 左轴: P-attn (%)，500G/630G 两条曲线（蓝色）
- 右轴: S-cont (avg SAS)，500G/630G 两条曲线（巧克力色）
- 带宽区分: 500G 实线、630G 虚线
坐标轴标题/刻度均为黑色，风格与 E10-E15 系列一致。
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

C_PATTN = '#1f77b4'   # LongLiu 蓝
C_SCONT = '#D2691E'   # chocolate


def load_summary():
    csv_path = Path(DATA_DIR, "e16_beta", "summary.csv")
    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data


def plot_e16_beta():
    data = load_summary()
    spine_bws = sorted(set(int(r["spine_bw"]) for r in data))
    betas = sorted(set(float(r["beta"]) for r in data))

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax2 = ax.twinx()

    x = np.arange(len(betas))
    for bw in spine_bws:
        bw_data = [r for r in data if int(r["spine_bw"]) == bw]
        bw_data = sorted(bw_data, key=lambda r: float(r["beta"]))

        p_attns = [float(r["p_attn_mean"]) * 100 for r in bw_data]
        p_stds = [float(r["p_attn_std"]) * 100 for r in bw_data]
        s_conts = [float(r["s_cont_cap_mean"]) for r in bw_data]

        ls = '-' if bw == spine_bws[0] else '--'  # 500G 实线, 630G 虚线

        # 左轴: P-attn（高位曲线）
        ax.plot(x, p_attns, ls, color=C_PATTN, lw=2.5, ms=8, marker='o',
                label=f"P-attn {bw}G")
        ax.errorbar(x, p_attns, yerr=p_stds, fmt='none', ecolor='black',
                    capsize=4, lw=1.5)

        # 右轴: S-cont（中低位曲线，与 P-attn 错开）
        ax2.plot(x, s_conts, ls, color=C_SCONT, lw=2.5, ms=8, marker='s',
                 label=f"S-cont {bw}G")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_xlabel(r"$\beta$", fontsize=28.6)
    ax.set_ylim(0, 115)
    ax.set_ylabel("P-attn (%)", fontsize=28.6)

    ax2.set_ylim(0.95, 1.02)
    ax2.set_ylabel("S-cont (avg SAS)", fontsize=28.6)

    # 图例：放图框上方，与 E10-E15 系列一致
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center",
              bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False,
              fontsize=20.8)
    fig.subplots_adjust(top=0.80, bottom=0.12)
    plt.savefig(str(Path(FIG_DIR, "fig_e16_beta.png")), dpi=300,
                bbox_inches='tight')
    plt.savefig(str(Path(FIG_DIR, "fig_e16_beta.pdf")), bbox_inches='tight')
    print("OK fig_e16_beta")


if __name__ == "__main__":
    plot_e16_beta()
