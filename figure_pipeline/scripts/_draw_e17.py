"""E17 混合集合通信：P-attn vs spine 带宽（7 策略，10 seeds）——与 fig3_e2_orthogonal 同风格。

复用 _draw_final_v3 的 POLICY_COLOR/LS/MARKER/LABEL/ORDER 与 fig3 布局：
figsize=(7.16, 2.8)、图例顶部两行透明、errorbar markersize=7 lw=1.4 capsize=2.5、
ylim(0,120)、grid、tight_layout(rect=[0,0.02,1,0.96])。
读 figure_pipeline/data/figure_registry/fig7_e17_mixed.csv，输出 fig7_e17_mixed.pdf/png。
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _draw_final_v3 import (  # noqa: E402
    POLICY_COLOR, POLICY_LS, POLICY_MARKER, POLICY_LABEL, POLICY_ORDER, FULL_W,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(ROOT, "figure_pipeline", "figs")
FIG_REG = os.path.join(ROOT, "figure_pipeline", "data", "figure_registry")
CSV = os.path.join(FIG_REG, "fig7_e17_mixed.csv")


def load():
    """{policy: {bw: (mean, std)}}。CSV 无 header：scene,sp_bw,policy,n_seeds,seeds,mean,std。"""
    data = {}
    with open(CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            policy, bw, mean, std = parts[2], int(parts[1]), float(parts[5]), float(parts[6])
            data.setdefault(policy, {})[bw] = (mean, std)
    return data


def draw():
    data = load()
    bws = [400, 500, 630, 800]

    fig, ax = plt.subplots(figsize=(FULL_W, 2.0))
    for pol in POLICY_ORDER:
        if pol not in data:
            continue
        m = np.array([data[pol][b][0] * 100 for b in bws])
        s = np.array([data[pol][b][1] * 100 for b in bws])
        ax.errorbar(bws, m, yerr=s, color=POLICY_COLOR[pol],
                    ls=POLICY_LS[pol], marker=POLICY_MARKER[pol],
                    markersize=7, lw=1.4, capsize=2.5, label=POLICY_LABEL[pol])

    ax.set_xlabel("Spine bandwidth (Gbps)", fontsize=12, labelpad=8)
    ax.set_ylabel("P-attn (%)", fontsize=12, labelpad=8)
    ax.set_ylim(0, 120)
    ax.set_xticks(bws)
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10.7, ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout(rect=[0, 0.02, 1, 0.96], pad=1.5)
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig7_e17_mixed.{ext}"), dpi=300,
                    bbox_inches="tight")
    print("saved fig7_e17_mixed.pdf/png (fig3 style)")


if __name__ == "__main__":
    draw()
