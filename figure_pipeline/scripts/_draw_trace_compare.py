"""
Lingjun trace 对照实验 P-attn 柱状图（论文 §6.6 robustness 章节）。

结构（单栏）：
- Lingjun 2023 trace 时段重放：P-attn 柱状图（5 策略 × 10 seeds，400-Gbps spine）

叙事：v4（LongLiu）在真实到达模式下保持最高 premium 达标率——
      trace 真实到达模式下 v4=75% 领先全部 baseline（Fair p=1.09e-06）。
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':         20,
    'axes.titlesize':    24,
    'axes.labelsize':    22,
    'xtick.labelsize':   18,
    'ytick.labelsize':   18,
    'legend.fontsize':   16,
    'axes.linewidth':    2.0,
    'axes.edgecolor':    'black',
    'savefig.dpi':       600,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# Okabe-Ito 色彩
POLICY_COLOR  = {"LongLiu":"#0072B2","CRUX":"#D55E00","DF":"#009E73",
                 "SP":"#E69F00","Fair":"#999999"}
POLICY_LABEL  = {"LongLiu":"LongLiu","CRUX":"CRUX","DF":"DF","SP":"SP","Fair":"Fair"}
POLICY_ORDER  = ["LongLiu","DF","CRUX","SP","Fair"]

FULL_W = 13.0  # 扁平宽图，匹配 E10-E15 系列
PROJ   = _BASE
PIPE_DIR = os.path.join(PROJ, "figure_pipeline")
REG    = os.path.join(PIPE_DIR, "data", "figure_registry")
TRACE_CSV = os.path.join(REG, "fig6_trace_compare.csv")  # 权威数据源
OUT_DIR = os.path.join(PIPE_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# trace CSV 中策略名 → 论文策略名
TRACE2POL = {"v4": "LongLiu", "D1": "DF"}


def load_trace():
    """Lingjun trace 重放：policy -> (mean, std)，从 FIGURE_REGISTRY 注册 CSV 读取。"""
    data = {}
    with open(TRACE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('scene') != 'TRACE':
                continue
            pol = TRACE2POL.get(row['policy'], row['policy'])
            data[pol] = (float(row['p_attn_mean']), float(row['p_attn_std']))
    return data


def save_both(fig, stem):
    fig.savefig(f"{stem}.pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f"{stem}.png", bbox_inches='tight', pad_inches=0.02)
    print(f"  -> {os.path.basename(stem)}.pdf + .png")


def draw():
    print("=== Fig-6: Lingjun Trace P-attn (single panel) ===")
    tr = load_trace()

    fig, ax = plt.subplots(1, 1, figsize=(FULL_W, 4.2))

    pols = [p for p in POLICY_ORDER if p in tr]
    means = np.array([tr[p][0] * 100 for p in pols])
    stds  = np.array([tr[p][1] * 100 for p in pols])
    colors = [POLICY_COLOR[p] for p in pols]

    bars = ax.bar(range(len(pols)), means, yerr=stds, capsize=4,
                  width=0.6, color=colors, edgecolor='black', lw=0.8,
                  error_kw=dict(ecolor='black', lw=1.0), zorder=5)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([POLICY_LABEL[p] for p in pols])
    ax.set_ylim(0, 100)
    ax.set_ylabel("P-attn (%)", fontsize=22)

    fig.tight_layout(pad=1.2)
    stem = os.path.join(OUT_DIR, "fig6_trace_compare")
    save_both(fig, stem)
    plt.close(fig)

    # 打印结果
    print("\n  Lingjun trace (10 seeds, P-attn%):")
    for pol in POLICY_ORDER:
        print(f"    {POLICY_LABEL[pol]:<9} {tr[pol][0]*100:5.1f} ± {tr[pol][1]*100:.1f}")
    return stem


if __name__ == "__main__":
    draw()
