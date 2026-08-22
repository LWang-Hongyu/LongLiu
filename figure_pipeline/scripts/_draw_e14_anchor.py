"""
绘制 E14 锚点冻结设计案例图（v2 叙事）。
对比 baseline / naive probe / passive recalibration / LongLiu closed-form 的 P-attn。
数据全部从 evidence/ 目录读取，不硬编码。
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(_THIS_DIR)        # figure_pipeline/
FIG_DIR = os.path.join(PIPE_DIR, "figs")
EVID_DIR = os.path.join(PIPE_DIR, "data", "evidence", "e14_anchor_frozen")

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


def read_csv_rows(path: str) -> dict:
    """读取 summary.csv，返回 {config_name: row_dict}。"""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return {r["config"] if "config" in r else r["strategy"] if "strategy" in r else r["probe_enabled"]: r
                for r in reader}


def mean_std(value_str: str) -> tuple:
    """把 '0.5333±0.1155' 或 '0.8367±0.069' 拆成 (mean, std)（百分比）。"""
    m, s = value_str.split("±")
    return float(m) * 100.0, float(s) * 100.0


def plot_e14_anchor():
    # 输出目录：优先命令行参数，否则系统临时目录（figs 目录可能被占用/沙箱限制）
    global out_dir
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", FIG_DIR)
    # baseline / passive: 10 seeds（同批次）
    base_rows = read_csv_rows(os.path.join(EVID_DIR, "summary_passive_10seeds.csv"))
    # naive probe: 2 seeds（probe_pass，T10D1 扫描最优）
    naive_rows = read_csv_rows(os.path.join(EVID_DIR, "naive_probe", "summary_2seeds.csv"))
    # v4 闭式解: 10 seeds
    v4_rows = read_csv_rows(os.path.join(EVID_DIR, "v4_closure", "summary_v4.csv"))

    labels = ["Baseline\n(control loop)", "Naive probe", "Passive\nrecalibration", "LongLiu\n(closed form)"]
    p_attns, p_stds = [], []
    for cfg in ["baseline", "probe_pass", "passive_low", "v4"]:
        if cfg == "probe_pass":
            row = naive_rows["probe_pass"]
            vals = [row["seed0_p_attn"], row["seed1_p_attn"]]
            m = float(row["mean"]) * 100.0
            s = float(np.std([float(v) for v in vals], ddof=1)) * 100.0
        else:
            row = (base_rows if cfg in base_rows else v4_rows)[cfg]
            m, s = mean_std(f"{row['p_attn_mean']}±{row['p_attn_std']}")
        p_attns.append(m)
        p_stds.append(s)

    colors = ['#808080', '#D2691E', '#1f77b4', '#2ca02c']

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))

    ax.bar(x, p_attns, 0.55, color=colors,
           edgecolor='black', linewidth=0.8,
           yerr=p_stds, capsize=5, ecolor='black')

    ax.set_ylabel("P-attn (%)", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_pdf = os.path.join(out_dir, "fig_e14_anchor.pdf")
    plt.savefig(out_pdf, bbox_inches='tight')
    try:
        plt.savefig(os.path.join(out_dir, "fig_e14_anchor.png"), dpi=300, bbox_inches='tight')
    except PermissionError:
        print("WARN: PNG locked (skipped), PDF saved")
    print("OK fig_e14_anchor ->", out_pdf,
          [f"{m:.1f}±{s:.1f}" for m, s in zip(p_attns, p_stds)])


if __name__ == "__main__":
    plot_e14_anchor()
