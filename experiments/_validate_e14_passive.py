"""
E14 被动校准 5-seed 验证：P4=0.2/P5=0.4 vs 默认权重 vs 新基线。

用法：
    python experiments/_validate_e14_passive.py
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import exp_e14_probe as E14
from exp_e14_probe import create_high_load_workload, load_e14_config, load_frozen, run_single

BASE = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # 默认权重（level 0-6）
PASSIVE_LOW = [0.05, 0.1, 0.2, 0.4, 0.2, 0.4, 1.0]  # P4=0.2, P5=0.4


def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")) as f:
        E14.CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    frozen = load_frozen()
    cfg = load_e14_config()
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9
    workload = create_high_load_workload(cfg["n_jobs"], seed=42)
    seeds = list(range(5))

    configs = [
        ("baseline",   False, None),
        ("passive_default", True, None),
        ("passive_low", True, PASSIVE_LOW),
    ]

    results = {name: [] for name, _, _ in configs}
    print(f"{'config':>16s} | " + " ".join(f"s{i:>5s}" for i in map(str, seeds)) + " | mean±std")
    print("-" * 76)
    for name, passive, weights in configs:
        vals = []
        for seed in seeds:
            r = run_single(False, workload, spine_bw, seed, frozen,
                           ema_passive=passive, ema_weights=weights)
            vals.append(r["p_attn"])
            results[name].append(r)
        mean = np.mean(vals) * 100
        std = np.std(vals) * 100
        print(f"{name:>16s} | " + " ".join(f"{v*100:5.1f}" for v in vals) + f" | {mean:.1f}±{std:.1f}")

    # 保存 summary CSV
    os.makedirs("outputs/e14_probe", exist_ok=True)
    with open("outputs/e14_probe/summary_passive.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_seeds", "p_attn_mean", "p_attn_std",
                         "n_frozen_jobs_mean", "p_cap_mean", "s_cont_cap_mean"])
        for name, _, _ in configs:
            runs = results[name]
            p_attns = [r["p_attn"] for r in runs]
            frozen_jobs = [r["n_frozen_jobs"] for r in runs]
            writer.writerow([name, len(runs),
                             round(np.mean(p_attns), 4), round(np.std(p_attns), 4),
                             round(np.mean(frozen_jobs), 2),
                             round(np.mean([r["p_cap"] for r in runs]), 4),
                             round(np.mean([r["s_cont_cap"] for r in runs]), 4)])
    print("\nSummary saved to outputs/e14_probe/summary_passive.csv")


if __name__ == "__main__":
    main()
