"""
E14 被动校准 10-seed 验证：passive_low (P4=0.2/P5=0.4) vs baseline，配对统计检验。

用法：
    python experiments/_validate_e14_passive_10seeds.py
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

PASSIVE_LOW = [0.05, 0.1, 0.2, 0.4, 0.2, 0.4, 1.0]  # P4=0.2, P5=0.4

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def paired_t_test(x, y):
    """配对 t 检验（x=passive_low, y=baseline）。"""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    se = std_d / np.sqrt(n)
    t = mean_d / se if se > 0 else 0.0
    if HAS_SCIPY:
        _, p = sp_stats.ttest_rel(x, y)
        return t, p
    # 无 scipy 时用正态近似（n>=10 可接受）
    p = 2.0 * (1.0 - _normal_cdf(abs(t)))
    return t, p


def _normal_cdf(z):
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _erf(x):
    # Abramowitz-Stegun 近似
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")) as f:
        E14.CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    frozen = load_frozen()
    cfg = load_e14_config()
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9
    workload = create_high_load_workload(cfg["n_jobs"], seed=42)
    seeds = list(range(10))

    configs = [
        ("baseline",   False, None),
        ("passive_low", True, PASSIVE_LOW),
    ]

    results = {name: [] for name, _, _ in configs}
    print(f"{'config':>13s} | " + " ".join(f"s{i:>4s}" for i in map(str, seeds)) + " | mean±std")
    print("-" * 78)
    for name, passive, weights in configs:
        vals = []
        for seed in seeds:
            r = run_single(False, workload, spine_bw, seed, frozen,
                           ema_passive=passive, ema_weights=weights)
            vals.append(r["p_attn"])
            results[name].append(r)
        mean = np.mean(vals) * 100
        std = np.std(vals, ddof=1) * 100
        print(f"{name:>13s} | " + " ".join(f"{v*100:4.1f}" for v in vals) + f" | {mean:.1f}±{std:.1f}")

    base = [r["p_attn"] for r in results["baseline"]]
    low = [r["p_attn"] for r in results["passive_low"]]
    t_stat, p_val = paired_t_test(low, base)
    mean_diff = (np.mean(low) - np.mean(base)) * 100

    print("-" * 78)
    print(f"Mean diff (passive_low - baseline) = {mean_diff:+.1f} pp")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f} "
          f"({'significant at 0.05' if p_val < 0.05 else 'NOT significant at 0.05'})")
    if HAS_SCIPY:
        w_stat, w_p = sp_stats.wilcoxon(low, base)
        print(f"Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")
        # Cohen's d (配对)
        d_vals = np.asarray(low) - np.asarray(base)
        cohen_d = np.mean(d_vals) / np.std(d_vals, ddof=1)
        print(f"Cohen's d (paired) = {cohen_d:.3f}")

    # 保存 summary CSV
    os.makedirs("outputs/e14_probe", exist_ok=True)
    with open("outputs/e14_probe/summary_passive_10seeds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_seeds", "p_attn_mean", "p_attn_std",
                         "n_frozen_jobs_mean", "p_cap_mean", "s_cont_cap_mean"])
        for name, _, _ in configs:
            runs = results[name]
            writer.writerow([name, len(runs),
                             round(np.mean([r["p_attn"] for r in runs]), 4),
                             round(np.std([r["p_attn"] for r in runs], ddof=1), 4),
                             round(np.mean([r["n_frozen_jobs"] for r in runs]), 2),
                             round(np.mean([r["p_cap"] for r in runs]), 4),
                             round(np.mean([r["s_cont_cap"] for r in runs]), 4)])
    print("Summary saved to outputs/e14_probe/summary_passive_10seeds.csv")


if __name__ == "__main__":
    main()
