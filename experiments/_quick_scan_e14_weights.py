"""
快速扫描 E14 被动校准信任权重表（P4/P5 网格，2 seeds）。

P0-P3 固定 [0.05, 0.1, 0.2, 0.4]，P6 固定 1.0，
扫 P4 ∈ {0.2, 0.4, 0.6} × P5 ∈ {0.4, 0.6, 0.8}。

用法：
    python experiments/_quick_scan_e14_weights.py
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import exp_e14_probe as E14
from exp_e14_probe import create_high_load_workload, load_e14_config, load_frozen, run_single

BASE = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # 默认权重（level 0-6）


def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")) as f:
        E14.CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    frozen = load_frozen()
    cfg = load_e14_config()
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9
    workload = create_high_load_workload(cfg["n_jobs"], seed=42)
    seeds = [0, 1]

    # 配置列表：(名称, ema_passive, ema_weights)
    configs = [("baseline", False, None)]
    for p4 in (0.2, 0.4, 0.6):
        for p5 in (0.4, 0.6, 0.8):
            w = list(BASE)
            w[4] = p4
            w[5] = p5
            configs.append((f"P4={p4:.1f} P5={p5:.1f}", True, w))

    print(f"{'config':>16s} | {'s0':>6s} {'s1':>6s} | {'mean':>6s} | {'frozen':>6s}")
    print("-" * 56)
    for name, passive, weights in configs:
        vals, frozens = [], []
        for seed in seeds:
            r = run_single(False, workload, spine_bw, seed, frozen,
                           ema_passive=passive, ema_weights=weights)
            vals.append(r["p_attn"])
            frozens.append(r["n_frozen_jobs"])
        mean = sum(vals) / len(vals)
        fr = sum(frozens) / len(frozens)
        print(f"{name:>16s} | {vals[0]*100:5.1f}% {vals[1]*100:5.1f}% | {mean*100:5.1f}% | {fr:>5.1f}")


if __name__ == "__main__":
    main()
