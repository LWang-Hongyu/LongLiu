"""
快速扫描 E14 被动弱更新（方案1+2，2 seeds）：对比基线 vs 被动校准。

用法：
    python experiments/_quick_scan_e14_passive.py
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import exp_e14_probe as E14
from exp_e14_probe import create_high_load_workload, load_e14_config, load_frozen, run_single


def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")) as f:
        E14.CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    frozen = load_frozen()
    cfg = load_e14_config()
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9
    workload = create_high_load_workload(cfg["n_jobs"], seed=42)
    seeds = [0, 1]

    configs = [
        ("baseline",    False, False, 0, 0),   # 无探测、无被动（当前 E14 基线）
        ("passive",     False, True,  0, 0),   # 方案1+2：被动弱更新
        ("probe_pass",  True,  True,  10, 1),  # 探测 + 被动叠加
        ("probe_only",  True,  False, 10, 1),  # 探测（对照）
    ]

    print(f"{'config':>12s} | {'s0':>6s} {'s1':>6s} | {'mean':>6s} | {'frozen':>6s}")
    print("-" * 52)
    for name, probe, passive, thr, dur in configs:
        vals, frozens = [], []
        for seed in seeds:
            r = run_single(probe, workload, spine_bw, seed, frozen,
                           probe_frozen_threshold=thr, probe_duration=dur,
                           ema_passive=passive)
            vals.append(r["p_attn"])
            frozens.append(r["n_frozen_jobs"])
        mean = sum(vals) / len(vals)
        fr = sum(frozens) / len(frozens)
        print(f"{name:>12s} | {vals[0]*100:5.1f}% {vals[1]*100:5.1f}% | {mean*100:5.1f}% | {fr:>5.1f}")


if __name__ == "__main__":
    main()
