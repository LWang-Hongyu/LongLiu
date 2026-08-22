"""
快速扫描 E14 探测参数（2 seeds）：threshold × duration 组合，找有效配置。

用法：
    python experiments/_quick_scan_e14_probe.py
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
        ("no_probe", False, 10, 0),
        ("T10D0",    True,  10, 0),
        ("T10D1",    True,  10, 1),
        ("T10D3",    True,  10, 3),
        ("T20D1",    True,  20, 1),
        ("T20D3",    True,  20, 3),
        ("T30D1",    True,  30, 1),
    ]

    print(f"{'config':>8s} | {'s0':>6s} {'s1':>6s} | {'mean':>6s}")
    print("-" * 40)
    for name, probe, thr, dur in configs:
        vals = []
        for seed in seeds:
            r = run_single(probe, workload, spine_bw, seed, frozen,
                           probe_frozen_threshold=thr, probe_duration=dur)
            vals.append(r["p_attn"])
        mean = sum(vals) / len(vals)
        print(f"{name:>8s} | {vals[0]*100:5.1f}% {vals[1]*100:5.1f}% | {mean*100:5.1f}%")


if __name__ == "__main__":
    main()
