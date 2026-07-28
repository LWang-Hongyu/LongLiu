"""
实验A：E3/E3' 补 D1 臂
双臂（E3 对照臂 @800G + E3' 杀伤臂 @630G）× D1 × 3 seeds = 12 run
目的：动态场景下 D1 的 W2 再收敛瞬态 vs v4 闭式无瞬态的决定性对照
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_e3_swap import (
    run_single, CONFIG_HASH, SEMANTICS_VERSION,
    FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD,
)

SEEDS = [0, 1, 2]
POLICIES = ["D1"]

CONFIGS = [
    {
        "name": "E3 (control arm, 800G)",
        "tag_prefix": "e3_swap",
        "workload": FEAS_BOUNDARY_V3_WORKLOAD,
        "spine_bw": 800,
    },
    {
        "name": "E3' (kill arm, 630G)",
        "tag_prefix": "e3p_swap",
        "workload": FEAS_BOUNDARY_V3_PRO_WORKLOAD,
        "spine_bw": 630,
    },
]


def main():
    print("=" * 80)
    print("实验A：E3/E3' D1 臂 3-seed 正式批")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}")
    print(f"CONFIG_HASH = {CONFIG_HASH}")
    print(f"Seeds: {SEEDS}")
    print(f"Policies: {POLICIES}")
    print(f"Total runs: {len(CONFIGS) * len(POLICIES) * len(SEEDS)}")
    print()

    t_start = time.time()
    all_results = {}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        tag = cfg["tag_prefix"]
        print("=" * 80)
        print(f"  {cfg_name}")
        print(f"  Workload: {len(cfg['workload'])} jobs, Spine: {cfg['spine_bw']}G")
        print("=" * 80)

        cfg_results = {}

        for pn in POLICIES:
            print(f"\n  [{pn}]")
            seed_results = {}

            for s in SEEDS:
                sys.stdout.write(f"    seed={s} ... ")
                sys.stdout.flush()

                try:
                    r = run_single(
                        pn, workload_raw=cfg["workload"],
                        spine_bw_gbps=cfg["spine_bw"],
                        tag_prefix=tag, seed=s
                    )
                    seed_results[s] = r

                    w1 = r["w1"]
                    w2 = r["w2"]
                    w3 = r["w3"]
                    print(f"W1 P-attn={w1['p_attn']*100:.1f}% "
                          f"W2 P-attn={w2['p_attn']*100:.1f}% "
                          f"W3 P-attn={w3['p_attn']*100:.1f}% "
                          f"starv={r['starv_post_swap']}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    print("\n  *** FAIL — 停跑上报 ***")
                    return 1

            cfg_results[pn] = seed_results

        all_results[tag] = cfg_results

    # ── Summary table ──
    t_elapsed = time.time() - t_start
    print()
    print("=" * 80)
    print("实验A SUMMARY TABLE")
    print("=" * 80)

    for tag, cfg_results in all_results.items():
        print(f"\n  [{tag}]")
        header = (f"  {'Policy':<6} {'Seed':<5} "
                  f"{'W1_P-attn':<10} {'W2_P-attn':<10} {'W3_P-attn':<10} "
                  f"{'W3_S-cap':<10} {'starv':<8}")
        print(header)
        print(f"  {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for pn in POLICIES:
            if pn not in cfg_results:
                continue
            for s in SEEDS:
                if s not in cfg_results[pn]:
                    continue
                r = cfg_results[pn][s]
                w1 = r["w1"]
                w2 = r["w2"]
                w3 = r["w3"]
                print(f"  {pn:<6} {s:<5} {w1['p_attn']*100:>9.1f}% "
                      f"{w2['p_attn']*100:>9.1f}% {w3['p_attn']*100:>9.1f}% "
                      f"{w3['s_cont_cap']:>10.3f} "
                      f"{r.get('starv_post_swap', 0):>8}")

        # 3-seed aggregate
        print(f"  {'─'*60}")
        for pn in POLICIES:
            if pn not in cfg_results:
                continue
            for win_name in ["w1", "w2", "w3"]:
                vals = [cfg_results[pn][s][win_name]["p_attn"] * 100
                        for s in SEEDS if s in cfg_results[pn]]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    print(f"  {pn:<6} {win_name}: {mean:.1f} ± {std:.1f}%")

    print(f"\nTotal time: {t_elapsed:.0f}s")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
