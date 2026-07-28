"""
实验B — E3/E3' v4+CRUX 补足至 5 seeds（增 8 run）
双臂（E3 @800G + E3' @630G）× v4/CRUX × seeds [4,5] = 8 runs
已有 seeds [0,1,2] 不重跑，只增 seeds [4,5]
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

NEW_SEEDS = [4, 5]
POLICIES = ["v4", "CRUX"]

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
    print("实验B — E3/E3' v4+CRUX 5-seed 加固（seeds 4,5 补充）")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}")
    print(f"CONFIG_HASH = {CONFIG_HASH}")
    print(f"New seeds: {NEW_SEEDS}")
    print(f"Policies: {POLICIES}")
    print(f"Total runs: {len(CONFIGS) * len(POLICIES) * len(NEW_SEEDS)}")
    print()

    t_start = time.time()
    all_results = {}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        tag = cfg["tag_prefix"]
        print("=" * 60)
        print(f"  {cfg_name}")
        print(f"  Workload: {len(cfg['workload'])} jobs, Spine: {cfg['spine_bw']}G")
        print("=" * 60)

        cfg_results = {}

        for pn in POLICIES:
            print(f"\n  [{pn}]")
            seed_results = {}

            for s in NEW_SEEDS:
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
                    w3 = r["w3"]
                    print(f"W1 P-attn={w1['p_attn']*100:.1f}% "
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

    # ── Summary ──
    t_elapsed = time.time() - t_start
    print()
    print("=" * 80)
    print("实验B E3/E3' 补充 SUMMARY")
    print("=" * 80)

    for tag, cfg_results in all_results.items():
        print(f"\n  [{tag}]")
        header = (f"  {'Policy':<6} {'Seed':<5} "
                  f"{'W1_P-attn':<10} {'W3_P-attn':<10} "
                  f"{'W3_S-cap':<10} {'starv':<8}")
        print(header)
        print(f"  {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for pn in POLICIES:
            if pn not in cfg_results:
                continue
            for s in NEW_SEEDS:
                if s not in cfg_results[pn]:
                    continue
                r = cfg_results[pn][s]
                w1 = r["w1"]
                w3 = r["w3"]
                print(f"  {pn:<6} {s:<5} {w1['p_attn']*100:>9.1f}% "
                      f"{w3['p_attn']*100:>9.1f}% "
                      f"{w3['s_cont_cap']:>10.3f} "
                      f"{r.get('starv_post_swap', 0):>8}")

    print(f"\nTotal time: {t_elapsed:.0f}s")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
