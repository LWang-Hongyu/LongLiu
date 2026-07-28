"""
E2'/E2-pro 5-seed 加固（增 60 run）
E2': 4 bw × 5 policies × seeds [4,5] = 40 runs
E2-pro: 2 bw × 5 policies × seeds [4,5] = 20 runs
已有 seeds [0,1,2] 不重跑，只增 seeds [4,5]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_v3_batch3_formal import (
    run_single, CONFIG_HASH, SEMANTICS_VERSION,
    SCENARIOS, POLICIES,
)

NEW_SEEDS = [4, 5]
SCENE_FILTER = ["E2'", "E2-pro"]


def main():
    print("=" * 80)
    print("E2'/E2-pro 5-seed 加固（seeds 4,5 补充）")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}")
    print(f"CONFIG_HASH = {CONFIG_HASH}")
    print(f"New seeds: {NEW_SEEDS}")
    print(f"Policies: {POLICIES}")
    print(f"Scenes: {SCENE_FILTER}")

    total_runs = 0
    for scene, workload, bws in SCENARIOS:
        if scene in SCENE_FILTER:
            total_runs += len(POLICIES) * len(bws) * len(NEW_SEEDS)
    print(f"Total runs: {total_runs}")
    print()

    t_start = time.time()
    all_results = {}

    for scene, workload, bws in SCENARIOS:
        if scene not in SCENE_FILTER:
            continue

        for bw in bws:
            cfg_key = f"{scene}_{int(bw)}g"
            print("-" * 60)
            print(f"  {cfg_key}")
            print("-" * 60)

            bw_results = {}
            for pn in POLICIES:
                print(f"  [{pn}]", end="")
                seed_results = {}
                for s in NEW_SEEDS:
                    sys.stdout.write(f" s={s}...")
                    sys.stdout.flush()
                    try:
                        r = run_single(scene, workload, bw, pn, s)
                        seed_results[s] = r
                        sys.stdout.write(f"P-attn={r['p_attn']*100:.1f}% ")
                    except Exception as e:
                        sys.stdout.write(f"ERROR:{e} ")
                        import traceback
                        traceback.print_exc()
                        print("\n  *** FAIL — 停跑上报 ***")
                        return 1
                print()
                bw_results[pn] = seed_results
            all_results[cfg_key] = bw_results

    # ── Summary ──
    t_elapsed = time.time() - t_start
    print()
    print("=" * 80)
    print("E2'/E2-pro 补充 SUMMARY")
    print("=" * 80)

    for cfg_key in sorted(all_results.keys()):
        bw_results = all_results[cfg_key]
        print(f"\n  [{cfg_key}]")
        print(f"  {'Policy':<6} {'s4 P-attn':<10} {'s5 P-attn':<10} "
              f"{'s4 P-cap':<10} {'s5 P-cap':<10}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for pn in POLICIES:
            if pn not in bw_results:
                continue
            s4 = bw_results[pn].get(4, {})
            s5 = bw_results[pn].get(5, {})
            print(f"  {pn:<6} "
                  f"{s4.get('p_attn', 0)*100:>9.1f}% "
                  f"{s5.get('p_attn', 0)*100:>9.1f}% "
                  f"{s4.get('p_cap', 0):>10.3f} "
                  f"{s5.get('p_cap', 0):>10.3f}")

    print(f"\nTotal time: {t_elapsed:.0f}s")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
