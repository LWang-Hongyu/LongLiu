"""
E3/E3' 3-seed 正式批：双臂 × v4/CRUX × 3 seeds = 12 runs.

判定规则（矩阵 v2.2）：
  - E3' 杀伤臂：v4 W3 P-attn=100%（下界）、starv=0；
    CRUX W3 P-attn ≪ v4（gap ≥10pp）
  - E3 对照臂：v4 W3 P-attn=100%；CRUX W3 观测行
  - 任一 FAIL 停跑上报
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
POLICIES = ["v4", "CRUX"]

CONFIGS = [
    {
        "name": "E3 (control arm)",
        "tag_prefix": "e3_swap",
        "workload": FEAS_BOUNDARY_V3_WORKLOAD,
        "spine_bw": 800,
        "checks": {
            "v4": {"w3_pattn_min": 1.0, "starv_post_max": 0},  # v4 must attain 100%
        },
    },
    {
        "name": "E3' (kill arm)",
        "tag_prefix": "e3p_swap",
        "workload": FEAS_BOUNDARY_V3_PRO_WORKLOAD,
        "spine_bw": 630,
        "checks": {
            "v4": {"w3_pattn_min": 1.0, "starv_post_max": 0},  # v4 must attain 100%
            "CRUX_vs_v4": {"gap_pp_min": 10.0},  # CRUX P-attn ≪ v4 by ≥10pp
        },
    },
]


def main():
    print("=" * 80)
    print("E3/E3' 3-SEED FORMAL BATCH")
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
                    w3 = r["w3"]
                    print(f"W1 P-attn={w1['p_attn']*100:.1f}% "
                          f"W3 P-attn={w3['p_attn']*100:.1f}% "
                          f"S-cap={w3['s_cont_cap']:.3f} "
                          f"starv={r['starv_post_swap']}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

            cfg_results[pn] = seed_results

        # ── Verification ──
        print()
        print("  " + "-" * 70)
        print("  Verification (Matrix v2.2)")
        print("  " + "-" * 70)

        checks_cfg = cfg["checks"]
        all_checks_pass = True

        # v4 checks (per-seed)
        if "v4" in cfg_results and "v4" in checks_cfg:
            v4_checks = checks_cfg["v4"]
            for s in SEEDS:
                if s not in cfg_results["v4"]:
                    continue
                w3 = cfg_results["v4"][s]["w3"]
                r = cfg_results["v4"][s]
                ok_pattn = w3["p_attn"] >= v4_checks.get("w3_pattn_min", 0.98)
                ok_starv = r.get("starv_post_swap", 0) <= v4_checks.get("starv_post_max", 0)
                ok = ok_pattn and ok_starv
                flag = "PASS" if ok else "FAIL"
                if not ok:
                    all_checks_pass = False
                print(f"  [{flag}] v4 seed={s}: W3 P-attn={w3['p_attn']*100:.1f}% "
                      f"starv_post={r.get('starv_post_swap', 0)}")

        # CRUX vs v4 gap check
        if "CRUX_vs_v4" in checks_cfg and "CRUX" in cfg_results and "v4" in cfg_results:
            gap_min = checks_cfg["CRUX_vs_v4"]["gap_pp_min"]
            for s in SEEDS:
                if s not in cfg_results["CRUX"] or s not in cfg_results["v4"]:
                    continue
                v4_pattn = cfg_results["v4"][s]["w3"]["p_attn"]
                crux_pattn = cfg_results["CRUX"][s]["w3"]["p_attn"]
                gap = (v4_pattn - crux_pattn) * 100
                ok = gap >= gap_min
                flag = "PASS" if ok else "FAIL"
                if not ok:
                    all_checks_pass = False
                print(f"  [{flag}] CRUX vs v4 seed={s}: gap={gap:.1f}pp "
                      f"(v4={v4_pattn*100:.1f}%, CRUX={crux_pattn*100:.1f}%)")

        # CRUX observation row (E3 control arm)
        if "CRUX" in cfg_results and "CRUX_vs_v4" not in checks_cfg:
            for s in SEEDS:
                if s not in cfg_results["CRUX"]:
                    continue
                w3 = cfg_results["CRUX"][s]["w3"]
                print(f"  [OBS] CRUX seed={s}: W3 P-attn={w3['p_attn']*100:.1f}% "
                      f"S-cap={w3['s_cont_cap']:.3f}")

        if all_checks_pass:
            print(f"\n  *** {cfg_name}: ALL CHECKS PASSED ***")
        else:
            print(f"\n  *** {cfg_name}: SOME CHECKS FAILED — STOPPING ***")
            # Save partial results before exit
            break

        all_results[tag] = cfg_results

    # ── Summary table ──
    t_elapsed = time.time() - t_start
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    for tag, cfg_results in all_results.items():
        print(f"\n  [{tag}]")
        print(f"  {'Policy':<6} {'Seed':<5} {'W1_P-attn':<10} {'W3_P-attn':<10} "
              f"{'W3_S-cap':<10} {'starv_post':<10}")
        print(f"  {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for pn in POLICIES:
            if pn not in cfg_results:
                continue
            for s in SEEDS:
                if s not in cfg_results[pn]:
                    continue
                r = cfg_results[pn][s]
                w1 = r["w1"]
                w3 = r["w3"]
                print(f"  {pn:<6} {s:<5} {w1['p_attn']*100:>9.1f}% "
                      f"{w3['p_attn']*100:>9.1f}% {w3['s_cont_cap']:>10.3f} "
                      f"{r.get('starv_post_swap', 0):>10}")

        # 3-seed aggregate
        print(f"  {'─'*50}")
        for pn in POLICIES:
            if pn not in cfg_results:
                continue
            w3_pattns = [cfg_results[pn][s]["w3"]["p_attn"] * 100
                         for s in SEEDS if s in cfg_results[pn]]
            if w3_pattns:
                mean = np.mean(w3_pattns)
                std = np.std(w3_pattns)
                print(f"  {pn:<6} 3-seed W3 P-attn = {mean:.1f} ± {std:.1f}%")

    print(f"\nTotal time: {t_elapsed:.0f}s")
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
