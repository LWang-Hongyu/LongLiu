"""
D1 臂补 seed 3,4（E3/E3' 双臂 × D1 × 2 seeds = 4 runs）
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp_e3_swap import (
    run_single, CONFIG_HASH, SEMANTICS_VERSION,
    FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD,
)

NEW_SEEDS = [3, 4]
POLICIES = ["D1"]
CONFIGS = [
    {"name": "E3 (control arm, 800G)", "tag_prefix": "e3_swap", "workload": FEAS_BOUNDARY_V3_WORKLOAD, "spine_bw": 800},
    {"name": "E3' (kill arm, 630G)", "tag_prefix": "e3p_swap", "workload": FEAS_BOUNDARY_V3_PRO_WORKLOAD, "spine_bw": 630},
]

def main():
    print("=" * 60)
    print("D1 臂 +2 seeds (3,4) — 4 runs")
    print("=" * 60)
    t0 = time.time()
    for cfg in CONFIGS:
        print(f"\n  [{cfg['name']}]")
        for s in NEW_SEEDS:
            sys.stdout.write(f"    D1 seed={s} ... "); sys.stdout.flush()
            r = run_single("D1", workload_raw=cfg["workload"],
                           spine_bw_gbps=cfg["spine_bw"],
                           tag_prefix=cfg["tag_prefix"], seed=s)
            w1, w2, w3 = r["w1"], r["w2"], r["w3"]
            print(f"W1={w1['p_attn']*100:.1f}% W2={w2['p_attn']*100:.1f}% "
                  f"W3={w3['p_attn']*100:.1f}% starv={r['starv_post_swap']}")
    print(f"\nDone in {time.time()-t0:.0f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
