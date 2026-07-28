"""
澄清脚本：E1 全漂移表 + v4@400G 逐seed
"""
import json, os, sys, numpy as np
from collections import defaultdict

E1_BASE = "outputs/v3_batch3_formal"
bws = [400, 500, 630, 800, 1000, 1200]
policies = ["Fair", "CRUX", "SP", "D1", "v4"]
all_seeds = [0, 1, 2, 4, 5]

print("=" * 100)
print("E1 LADDER 完整逐seed P-attn 漂移表")
print("=" * 100)

# Header
print(f"\n{'BW':<6} {'Policy':<6} ", end="")
for s in all_seeds:
    print(f"{'s'+str(s):>7} ", end="")
print(f"{'3s_μ':>7} {'5s_μ':>7} {'Δ':>7} {'|Δ|>10pp':>8}")
print("-" * 100)

drift_points = []
for bw in bws:
    for pn in policies:
        vals = {}
        for s in all_seeds:
            path = f"{E1_BASE}/E1_{pn}_{bw}g_s{s}/run_meta.json"
            if os.path.exists(path):
                with open(path) as f:
                    meta = json.load(f)
                vals[s] = meta["p_attn"] * 100
            else:
                vals[s] = None

        # Print row
        print(f"{bw:<6} {pn:<6} ", end="")
        for s in all_seeds:
            if vals[s] is not None:
                print(f"{vals[s]:>6.1f}% ", end="")
            else:
                print(f"{'?':>6}  ", end="")

        vals_3s = [vals[s] for s in [0,1,2] if vals[s] is not None]
        vals_5s = [vals[s] for s in all_seeds if vals[s] is not None]
        m3 = np.mean(vals_3s) if vals_3s else float('nan')
        m5 = np.mean(vals_5s) if vals_5s else float('nan')
        delta = m5 - m3 if not np.isnan(m3) else 0

        flag = ""
        if abs(delta) > 10:
            drift_points.append(f"{bw}g_{pn}")
            flag = " !DRIFT"

        print(f"{m3:>7.1f} {m5:>7.1f} {delta:>+7.1f}{flag}")

print()
print(f"|Δ|>10pp 漂移点: {len(drift_points)}")
for dp in drift_points:
    print(f"  - {dp}")
print()

# ── v4@400G 专项 ──
print("=" * 60)
print("v4@400G 专项逐seed核查")
print("=" * 60)
for s in all_seeds:
    path = f"{E1_BASE}/E1_v4_400g_s{s}/run_meta.json"
    with open(path) as f:
        meta = json.load(f)
    print(f"  seed{s}: P-attn={meta['p_attn']*100:.1f}%  "
          f"P-cap={meta.get('p_cap',0)*100:.1f}%  "
          f"starv={meta.get('starv',0)}  "
          f"hash={meta.get('config_hash','?')}")

vals_3s = []
vals_5s = []
for s in all_seeds:
    path = f"{E1_BASE}/E1_v4_400g_s{s}/run_meta.json"
    with open(path) as f:
        meta = json.load(f)
    v = meta["p_attn"] * 100
    if s in [0,1,2]:
        vals_3s.append(v)
    vals_5s.append(v)

print(f"\n  3-seed: {np.mean(vals_3s):.1f}±{np.std(vals_3s):.1f}% (seeds {[f'{v:.1f}' for v in vals_3s]})")
print(f"  5-seed: {np.mean(vals_5s):.1f}±{np.std(vals_5s):.1f}% (seeds {[f'{v:.1f}' for v in vals_5s]})")
print(f"  Δ = {np.mean(vals_5s)-np.mean(vals_3s):+.1f}pp")
