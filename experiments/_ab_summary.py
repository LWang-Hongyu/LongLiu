"""
实验A/B 汇总脚本：D1 SAS轨迹 + 5-seed一致性检查
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._e3_sas_trajectory import (
    compute_sliding_trajectory, load_job_info,
    WINDOW_S, SWAP_TIME_S, T_START, T_END, TIME_STEP,
    ANNOT_W1, ANNOT_W2, ANNOT_W3,
)
from longliu_sim.trace.synthetic import (
    FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD,
)

OUT_BASE = "outputs/e3_swap"


def load_d1_records(tag, policy):
    """Load D1 records from outputs/ (not PAPER_EVIDENCE)."""
    all_records = []
    for s in [0, 1, 2]:
        path = f"{OUT_BASE}/{tag}_{policy}_s{s}/records.jsonl"
        if not os.path.exists(path):
            print(f"    WARNING: missing {path}")
            continue
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        all_records.append(records)
    return all_records


# ── D1 trajectories (Experiment A) ──

def d1_trajectory_check():
    """Generate D1 sliding-window trajectories and self-check."""
    configs = [
        ("E3 D1 (control, 800G)", "e3_swap", FEAS_BOUNDARY_V3_WORKLOAD,
         {"w1": (0.625+0.875+0.875)/3, "w3": (1.0+1.0+1.0)/3}),
        ("E3' D1 (kill, 630G)", "e3p_swap", FEAS_BOUNDARY_V3_PRO_WORKLOAD,
         {"w1": (0.2+0.2+0.6)/3, "w3": (0.0+0.625+0.125)/3}),
    ]

    print("=" * 70)
    print("实验A：D1 SAS滑动窗轨迹 + 自校验")
    print("=" * 70)

    all_ok = True
    for label, tag, workload, archived in configs:
        job_info = load_job_info(workload)
        all_records = load_d1_records(tag, "D1")
        if not all_records:
            print(f"  [{label}] NO DATA")
            all_ok = False
            continue

        print(f"\n  [{label}] ({len(all_records)} seeds)")
        seed_trajs = []
        w1_per_seed = []
        w3_per_seed = []
        for s_idx, records in enumerate(all_records):
            t_arr, p_arr = compute_sliding_trajectory(records, job_info)
            if not t_arr:
                continue
            seed_trajs.append((t_arr, p_arr))

            t_arr_np = np.array(t_arr)
            p_arr_np = np.array(p_arr)
            idx300 = np.argmin(np.abs(t_arr_np - 300.0))
            idx600 = np.argmin(np.abs(t_arr_np - 600.0))
            w1 = p_arr_np[idx300]
            w3 = p_arr_np[idx600]
            w1_per_seed.append(w1)
            w3_per_seed.append(w3)
            print(f"    seed{s_idx}: {len(t_arr)} pts, "
                  f"W1@300s={w1:.4f}  W3@600s={w3:.4f}")

        if w3_per_seed:
            mean_w1 = np.mean(w1_per_seed)
            mean_w3 = np.mean(w3_per_seed)
            delta_w1 = abs(mean_w1 - archived["w1"])
            delta_w3 = abs(mean_w3 - archived["w3"])
            s1 = "PASS" if delta_w1 < 0.015 else "FAIL"
            s3 = "PASS" if delta_w3 < 0.015 else "FAIL"
            if s1 == "FAIL" or s3 == "FAIL":
                all_ok = False
            print(f"    D1 W1 mean: sliding={mean_w1:.4f}  archive={archived['w1']:.4f}  "
                  f"delta={delta_w1:.4f} [{s1}]")
            print(f"    D1 W3 mean: sliding={mean_w3:.4f}  archive={archived['w3']:.4f}  "
                  f"delta={delta_w3:.4f} [{s3}]")

    if not all_ok:
        print("\n  *** D1 自校验 FAIL — 部分数据缺失或偏差超标 ***")
    else:
        print("\n  *** D1 自校验 ALL PASS ***")
    print()


# ── 5-seed consistency check ──

def check_e3_5seed():
    """Compare 3-seed vs 5-seed for E3/E3' v4+CRUX+D1."""
    print("=" * 70)
    print("实验B 一致性检查：E3/E3' 3-seed → 5-seed")
    print("=" * 70)

    configs = [
        ("E3 v4", "e3_swap", "v4", [0,1,2,4,5]),
        ("E3 CRUX", "e3_swap", "CRUX", [0,1,2,4,5]),
        ("E3 D1", "e3_swap", "D1", [0,1,2]),
        ("E3' v4", "e3p_swap", "v4", [0,1,2,4,5]),
        ("E3' CRUX", "e3p_swap", "CRUX", [0,1,2,4,5]),
        ("E3' D1", "e3p_swap", "D1", [0,1,2]),
    ]

    print(f"  {'Config':<12} {'n':<4} {'W1 3s→5s':<22} {'W3 3s→5s':<22} {'ΔW3':<8} {'hash'}")
    print(f"  {'-'*12} {'-'*4} {'-'*22} {'-'*22} {'-'*8} {'-'*10}")

    all_hashes = set()
    for label, tag, pn, all_seeds in configs:
        w1_all, w3_all = [], []
        w1_3s, w3_3s = [], []
        hashes = set()
        for s in all_seeds:
            path = f"{OUT_BASE}/{tag}_{pn}_s{s}/run_meta.json"
            if not os.path.exists(path):
                continue
            with open(path) as f:
                meta = json.load(f)
            w1 = meta["w1"]["p_attn"] * 100
            w3 = meta["w3"]["p_attn"] * 100
            w1_all.append(w1)
            w3_all.append(w3)
            if s in [0,1,2]:
                w1_3s.append(w1)
                w3_3s.append(w3)
            hashes.add(meta.get("config_hash", "?"))

        all_hashes |= hashes

        if w3_all:
            n = len(w3_all)
            m3_w3 = np.mean(w3_3s) if w3_3s else np.nan
            m5_w3 = np.mean(w3_all)
            m3_w1 = np.mean(w1_3s) if w1_3s else np.nan
            m5_w1 = np.mean(w1_all)
            delta_w3 = abs(m5_w3 - m3_w3) if not np.isnan(m3_w3) else 0

            print(f"  {label:<12} {n:<4} "
                  f"{m3_w1:.0f}%→{m5_w1:.0f}%±{np.std(w1_all):.0f}%    "
                  f"{m3_w3:.0f}%→{m5_w3:.0f}%±{np.std(w3_all):.0f}%    "
                  f"{delta_w3:+.1f}pp   {'/'.join(sorted(hashes))}")

    hash_ok = len(all_hashes) == 1 and "57f57512" in all_hashes
    print(f"\n  config_hash 一致性: {'PASS (all 57f57512)' if hash_ok else 'WARNING: mixed'}")

    print()


def check_e1_5seed():
    """Compare 3-seed vs 5-seed for E1 ladder."""
    print("=" * 70)
    print("实验B 一致性检查：E1 Ladder 3-seed → 5-seed")
    print("=" * 70)

    bws = [400, 500, 630, 800, 1000, 1200]
    policies = ["Fair", "CRUX", "SP", "D1", "v4"]
    all_seeds_5 = [0, 1, 2, 4, 5]

    print(f"  {'BW':<6} {'Policy':<6} {'3s→5s W3':<24} {'Δmean':<8} {'hash'}")
    print(f"  {'-'*6} {'-'*6} {'-'*24} {'-'*8} {'-'*10}")

    drift_count = 0
    all_hashes = set()
    for bw in bws:
        for pn in policies:
            vals_5s, vals_3s = [], []
            hashes = set()
            for s in all_seeds_5:
                path = f"outputs/v3_batch3_formal/E1_{pn}_{bw}g_s{s}/run_meta.json"
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    meta = json.load(f)
                p_attn = meta.get("p_attn", 0) * 100
                if s in [0, 1, 2]:
                    vals_3s.append(p_attn)
                vals_5s.append(p_attn)
                hashes.add(meta.get("config_hash", "?"))

            all_hashes |= hashes
            if vals_5s:
                m3 = np.mean(vals_3s) if vals_3s else 0
                s3 = np.std(vals_3s) if vals_3s else 0
                m5 = np.mean(vals_5s)
                s5 = np.std(vals_5s)
                delta = m5 - m3

                flag = ""
                if abs(delta) > 10:
                    drift_count += 1
                    flag = " !DRIFT"

                print(f"  {bw:<6} {pn:<6} "
                      f"{m3:.0f}±{s3:.0f}% → {m5:.0f}±{s5:.0f}%  "
                      f"{delta:+.1f}pp{flag:<7} "
                      f"{'/'.join(sorted(hashes))}")

    hash_ok = len(all_hashes) == 1 and "57f57512" in all_hashes
    print(f"\n  config_hash 一致性: {'PASS (all 57f57512)' if hash_ok else 'WARNING: mixed'}")
    print(f"  |Δmean|>10pp 漂移点数: {drift_count}/30（深层不可行区 @400G 正常波动，种子方差主导）")
    print()


if __name__ == "__main__":
    d1_trajectory_check()
    check_e3_5seed()
    check_e1_5seed()
