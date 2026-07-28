"""
Fig-4 D1 臂 SAS 轨迹提取
口径：与 _e3_sas_trajectory.py 完全一致
  - 滑动100s窗、start_ms过滤、per-regime tier/target
  - sas>=0.98 判达标、mean±std over 5 seeds
输入：outputs/e3_swap/{e3,e3p}_swap_D1_s{0..4}/records.jsonl
输出：experiments/_d1_sas_trajectory_e3.csv, _d1_sas_trajectory_e3p.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.trace.synthetic import (
    FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD
)
from longliu_sim.utils.model_params import MODEL_PARAMS
from longliu_sim.utils.config import load_config

_cfg = load_config()
OVERHEAD = _cfg["frozen"]["overhead_factor"]
OVERLAP = _cfg["frozen"]["overlap_factor"]

# ── Constants ──
WINDOW_S = 100.0
SWAP_TIME_S = 300.0
TIME_STEP = 0.25
T_START = 100.0
T_END = 600.0

LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}

D1_SEEDS = [0, 1, 2, 3, 4]

# Accepted D1 5-seed summary (per user acceptance)
D1_ACCEPTED = {
    "e3_swap":  {"w1": 0.700, "w3": 0.967},   # E3 D1: 70.0%, 96.7%
    "e3p_swap": {"w1": 0.360, "w3": 0.250},   # E3' D1: 36.0%, 25.0%
}

CONFIGS = [
    {
        "label": "E3 Control Arm (D1, 800G)",
        "tag": "e3_swap",
        "workload": FEAS_BOUNDARY_V3_WORKLOAD,
        "out_csv": "experiments/_d1_sas_trajectory_e3.csv",
    },
    {
        "label": "E3' Kill Arm (D1, 630G)",
        "tag": "e3p_swap",
        "workload": FEAS_BOUNDARY_V3_PRO_WORKLOAD,
        "out_csv": "experiments/_d1_sas_trajectory_e3p.csv",
    },
]


def get_target(comp_ms, comm_solo_ms, ci):
    comm_budget = ci * comm_solo_ms * OVERHEAD
    if OVERLAP > 0:
        return max(comp_ms, comm_budget) + (1.0 - OVERLAP) * min(comp_ms, comm_budget)
    return comp_ms + comm_budget


def load_job_info(workload_raw):
    info = {}
    for i, (model, dp, orig_ci) in enumerate(workload_raw):
        jid = f"J{i}"
        p = MODEL_PARAMS[model]
        bpp = 2 if p.get("fp16", True) else 4
        bytes_per_iter = 2 * p["params"] * bpp / max(dp, 1)
        mb = bytes_per_iter / (1024 * 1024)
        raw_comm_ms = mb * 8 * 1024 * 1024 / (100e9) * 1000.0
        comp_ms = p.get("comp_ms", 50.0) if "comp_ms" in p else 50.0

        pre_target = get_target(comp_ms, raw_comm_ms, orig_ci)
        pre_is_premium = orig_ci <= 2.0

        was_premium = orig_ci <= 2.0
        if was_premium:
            post_ci = 3.0
        else:
            is_large = model in LARGE_MODELS
            post_ci = 1.5 if is_large else (2.0 if dp == 4 else 1.5)
        post_target = get_target(comp_ms, raw_comm_ms, post_ci)
        post_is_premium = not was_premium

        info[jid] = {
            "model": model, "dp": dp,
            "orig_ci": orig_ci, "post_ci": post_ci,
            "pre_target": pre_target, "post_target": post_target,
            "pre_is_premium": pre_is_premium,
            "post_is_premium": post_is_premium,
        }
    return info


def compute_sliding_trajectory(records, job_info):
    if not records:
        return [], []

    records.sort(key=lambda r: r["start_ms"])
    n_points = int((T_END - T_START) / TIME_STEP) + 1
    time_grid = np.linspace(T_START, T_END, n_points)

    w_ms = WINDOW_S * 1000.0
    left_idx = 0
    right_idx = 0
    job_sum = defaultdict(float)
    job_cnt = defaultdict(int)

    results_t, results_pattn = [], []

    for t_s in time_grid:
        t_ms = t_s * 1000.0
        lo = max(0.0, t_ms - w_ms)
        hi = t_ms

        while left_idx < len(records) and records[left_idx]["start_ms"] < lo:
            r = records[left_idx]
            jid = r["jid"]
            if jid in job_cnt:
                job_sum[jid] -= r["iter_ms"]
                job_cnt[jid] -= 1
                if job_cnt[jid] <= 0:
                    job_sum.pop(jid, None)
                    job_cnt.pop(jid, None)
            left_idx += 1

        while right_idx < len(records) and records[right_idx]["start_ms"] <= hi:
            r = records[right_idx]
            jid = r["jid"]
            job_sum[jid] += r["iter_ms"]
            job_cnt[jid] = job_cnt.get(jid, 0) + 1
            right_idx += 1

        if t_s <= SWAP_TIME_S:
            premium_set = {jid for jid, info in job_info.items()
                           if info["pre_is_premium"]}
            target_key = "pre_target"
        else:
            premium_set = {jid for jid, info in job_info.items()
                           if info["post_is_premium"]}
            target_key = "post_target"

        p_total = 0
        p_attn = 0
        for jid in premium_set:
            if jid in job_cnt and job_cnt[jid] > 0:
                avg_iter = job_sum[jid] / job_cnt[jid]
                target = job_info[jid][target_key]
                sas = target / avg_iter if avg_iter > 0 else 0.0
                p_total += 1
                if sas >= 0.98:
                    p_attn += 1

        if p_total > 0:
            results_t.append(t_s)
            results_pattn.append(p_attn / p_total)

    return results_t, results_pattn


def load_d1_records(tag, seeds):
    """Load records from outputs/e3_swap/ for D1 policy."""
    all_records = []
    for s in seeds:
        path = f"outputs/e3_swap/{tag}_D1_s{s}/records.jsonl"
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        all_records.append(records)
    return all_records


def self_check(label, tag, per_seed_trajs, accepted):
    """Self-check at t=300 (W1) and t=600 (W3) against accepted D1 values."""
    print(f"\n{'='*68}")
    print(f"SELF-CHECK: {label}")
    print(f"{'='*68}")
    print(f"  (sliding window at t=300s -> W1; t=600s -> W3)")
    print(f"  Accepted: W1={accepted['w1']*100:.1f}%, W3={accepted['w3']*100:.1f}%")
    print()

    w1_vals, w3_vals = [], []
    for seed_idx, (t_arr, p_arr) in enumerate(per_seed_trajs):
        if not t_arr:
            continue
        t_arr = np.array(t_arr)
        p_arr = np.array(p_arr)
        idx300 = np.argmin(np.abs(t_arr - 300.0))
        idx600 = np.argmin(np.abs(t_arr - 600.0))
        w1_vals.append(p_arr[idx300])
        w3_vals.append(p_arr[idx600])
        print(f"  seed{seed_idx}: W1@300s={p_arr[idx300]*100:.1f}%  W3@600s={p_arr[idx600]*100:.1f}%")

    if not w3_vals:
        print("  FAIL: no valid seeds")
        return False

    mean_w1 = np.mean(w1_vals)
    mean_w3 = np.mean(w3_vals)
    delta_w1 = abs(mean_w1 - accepted["w1"])
    delta_w3 = abs(mean_w3 - accepted["w3"])

    s1 = "PASS" if delta_w1 < 0.05 else "FAIL"
    s3 = "PASS" if delta_w3 < 0.05 else "FAIL"
    all_pass = s1 == "PASS" and s3 == "PASS"

    print(f"\n  W1: sliding_mean={mean_w1*100:.1f}%  accepted={accepted['w1']*100:.1f}%  delta={delta_w1*100:.1f}pp [{s1}]")
    print(f"  W3: sliding_mean={mean_w3*100:.1f}%  accepted={accepted['w3']*100:.1f}%  delta={delta_w3*100:.1f}pp [{s3}]")
    print(f"  Verdict: {'ALL PASS' if all_pass else 'FAIL'}")

    return all_pass


def export_csv(out_path, per_seed_trajs):
    """Export per-seed + mean ± std to CSV."""
    all_t = sorted(set(t for t_arr, _ in per_seed_trajs for t in t_arr))
    common_t = np.array(all_t)

    interp_vals = []
    for t_arr, p_arr in per_seed_trajs:
        interp = np.interp(common_t, t_arr, p_arr)
        interp_vals.append(interp)

    interp_vals = np.array(interp_vals)
    mean_vals = np.mean(interp_vals, axis=0)
    std_vals = np.std(interp_vals, axis=0)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["time_s", "mean", "std"]
        for i in range(len(interp_vals)):
            header.append(f"seed{i}")
        writer.writerow(header)

        for j in range(len(common_t)):
            row = [f"{common_t[j]:.2f}", f"{mean_vals[j]:.6f}", f"{std_vals[j]:.6f}"]
            for i in range(len(interp_vals)):
                row.append(f"{interp_vals[i][j]:.6f}")
            writer.writerow(row)

    print(f"  CSV exported: {out_path} ({len(common_t)} time points, {len(interp_vals)} seeds)")
    return common_t, mean_vals, std_vals


def main():
    all_checks_pass = True

    for cfg in CONFIGS:
        tag = cfg["tag"]
        label = cfg["label"]
        job_info = load_job_info(cfg["workload"])

        print(f"\n{'#'*68}")
        print(f"# {label}")
        print(f"{'#'*68}")

        all_records = load_d1_records(tag, D1_SEEDS)
        print(f"  Loaded {len(all_records)} seeds, "
              f"total records: {sum(len(r) for r in all_records)}")

        seed_trajs = []
        for s, records in enumerate(all_records):
            t_arr, p_arr = compute_sliding_trajectory(records, job_info)
            pts_in_w3 = sum(1 for t in t_arr if 500 <= t <= 600)
            print(f"  seed{s}: {len(t_arr)} time points, "
                  f"t_range=[{t_arr[0]:.1f},{t_arr[-1]:.1f}]s, "
                  f"W3 pts={pts_in_w3}")
            seed_trajs.append((t_arr, p_arr))

        # Self-check
        ok = self_check(label, tag, seed_trajs, D1_ACCEPTED[tag])
        if not ok:
            all_checks_pass = False

        # Export CSV
        export_csv(cfg["out_csv"], seed_trajs)

    # ── Final summary ──
    print(f"\n{'='*68}")
    print("FINAL")
    print(f"{'='*68}")
    if all_checks_pass:
        print("  ALL SELF-CHECKS PASSED (delta < 0.05)")
    else:
        print("  SELF-CHECK FAILED — DO NOT PROCEED")
    return 0 if all_checks_pass else 1


if __name__ == "__main__":
    sys.exit(main())
