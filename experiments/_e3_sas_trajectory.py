"""
sas-t 轨迹图 v2：滑动100s窗 + start_ms语义 + per-regime口径
满足【主图重绘指令】全部要求。

数据源：PAPER_EVIDENCE/05_E3_swap_main records.jsonl（归档正式批）
输出：
  - outputs/e3_swap/sas_t_trajectory.png
  - outputs/e3_swap/sas_t_data_e3.csv
  - outputs/e3_swap/sas_t_data_e3p.csv
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

# ── Paths ──
DATA_BASE = "PAPER_EVIDENCE/05_E3_swap_main"
OUT_BASE = "outputs/e3_swap"

# ── Constants ──
WINDOW_S = 100.0         # sliding window size
SWAP_TIME_S = 300.0       # ci swap time
TIME_STEP = 0.25          # time grid resolution (seconds)
T_START = 100.0           # start of time grid
T_END = 600.0             # end of time grid

# Window annotation regions (for chart display only)
ANNOT_W1 = (200, 300)
ANNOT_W2 = (300, 320)
ANNOT_W3 = (500, 600)

# Models classified as "large" (mirrors _do_ci_swap)
LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}

# ── Configs ──
CONFIGS = [
    {
        "label": "E3 Control Arm (CRUX-advantaging swap, 800G)",
        "tag": "e3_swap",
        "workload": FEAS_BOUNDARY_V3_WORKLOAD,
        "out_csv": f"{OUT_BASE}/sas_t_data_e3.csv",
    },
    {
        "label": "E3' Kill Arm (CRUX-disadvantaging swap, 630G)",
        "tag": "e3p_swap",
        "workload": FEAS_BOUNDARY_V3_PRO_WORKLOAD,
        "out_csv": f"{OUT_BASE}/sas_t_data_e3p.csv",
    },
]

# ── archived 3-seed summary (self-check targets) ──
ARCHIVED = {
    "e3_swap": {
        # 3-seed mean of archive W1/W3 fixed-window P-attn
        "v4":   {"w1": 1.0, "w3": 1.0},
        "CRUX": {"w1": (0.375+0.5+0.5)/3, "w3": (0.5+0.8333+0.5)/3},
    },
    "e3p_swap": {
        "v4":   {"w1": 1.0, "w3": 1.0},
        "CRUX": {"w1": (0.4+0.4+1.0)/3, "w3": (0.0+0.25+0.0)/3},
    },
}


# ═══════════════════════════════════════════════════════════════
# Target computation
# ═══════════════════════════════════════════════════════════════

def get_target(comp_ms, comm_solo_ms, ci):
    comm_budget = ci * comm_solo_ms * OVERHEAD
    if OVERLAP > 0:
        return max(comp_ms, comm_budget) + (1.0 - OVERLAP) * min(comp_ms, comm_budget)
    return comp_ms + comm_budget


# ═══════════════════════════════════════════════════════════════
# Workload info (mirrors _do_ci_swap for post-swap ci)
# ═══════════════════════════════════════════════════════════════

def load_job_info(workload_raw):
    """Build per-job info with correct post-swap ci mapping."""
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
            if is_large:
                post_ci = 1.5
            elif dp == 4:
                post_ci = 2.0
            else:
                post_ci = 1.5
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


# ═══════════════════════════════════════════════════════════════
# Sliding window P-attn trajectory
# ═══════════════════════════════════════════════════════════════

def compute_sliding_trajectory(records, job_info):
    """Sliding 100s window, start_ms filter, per-regime tier/target.

    For each t on the time grid:
      window = [t - WINDOW_S, t]  (in seconds)
      filter: start_ms in window
      if t <= SWAP_TIME_S → pre-swap premium set + pre_target
      if t >  SWAP_TIME_S → post-swap premium set + post_target
      P-attn = fraction of premium jobs with SAS >= 0.98

    Returns (times_seconds, pattn_values).
    """
    if not records:
        return [], []

    records.sort(key=lambda r: r["start_ms"])
    n_points = int((T_END - T_START) / TIME_STEP) + 1
    time_grid = np.linspace(T_START, T_END, n_points)

    w_ms = WINDOW_S * 1000.0
    left_idx = 0
    right_idx = 0
    # job_agg: {jid: [sum_iter_ms, count]}
    job_sum = defaultdict(float)
    job_cnt = defaultdict(int)

    results_t = []
    results_pattn = []

    for t_s in time_grid:
        t_ms = t_s * 1000.0
        lo = max(0.0, t_ms - w_ms)
        hi = t_ms

        # Slide left: remove records that fell out of window
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

        # Slide right: add new records entering window
        while right_idx < len(records) and records[right_idx]["start_ms"] <= hi:
            r = records[right_idx]
            jid = r["jid"]
            job_sum[jid] += r["iter_ms"]
            job_cnt[jid] = job_cnt.get(jid, 0) + 1
            right_idx += 1

        # Determine regime → premium set + target key
        if t_s <= SWAP_TIME_S:
            premium_set = {jid for jid, info in job_info.items()
                           if info["pre_is_premium"]}
            target_key = "pre_target"
        else:
            premium_set = {jid for jid, info in job_info.items()
                           if info["post_is_premium"]}
            target_key = "post_target"

        # Compute P-attn
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


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_all_records(tag, policy):
    """Load records from 3 seeds for a given (tag, policy)."""
    all_records = []
    for s in [0, 1, 2]:
        path = f"{DATA_BASE}/{tag}_{policy}_s{s}/records.jsonl"
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


# ═══════════════════════════════════════════════════════════════
# Self-check
# ═══════════════════════════════════════════════════════════════

def self_check(config_label, tag, per_seed_trajectories, archived):
    """Verify sliding-window P-attn against archived 3-seed summary.

    Check at t=300 (sliding window [200,300] = archive W1) and
    t=600 (sliding window [500,600] = archive W3).
    """
    print(f"\n{'='*68}")
    print(f"SELF-CHECK: {config_label}")
    print(f"{'='*68}")
    print(f"  (sliding window at t=300s → archive W1; t=600s → archive W3)")
    print()

    all_pass = True
    for pn in ["v4", "CRUX"]:
        seeds = per_seed_trajectories.get(pn, [])
        if not seeds:
            continue

        # Per-seed P-attn at t=300 and t=600
        w1_vals, w3_vals = [], []
        for seed_idx, (t_arr, p_arr) in enumerate(seeds):
            if not t_arr:
                continue
            t_arr = np.array(t_arr)
            p_arr = np.array(p_arr)

            # t=300
            idx300 = np.argmin(np.abs(t_arr - 300.0))
            w1_vals.append(p_arr[idx300])

            # t=600
            idx600 = np.argmin(np.abs(t_arr - 600.0))
            w3_vals.append(p_arr[idx600])

        if not w3_vals:
            continue

        mean_w1 = np.mean(w1_vals)
        mean_w3 = np.mean(w3_vals)

        arch_w1 = archived[pn]["w1"]
        arch_w3 = archived[pn]["w3"]

        delta_w1 = abs(mean_w1 - arch_w1)
        delta_w3 = abs(mean_w3 - arch_w3)

        s1 = "PASS" if delta_w1 < 0.015 else "FAIL"
        s3 = "PASS" if delta_w3 < 0.015 else "FAIL"
        if s1 == "FAIL" or s3 == "FAIL":
            all_pass = False

        print(f"  {pn:<6} W1: sliding_mean={mean_w1:.4f}  archive={arch_w1:.4f}  "
              f"delta={delta_w1:.4f} [{s1}]  seeds={[f'{v:.4f}' for v in w1_vals]}")
        print(f"  {pn:<6} W3: sliding_mean={mean_w3:.4f}  archive={arch_w3:.4f}  "
              f"delta={delta_w3:.4f} [{s3}]  seeds={[f'{v:.4f}' for v in w3_vals]}")
        print()

    verdict = "ALL PASS" if all_pass else "FAIL — DO NOT POST"
    print(f"  Verdict: {verdict}")
    return all_pass


# ═══════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════

def export_csv(out_path, time_grid, per_policy):
    """Export mean ± std to CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["time_s"]
        for pn in ["v4", "CRUX"]:
            header += [f"{pn}_mean", f"{pn}_std"]
        writer.writerow(header)

        n = len(time_grid)
        for i in range(n):
            row = [f"{time_grid[i]:.2f}"]
            for pn in ["v4", "CRUX"]:
                mean, std = per_policy.get(pn, ([], []))
                row.append(f"{mean[i]:.6f}" if i < len(mean) else "")
                row.append(f"{std[i]:.6f}" if i < len(std) else "")
            writer.writerow(row)
    print(f"  CSV exported: {out_path} ({n} time points)")


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def make_plot(all_results):
    """Generate the 2-panel trajectory figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    colors = {"v4": "#2196F3", "CRUX": "#F44336"}

    for ax, cfg, res in zip([ax1, ax2], CONFIGS, all_results):
        tag = cfg["tag"]
        label = cfg["label"]

        for pn in ["v4", "CRUX"]:
            per_policy = res.get(pn, {})
            time_grid = per_policy.get("time_grid", None)
            means = per_policy.get("means", None)
            stds = per_policy.get("stds", None)
            if time_grid is None or means is None:
                continue

            ax.plot(time_grid, means, color=colors[pn], linewidth=1.6,
                    label=f"{pn} (n=3)", alpha=0.9)
            ax.fill_between(time_grid,
                            np.clip(means - stds, 0, None),
                            np.clip(means + stds, 0, 1),
                            color=colors[pn], alpha=0.12)

        # Window annotation regions
        for ws, we, wl in [(ANNOT_W1[0], ANNOT_W1[1], "W1"),
                            (ANNOT_W2[0], ANNOT_W2[1], "W2"),
                            (ANNOT_W3[0], ANNOT_W3[1], "W3")]:
            ax.axvspan(ws, we, alpha=0.07, color='gray')
            ax.text((ws + we) / 2, 1.02, wl, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#555555')

        # Swap line
        ax.axvline(x=SWAP_TIME_S, color='black', linestyle='--',
                   linewidth=1.0, alpha=0.45)
        ax.text(SWAP_TIME_S + 3, 0.04, 'swap', fontsize=8,
                color='black', alpha=0.55)

        ax.set_ylabel("P-attn", fontsize=10)
        ax.set_ylim(-0.05, 1.18)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.25)

    ax2.set_xlabel("Time (s)", fontsize=10)

    caption = ("sliding 100s window, start_ms semantics, per-regime tier/target; "
               "3 seeds mean ± std; swap at t=300s")
    fig.suptitle("P-attn Trajectory: v4 vs CRUX",
                 fontsize=13, fontweight='bold', y=0.997)
    fig.text(0.5, 0.005, caption, ha='center', fontsize=8,
             color='#555555', style='italic')

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    out_path = f"{OUT_BASE}/sas_t_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    all_results = []
    all_checks_pass = True

    for cfg in CONFIGS:
        tag = cfg["tag"]
        label = cfg["label"]
        workload = cfg["workload"]
        job_info = load_job_info(workload)

        print(f"\n{'#'*68}")
        print(f"# {label}")
        print(f"{'#'*68}")

        per_seed_trajectories = {}
        all_policy_results = {}

        for pn in ["v4", "CRUX"]:
            print(f"\n  [{pn}]")
            all_records = load_all_records(tag, pn)
            print(f"    Loaded {len(all_records)} seeds, "
                  f"total records: {sum(len(r) for r in all_records)}")

            seed_trajs = []
            for s, records in enumerate(all_records):
                t_arr, p_arr = compute_sliding_trajectory(records, job_info)
                print(f"    seed{s}: {len(t_arr)} time points, "
                      f"t_range=[{t_arr[0]:.1f}, {t_arr[-1]:.1f}]s")
                seed_trajs.append((t_arr, p_arr))
            per_seed_trajectories[pn] = seed_trajs

            # Align to common grid for mean/std
            if seed_trajs:
                # Use finest common grid
                all_t = sorted(set(
                    t for t_arr, _ in seed_trajs for t in t_arr
                ))
                common_t = np.array(all_t)

                interp_vals = []
                for t_arr, p_arr in seed_trajs:
                    interp = np.interp(common_t, t_arr, p_arr)
                    interp_vals.append(interp)

                interp_vals = np.array(interp_vals)
                mean_vals = np.mean(interp_vals, axis=0)
                std_vals = np.std(interp_vals, axis=0)

                all_policy_results[pn] = {
                    "time_grid": common_t,
                    "means": mean_vals,
                    "stds": std_vals,
                }

        # Self-check
        ok = self_check(label, tag, per_seed_trajectories, ARCHIVED[tag])
        if not ok:
            all_checks_pass = False

        all_results.append(all_policy_results)

        # Export CSV
        export_csv(cfg["out_csv"],
                   all_policy_results.get("v4", {}).get("time_grid", np.array([])),
                   {pn: (r["means"], r["stds"])
                    for pn, r in all_policy_results.items()})

    # ── Plot ──
    if not all_checks_pass:
        print("\n" + "!"*68)
        print("! SELF-CHECK FAILED — plot suppressed per instructions")
        print("!"*68)
        return 1

    make_plot(all_results)

    # ── Final summary ──
    print(f"\n{'='*68}")
    print("FINAL SUMMARY")
    print(f"{'='*68}")
    for cfg in CONFIGS:
        tag = cfg["tag"]
        arch = ARCHIVED[tag]
        print(f"\n  {cfg['label']}:")
        for pn in ["v4", "CRUX"]:
            a = arch[pn]
            print(f"    {pn:<6} archive W1={a['w1']:.4f}  W3={a['w3']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
