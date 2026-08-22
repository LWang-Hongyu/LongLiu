"""
sas-t 轨迹图：E3/E3' 双臂四线，W1/W2/W3 标注，逐 epoch 分辨率。

输入：records.jsonl（3 seeds × 双臂 × v4/CRUX）
输出：outputs/e3_swap/sas_t_trajectory.png
"""

from __future__ import annotations

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

BASE = "outputs/e3_swap"
SWAP_TIME_S = 300.0

# Window definitions
WINDOWS = [
    (200, 300, "W1", "pre-swap"),
    (300, 320, "W2", "transient"),
    (500, 600, "W3", "post-swap"),
]

# ── Helpers ──

def get_target(comp_ms, comm_solo_ms, ci):
    comm_budget = ci * comm_solo_ms * OVERHEAD
    if OVERLAP > 0:
        return max(comp_ms, comm_budget) + (1.0 - OVERLAP) * min(comp_ms, comm_budget)
    return comp_ms + comm_budget


def load_workload_info(workload_raw):
    """Build per-job model params and post-swap target info."""
    info = {}
    for i, (model, dp, orig_ci) in enumerate(workload_raw):
        jid = f"J{i}"
        p = MODEL_PARAMS[model]
        # comm_solo (logical)
        bpp = 2 if p.get("fp16", True) else 4
        bytes_per_iter = 2 * p["params"] * bpp / max(dp, 1)
        mb = bytes_per_iter / (1024 * 1024)
        raw_comm_ms = mb * 8 * 1024 * 1024 / (100e9) * 1000.0
        comp_ms = p.get("comp_ms", 50.0) if "comp_ms" in p else 50.0

        # Pre-swap target
        pre_target = get_target(comp_ms, raw_comm_ms, orig_ci)

        # Post-swap ci
        was_premium = orig_ci <= 2.0
        post_ci = 3.0 if was_premium else 1.5
        post_target = get_target(comp_ms, raw_comm_ms, post_ci)

        # Tier classification
        pre_is_premium = orig_ci <= 2.0
        post_is_premium = orig_ci > 2.0  # was standard → post-swap premium

        info[jid] = {
            "model": model, "dp": dp,
            "orig_ci": orig_ci, "post_ci": post_ci,
            "pre_target": pre_target, "post_target": post_target,
            "pre_is_premium": pre_is_premium,
            "post_is_premium": post_is_premium,
            "comp_ms": comp_ms, "comm_solo_ms": raw_comm_ms,
        }
    return info


def compute_trajectory(records, job_info, post_swap_premium):
    """Compute P-attn trajectory from iteration records.
    
    Uses cumulative average iteration time up to each time point.
    P-attn = fraction of premium jobs with SAS ≥ 0.98 at that time.
    SAS = post_target / cumulative_avg_iter_ms.
    """
    # Sort records by end_ms
    sorted_records = sorted(records, key=lambda r: r["end_ms"])

    # Per-job cumulative stats
    job_cum = {jid: {"sum_iter": 0.0, "count": 0, "times": [], "sas_vals": []}
               for jid in job_info}

    times = []
    pattn_vals = []

    for rec in sorted_records:
        jid = rec["jid"]
        if jid not in job_cum:
            continue
        job_cum[jid]["sum_iter"] += rec["iter_ms"]
        job_cum[jid]["count"] += 1
        t = rec["end_ms"]

        # Compute SAS for all premium jobs at this time
        p_total = 0
        p_attn = 0
        for jid_p in post_swap_premium:
            info = job_info[jid_p]
            cum = job_cum[jid_p]
            if cum["count"] == 0:
                continue
            avg_iter = cum["sum_iter"] / cum["count"]
            sas = info["post_target"] / avg_iter if avg_iter > 0 else 0.0
            p_total += 1
            if sas >= 0.98:
                p_attn += 1

        if p_total > 0:
            times.append(t / 1000.0)  # convert to seconds
            pattn_vals.append(p_attn / p_total)

    return times, pattn_vals


def load_records(tag_prefix, policy, seed):
    path = f"{BASE}/{tag_prefix}_{policy}_s{seed}/records.jsonl"
    if not os.path.exists(path):
        return None
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def plot_trajectories(ax, config_label, tag_prefix, workload_raw):
    """Plot v4 and CRUX trajectories on given axis."""
    job_info = load_workload_info(workload_raw)
    post_swap_premium = {jid for jid, info in job_info.items()
                         if info["post_is_premium"]}

    colors = {"v4": "#2196F3", "CRUX": "#F44336"}
    linestyles = {0: "-", 1: "--", 2: ":"}

    for pn in ["v4", "CRUX"]:
        all_times = []
        for s in [0, 1, 2]:
            records = load_records(tag_prefix, pn, s)
            if records is None:
                continue
            times, pattn = compute_trajectory(records, job_info, post_swap_premium)
            if times:
                all_times.append((times, pattn))

        if not all_times:
            continue

        # Align to common time grid for mean/std
        # Use the finest time grid
        common_t = sorted(set(t for times, _ in all_times for t in times))
        common_t = [t for t in common_t if t >= 50]  # skip warmup

        # Interpolate each seed to common grid
        interp_vals = []
        for times, pattn in all_times:
            interp = np.interp(common_t, times, pattn)
            interp_vals.append(interp)

        interp_vals = np.array(interp_vals)
        mean = np.mean(interp_vals, axis=0)
        std = np.std(interp_vals, axis=0)

        ax.plot(common_t, mean, color=colors[pn], linewidth=1.8,
                label=f"{pn} (n=3)", alpha=0.9)
        ax.fill_between(common_t, mean - std, mean + std,
                        color=colors[pn], alpha=0.12)

    # Draw window regions
    for ws, we, wl, _ in WINDOWS:
        ax.axvspan(ws, we, alpha=0.08, color='gray')
        ax.text((ws + we) / 2, 1.02, wl, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='gray')

    # Swap line
    ax.axvline(x=SWAP_TIME_S, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.text(SWAP_TIME_S + 2, 0.05, 'swap', fontsize=7, color='black', alpha=0.6)

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("P-attn", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(config_label, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # E3 control arm
    plot_trajectories(ax1,
                      "E3 Control Arm (CRUX-advantaging swap, 800G)",
                      "e3_swap", FEAS_BOUNDARY_V3_WORKLOAD)

    # E3' kill arm
    plot_trajectories(ax2,
                      "E3' Kill Arm (CRUX-disadvantaging swap, 630G)",
                      "e3p_swap", FEAS_BOUNDARY_V3_PRO_WORKLOAD)

    fig.suptitle("P-attn Trajectory: v4 vs CRUX (3 seeds, mean ± std)",
                 fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout()

    out_path = f"{BASE}/sas_t_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # ── Also print numeric summary for paper ──
    print()
    print("=" * 60)
    print("NUMERIC SUMMARY (3-seed mean ± std)")
    print("=" * 60)

    for cfg_label, tag, wl in [
        ("E3 (control, 800G)", "e3_swap", FEAS_BOUNDARY_V3_WORKLOAD),
        ("E3' (kill, 630G)", "e3p_swap", FEAS_BOUNDARY_V3_PRO_WORKLOAD),
    ]:
        job_info = load_workload_info(wl)
        post_swap_premium = {jid for jid, info in job_info.items()
                             if info["post_is_premium"]}
        pre_swap_premium = {jid for jid, info in job_info.items()
                            if info["pre_is_premium"]}

        print(f"\n  [{cfg_label}]")
        print(f"  {'Policy':<6} {'W1 P-attn':<12} {'W2 P-attn':<12} "
              f"{'W3 P-attn':<12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

        for pn in ["v4", "CRUX"]:
            w1_vals, w2_vals, w3_vals = [], [], []
            for s in [0, 1, 2]:
                meta_path = f"{BASE}/{tag}_{pn}_s{s}/run_meta.json"
                if not os.path.exists(meta_path):
                    continue
                meta = json.load(open(meta_path))
                w1_vals.append(meta["w1"]["p_attn"] * 100)
                w2_vals.append(meta["w2"]["p_attn"] * 100)
                w3_vals.append(meta["w3"]["p_attn"] * 100)

            if w3_vals:
                print(f"  {pn:<6} {np.mean(w1_vals):>7.1f}±{np.std(w1_vals):.1f}%  "
                      f"{np.mean(w2_vals):>7.1f}±{np.std(w2_vals):.1f}%  "
                      f"{np.mean(w3_vals):>7.1f}±{np.std(w3_vals):.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
