#!/usr/bin/env python3
"""
Experiment C — Trajectory plot generator
=========================================
Generates per-regime slowdown trajectory plots comparing longliu/static/fair arms.
"""
import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_C_DIR = SCRIPT_DIR.parent
DATA_DIR = EXP_C_DIR / "data"
ANALYSIS_DIR = SCRIPT_DIR
SCENARIOS_FILE = EXP_C_DIR / "scenarios/scenarios.json"

WARMUP_EPOCHS = 5
NUM_EPOCHS = 25
ANALYSIS_LO = WARMUP_EPOCHS
ANALYSIS_HI = NUM_EPOCHS - 5 - 1


def load_ttarget(jid):
    f = Path(f"/tmp/expC_ttarget_{jid}.json")
    if not f.exists():
        return None
    return json.load(open(f))["target_comm_time_ms"] / 1000.0


def parse_run_dir(run_dir: Path):
    name = run_dir.name
    m = re.match(r"^(.+)_(longliu|static|fair)_r(\d+)_\d{8}_\d{6}$", name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def main():
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)

    run_dirs = sorted(DATA_DIR.glob("*_r*_20*"))
    print(f"[plot] Found {len(run_dirs)} run directories")

    # Aggregate per-epoch slowdown across rounds: traj[regime][arm][jid][epoch] = [sd_round1, sd_round2, ...]
    traj = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for run_dir in run_dirs:
        parsed = parse_run_dir(run_dir)
        if not parsed:
            continue
        regime, arm, rnd = parsed
        regime_cfg = scenarios["regimes"].get(regime)
        if not regime_cfg:
            continue

        for jc in regime_cfg["jobs"]:
            jid = jc["job_id"]
            stats_path = run_dir / f"job{jid}_stats.csv"
            if not stats_path.exists():
                continue
            ttarget_s = load_ttarget(jid)
            if ttarget_s is None:
                continue
            ttarget_us = ttarget_s * 1e6

            by_epoch = defaultdict(list)
            with open(stats_path) as f:
                for row in csv.DictReader(f):
                    ep = int(row["epoch"])
                    by_epoch[ep].append(float(row["comm_us"]))

            for ep, comm_list in by_epoch.items():
                avg_us = float(np.mean(comm_list))
                sd = avg_us / ttarget_us
                traj[regime][arm][jid][ep].append(sd)

    # Plot: 3 subplots (one per regime), each with N_jobs × 3 arms lines
    regimes = sorted(traj.keys())
    fig, axes = plt.subplots(len(regimes), 1, figsize=(12, 4 * len(regimes)),
                              sharex=True)
    if len(regimes) == 1:
        axes = [axes]

    arm_colors = {"longliu": "C0", "static": "C1", "fair": "C2"}
    arm_styles = {"longliu": "-", "static": "--", "fair": ":"}

    for ax, regime in zip(axes, regimes):
        regime_cfg = scenarios["regimes"][regime]
        jobs = regime_cfg["jobs"]
        n_jobs = len(jobs)

        # For each job, plot 3 arms (mean across rounds)
        for jidx, jc in enumerate(jobs):
            jid = jc["job_id"]
            for arm in ["longliu", "static", "fair"]:
                if arm not in traj[regime] or jid not in traj[regime][arm]:
                    continue
                ep_sds = traj[regime][arm][jid]
                epochs = sorted(ep_sds.keys())
                means = [float(np.mean(ep_sds[ep])) if ep_sds[ep] else float("nan") for ep in epochs]
                stds = [float(np.std(ep_sds[ep])) if len(ep_sds[ep]) > 1 else 0 for ep in epochs]

                label = f"J{jid}({jc.get('class','?')[0:3]}) {arm}"
                ax.plot(epochs, means, color=arm_colors[arm],
                        linestyle=arm_styles[arm], linewidth=1.5,
                        alpha=0.8, label=label)
                ax.fill_between(epochs,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=arm_colors[arm], alpha=0.1)

        ax.axvspan(ANALYSIS_LO, ANALYSIS_HI, alpha=0.05, color="green",
                   label="analysis window")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
        ax.set_ylabel("Slowdown")
        ax.set_title(f"Regime: {regime} (Σb^att/B≈{regime_cfg.get('expected_ratio','?')}, "
                     f"{n_jobs} jobs)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=3)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Experiment C: Per-epoch slowdown trajectory (3 regimes × 3 arms, mean±std across 3 rounds)",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = ANALYSIS_DIR / "expC_trajectory.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[plot] Wrote {out_path}")

    # Also a P-attn cross-regime bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    p_attn_data = defaultdict(lambda: defaultdict(list))
    for regime in regimes:
        regime_cfg = scenarios["regimes"][regime]
        for run_dir in run_dirs:
            parsed = parse_run_dir(run_dir)
            if not parsed:
                continue
            r_regime, arm, rnd = parsed
            if r_regime != regime:
                continue
            # Compute P-attn for this run
            p_attn = 0.0
            for jc in regime_cfg["jobs"]:
                if jc.get("class") != "premium":
                    continue
                jid = jc["job_id"]
                stats_path = run_dir / f"job{jid}_stats.csv"
                if not stats_path.exists():
                    continue
                ttarget_s = load_ttarget(jid)
                if ttarget_s is None:
                    continue
                ttarget_us = ttarget_s * 1e6
                sds = []
                by_epoch = defaultdict(list)
                with open(stats_path) as f:
                    for row in csv.DictReader(f):
                        ep = int(row["epoch"])
                        if ANALYSIS_LO <= ep <= ANALYSIS_HI:
                            by_epoch[ep].append(float(row["comm_us"]))
                if not by_epoch:
                    continue
                ep_means = [float(np.mean(v)) / ttarget_us for v in by_epoch.values()]
                sd_mean = float(np.mean(ep_means))
                p_attn += max(0.0, sd_mean - 1.0)
            p_attn_data[regime][arm].append(p_attn)

    arms = ["longliu", "static", "fair"]
    x = np.arange(len(regimes))
    width = 0.25
    for i, arm in enumerate(arms):
        means = [float(np.mean(p_attn_data[r][arm])) if p_attn_data[r][arm] else 0
                 for r in regimes]
        stds = [float(np.std(p_attn_data[r][arm])) if len(p_attn_data[r][arm]) > 1 else 0
                for r in regimes]
        ax2.bar(x + i * width, means, width, yerr=stds, label=arm,
                color=arm_colors[arm], alpha=0.8, capsize=3)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(regimes)
    ax2.set_ylabel("P-attn (lower = better)")
    ax2.set_title("Experiment C: P-attn cross-regime comparison (mean ± std, 3 rounds)")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    out_path2 = ANALYSIS_DIR / "expC_p_attn_bars.png"
    fig2.savefig(out_path2, dpi=120, bbox_inches="tight")
    print(f"[plot] Wrote {out_path2}")


if __name__ == "__main__":
    main()
