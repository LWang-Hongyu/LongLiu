#!/usr/bin/env python3
"""Quick check: compute slowdown per job for each regime × arm × round."""
import csv, json, os, sys
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WARMUP_EPOCHS = 5
TAIL_EPOCHS = 5
NUM_EPOCHS = 25
ANALYSIS_LO = WARMUP_EPOCHS
ANALYSIS_HI = NUM_EPOCHS - TAIL_EPOCHS - 1  # epochs 5-19

def load_ttarget(jid):
    f = Path(f"/tmp/expC_ttarget_{jid}.json")
    if not f.exists():
        return None
    d = json.load(open(f))
    return d["target_comm_time_ms"] / 1000.0  # convert ms → s

def load_stats_csv(path):
    recs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            recs.append({
                "epoch": int(row["epoch"]),
                "comm_us": float(row["comm_us"]),
                "dscp": int(row["dscp"]),
            })
    return recs

def main():
    # Collect all run dirs grouped by regime_arm
    run_dirs = sorted(DATA_DIR.glob("*_r1_*"))
    print(f"Found {len(run_dirs)} r1 runs")
    print()
    print(f"{'regime':<15} {'arm':<10} {'job':<4} {'class':<10} {'c_i':<5} "
          f"{'T_target_us':<12} {'avg_comm_us':<12} {'slowdown':<10} {'final_dscp':<10}")
    print("-" * 100)

    scenarios = json.load(open(Path(__file__).resolve().parent.parent /
                                "scenarios/scenarios_snapshot.json" if False else
                                Path(__file__).resolve().parent.parent / "scenarios/scenarios.json"))

    for run_dir in run_dirs:
        name = run_dir.name  # e.g. deep_scarcity_longliu_r1_20260730_023315
        parts = name.split("_")
        # Parse: regime_arm_rN_timestamp
        # regime may contain underscores (deep_scarcity)
        # Find the r<N> marker
        ridx = None
        for i, p in enumerate(parts):
            if p.startswith("r") and p[1:].isdigit():
                ridx = i
                break
        if ridx is None or ridx < 2:
            continue
        regime = "_".join(parts[:ridx-1])
        arm = parts[ridx-1]
        round_num = parts[ridx]

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
            recs = load_stats_csv(stats_path)
            window = [r for r in recs if ANALYSIS_LO <= r["epoch"] <= ANALYSIS_HI]
            if not window:
                continue
            avg_comm_us = np.mean([r["comm_us"] for r in window])
            ttarget_us = ttarget_s * 1e6
            slowdown = avg_comm_us / ttarget_us
            final_dscp = window[-1]["dscp"]
            cls = jc.get("class", "?")
            print(f"{regime:<15} {arm:<10} {jid:<4} {cls:<10} {jc['c_i']:<5} "
                  f"{ttarget_us:<12.1f} {avg_comm_us:<12.1f} {slowdown:<10.3f} "
                  f"{final_dscp:<10}")
        print()

if __name__ == "__main__":
    main()
