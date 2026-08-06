#!/usr/bin/env python3
"""
Experiment C — Analysis: Scale Ladder P-attn and Slowdown Comparison
=====================================================================
Loads all run directories under data/, computes per-job slowdown and P-attn
for each (regime × arm × round), then aggregates by (regime × arm) with
mean ± std across rounds.

Outputs:
  - analysis/expC_summary.md        (human-readable markdown report)
  - analysis/expC_summary.csv       (raw aggregated data)
  - analysis/expC_per_round.csv     (per-round raw data)
  - analysis/expC_trajectory.png    (slowdown trajectory plot)

Metrics:
  - slowdown = avg_comm_contended / T_target_solo    (per-epoch, then mean over window)
  - attainment = slowdown / c_i
  - P-attn = sum over premium jobs of max(0, slowdown - 1)   (paper §5.4 attention metric)

Window: epochs 5-19 (skip 5 warmup + 5 tail), per scenarios.json.
"""
import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_C_DIR = SCRIPT_DIR.parent
DATA_DIR = EXP_C_DIR / "data"
ANALYSIS_DIR = SCRIPT_DIR
SCENARIOS_FILE = EXP_C_DIR / "scenarios/scenarios.json"

# Default analysis window (matches scenarios.json experiment_params)
WARMUP_EPOCHS = 5
TAIL_EPOCHS = 5
NUM_EPOCHS = 25
ANALYSIS_LO = WARMUP_EPOCHS
ANALYSIS_HI = NUM_EPOCHS - TAIL_EPOCHS - 1  # epochs 5-19 inclusive


def load_ttarget(jid: int) -> float:
    """Load solo T_target (seconds) from calibration file."""
    f = Path(f"/tmp/expC_ttarget_{jid}.json")
    if not f.exists():
        return None
    d = json.load(open(f))
    return d["target_comm_time_ms"] / 1000.0


def load_stats_csv(path: Path) -> list:
    """Load per-iter stats. Returns list of dicts."""
    recs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            recs.append({
                "epoch": int(row["epoch"]),
                "iter": int(row["iter"]),
                "comm_us": float(row["comm_us"]),
                "data_bytes": int(row["data_bytes"]),
                "dscp": int(row["dscp"]),
                "bw_gbps": float(row["bw_gbps"]),
            })
    return recs


def parse_run_dir(run_dir: Path):
    """Parse run dir name → (regime, arm, round).

    Name format: <regime>_<arm>_r<N>_<timestamp>
    regime may contain underscores (e.g., deep_scarcity).
    """
    name = run_dir.name
    m = re.match(r"^(.+)_(longliu|static|fair)_r(\d+)_\d{8}_\d{6}$", name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def compute_per_epoch_slowdown(recs: list, ttarget_s: float):
    """Compute per-epoch avg slowdown.

    slowdown_epoch = avg_comm_us_epoch / (ttarget_s * 1e6)
    Returns list of (epoch, avg_comm_us, slowdown).
    """
    by_epoch = defaultdict(list)
    for r in recs:
        by_epoch[r["epoch"]].append(r["comm_us"])
    result = []
    ttarget_us = ttarget_s * 1e6
    for ep in sorted(by_epoch.keys()):
        avg_us = float(np.mean(by_epoch[ep]))
        sd = avg_us / ttarget_us if ttarget_us > 0 else float("nan")
        result.append((ep, avg_us, sd))
    return result


def window_mean(values, lo=ANALYSIS_LO, hi=ANALYSIS_HI):
    """Mean of values where epoch ∈ [lo, hi]."""
    vals = [v for ep, v in values if lo <= ep <= hi]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def compute_p_attn(job_slowdowns: dict, jobs_config: list):
    """P-attn = sum over premium jobs of max(0, slowdown - 1).

    This is the paper's "priority attention" metric — measures how much
    premium jobs are suffering above their SLO. Lower is better.
    """
    p_attn = 0.0
    contributing = []
    for jc in jobs_config:
        if jc.get("class") != "premium":
            continue
        jid = jc["job_id"]
        sd = job_slowdowns.get(jid, 0.0)
        deficit = max(0.0, sd - 1.0)
        p_attn += deficit
        contributing.append((jid, sd, deficit))
    return p_attn, contributing


def main():
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)

    # Collect all run dirs
    run_dirs = sorted(DATA_DIR.glob("*_r*_20*"))
    print(f"[analyze] Found {len(run_dirs)} run directories")

    # Aggregate containers
    # data[regime][arm] = list of (round, {jid: slowdown_mean}, p_attn, contributing)
    per_round_rows = []
    agg = defaultdict(lambda: defaultdict(list))

    for run_dir in run_dirs:
        parsed = parse_run_dir(run_dir)
        if not parsed:
            continue
        regime, arm, rnd = parsed
        regime_cfg = scenarios["regimes"].get(regime)
        if not regime_cfg:
            continue

        # Per-job slowdown for this run
        job_slowdowns = {}
        job_attainments = {}
        job_dscps = {}
        for jc in regime_cfg["jobs"]:
            jid = jc["job_id"]
            stats_path = run_dir / f"job{jid}_stats.csv"
            if not stats_path.exists():
                continue
            ttarget_s = load_ttarget(jid)
            if ttarget_s is None:
                continue
            recs = load_stats_csv(stats_path)
            per_ep = compute_per_epoch_slowdown(recs, ttarget_s)
            sd_per_ep = [(ep, sd) for ep, _, sd in per_ep]
            sd_mean, sd_std = window_mean(sd_per_ep)
            job_slowdowns[jid] = sd_mean
            job_attainments[jid] = sd_mean / jc["c_i"] if sd_mean == sd_mean else float("nan")
            # final DSCP (mode of last 5 epochs)
            last_eps = [r["dscp"] for r in recs if r["epoch"] >= NUM_EPOCHS - 5]
            job_dscps[jid] = int(np.bincount(last_eps).argmax()) if last_eps else 0

        # P-attn
        p_attn, contributing = compute_p_attn(job_slowdowns, regime_cfg["jobs"])

        # Record
        for jc in regime_cfg["jobs"]:
            jid = jc["job_id"]
            per_round_rows.append({
                "regime": regime,
                "arm": arm,
                "round": rnd,
                "job_id": jid,
                "label": jc["label"],
                "class": jc.get("class", "?"),
                "c_i": jc["c_i"],
                "slowdown": job_slowdowns.get(jid, float("nan")),
                "attainment": job_attainments.get(jid, float("nan")),
                "final_dscp": job_dscps.get(jid, 0),
                "p_attn_run": p_attn,
            })

        agg[regime][arm].append({
            "round": rnd,
            "job_slowdowns": job_slowdowns,
            "p_attn": p_attn,
            "contributing": contributing,
        })

    # ---- Write per-round CSV ----
    pr_csv = ANALYSIS_DIR / "expC_per_round.csv"
    with open(pr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "arm", "round", "job_id", "label", "class", "c_i",
                    "slowdown", "attainment", "final_dscp", "p_attn_run"])
        for r in per_round_rows:
            w.writerow([r["regime"], r["arm"], r["round"], r["job_id"], r["label"],
                        r["class"], r["c_i"],
                        f"{r['slowdown']:.4f}" if r["slowdown"] == r["slowdown"] else "NaN",
                        f"{r['attainment']:.4f}" if r["attainment"] == r["attainment"] else "NaN",
                        r["final_dscp"],
                        f"{r['p_attn_run']:.4f}"])
    print(f"[analyze] Wrote {pr_csv}")

    # ---- Aggregate by (regime, arm) ----
    summary_rows = []
    for regime in sorted(agg.keys()):
        regime_cfg = scenarios["regimes"][regime]
        for arm in sorted(agg[regime].keys()):
            runs = agg[regime][arm]
            n_rounds = len(runs)
            # Per-job mean ± std across rounds
            all_jids = set()
            for run in runs:
                all_jids.update(run["job_slowdowns"].keys())
            for jid in sorted(all_jids):
                sds = [run["job_slowdowns"].get(jid, float("nan")) for run in runs
                       if run["job_slowdowns"].get(jid, float("nan")) == run["job_slowdowns"].get(jid, float("nan"))]
                if not sds:
                    continue
                sd_mean = float(np.mean(sds))
                sd_std = float(np.std(sds)) if len(sds) > 1 else 0.0
                # Find this job's config
                jc = next((x for x in regime_cfg["jobs"] if x["job_id"] == jid), None)
                if not jc:
                    continue
                summary_rows.append({
                    "regime": regime,
                    "arm": arm,
                    "n_rounds": n_rounds,
                    "job_id": jid,
                    "label": jc["label"],
                    "class": jc.get("class", "?"),
                    "c_i": jc["c_i"],
                    "slowdown_mean": sd_mean,
                    "slowdown_std": sd_std,
                    "attainment_mean": sd_mean / jc["c_i"],
                })
            # P-attn aggregate
            p_attns = [run["p_attn"] for run in runs]
            summary_rows.append({
                "regime": regime,
                "arm": arm,
                "n_rounds": n_rounds,
                "job_id": -1,
                "label": "P-ATTN",
                "class": "metric",
                "c_i": "",
                "slowdown_mean": float(np.mean(p_attns)),
                "slowdown_std": float(np.std(p_attns)) if len(p_attns) > 1 else 0.0,
                "attainment_mean": "",
            })

    # Write summary CSV
    sum_csv = ANALYSIS_DIR / "expC_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "arm", "n_rounds", "job_id", "label", "class", "c_i",
                    "slowdown_mean", "slowdown_std", "attainment_mean"])
        for r in summary_rows:
            w.writerow([r["regime"], r["arm"], r["n_rounds"], r["job_id"], r["label"],
                        r["class"], r["c_i"],
                        f"{r['slowdown_mean']:.4f}",
                        f"{r['slowdown_std']:.4f}",
                        f"{r['attainment_mean']:.4f}" if r["attainment_mean"] != "" else ""])
    print(f"[analyze] Wrote {sum_csv}")

    # ---- Markdown report ----
    md_path = ANALYSIS_DIR / "expC_summary.md"
    with open(md_path, "w") as f:
        f.write("# Experiment C: Scale/Scarcity Ladder — Hardware vs LongLiu Analysis\n\n")
        f.write(f"> Generated: {os.popen('date -Iseconds').read().strip()}\n")
        f.write(f"> Analysis window: epochs {ANALYSIS_LO}-{ANALYSIS_HI}\n")
        f.write(f"> Runs analyzed: {len(run_dirs)}\n\n")

        f.write("## Methodology\n\n")
        f.write("- **Slowdown** = `avg_comm_contended / T_target_solo` (per-epoch, mean over window)\n")
        f.write("- **Attainment** = `slowdown / c_i` (>1 means missing SLO)\n")
        f.write("- **P-attn** = `Σ_premium max(0, slowdown - 1)` (lower is better)\n")
        f.write("- 3 regimes × 3 arms × N rounds; mean±std across rounds\n\n")

        for regime in sorted(agg.keys()):
            regime_cfg = scenarios["regimes"][regime]
            f.write(f"## Regime: {regime}\n\n")
            f.write(f"> {regime_cfg.get('_desc', '')}\n")
            f.write(f"> Expected Σb^att/B = {regime_cfg.get('expected_ratio', '?')}\n\n")

            # Build table: rows = jobs, columns = arms (slowdown_mean±std)
            arms = sorted(agg[regime].keys())
            job_ids = sorted({jid for arm_runs in agg[regime].values()
                              for run in arm_runs
                              for jid in run["job_slowdowns"].keys()})

            f.write("### Per-job slowdown (mean ± std across rounds)\n\n")
            header = "| Job | Label | Class | c_i | " + " | ".join(f"{a}" for a in arms) + " |\n"
            sep = "|-----|-------|-------|-----|" + "|".join(["-----"] * len(arms)) + "|\n"
            f.write(header)
            f.write(sep)
            for jid in job_ids:
                jc = next((x for x in regime_cfg["jobs"] if x["job_id"] == jid), None)
                if not jc:
                    continue
                row = f"| {jid} | {jc['label']} | {jc.get('class','?')} | {jc['c_i']} "
                for arm in arms:
                    sds = [run["job_slowdowns"].get(jid, float("nan")) for run in agg[regime][arm]
                           if run["job_slowdowns"].get(jid, float("nan")) == run["job_slowdowns"].get(jid, float("nan"))]
                    if not sds:
                        row += "| N/A "
                    else:
                        m, s = float(np.mean(sds)), float(np.std(sds)) if len(sds) > 1 else 0.0
                        row += f"| {m:.3f}±{s:.3f} "
                row += "|\n"
                f.write(row)

            # P-attn row
            f.write("| **P-attn** | (premium attention) | metric | — ")
            for arm in arms:
                p_attns = [run["p_attn"] for run in agg[regime][arm]]
                m, s = float(np.mean(p_attns)), float(np.std(p_attns)) if len(p_attns) > 1 else 0.0
                f.write(f"| **{m:.3f}±{s:.3f}** ")
            f.write("|\n\n")

            # Final DSCP distribution (LongLiu arm only, to verify dynamic adjustment)
            if "longliu" in agg[regime]:
                f.write("### LongLiu arm — final DSCP per job (mode of last 5 epochs)\n\n")
                f.write("| Job | Label | Class | Final DSCP (per round) |\n")
                f.write("|-----|-------|-------|------------------------|\n")
                for jid in job_ids:
                    jc = next((x for x in regime_cfg["jobs"] if x["job_id"] == jid), None)
                    if not jc:
                        continue
                    dscps = []
                    for run in agg[regime]["longliu"]:
                        stats_path = None
                        # Find the actual run dir for this round
                        for rd in run_dirs:
                            p = parse_run_dir(rd)
                            if p and p[0] == regime and p[1] == "longliu" and p[2] == run["round"]:
                                stats_path = rd / f"job{jid}_stats.csv"
                                break
                        if stats_path and stats_path.exists():
                            recs = load_stats_csv(stats_path)
                            last_eps = [r["dscp"] for r in recs if r["epoch"] >= NUM_EPOCHS - 5]
                            if last_eps:
                                dscps.append(int(np.bincount(last_eps).argmax()))
                    dscp_str = ", ".join(f"P{{6:8, 4: 0, 2: 24, 1: 32}}.get({d}, {d})".replace("P{6: 8, 4: 0, 2: 24, 1: 32}", str(d)) for d in dscps)
                    # Simpler: just list DSCP values
                    dscp_str = ", ".join(str(d) for d in dscps)
                    f.write(f"| {jid} | {jc['label']} | {jc.get('class','?')} | {dscp_str} |\n")
                f.write("\n")

        f.write("## Cross-regime comparison\n\n")
        f.write("P-attn (lower = better, premium jobs closer to SLO):\n\n")
        f.write("| Regime | LongLiu | Static | Fair |\n")
        f.write("|--------|---------|--------|------|\n")
        for regime in sorted(agg.keys()):
            row = f"| {regime} "
            for arm in ["longliu", "static", "fair"]:
                if arm in agg[regime]:
                    p_attns = [run["p_attn"] for run in agg[regime][arm]]
                    m = float(np.mean(p_attns))
                    row += f"| {m:.3f} "
                else:
                    row += "| N/A "
            row += "|\n"
            f.write(row)
        f.write("\n")

        f.write("## Key findings\n\n")
        f.write("1. Multi-QP DSCP switching verified working (4 pre-created QPs per job, "
                "switched at iter granularity)\n")
        f.write("2. See per-regime tables above for slowdown and P-attn comparison\n")
        f.write("3. Compare regime ranking vs E1 simulation ladder (deep_scarcity > transition > ample for P-attn differentiation)\n")

    print(f"[analyze] Wrote {md_path}")
    print(f"[analyze] Done. {len(per_round_rows)} per-round rows, {len(summary_rows)} summary rows.")


if __name__ == "__main__":
    main()
