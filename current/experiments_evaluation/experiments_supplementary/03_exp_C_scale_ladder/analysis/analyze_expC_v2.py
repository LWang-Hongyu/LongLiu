#!/usr/bin/env python3
"""
Experiment C v2 — Analysis Script
==================================
Recalculates all runs with CORRECT v2 metrics:
  - **Iteration-level slowdown**: s = (Tcomp + comm) / (Tcomp + Tcomm_solo)
    (NOT communication-only: comm/Tcomm_solo, which was the v1 bug)
  - **Attainment**: att = s / c_eval (≤1 = SLO met, >1 = SLO violated)
  - **P-attn**: Σ_premium max(0, s - 1) — premium attention needed
  - **S-cont**: Σ_standard max(0, s - 1) — standard contention suffered
  - **SLO rate**: fraction of epochs where att ≤ 1.0
  - **Max slowdown**: across all jobs (key for S2's bounded degradation story)

Key v2 design (from 实验C_场景参数表_v2.md):
  - c_policy=1.35 used by scheduler for π computation
  - c_eval=1.5 used for attainment reporting
  - Standard c=2.0 for both policy and eval
  - Analysis window: epochs 5 to (num_epochs - tail_epochs - 1)

Usage:
  python3 analyze_expC_v2.py [--data-dir DIR] [--scenarios FILE] [--output DIR]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_C_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = EXP_C_DIR / "data_v2"
DEFAULT_SCENARIOS = EXP_C_DIR / "scenarios" / "scenarios_v2.json"
DEFAULT_OUTPUT = SCRIPT_DIR


def load_ttarget(jid: int, run_dir: Path = None) -> dict:
    """Load T_target calibration for a job. Try run_dir first, then /tmp."""
    # Try from run_dir first (archived copy)
    if run_dir:
        f = run_dir / f"ttarget_{jid}.json"
        if f.exists():
            return json.load(open(f))
    # Fallback to /tmp
    f = Path(f"/tmp/expC_ttarget_{jid}.json")
    if f.exists():
        return json.load(open(f))
    return None


def parse_run_dir(run_dir: Path) -> tuple:
    """Parse run directory name: {regime}_{arm}_r{round}_{timestamp}"""
    name = run_dir.name
    # Try v2 format: S1_moderate_longliu_r1_20260730_120000
    # or S2_starvation_static_r3_20260730_120000
    import re
    m = re.match(r"^(S\d+_\w+)_(longliu|static|fair)_r(\d+)_\d{8}_\d{6}$", name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


def resolve_regime_config(scenarios: dict, regime_name: str) -> dict:
    """Resolve full regime config from scenarios_v2.json, including jobs with d_scale.
    
    v2.1: Uses payload_kb directly (not d_per_epoch_mb which was removed).
    """
    for sc_name, sc in scenarios["scenarios"].items():
        if regime_name in sc["regimes"]:
            regime = sc["regimes"][regime_name]
            d_scale = regime.get("d_scale", 1.0)
            jobs = []
            for bj in sc["base_jobs"]:
                payload_kb_scaled = int(bj["payload_kb"] * d_scale)
                jobs.append({
                    "job_id": bj["job_id"],
                    "label": bj["label"],
                    "tier": bj["tier"],
                    "c_policy": bj["c_policy"],
                    "c_eval": bj["c_eval"],
                    "payload_bytes": payload_kb_scaled * 1024,
                    "payload_kb": payload_kb_scaled,
                    "phi_target": bj["phi_target"],
                })
            return {
                "_desc": regime.get("_desc", ""),
                "scenario": sc_name,
                "d_scale": d_scale,
                "jobs": jobs,
            }
    return None


def compute_per_epoch_metrics(recs: list, ttarget_s: float, c_eval: float,
                              tcomp_per_iter_s: float) -> list:
    """Compute per-epoch metrics using ITERATION-LEVEL slowdown formula.

    v2 formula: s = (Tcomp + comm) / (Tcomp + Tcomm_solo)
    where Tcomp and Tcomm are per-iteration averages.

    This aligns with the paper's c_i definition (total iteration time,
    not just communication time).
    """
    by_epoch = defaultdict(list)
    for r in recs:
        by_epoch[r["epoch"]].append(r)

    tcomm_solo_s = ttarget_s  # per-iter solo comm time
    t_iter_solo = tcomp_per_iter_s + tcomm_solo_s  # solo iteration time

    result = []
    for ep in sorted(by_epoch.keys()):
        epoch_recs = by_epoch[ep]
        avg_comm_us = float(np.mean([r["comm_us"] for r in epoch_recs]))
        avg_comm_s = avg_comm_us / 1e6

        # Iteration-level slowdown
        t_iter_actual = tcomp_per_iter_s + avg_comm_s
        slowdown = t_iter_actual / t_iter_solo if t_iter_solo > 0 else float("nan")

        # Attainment = slowdown / c_eval
        attainment = slowdown / c_eval if c_eval > 0 else float("nan")

        # Communication-only slowdown (v1 formula, for comparison)
        comm_slowdown = avg_comm_s / tcomm_solo_s if tcomm_solo_s > 0 else float("nan")

        result.append({
            "epoch": ep,
            "avg_comm_us": avg_comm_us,
            "slowdown": slowdown,           # v2 iteration-level
            "attainment": attainment,        # v2 based on c_eval
            "comm_slowdown": comm_slowdown,  # v1 communication-only (legacy)
            "slo_met": attainment <= 1.0 if attainment == attainment else None,
        })
    return result


def main():
    ap = argparse.ArgumentParser(description="Experiment C v2 analysis")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    scenarios_file = Path(args.scenarios)
    output_dir = Path(args.output)

    with open(scenarios_file) as f:
        scenarios = json.load(f)

    exp_params = scenarios["experiment_params"]
    analysis_lo = exp_params.get("warmup_epochs", 5)
    analysis_hi = exp_params["num_epochs"] - exp_params.get("tail_epochs", 5) - 1

    run_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and "_r" in d.name)
    print(f"[analyze] Found {len(run_dirs)} run directories in {data_dir}")

    # ---- Per-round data collection ----
    all_rows = []
    per_epoch_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_dir in run_dirs:
        parsed = parse_run_dir(run_dir)
        if not parsed:
            continue
        regime_name, arm, rnd = parsed

        regime_cfg = resolve_regime_config(scenarios, regime_name)
        if not regime_cfg:
            print(f"[WARN] regime '{regime_name}' not found in scenarios")
            continue

        for jc in regime_cfg["jobs"]:
            jid = jc["job_id"]
            tier = jc["tier"]
            c_eval = jc["c_eval"]

            # Load T_target (contains tcomp_per_iter_ms from calibration)
            ttarget = load_ttarget(jid, run_dir)
            if ttarget is None:
                continue
            ttarget_s = ttarget["target_comm_time_ms"] / 1000.0
            # tcomp from calibration file (adjusted for φ target)
            tcomp_per_iter_ms = ttarget.get("tcomp_per_iter_ms", ttarget.get("sleep_us_adjusted", 5000) / 1000.0)
            tcomp_per_iter_s = tcomp_per_iter_ms / 1000.0

            # Stats path
            stats_path = run_dir / f"job{jid}_stats.csv"
            if not stats_path.exists():
                continue
            recs = []
            with open(stats_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    recs.append({
                        "epoch": int(row["epoch"]),
                        "comm_us": float(row["comm_us"]),
                        "dscp": int(row["dscp"]),
                    })

            per_ep = compute_per_epoch_metrics(recs, ttarget_s, c_eval, tcomp_per_iter_s)
            window = [e for e in per_ep if analysis_lo <= e["epoch"] <= analysis_hi]

            if not window:
                continue

            sd_mean = float(np.mean([e["slowdown"] for e in window]))
            sd_std = float(np.std([e["slowdown"] for e in window])) if len(window) > 1 else 0
            att_mean = float(np.mean([e["attainment"] for e in window]))
            slo_met_frac = float(np.mean([1 if e["slo_met"] else 0 for e in window
                                           if e["slo_met"] is not None]))

            # Legacy comm-only slowdown for comparison
            comm_sd_mean = float(np.mean([e["comm_slowdown"] for e in window]))

            all_rows.append({
                "regime": regime_name,
                "scenario": regime_cfg["scenario"],
                "arm": arm,
                "round": rnd,
                "job_id": jid,
                "label": jc["label"],
                "tier": tier,
                "c_policy": jc["c_policy"],
                "c_eval": c_eval,
                "slowdown_mean": sd_mean,
                "slowdown_std": sd_std,
                "attainment_mean": att_mean,
                "slo_met_frac": slo_met_frac,
                "comm_slowdown_mean": comm_sd_mean,  # v1 legacy
                "slowdown_lt1": sd_mean < 1.0,
            })

            for e in window:
                per_epoch_data[regime_name][arm][jid].append({
                    "epoch": e["epoch"],
                    "slowdown": e["slowdown"],
                    "attainment": e["attainment"],
                    "comm_slowdown": e["comm_slowdown"],
                    "round": rnd,
                })

    if not all_rows:
        print("[ERR] No data rows found. Check data directory and scenario config.")
        return

    # ---- Aggregate by (regime, arm) ----
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_rows:
        agg[r["regime"]][r["arm"]][r["job_id"]].append(r)

    # ---- Compute P-attn, S-cont, SLO rate per (regime, arm, round) ----
    regime_arm_round = defaultdict(list)
    for r in all_rows:
        regime_arm_round[(r["regime"], r["arm"], r["round"])].append(r)

    metric_rows = []
    for (regime, arm, rnd), rows in sorted(regime_arm_round.items()):
        p_attn = sum(max(0, r["slowdown_mean"] - 1.0) for r in rows
                     if r["tier"] == "premium")
        s_cont = sum(max(0, r["slowdown_mean"] - 1.0) for r in rows
                     if r["tier"] == "standard")
        premium_slo_met = [r["slo_met_frac"] for r in rows if r["tier"] == "premium"]
        premium_slo_rate = float(np.mean(premium_slo_met)) if premium_slo_met else 0.0
        standard_slo_met = [r["slo_met_frac"] for r in rows if r["tier"] == "standard"]
        standard_slo_rate = float(np.mean(standard_slo_met)) if standard_slo_met else 0.0

        n_slow_lt1 = sum(1 for r in rows if r["slowdown_lt1"])
        max_slowdown = max(r["slowdown_mean"] for r in rows)

        metric_rows.append({
            "regime": regime,
            "arm": arm,
            "round": rnd,
            "p_attn": p_attn,
            "s_cont": s_cont,
            "premium_slo_rate": premium_slo_rate,
            "standard_slo_rate": standard_slo_rate,
            "n_jobs_slow_lt1": n_slow_lt1,
            "max_slowdown": max_slowdown,
            "n_jobs": len(rows),
        })

    # Aggregate metrics across rounds
    metric_agg = defaultdict(lambda: defaultdict(list))
    for r in metric_rows:
        metric_agg[r["regime"]][r["arm"]].append(r)

    # ---- Per-round CSV ----
    pr_csv = output_dir / "expC_v2_per_round.csv"
    with open(pr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "scenario", "arm", "round", "job_id", "label", "tier",
                     "c_policy", "c_eval", "slowdown_mean", "slowdown_std",
                     "attainment_mean", "slo_met_frac", "comm_slowdown_v1",
                     "slowdown_lt1"])
        for r in all_rows:
            w.writerow([r["regime"], r["scenario"], r["arm"], r["round"],
                        r["job_id"], r["label"], r["tier"],
                        r["c_policy"], r["c_eval"],
                        f"{r['slowdown_mean']:.4f}", f"{r['slowdown_std']:.4f}",
                        f"{r['attainment_mean']:.4f}", f"{r['slo_met_frac']:.3f}",
                        f"{r['comm_slowdown_mean']:.4f}",
                        "Y" if r["slowdown_lt1"] else "N"])
    print(f"[analyze] Wrote {pr_csv}")

    # ---- Markdown report ----
    md_path = output_dir / "expC_v2_analysis.md"
    with open(md_path, "w") as f:
        f.write("# Experiment C v2: Iteration-Level Slowdown Analysis\n\n")
        f.write(f"> Generated: {datetime.now().isoformat()}\n")
        f.write(f"> Data dir: {data_dir}\n")
        f.write(f"> Analysis window: epochs {analysis_lo}–{analysis_hi}\n")
        f.write(f"> Total runs: {len(run_dirs)}\n\n")
        f.write("## Metric Definitions (v2)\n\n")
        f.write("- **Slowdown (iter-level)**: s = (Tcomp + comm) / (Tcomp + Tcomm_solo)\n")
        f.write("  - This aligns with the paper's c_i definition (total iteration time)\n")
        f.write("  - v1 used comm-only: comm / Tcomm_solo (incorrect per v2 spec)\n")
        f.write("- **Attainment**: att = s / c_eval (≤1 = SLO met, >1 = SLO violated)\n")
        f.write("- **c_policy / c_eval split**: scheduler uses c_policy=1.35 for π; "
                "attainment uses c_eval=1.5\n")
        f.write("- **P-attn** = Σ_premium max(0, s-1) — premium attention needed (lower=better)\n")
        f.write("- **S-cont** = Σ_standard max(0, s-1) — standard contention (lower=better)\n")
        f.write("- **Max slowdown** = max across all jobs (key for S2 bounded degradation)\n\n")

        # ---- Per-regime tables ----
        for regime in sorted(agg.keys()):
            regime_cfg = resolve_regime_config(scenarios, regime)
            if not regime_cfg:
                continue
            scenario = regime_cfg["scenario"]
            d_scale = regime_cfg["d_scale"]

            f.write(f"## Regime: {regime}\n\n")
            f.write(f"> Scenario: {scenario} | D scale: ×{d_scale}\n")
            f.write(f"> {regime_cfg.get('_desc', '')}\n\n")

            arms = sorted(agg[regime].keys())
            job_ids = sorted(set(jid for arm in arms for jid in agg[regime][arm]))

            # Table: per-job slowdown + attainment
            f.write("### Per-job metrics (mean ± std across rounds)\n\n")
            f.write("| Job | Label | Tier | c_policy | c_eval | ")
            for arm in arms:
                f.write(f"{arm} SD | {arm} Att | ")
            f.write("\n")
            f.write("|-----|-------|------|----------|--------|")
            for _ in arms:
                f.write("------|------|")
            f.write("\n")

            for jid in job_ids:
                jc = next((x for x in regime_cfg["jobs"] if x["job_id"] == jid), None)
                if not jc:
                    continue
                f.write(f"| {jid} | {jc['label']} | {jc['tier']} | "
                        f"{jc['c_policy']} | {jc['c_eval']} | ")
                for arm in arms:
                    rows = agg[regime][arm].get(jid, [])
                    if not rows:
                        f.write("N/A | N/A | ")
                        continue
                    sds = [r["slowdown_mean"] for r in rows]
                    atts = [r["attainment_mean"] for r in rows]
                    sd_m = float(np.mean(sds))
                    sd_s = float(np.std(sds)) if len(sds) > 1 else 0
                    att_m = float(np.mean(atts))
                    att_s = float(np.std(atts)) if len(atts) > 1 else 0
                    f.write(f"{sd_m:.3f}±{sd_s:.3f} | {att_m:.3f}±{att_s:.3f} | ")
                f.write("\n")

            # Aggregate metrics
            f.write("\n### Aggregate metrics (mean ± std across rounds)\n\n")
            f.write("| Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Max SD | N slow<1 |\n")
            f.write("|-----|--------|--------|-------------|---------|--------|----------|\n")
            for arm in arms:
                mrows = metric_agg[regime][arm]
                pa = [r["p_attn"] for r in mrows]
                sc = [r["s_cont"] for r in mrows]
                psr = [r["premium_slo_rate"] for r in mrows]
                ssr = [r["standard_slo_rate"] for r in mrows]
                msd = [r["max_slowdown"] for r in mrows]
                nlt = [r["n_jobs_slow_lt1"] for r in mrows]
                f.write(f"| {arm} | "
                        f"{np.mean(pa):.3f}±{np.std(pa):.3f} | "
                        f"{np.mean(sc):.3f}±{np.std(sc):.3f} | "
                        f"{np.mean(psr):.3f} | "
                        f"{np.mean(ssr):.3f} | "
                        f"{np.mean(msd):.2f} | "
                        f"{np.mean(nlt):.1f}/{np.mean([r['n_jobs'] for r in mrows]):.0f} |\n")
            f.write("\n")

        # ---- Cross-regime comparison ----
        f.write("## Cross-regime comparison\n\n")

        # P-attn
        f.write("### P-attn (lower = better, premium jobs closer to SLO)\n\n")
        f.write("| Regime | LongLiu | Static | Fair | LL vs Static | LL vs Fair |\n")
        f.write("|--------|---------|--------|------|--------------|------------|\n")
        for regime in sorted(agg.keys()):
            row = f"| {regime} "
            vals = {}
            for arm in ["longliu", "static", "fair"]:
                if arm in metric_agg[regime]:
                    pa = [r["p_attn"] for r in metric_agg[regime][arm]]
                    m = float(np.mean(pa))
                    vals[arm] = m
                    row += f"| {m:.3f} "
                else:
                    row += "| N/A "
            if "longliu" in vals and "static" in vals:
                diff = (vals["static"] - vals["longliu"]) / vals["static"] * 100
                row += f"| LL {'优' if diff > 0 else '劣'} {abs(diff):.0f}% "
            else:
                row += "| — "
            if "longliu" in vals and "fair" in vals:
                diff = (vals["fair"] - vals["longliu"]) / vals["fair"] * 100
                row += f"| LL {'优' if diff > 0 else '劣'} {abs(diff):.0f}% "
            else:
                row += "| — "
            row += "|\n"
            f.write(row)

        # S-cont
        f.write("\n### S-cont (lower = better, standard jobs less contended)\n\n")
        f.write("| Regime | LongLiu | Static | Fair | LL vs Static | LL vs Fair |\n")
        f.write("|--------|---------|--------|------|--------------|------------|\n")
        for regime in sorted(agg.keys()):
            row = f"| {regime} "
            vals = {}
            for arm in ["longliu", "static", "fair"]:
                if arm in metric_agg[regime]:
                    sc = [r["s_cont"] for r in metric_agg[regime][arm]]
                    m = float(np.mean(sc))
                    vals[arm] = m
                    row += f"| {m:.3f} "
                else:
                    row += "| N/A "
            if "longliu" in vals and "static" in vals:
                d = (vals["static"] - vals["longliu"])
                denom = vals["static"] if vals["static"] != 0 else 1
                diff = d / abs(denom) * 100
                row += f"| LL {'优' if diff > 0 else '劣'} {abs(diff):.0f}% "
            else:
                row += "| — "
            if "longliu" in vals and "fair" in vals:
                d = (vals["fair"] - vals["longliu"])
                denom = vals["fair"] if vals["fair"] != 0 else 1
                diff = d / abs(denom) * 100
                row += f"| LL {'优' if diff > 0 else '劣'} {abs(diff):.0f}% "
            else:
                row += "| — "
            row += "|\n"
            f.write(row)

        # Max slowdown (S2 key metric)
        f.write("\n### Max slowdown across jobs (lower = better, S2 key metric)\n\n")
        f.write("| Regime | LongLiu | Static | Fair | Narrative |\n")
        f.write("|--------|---------|--------|------|------------|\n")
        for regime in sorted(agg.keys()):
            row = f"| {regime} "
            vals = {}
            for arm in ["longliu", "static", "fair"]:
                if arm in metric_agg[regime]:
                    msd = [r["max_slowdown"] for r in metric_agg[regime][arm]]
                    m = float(np.mean(msd))
                    vals[arm] = m
                    row += f"| {m:.2f} "
                else:
                    row += "| N/A "
            # Narrative
            if "longliu" in vals and "static" in vals and "fair" in vals:
                if vals["longliu"] <= min(vals["static"], vals["fair"]):
                    row += "| LL 最优 (有界退化) "
                elif vals["longliu"] < vals["static"]:
                    row += "| LL < Static "
                else:
                    row += "| — "
            else:
                row += "| — "
            row += "|\n"
            f.write(row)

        # ---- v1 vs v2 slowdown comparison ----
        f.write("\n## v1 (comm-only) vs v2 (iter-level) Slowdown Comparison\n\n")
        f.write("| Regime | Arm | Job | Tier | v1 SD | v2 SD | v2/v1 ratio |\n")
        f.write("|--------|-----|-----|------|-------|-------|-------------|\n")
        for regime in sorted(agg.keys()):
            for arm in sorted(agg[regime].keys()):
                for jid in sorted(agg[regime][arm].keys()):
                    rows = agg[regime][arm][jid]
                    v1_sd = float(np.mean([r["comm_slowdown_mean"] for r in rows]))
                    v2_sd = float(np.mean([r["slowdown_mean"] for r in rows]))
                    tier = rows[0]["tier"]
                    label = rows[0]["label"]
                    ratio = v2_sd / v1_sd if v1_sd > 0 else float("nan")
                    f.write(f"| {regime} | {arm} | {jid}({label}) | {tier} | "
                            f"{v1_sd:.3f} | {v2_sd:.3f} | {ratio:.2f} |\n")

        # ---- Iron rule verification ----
        f.write("\n## Iron Rule Verification\n\n")
        f.write("| Regime | Rule 1: Real contention? | Rule 2: Fair must fail? | "
                "Rule 3: LL passes? |\n")
        f.write("|--------|--------------------------|--------------------------|"
                "----------------------|\n")
        for regime in sorted(agg.keys()):
            regime_cfg_r = resolve_regime_config(scenarios, regime)
            if not regime_cfg_r:
                continue

            # Rule 1: Check if any job has slowdown > 1 (real contention)
            all_sds = []
            for arm in agg[regime]:
                for jid in agg[regime][arm]:
                    rows = agg[regime][arm][jid]
                    all_sds.extend([r["slowdown_mean"] for r in rows])
            max_sd = max(all_sds) if all_sds else 0
            rule1 = "✓" if max_sd > 1.05 else "✗ (slowdown≤1)"

            # Rule 2: Fair arm premium b^att ≥ 1.5×(B/N)?
            # Proxy: fair arm premium slowdown > 1
            fair_prem_sds = []
            if "fair" in agg[regime]:
                for jid in agg[regime]["fair"]:
                    rows = agg[regime]["fair"][jid]
                    if rows[0]["tier"] == "premium":
                        fair_prem_sds.extend([r["slowdown_mean"] for r in rows])
            fair_prem_mean = float(np.mean(fair_prem_sds)) if fair_prem_sds else 0
            c_eval_threshold = 1.5  # premium c_eval
            rule2 = f"✓ (fair P SD={fair_prem_mean:.2f})" if fair_prem_mean > c_eval_threshold else "✗"

            # Rule 3: LongLiu premium attainment ≤ c_eval?
            ll_prem_atts = []
            if "longliu" in agg[regime]:
                for jid in agg[regime]["longliu"]:
                    rows = agg[regime]["longliu"][jid]
                    if rows[0]["tier"] == "premium":
                        ll_prem_atts.extend([r["attainment_mean"] for r in rows])
            ll_prem_mean = float(np.mean(ll_prem_atts)) if ll_prem_atts else 0
            rule3 = f"✓ (LL P att={ll_prem_mean:.2f})" if ll_prem_mean <= 1.0 else f"✗ (att={ll_prem_mean:.2f})"

            f.write(f"| {regime} | {rule1} | {rule2} | {rule3} |\n")

    print(f"[analyze] Wrote {md_path}")
    print(f"\n[analyze] Summary:")
    print(f"  Runs analyzed: {len(run_dirs)}")
    print(f"  Data rows: {len(all_rows)}")
    print(f"  Regimes: {sorted(set(r['regime'] for r in all_rows))}")


if __name__ == "__main__":
    main()
