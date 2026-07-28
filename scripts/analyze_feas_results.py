#!/usr/bin/env python3
"""Analyze feasibility boundary results and produce a comprehensive Markdown report."""

import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean as stat_mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load points ────────────────────────────────────────────────
LOAD_POINTS = {
    "ci15": {
        "label": "1.02×, ci=1.5/2.5, spine=800G",
        "path": PROJECT_ROOT / "outputs/feas_boundary_v1_ci15/results.json",
    },
    "final": {
        "label": "1.21×, ci=1.3/2.0, spine=800G",
        "path": PROJECT_ROOT / "outputs/feas_boundary_v1_final/results.json",
    },
    "v1": {
        "label": "1.27×, ci=1.2/2.0, spine=800G",
        "path": PROJECT_ROOT / "outputs/feas_boundary_v1/results.json",
    },
    "630g": {
        "label": "1.54×, ci=1.3/2.0, spine=630G",
        "path": PROJECT_ROOT / "outputs/feas_boundary_v1_630g/results.json",
    },
}

# ── Tier & contest classification ──────────────────────────────
PREMIUM_CI = {1.2, 1.3, 1.5}
STANDARD_CI = {2.0, 2.5}

CONTESTED_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}  # battlefield


def is_premium(ci):
    return ci in PREMIUM_CI


def is_standard(ci):
    return ci in STANDARD_CI


def is_contested(model):
    return model in CONTESTED_MODELS


# ── Job column mapping ─────────────────────────────────────────
# P1: LLaMA-2-13B premium  | P2: LLaMA-2-7B premium  | P3: BERT-* premium
# S1: LLaMA-2-13B standard | S2: T5-11B standard     | S3: BERT-* standard
# S4: ViT-* standard

def classify_job(model, ci):
    """Return column key like 'P1', 'P2', etc., or None if unmatched."""
    if is_premium(ci):
        if model == "LLaMA-2-13B":
            return "P1"
        if model == "LLaMA-2-7B":
            return "P2"
        if model.startswith("BERT"):
            return "P3"
    elif is_standard(ci):
        if model == "LLaMA-2-13B":
            return "S1"
        if model == "T5-11B-fp16":
            return "S2"
        if model.startswith("BERT"):
            return "S3"
        if model.startswith("ViT"):
            return "S4"
    return None


COLUMN_ORDER = ["P1", "P2", "P3", "S1", "S2", "S3", "S4"]
COLUMN_LABELS = {
    "P1": "P1(13B-p)",
    "P2": "P2(7B-p)",
    "P3": "P3(BERT-p)",
    "S1": "S1(13B-s)",
    "S2": "S2(T5-s)",
    "S3": "S3(BERT-s)",
    "S4": "S4(ViT-s)",
}


def fmt_val(values):
    """Format mean ± half_range.  Single value: no range.  Empty: 'N/A'."""
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    m = stat_mean(values)
    rng = max(values) - min(values)
    return f"{m:.3f}±{rng / 2:.3f}"


def fmt_pct(values):
    """Format a percentage."""
    if not values:
        return "N/A"
    return f"{stat_mean(values) * 100:.1f}%"


# ── Main ───────────────────────────────────────────────────────

def load_data():
    """Load all four results files."""
    data = {}
    for key, lp in LOAD_POINTS.items():
        with open(lp["path"]) as f:
            data[key] = json.load(f)
    return data


def build_job_table(data, lp_key):
    """Build per-job SAS table for a single load point.

    Returns: dict[policy_name][column] = list of sas_eval values across seeds
    """
    lp_data = data[lp_key]
    table = defaultdict(lambda: defaultdict(list))

    for policy_name, seeds in lp_data.items():
        for seed_data in seeds:
            for job in seed_data["per_job"]:
                col = classify_job(job["model"], job["ci"])
                if col is None:
                    continue
                table[policy_name][col].append(job["sas_eval"])

    return table


def compute_battlefield_metrics(data, lp_key):
    """Compute battlefield metrics for a single load point.

    Returns: dict[policy_name] = {
        'prem_contested_mean': float,
        'prem_contested_capped': float,
        'prem_contested_attainment': float,
        'std_contested_mean': float,
        'std_contested_capped': float,
        'filler_prem_mean': float,
        'filler_std_mean': float,
        'prem_contested_per_seed': [float, ...],
        'prem_contested_attainment_flags': [bool, ...],
    }
    """
    lp_data = data[lp_key]
    metrics = {}

    for policy_name, seeds in lp_data.items():
        # Per-seed accumulators
        prem_contested_per_seed = []  # average of P1+P2 sas per seed
        std_contested_per_seed = []   # average of S1+S2 sas per seed
        filler_prem_per_seed = []     # P3 sas per seed
        filler_std_per_seed = []      # S3+S4 average per seed
        prem_both_attained = []       # bool: P1 & P2 both attained

        p1_vals = []
        p2_vals = []
        s1_vals = []
        s2_vals = []

        for seed_data in seeds:
            seed_jobs = {}
            for job in seed_data["per_job"]:
                col = classify_job(job["model"], job["ci"])
                if col is not None:
                    seed_jobs[col] = job

            # Premium contested: P1 + P2
            p1 = seed_jobs.get("P1", None)
            p2 = seed_jobs.get("P2", None)
            if p1 is not None and p2 is not None:
                prem_contested = (p1["sas_eval"] + p2["sas_eval"]) / 2
                prem_contested_per_seed.append(prem_contested)
                p1_vals.append(p1["sas_eval"])
                p2_vals.append(p2["sas_eval"])
                prem_both_attained.append(p1["attained"] and p2["attained"])

            # Standard contested: S1 + S2
            s1 = seed_jobs.get("S1", None)
            s2 = seed_jobs.get("S2", None)
            if s1 is not None and s2 is not None:
                std_contested = (s1["sas_eval"] + s2["sas_eval"]) / 2
                std_contested_per_seed.append(std_contested)
                s1_vals.append(s1["sas_eval"])
                s2_vals.append(s2["sas_eval"])

            # Filler premium: P3
            p3 = seed_jobs.get("P3", None)
            if p3 is not None:
                filler_prem_per_seed.append(p3["sas_eval"])

            # Filler standard: S3 + S4
            s3 = seed_jobs.get("S3", None)
            s4 = seed_jobs.get("S4", None)
            if s3 is not None and s4 is not None:
                filler_std_per_seed.append((s3["sas_eval"] + s4["sas_eval"]) / 2)

        # Compute aggregate metrics
        m = {}
        m["prem_contested_mean"] = stat_mean(prem_contested_per_seed) if prem_contested_per_seed else None
        m["prem_contested_capped"] = min(stat_mean(p1_vals), stat_mean(p2_vals), 1.0) if p1_vals else None
        m["prem_contested_attainment"] = stat_mean([int(f) for f in prem_both_attained]) if prem_both_attained else None
        m["std_contested_mean"] = stat_mean(std_contested_per_seed) if std_contested_per_seed else None
        m["std_contested_capped"] = min(stat_mean(s1_vals), stat_mean(s2_vals), 1.0) if s1_vals else None
        m["filler_prem_mean"] = stat_mean(filler_prem_per_seed) if filler_prem_per_seed else None
        m["filler_std_mean"] = stat_mean(filler_std_per_seed) if filler_std_per_seed else None
        m["prem_contested_per_seed"] = prem_contested_per_seed
        m["prem_contested_attainment_flags"] = prem_both_attained
        m["p1_vals"] = p1_vals
        m["p2_vals"] = p2_vals
        m["p3_vals"] = filler_prem_per_seed
        m["s1_vals"] = s1_vals
        m["s2_vals"] = s2_vals
        m["s3_vals"] = None  # will fill below
        m["s4_vals"] = None

        # Fill standard filler raw vals for Section D analysis
        for seed_data in seeds:
            seed_jobs = {}
            for job in seed_data["per_job"]:
                col = classify_job(job["model"], job["ci"])
                if col is not None:
                    seed_jobs[col] = job

        # Actually let me redo this more cleanly for s3/s4
        s3_raw = []
        s4_raw = []
        for seed_data in seeds:
            seed_jobs = {}
            for job in seed_data["per_job"]:
                col = classify_job(job["model"], job["ci"])
                if col is not None:
                    seed_jobs[col] = job
            s3 = seed_jobs.get("S3", None)
            s4 = seed_jobs.get("S4", None)
            if s3 is not None:
                s3_raw.append(s3["sas_eval"])
            if s4 is not None:
                s4_raw.append(s4["sas_eval"])
        m["s3_vals"] = s3_raw
        m["s4_vals"] = s4_raw

        metrics[policy_name] = m

    return metrics


def main():
    data = load_data()

    # Collect all policies across all load points
    all_policies = set()
    for lp_data in data.values():
        all_policies.update(lp_data.keys())
    # Sort: Fair, CRUX, D1, LongLiu-SP (or however they appear)
    POLICY_ORDER = ["Fair", "CRUX", "D1", "LongLiu-SP"]
    policies = [p for p in POLICY_ORDER if p in all_policies]

    lines = []
    def emit(s=""):
        lines.append(s)

    emit("# Feasibility Boundary Analysis")
    emit()
    emit(f"**Analysis date**: auto-generated")
    emit(f"**Data source**: 4 load points × {len(policies)} policies × 3 seeds (CRN)")
    emit()
    emit("---")
    emit()

    # ══════════════════════════════════════════════════════════════
    # Section A: Per-job SAS tables
    # ══════════════════════════════════════════════════════════════
    emit("## Section A: Per-job SAS (all load points)")
    emit()

    for lp_key, lp_info in LOAD_POINTS.items():
        emit(f"### Load point: {lp_info['label']}")
        emit()

        table = build_job_table(data, lp_key)

        # Header
        header = "| Policy | " + " | ".join(COLUMN_LABELS[c] for c in COLUMN_ORDER) + " |"
        emit(header)
        sep = "|--------|" + "|".join("----------" for _ in COLUMN_ORDER) + "|"
        emit(sep)

        for policy in policies:
            row = f"| {policy} "
            for col in COLUMN_ORDER:
                vals = table[policy].get(col, [])
                row += f"| {fmt_val(vals)} "
            row += "|"
            emit(row)

        emit()

    # ══════════════════════════════════════════════════════════════
    # Section B: Battlefield metrics
    # ══════════════════════════════════════════════════════════════
    emit("---")
    emit()
    emit("## Section B: Battlefield Metrics")
    emit()

    for lp_key, lp_info in LOAD_POINTS.items():
        emit(f"### Load point: {lp_info['label']}")
        emit()
        metrics = compute_battlefield_metrics(data, lp_key)

        # Table header
        header_cols = [
            "Policy",
            "Prem Contest Mean",
            "Prem Contest Capped",
            "Prem Contest Attain%",
            "Std Contest Mean",
            "Std Contest Capped",
            "Filler Prem (P3) Mean",
            "Filler Std (S3+S4) Mean",
        ]
        emit("| " + " | ".join(header_cols) + " |")
        emit("|" + "|".join("----------" for _ in header_cols) + "|")

        for policy in policies:
            m = metrics[policy]
            row = [
                policy,
                f"{m['prem_contested_mean']:.3f}" if m['prem_contested_mean'] is not None else "N/A",
                f"{m['prem_contested_capped']:.3f}" if m['prem_contested_capped'] is not None else "N/A",
                f"{m['prem_contested_attainment']*100:.1f}%" if m['prem_contested_attainment'] is not None else "N/A",
                f"{m['std_contested_mean']:.3f}" if m['std_contested_mean'] is not None else "N/A",
                f"{m['std_contested_capped']:.3f}" if m['std_contested_capped'] is not None else "N/A",
                f"{m['filler_prem_mean']:.3f}" if m['filler_prem_mean'] is not None else "N/A",
                f"{m['filler_std_mean']:.3f}" if m['filler_std_mean'] is not None else "N/A",
            ]
            emit("| " + " | ".join(row) + " |")

        emit()

    # ══════════════════════════════════════════════════════════════
    # Section C: Paired differences (D1-Fair, D1-CRUX, CRUX-Fair)
    # ══════════════════════════════════════════════════════════════
    emit("---")
    emit()
    emit("## Section C: Paired Differences (CRN, 3 seeds, 功效不足)")
    emit()

    DIFF_PAIRS = [
        ("D1", "Fair"),
        ("D1", "CRUX"),
        ("CRUX", "Fair"),
    ]

    for lp_key, lp_info in LOAD_POINTS.items():
        emit(f"### Load point: {lp_info['label']}")
        emit()
        metrics = compute_battlefield_metrics(data, lp_key)

        for pol_a, pol_b in DIFF_PAIRS:
            if pol_a not in metrics or pol_b not in metrics:
                emit(f"**{pol_a} − {pol_b}**: N/A (policy missing)")
                emit()
                continue

            a_seeds = metrics[pol_a]["prem_contested_per_seed"]
            b_seeds = metrics[pol_b]["prem_contested_per_seed"]

            # They should be aligned by seed index (CRN), same length
            if len(a_seeds) != len(b_seeds):
                emit(f"**{pol_a} − {pol_b}**: seed count mismatch ({len(a_seeds)} vs {len(b_seeds)})")
                emit()
                continue

            diffs = [a - b for a, b in zip(a_seeds, b_seeds)]
            diff_mean = stat_mean(diffs)
            diff_half_range = (max(diffs) - min(diffs)) / 2 if len(diffs) > 1 else 0.0

            emit(f"#### {pol_a} − {pol_b} (premium contested mean difference)")
            emit()
            emit(f"Per-seed: " + ", ".join(f"seed{i}=Δ{d:.3f}" for i, d in enumerate(diffs)))
            emit(f"Mean ± half-range: **{diff_mean:.3f} ± {diff_half_range:.3f}**")
            emit()

    # ══════════════════════════════════════════════════════════════
    # Section D: Key Questions
    # ══════════════════════════════════════════════════════════════
    emit("---")
    emit()
    emit("## Section D: Key Questions")
    emit()

    for lp_key, lp_info in LOAD_POINTS.items():
        emit(f"### Load point: {lp_info['label']}")
        emit()
        metrics = compute_battlefield_metrics(data, lp_key)

        # ── D(a): Does CRUX's premium mean advantage come entirely from P3 overshoot? ──
        emit("#### D(a): CRUX premium mean advantage — P3 (BERT) overshoot?")
        emit()

        # Compare CRUX vs Fair: premium_mean from the JSON (includes P1+P2+P3)
        # vs contested-only (P1+P2)
        fair = metrics.get("Fair", {})
        crux = metrics.get("CRUX", {})
        d1 = metrics.get("D1", {})

        if fair and crux:
            # Premium mean from JSON (full, including BERT)
            lp_data = data[lp_key]
            fair_prem_full = stat_mean([s["premium_mean"] for s in lp_data["Fair"]])
            crux_prem_full = stat_mean([s["premium_mean"] for s in lp_data["CRUX"]])

            fair_prem_contested = fair.get("prem_contested_mean")
            crux_prem_contested = crux.get("prem_contested_mean")
            fair_p3_mean = fair.get("filler_prem_mean")
            crux_p3_mean = crux.get("filler_prem_mean")

            emit(f"| Metric | Fair | CRUX | Δ (CRUX−Fair) |")
            emit(f"|--------|------|------|---------------|")
            emit(f"| premium_mean (full, from JSON) | {fair_prem_full:.3f} | {crux_prem_full:.3f} | {crux_prem_full - fair_prem_full:+.3f} |")
            emit(f"| contested only (P1+P2)/2 | {fair_prem_contested:.3f} | {crux_prem_contested:.3f} | {crux_prem_contested - fair_prem_contested:+.3f} |")
            emit(f"| filler P3 (BERT-p) | {fair_p3_mean:.3f} | {crux_p3_mean:.3f} | {crux_p3_mean - fair_p3_mean:+.3f} |")
            emit()

            # Analysis
            full_delta = crux_prem_full - fair_prem_full
            contested_delta = crux_prem_contested - fair_prem_contested
            p3_delta = crux_p3_mean - fair_p3_mean

            if abs(full_delta) < 0.001:
                emit("CRUX and Fair have essentially identical premium_mean. No advantage to decompose.")
            else:
                pct_contested = abs(contested_delta / full_delta) * 100 if abs(full_delta) > 1e-6 else 0
                pct_p3 = abs(p3_delta / full_delta) * 100 if abs(full_delta) > 1e-6 else 0
                emit(f"**Interpretation**: Of CRUX's {full_delta:+.3f} premium_mean advantage over Fair, "
                     f"{contested_delta:+.3f} ({pct_contested:.0f}%) comes from contested jobs (P1+P2), "
                     f"while P3 (BERT) contributes {p3_delta:+.3f} ({pct_p3:.0f}%).")
                if p3_delta > contested_delta:
                    emit(f"**→ CRUX's premium advantage is disproportionately driven by P3 (BERT) overshoot.**")
                else:
                    emit(f"**→ CRUX's premium advantage is primarily from contested (battlefield) jobs, not just BERT overshoot.**")
                emit()

        # ── D(b): D1 takes bandwidth from whom, gives to whom vs Fair? ──
        emit("#### D(b): D1 vs Fair — bandwidth reallocation")
        emit()

        if d1 and fair:
            # Compare S1 (13B-s), S2 (T5-s), P1 (13B-p), P2 (7B-p)
            for col_name, col_label in [
                ("P1", "P1(13B-p)"),
                ("P2", "P2(7B-p)"),
                ("S1", "S1(13B-s)"),
                ("S2", "S2(T5-s)"),
            ]:
                # Get per-seed values
                table = build_job_table(data, lp_key)
                fair_vals = table.get("Fair", {}).get(col_name, [])
                d1_vals = table.get("D1", {}).get(col_name, [])

                if fair_vals and d1_vals and len(fair_vals) == len(d1_vals):
                    fair_m = stat_mean(fair_vals)
                    d1_m = stat_mean(d1_vals)
                    delta = d1_m - fair_m
                    emit(f"| {col_label} | {fair_m:.3f} | {d1_m:.3f} | {delta:+.3f} |")

            emit()
            emit("**Where D1 takes from / gives to:**")
            emit()

            # P1, P2 should improve (D1 gives to premium), S1, S2 should degrade (D1 takes from standard)
            table = build_job_table(data, lp_key)
            d1_p1 = stat_mean(table.get("D1", {}).get("P1", [0]))
            fair_p1 = stat_mean(table.get("Fair", {}).get("P1", [0]))
            d1_p2 = stat_mean(table.get("D1", {}).get("P2", [0]))
            fair_p2 = stat_mean(table.get("Fair", {}).get("P2", [0]))
            d1_s1 = stat_mean(table.get("D1", {}).get("S1", [0]))
            fair_s1 = stat_mean(table.get("Fair", {}).get("S1", [0]))
            d1_s2 = stat_mean(table.get("D1", {}).get("S2", [0]))
            fair_s2 = stat_mean(table.get("Fair", {}).get("S2", [0]))

            emit(f"- **Premium contested (P1+P2)**: D1 = {(d1_p1 + d1_p2)/2:.3f}, Fair = {(fair_p1 + fair_p2)/2:.3f}, "
                 f"Δ = {(d1_p1 + d1_p2)/2 - (fair_p1 + fair_p2)/2:+.3f}")
            emit(f"- **Standard contested (S1+S2)**: D1 = {(d1_s1 + d1_s2)/2:.3f}, Fair = {(fair_s1 + fair_s2)/2:.3f}, "
                 f"Δ = {(d1_s1 + d1_s2)/2 - (fair_s1 + fair_s2)/2:+.3f}")

            # Per-job breakdown
            for col_name, col_label in [
                ("S1", "S1(13B-s)"),
                ("S2", "S2(T5-s)"),
                ("P1", "P1(13B-p)"),
                ("P2", "P2(7B-p)"),
            ]:
                d1_vals = table.get("D1", {}).get(col_name, [])
                fair_vals = table.get("Fair", {}).get(col_name, [])
                if d1_vals and fair_vals:
                    emit(f"  - {col_label}: Fair={stat_mean(fair_vals):.3f} → D1={stat_mean(d1_vals):.3f} "
                         f"(Δ={stat_mean(d1_vals) - stat_mean(fair_vals):+.3f})")
            emit()

    # ── Output ───────────────────────────────────────────────────
    output = "\n".join(lines)
    # Save to file
    out_path = PROJECT_ROOT / "outputs/feas_analysis.md"
    with open(out_path, "w") as f:
        f.write(output + "\n")

    print(output)
    print(f"\n--- Saved to {out_path} ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
