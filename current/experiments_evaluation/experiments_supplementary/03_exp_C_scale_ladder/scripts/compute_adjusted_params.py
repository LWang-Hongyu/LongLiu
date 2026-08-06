#!/usr/bin/env python3
"""
Compute adjusted sleep_us and payload_kb for each regime based on calibration data.

The v2 parameter table assumes 50Gbps solo BW, but actual measured BW is ~30Gbps.
This script reads calibration data and computes:
  - sleep_us = Tcomm_solo * (1 - φ_target) / φ_target  (per iter)
  - Adjusted payload if needed

Usage:
  python3 compute_adjusted_params.py [--scenario S1] [--regime S1_moderate]
  python3 compute_adjusted_params.py --all   # compute for all regimes
"""
import argparse
import json
import sys
from pathlib import Path

SCENARIOS_FILE = Path(__file__).resolve().parent.parent / "scenarios" / "scenarios_v2.json"
TTARGET_FMT = "/tmp/expC_ttarget_{}.json"


def compute_for_regime(scenarios: dict, scenario_name: str, regime_name: str):
    """Compute adjusted parameters for one regime."""
    scenario = scenarios["scenarios"][scenario_name]
    regime = scenario["regimes"][regime_name]
    d_scale = regime.get("d_scale", 1.0)
    iters_per_epoch = scenarios["experiment_params"]["iters_per_epoch"]

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name} | Regime: {regime_name} | d_scale: {d_scale}")
    print(f"{'='*70}")

    adjusted_jobs = []
    for bj in scenario["base_jobs"]:
        jid = bj["job_id"]
        phi_target = bj["phi_target"]
        tcomp_orig_ms = bj["tcomp_per_iter_ms"]

        # Read calibration data
        ttarget_file = Path(TTARGET_FMT.format(jid))
        if not ttarget_file.exists():
            print(f"  [WARN] Job {jid}: no calibration file, skipping")
            continue

        calib = json.load(open(ttarget_file))
        tcomm_solo_ms = calib["target_comm_time_ms"]  # per-iter solo comm time
        solo_bw = calib.get("solo_bw_gbps", 30.0)

        # Scale Tcomm by d_scale (linear approximation)
        tcomm_scaled_ms = tcomm_solo_ms * d_scale

        # Compute required Tcomp for target φ
        # φ = Tcomp/(Tcomp+Tcomm) → Tcomp = Tcomm * (1-φ)/φ
        if phi_target > 0 and phi_target < 1:
            tcomp_required_ms = tcomm_scaled_ms * (1 - phi_target) / phi_target
        else:
            tcomp_required_ms = tcomp_orig_ms

        sleep_us = int(tcomp_required_ms * 1000)

        # Compute payload per iter
        d_per_epoch_mb = bj["d_per_epoch_mb"] * d_scale
        d_per_iter_bytes = int(d_per_epoch_mb * 1024 * 1024 / iters_per_epoch)
        payload_kb = d_per_iter_bytes // 1024

        # Verify φ
        phi_actual = tcomm_scaled_ms / (tcomp_required_ms + tcomm_scaled_ms)

        # Compute bandwidth metrics
        b_bar_gbps = phi_actual * scenarios["link_bw_gbps"]
        b_att_gbps = bj["c_policy"] * b_bar_gbps

        # Epoch duration estimate
        epoch_duration_ms = (tcomp_required_ms + tcomm_scaled_ms) * iters_per_epoch

        print(f"\n  Job {jid} ({bj['tier']}, {bj['label']}):")
        print(f"    Tcomm_solo(per-iter): {tcomm_scaled_ms:.3f}ms (scaled by ×{d_scale})")
        print(f"    Tcomp_orig: {tcomp_orig_ms:.1f}ms → Tcomp_required: {tcomp_required_ms:.2f}ms")
        print(f"    sleep_us: {sleep_us}")
        print(f"    payload_kb: {payload_kb}")
        print(f"    φ_actual: {phi_actual:.4f} (target: {phi_target})")
        print(f"    b̄ = {b_bar_gbps:.1f}G, b^att = {b_att_gbps:.1f}G")
        print(f"    epoch_duration ≈ {epoch_duration_ms:.0f}ms")

        adjusted_jobs.append({
            "job_id": jid,
            "label": bj["label"],
            "tier": bj["tier"],
            "c_policy": bj["c_policy"],
            "c_eval": bj["c_eval"],
            "sleep_us": sleep_us,
            "payload_kb": payload_kb,
            "phi_target": phi_target,
            "phi_actual": phi_actual,
            "tcomm_solo_ms": tcomm_scaled_ms,
            "tcomp_ms": tcomp_required_ms,
            "b_bar_gbps": b_bar_gbps,
            "b_att_gbps": b_att_gbps,
            "epoch_duration_ms": epoch_duration_ms,
        })

    # Compute aggregate regime metrics
    if adjusted_jobs:
        b_link = scenarios["link_bw_gbps"]
        beta = scenarios.get("beta", 0.5)
        n_jobs = len(adjusted_jobs)
        b_per_job = b_link / n_jobs

        sigma_b_bar = sum(j["b_bar_gbps"] for j in adjusted_jobs)
        sigma_b_att = sum(j["b_att_gbps"] for j in adjusted_jobs)
        # C* = Σ_P b^att + β * Σ_S b^att
        c_star = sum(j["b_att_gbps"] for j in adjusted_jobs if j["tier"] == "premium") + \
                 beta * sum(j["b_att_gbps"] for j in adjusted_jobs if j["tier"] == "standard")
        # λ = C* / B (effective capacity ratio)
        lam = c_star / b_link if b_link > 0 else 0
        offered = sigma_b_bar / b_link

        # fair premium check: b^att ≥ 1.5 × (B/N)?
        premium_batt = [j["b_att_gbps"] for j in adjusted_jobs if j["tier"] == "premium"]
        fair_premium_check = all(b >= 1.5 * b_per_job for b in premium_batt) if premium_batt else False

        print(f"\n  --- Regime Aggregate ---")
        print(f"  Σb̄ = {sigma_b_bar:.1f}G ({offered:.2f}B)")
        print(f"  Σb^att = {sigma_b_att:.1f}G")
        print(f"  C* = {c_star:.1f}G ({c_star/b_link:.2f}B)")
        print(f"  λ = {lam:.2f}")
        print(f"  B/N = {b_per_job:.1f}G, premium b^att ≥ 1.5×(B/N)? {fair_premium_check}")
        print(f"  Iron Rule 1 (real contention): {'✓' if offered >= 1.0 else '✗'} Σb̄/B={offered:.2f}")
        print(f"  Iron Rule 2 (fair must fail):  {'✓' if fair_premium_check else '✗'}")
        print(f"  Iron Rule 3 (LL passes):       λ={lam:.2f}, need ≥0.8 for LL to pass")

    return adjusted_jobs


def write_regime_params(adjusted_jobs, scenario_name, regime_name, output_dir):
    """Write adjusted parameters to a JSON file for the run script to read."""
    params = {
        "scenario": scenario_name,
        "regime": regime_name,
        "jobs": adjusted_jobs,
    }
    out_file = Path(output_dir) / f"params_{scenario_name}_{regime_name}.json"
    json.dump(params, open(out_file, "w"), indent=2)
    print(f"\n  [saved] {out_file}")
    return out_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--output-dir", default="/tmp")
    args = ap.parse_args()

    scenarios = json.load(open(SCENARIOS_FILE))

    if args.all:
        for sc_name, sc in scenarios["scenarios"].items():
            for reg_name in sc["regimes"]:
                jobs = compute_for_regime(scenarios, sc_name, reg_name)
                write_regime_params(jobs, sc_name, reg_name, args.output_dir)
    elif args.scenario and args.regime:
        jobs = compute_for_regime(scenarios, args.scenario, args.regime)
        write_regime_params(jobs, args.scenario, args.regime, args.output_dir)
    else:
        print("Specify --all or --scenario SC --regime REG")
        sys.exit(1)


if __name__ == "__main__":
    main()
