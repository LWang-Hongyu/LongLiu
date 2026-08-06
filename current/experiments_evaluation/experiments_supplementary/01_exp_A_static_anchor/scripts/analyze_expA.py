#!/usr/bin/env python3
"""
Experiment A — Analysis: HW-vs-Sim fidelity evidence table
==========================================================
Loads HW per-epoch CSVs and Sim per-epoch CSVs, computes per-scenario:
  - mean slowdown_hw (epochs 5-19, skip warmup + tail)
  - mean slowdown_sim (same window)
  - relative error |sim - hw| / hw
  - attainment_hw = slowdown_hw / c_i
  - attainment_sim = slowdown_sim / c_i
  - attainment diff (percentage points)

Outputs:
  - analysis/expA_evidence_table.md  (markdown table for paper)
  - analysis/expA_evidence_table.csv (raw data)
  - analysis/expA_trajectory.png     (per-scenario trajectory plot)
"""
import json
import csv
import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Match expA_scenarios.json windowing
WARMUP_EPOCHS = 5     # skip epochs 0-4
TAIL_EPOCHS = 5       # skip epochs 20-24
# For 25 epochs: analysis window = [5, 19]


def load_epoch_csv(path: Path) -> list:
    """Load p4_job_reverse.py epoch CSV. Returns list of dicts with int epoch."""
    if not path.exists():
        return []
    recs = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                recs.append({
                    "epoch": int(row["epoch"]),
                    "c_i": float(row.get("c_i", 0) or 0),
                    "avg_comm_s": float(row.get("avg_comm_s", 0) or 0),
                    "avg_bw_gbps": float(row.get("avg_bw_gbps", 0) or 0),
                    "pi": float(row.get("pi", 0) or 0),
                    "priority": int(row.get("priority", 0) or 0),
                    "dscp": int(row.get("dscp", 0) or 0),
                    "slowdown": float(row.get("slowdown", "nan") or "nan"),
                    "t_target_ms": float(row.get("t_target_ms", 0) or 0),
                })
            except (ValueError, KeyError) as e:
                print(f"[WARN] bad row in {path}: {row} ({e})")
    return recs


def window_slowdown(recs: list, lo: int = WARMUP_EPOCHS, hi: int = 24 - TAIL_EPOCHS):
    """Mean slowdown over epoch window [lo, hi]."""
    sd = [r["slowdown"] for r in recs if lo <= r["epoch"] <= hi
          and r["slowdown"] == r["slowdown"]]  # filter NaN
    if not sd:
        return float("nan"), float("nan")
    return float(np.mean(sd)), float(np.std(sd))


def find_hw_dirs(data_dir: Path) -> dict:
    """Find the latest run_* directory and return {scenario_id: path}."""
    run_dirs = sorted(data_dir.glob("run_*"))
    if not run_dirs:
        return {}
    latest = run_dirs[-1]
    scen_dirs = {}
    for d in latest.iterdir():
        if d.is_dir():
            # Dir name like "S1_2job_50G_ci1.2"
            sid = d.name.split("_")[0]
            if sid.startswith("S"):
                scen_dirs[sid] = d
    return scen_dirs


def find_sim_dir(data_dir: Path) -> Path:
    """Find the latest sim_run_* directory."""
    latest_file = data_dir / "latest_sim.txt"
    if latest_file.exists():
        p = Path(latest_file.read_text().strip())
        if p.exists():
            return p
    sim_dirs = sorted(data_dir.glob("sim_run_*"))
    return sim_dirs[-1] if sim_dirs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--analysis-dir", default=str(Path(__file__).resolve().parent.parent / "analysis"))
    ap.add_argument("--scenarios-file", default=str(Path(__file__).resolve().parent / "expA_scenarios.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenarios_file) as f:
        scen_def = json.load(f)

    hw_dirs = find_hw_dirs(data_dir)
    sim_dir = find_sim_dir(data_dir)

    print(f"[analyze] HW run dirs: {hw_dirs}")
    print(f"[analyze] Sim dir: {sim_dir}")

    rows = []
    trajectory_data = {}

    for scen in scen_def["scenarios"]:
        sid = scen["id"]
        label = scen["label"]
        c_i = scen["c_i"]
        payload = scen["payload_mb"]
        cap = scen["capacity_label"]
        holdout = scen.get("holdout", False)
        anchor_src = scen.get("anchor_source", "?")

        # HW side
        hw_sd_a_mean = hw_sd_b_mean = float("nan")
        hw_sd_a_std = hw_sd_b_std = float("nan")
        hw_recs_a = hw_recs_b = []
        if sid in hw_dirs:
            hd = hw_dirs[sid]
            # Try common CSV name patterns
            for pat in [f"jobA_rank0_epoch.csv", f"p4_jobA_rank0_epoch.csv"]:
                p = hd / pat
                if p.exists():
                    hw_recs_a = load_epoch_csv(p)
                    break
            for pat in [f"jobB_rank0_epoch.csv", f"p4_jobB_rank0_epoch.csv"]:
                p = hd / pat
                if p.exists():
                    hw_recs_b = load_epoch_csv(p)
                    break
            hw_sd_a_mean, hw_sd_a_std = window_slowdown(hw_recs_a)
            hw_sd_b_mean, hw_sd_b_std = window_slowdown(hw_recs_b)

        # Sim side
        sim_sd_a_mean = sim_sd_b_mean = float("nan")
        sim_sd_a_std = sim_sd_b_std = float("nan")
        sim_recs_a = sim_recs_b = []
        sim_manifest = {}
        if sim_dir:
            sd_path = sim_dir / f"{sid}_{label}"
            if sd_path.exists():
                for pat in ["jobA_sim_epoch.csv"]:
                    p = sd_path / pat
                    if p.exists():
                        sim_recs_a = load_epoch_csv(p)
                        break
                for pat in ["jobB_sim_epoch.csv"]:
                    p = sd_path / pat
                    if p.exists():
                        sim_recs_b = load_epoch_csv(p)
                        break
                mp = sd_path / "sim_manifest.json"
                if mp.exists():
                    sim_manifest = json.load(open(mp))
                sim_sd_a_mean, sim_sd_a_std = window_slowdown(sim_recs_a)
                sim_sd_b_mean, sim_sd_b_std = window_slowdown(sim_recs_b)

        # Use mean of A and B as the scenario slowdown (symmetric static scenario)
        hw_mean = np.nanmean([hw_sd_a_mean, hw_sd_b_mean]) if not (np.isnan(hw_sd_a_mean) and np.isnan(hw_sd_b_mean)) else float("nan")
        sim_mean = np.nanmean([sim_sd_a_mean, sim_sd_b_mean]) if not (np.isnan(sim_sd_a_mean) and np.isnan(sim_sd_b_mean)) else float("nan")

        rel_err = abs(sim_mean - hw_mean) / hw_mean if hw_mean > 0 and not np.isnan(sim_mean) else float("nan")
        attain_hw = hw_mean / c_i if not np.isnan(hw_mean) else float("nan")
        attain_sim = sim_mean / c_i if not np.isnan(sim_mean) else float("nan")
        attain_diff_pp = (attain_sim - attain_hw) * 100 if not (np.isnan(attain_hw) or np.isnan(attain_sim)) else float("nan")

        rows.append({
            "scenario": sid,
            "label": label,
            "capacity": cap,
            "payload_mb": payload,
            "c_i": c_i,
            "slowdown_hw": hw_mean,
            "slowdown_sim": sim_mean,
            "rel_error": rel_err,
            "attain_hw": attain_hw,
            "attain_sim": attain_sim,
            "attain_diff_pp": attain_diff_pp,
            "holdout": holdout,
            "anchor_source": anchor_src,
            "hw_sd_a": hw_sd_a_mean, "hw_sd_b": hw_sd_b_mean,
            "sim_sd_a": sim_sd_a_mean, "sim_sd_b": sim_sd_b_mean,
        })

        trajectory_data[sid] = {
            "hw_a": hw_recs_a, "hw_b": hw_recs_b,
            "sim_a": sim_recs_a, "sim_b": sim_recs_b,
        }

    # ---- Write CSV ----
    csv_path = analysis_dir / "expA_evidence_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "label", "capacity", "payload_mb", "c_i",
                    "slowdown_hw", "slowdown_sim", "rel_error",
                    "attain_hw", "attain_sim", "attain_diff_pp",
                    "holdout", "anchor_source"])
        for r in rows:
            w.writerow([
                r["scenario"], r["label"], r["capacity"], r["payload_mb"], r["c_i"],
                f"{r['slowdown_hw']:.4f}" if not np.isnan(r["slowdown_hw"]) else "NaN",
                f"{r['slowdown_sim']:.4f}" if not np.isnan(r["slowdown_sim"]) else "NaN",
                f"{r['rel_error']:.4f}" if not np.isnan(r["rel_error"]) else "NaN",
                f"{r['attain_hw']:.4f}" if not np.isnan(r["attain_hw"]) else "NaN",
                f"{r['attain_sim']:.4f}" if not np.isnan(r["attain_sim"]) else "NaN",
                f"{r['attain_diff_pp']:.2f}" if not np.isnan(r["attain_diff_pp"]) else "NaN",
                "Y" if r["holdout"] else "N",
                r["anchor_source"],
            ])

    # ---- Write Markdown report ----
    md_path = analysis_dir / "expA_evidence_table.md"
    with open(md_path, "w") as f:
        f.write("# Experiment A: Static Anchor — HW-vs-Sim Fidelity Evidence Table\n\n")
        f.write(f"> Generated: {os.popen('date -Iseconds').read().strip()}\n")
        f.write(f"> Analysis window: epochs {WARMUP_EPOCHS}-{24-TAIL_EPOCHS} (skip {WARMUP_EPOCHS} warmup + {TAIL_EPOCHS} tail)\n")
        f.write(f"> Scenarios: {len(rows)} (hold-out: {sum(1 for r in rows if r['holdout'])})\n\n")

        # Aggregate stats
        non_holdout = [r for r in rows if not r["holdout"] and not np.isnan(r["rel_error"])]
        all_with_data = [r for r in rows if not np.isnan(r["rel_error"])]
        if non_holdout:
            errs = [r["rel_error"] for r in non_holdout]
            max_err = max(errs)
            mean_err = float(np.mean(errs))
            f.write(f"## Aggregate Fidelity (non-holdout, N={len(non_holdout)})\n\n")
            f.write(f"- **Max relative error**: {max_err*100:.2f}%\n")
            f.write(f"- **Mean relative error**: {mean_err*100:.2f}%\n")
            f.write(f"- Paper claim threshold: 0.2% — **{'PASS' if max_err <= 0.002 else 'FAIL'}**\n\n")

        f.write("## Evidence Table (§A.3)\n\n")
        f.write("| # | Capacity | c_i | slowdown_hw | slowdown_sim | rel.err | attain_hw | attain_sim | diff(pp) | holdout | anchor |\n")
        f.write("|---|----------|-----|-------------|--------------|---------|-----------|------------|----------|---------|--------|\n")
        for r in rows:
            f.write(f"| {r['scenario']} | {r['capacity']} | {r['c_i']} | "
                    f"{r['slowdown_hw']:.4f} | {r['slowdown_sim']:.4f} | "
                    f"{r['rel_error']*100:.2f}% | "
                    f"{r['attain_hw']:.4f} | {r['attain_sim']:.4f} | "
                    f"{r['attain_diff_pp']:+.2f} | "
                    f"{'†' if r['holdout'] else ''} | {r['anchor_source']} |\n")

        f.write("\n† = hold-out scenario (anchor from theoretical BW, not measured solo)\n\n")

        # Per-scenario details
        f.write("## Per-Scenario Breakdown\n\n")
        for r in rows:
            f.write(f"### {r['scenario']} — {r['label']}\n\n")
            f.write(f"- Payload: {r['payload_mb']}MB, c_i: {r['c_i']}, Capacity: {r['capacity']}\n")
            f.write(f"- HW slowdown: A={r['hw_sd_a']:.4f}, B={r['hw_sd_b']:.4f} → mean={r['slowdown_hw']:.4f}\n")
            f.write(f"- Sim slowdown: A={r['sim_sd_a']:.4f}, B={r['sim_sd_b']:.4f} → mean={r['slowdown_sim']:.4f}\n")
            f.write(f"- Relative error: {r['rel_error']*100:.2f}%\n")
            f.write(f"- Attainment: hw={r['attain_hw']:.4f}, sim={r['attain_sim']:.4f}, diff={r['attain_diff_pp']:+.2f}pp\n\n")

    # ---- Trajectory plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        axes = axes.flatten()
        for i, sid in enumerate(trajectory_data):
            if i >= 6:
                break
            ax = axes[i]
            td = trajectory_data[sid]
            for recs, lbl, sty in [
                (td["hw_a"], "HW A", "-o"),
                (td["hw_b"], "HW B", "--o"),
                (td["sim_a"], "Sim A", "-s"),
                (td["sim_b"], "Sim B", "--s"),
            ]:
                if recs:
                    xs = [r["epoch"] for r in recs]
                    ys = [r["slowdown"] for r in recs]
                    ax.plot(xs, ys, sty, label=lbl, markersize=3, alpha=0.7)
            ax.axvspan(WARMUP_EPOCHS, 24 - TAIL_EPOCHS, alpha=0.1, color="green", label="analysis window")
            ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
            ax.set_title(f"{sid}: {rows[i]['label']}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("slowdown")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = analysis_dir / "expA_trajectory.png"
        plt.savefig(plot_path, dpi=120)
        print(f"[analyze] trajectory plot: {plot_path}")
    except ImportError:
        print("[WARN] matplotlib not available, skipping trajectory plot")

    print(f"[analyze] CSV: {csv_path}")
    print(f"[analyze] MD:  {md_path}")

    # Print summary to stdout
    print("\n=== Summary ===")
    for r in rows:
        h = f"{r['slowdown_hw']:.3f}" if not np.isnan(r['slowdown_hw']) else "  NaN"
        s = f"{r['slowdown_sim']:.3f}" if not np.isnan(r['slowdown_sim']) else "  NaN"
        e = f"{r['rel_error']*100:5.2f}%" if not np.isnan(r['rel_error']) else "   NaN"
        print(f"  {r['scenario']} | hw={h} sim={s} err={e} {'†' if r['holdout'] else ''}")


if __name__ == "__main__":
    main()
