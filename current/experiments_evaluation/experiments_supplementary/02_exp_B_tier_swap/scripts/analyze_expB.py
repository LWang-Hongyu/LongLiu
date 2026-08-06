#!/usr/bin/env python3
"""
Experiment B: Hardware Tier Swap — Window Analysis & Trajectory Plot

Reads per-epoch CSV files from all rounds and computes:
  - W1/W2/W3 average slowdown for the tight SLO job
  - Cross-arm comparison (LongLiu vs CRUX static)
  - Trajectory plot (slowdown vs epoch) for both arms

Window definitions (aligned with simulation EQ3):
  W1 = epochs [3, 7]   — tight job = A (before swap)
  W2 = epochs [8, 12]  — tight job = B (just after swap, T_swap=8)
  W3 = epochs [18, 22] — tight job = B (later after swap)

Success criteria (qualitative, per §B.4):
  LongLiu arm: W3 slowdown ≈ W1 (no transient crash)
  Static  arm: W3 slowdown >> W1 (lost lock after swap)

Usage:
  python3 analyze_expB.py [--data-dir <path>] [--rounds 1,2,3,4]
"""

import os
import sys
import csv
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

# ============================================================
# Configuration (must match expB_config.sh)
# ============================================================
REVERSE_EPOCH = 8
W1_RANGE = range(3, 8)    # epochs 3-7
W2_RANGE = range(8, 13)   # epochs 8-12
W3_RANGE = range(18, 23)  # epochs 18-22

CI_PREMIUM = 1.2   # tight SLO
CI_STANDARD = 2.0  # loose SLO

EXP_B_DIR = Path("/home/why/LongLiu_rebuild/experiments_supplementary/02_exp_B_tier_swap")
DEFAULT_DATA_DIR = EXP_B_DIR / "data"
ANALYSIS_DIR = EXP_B_DIR / "analysis"


# ============================================================
# Data loading
# ============================================================
def load_epoch_csv(csv_path):
    """Load per-epoch CSV into list of dicts."""
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'epoch': int(row['epoch']),
                'phase': row['phase'],
                'c_i': float(row['c_i']),
                'avg_comm_s': float(row['avg_comm_s']),
                'avg_bw_gbps': float(row['avg_bw_gbps']),
                'pi': float(row['pi']) if row['pi'] != 'nan' else float('nan'),
                'priority': int(row['priority']),
                'dscp': int(row['dscp']),
                'slowdown': float(row['slowdown']) if row['slowdown'] != 'nan' else float('nan'),
                't_target_ms': float(row['t_target_ms']) if row['t_target_ms'] != 'nan' else float('nan'),
            })
    return records


def identify_tight_job(job_a_records, job_b_records, epoch):
    """Identify which job is the tight SLO job at a given epoch.
    The tight job has c_i = CI_PREMIUM (1.2).
    """
    a_ci = next((r['c_i'] for r in job_a_records if r['epoch'] == epoch), None)
    b_ci = next((r['c_i'] for r in job_b_records if r['epoch'] == epoch), None)
    if a_ci is not None and abs(a_ci - CI_PREMIUM) < 0.01:
        return 'A', job_a_records
    if b_ci is not None and abs(b_ci - CI_PREMIUM) < 0.01:
        return 'B', job_b_records
    return None, None


def get_tight_job_slowdown(job_a_records, job_b_records, epoch_range):
    """Get slowdown values for the tight job across a range of epochs."""
    slowdowns = []
    tight_job = None
    for epoch in epoch_range:
        job_label, records = identify_tight_job(job_a_records, job_b_records, epoch)
        if job_label is None:
            continue
        if tight_job is None:
            tight_job = job_label
        elif tight_job != job_label:
            # Tight job changed mid-window (shouldn't happen if windows are aligned)
            pass
        rec = next((r for r in records if r['epoch'] == epoch), None)
        if rec and rec['slowdown'] == rec['slowdown']:  # not NaN
            slowdowns.append(rec['slowdown'])
    return tight_job, slowdowns


def compute_window_stats(slowdowns):
    """Compute mean, std, min, max for a list of slowdown values."""
    if not slowdowns:
        return {'mean': float('nan'), 'std': float('nan'),
                'min': float('nan'), 'max': float('nan'), 'n': 0}
    return {
        'mean': statistics.mean(slowdowns),
        'std': statistics.stdev(slowdowns) if len(slowdowns) > 1 else 0.0,
        'min': min(slowdowns),
        'max': max(slowdowns),
        'n': len(slowdowns),
    }


# ============================================================
# Round analysis
# ============================================================
def analyze_round(round_dir):
    """Analyze one round's data. Returns dict with results for both arms."""
    result = {
        'round_dir': str(round_dir),
        'arms': {},
    }

    for arm_label, mode_label in [('LL', 'longliu'), ('CX', 'crux')]:
        csv_a = round_dir / f"jobA_{arm_label}_epoch.csv"
        csv_b = round_dir / f"jobB_{arm_label}_epoch.csv"

        if not csv_a.exists() or not csv_b.exists():
            print(f"  WARNING: {arm_label} arm CSV missing in {round_dir}")
            result['arms'][arm_label] = None
            continue

        job_a = load_epoch_csv(csv_a)
        job_b = load_epoch_csv(csv_b)

        # Compute window stats for tight job
        w1_job, w1_slowdowns = get_tight_job_slowdown(job_a, job_b, W1_RANGE)
        w2_job, w2_slowdowns = get_tight_job_slowdown(job_a, job_b, W2_RANGE)
        w3_job, w3_slowdowns = get_tight_job_slowdown(job_a, job_b, W3_RANGE)

        # Also get loose job stats for reference
        def get_loose_slowdowns(epoch_range, tight_job):
            loose_records = job_b if tight_job == 'A' else job_a
            slowdowns = []
            for epoch in epoch_range:
                rec = next((r for r in loose_records if r['epoch'] == epoch), None)
                if rec and rec['slowdown'] == rec['slowdown']:
                    slowdowns.append(rec['slowdown'])
            return slowdowns

        w1_loose = get_loose_slowdowns(W1_RANGE, w1_job)
        w2_loose = get_loose_slowdowns(W2_RANGE, w2_job)
        w3_loose = get_loose_slowdowns(W3_RANGE, w3_job)

        arm_result = {
            'mode': mode_label,
            'w1': {
                'tight_job': w1_job,
                'tight_slowdown': compute_window_stats(w1_slowdowns),
                'loose_slowdown': compute_window_stats(w1_loose),
            },
            'w2': {
                'tight_job': w2_job,
                'tight_slowdown': compute_window_stats(w2_slowdowns),
                'loose_slowdown': compute_window_stats(w2_loose),
            },
            'w3': {
                'tight_job': w3_job,
                'tight_slowdown': compute_window_stats(w3_slowdowns),
                'loose_slowdown': compute_window_stats(w3_loose),
            },
            # Full trajectory for plotting
            'trajectory': {
                'jobA': [(r['epoch'], r['slowdown'], r['c_i'], r['priority'])
                         for r in job_a],
                'jobB': [(r['epoch'], r['slowdown'], r['c_i'], r['priority'])
                         for r in job_b],
            },
        }
        result['arms'][arm_label] = arm_result

    return result


# ============================================================
# Cross-round aggregation
# ============================================================
def aggregate_rounds(all_results):
    """Aggregate window stats across all rounds (mean ± std)."""
    agg = {}
    for arm_label in ['LL', 'CX']:
        arm_data = [r['arms'][arm_label] for r in all_results
                    if r['arms'].get(arm_label)]
        if not arm_data:
            agg[arm_label] = None
            continue

        for window in ['w1', 'w2', 'w3']:
            tight_means = [d[window]['tight_slowdown']['mean'] for d in arm_data
                           if d[window]['tight_slowdown']['mean'] == d[window]['tight_slowdown']['mean']]
            loose_means = [d[window]['loose_slowdown']['mean'] for d in arm_data
                           if d[window]['loose_slowdown']['mean'] == d[window]['loose_slowdown']['mean']]

            agg.setdefault(arm_label, {})[window] = {
                'tight_mean_of_means': statistics.mean(tight_means) if tight_means else float('nan'),
                'tight_std_of_means': statistics.stdev(tight_means) if len(tight_means) > 1 else 0.0,
                'tight_n_rounds': len(tight_means),
                'loose_mean_of_means': statistics.mean(loose_means) if loose_means else float('nan'),
                'loose_std_of_means': statistics.stdev(loose_means) if len(loose_means) > 1 else 0.0,
            }
    return agg


# ============================================================
# Report generation
# ============================================================
def generate_report(all_results, agg, output_dir):
    """Generate markdown report with tables and success criteria check."""
    lines = []
    lines.append("# Experiment B: Hardware Tier Swap — Analysis Report")
    lines.append("")
    lines.append(f"> Generated: {__import__('datetime').datetime.now().isoformat()}")
    lines.append(f"> Swap epoch: {REVERSE_EPOCH}, c_i: {CI_PREMIUM}↔{CI_STANDARD}")
    lines.append(f"> Windows: W1={list(W1_RANGE)}, W2={list(W2_RANGE)}, W3={list(W3_RANGE)}")
    lines.append("")

    # ---- Per-round table ----
    lines.append("## Per-Round Window Slowdown (Tight SLO Job)")
    lines.append("")
    lines.append("| Round | Arm | W1 mean | W2 mean | W3 mean | W1→W3 ratio | Tight job W1→W2→W3 |")
    lines.append("|-------|-----|---------|---------|---------|-------------|-------------------|")

    for i, result in enumerate(all_results, 1):
        round_name = Path(result['round_dir']).name
        for arm_label in ['LL', 'CX']:
            arm = result['arms'].get(arm_label)
            if arm is None:
                lines.append(f"| {round_name} | {arm_label} | N/A | N/A | N/A | N/A | N/A |")
                continue
            w1m = arm['w1']['tight_slowdown']['mean']
            w2m = arm['w2']['tight_slowdown']['mean']
            w3m = arm['w3']['tight_slowdown']['mean']
            ratio = w3m / w1m if w1m == w1m and w3m == w3m and w1m > 0 else float('nan')
            jobs = f"{arm['w1']['tight_job']}→{arm['w2']['tight_job']}→{arm['w3']['tight_job']}"
            lines.append(f"| {round_name} | {arm_label} | {w1m:.3f} | {w2m:.3f} | {w3m:.3f} | {ratio:.2f}× | {jobs} |")

    lines.append("")

    # ---- Aggregated table ----
    lines.append("## Aggregated (Mean ± Std across rounds)")
    lines.append("")
    lines.append("| Arm | W1 tight | W2 tight | W3 tight | W3/W1 ratio | Success? |")
    lines.append("|-----|----------|----------|----------|-------------|----------|")

    for arm_label in ['LL', 'CX']:
        if agg.get(arm_label) is None:
            lines.append(f"| {arm_label} | N/A | N/A | N/A | N/A | N/A |")
            continue
        a = agg[arm_label]
        w1 = a['w1']['tight_mean_of_means']
        w2 = a['w2']['tight_mean_of_means']
        w3 = a['w3']['tight_mean_of_means']
        w1s = a['w1']['tight_std_of_means']
        w2s = a['w2']['tight_std_of_means']
        w3s = a['w3']['tight_std_of_means']
        ratio = w3 / w1 if w1 == w1 and w3 == w3 and w1 > 0 else float('nan')
        # Success criteria (§B.4):
        # LL: W3 ≈ W1 (no transient crash) → ratio < 1.5
        # CX: W3 >> W1 (lost lock) → ratio > 1.5
        if arm_label == 'LL':
            success = "✅ PASS" if ratio < 1.5 else "❌ FAIL"
            crit = f"(W3/W1={ratio:.2f}, expect <1.5)"
        else:
            success = "✅ PASS" if ratio > 1.5 else "❌ FAIL"
            crit = f"(W3/W1={ratio:.2f}, expect >1.5)"
        lines.append(f"| {arm_label} | {w1:.3f}±{w1s:.3f} | {w2:.3f}±{w2s:.3f} | "
                     f"{w3:.3f}±{w3s:.3f} | {ratio:.2f}× | {success} {crit} |")

    lines.append("")

    # ---- Success criteria summary ----
    lines.append("## Success Criteria Assessment (§B.4)")
    lines.append("")
    lines.append("- **LongLiu arm**: swap后 J_B 的 slowdown 在 W3 回到与 W1 中 J_A 相当的水平 — 无瞬态崩溃")
    lines.append("- **Static arm**: swap后失锁，W3 slowdown 显著高于 W1")
    lines.append("")
    ll_agg = agg.get('LL')
    cx_agg = agg.get('CX')
    if ll_agg and cx_agg:
        ll_ratio = ll_agg['w3']['tight_mean_of_means'] / ll_agg['w1']['tight_mean_of_means']
        cx_ratio = cx_agg['w3']['tight_mean_of_means'] / cx_agg['w1']['tight_mean_of_means']
        lines.append(f"- LongLiu W3/W1 = {ll_ratio:.2f}× (target: ≈1.0, threshold <1.5)")
        lines.append(f"- Static  W3/W1 = {cx_ratio:.2f}× (target: >>1.0, threshold >1.5)")
        lines.append(f"- **Overall**: {'✅ Both criteria met' if ll_ratio < 1.5 and cx_ratio > 1.5 else '⚠️ Criteria not fully met — see per-round details'}")
    lines.append("")

    # ---- Per-round trajectory data ----
    lines.append("## Per-Round Trajectory Data (for plotting)")
    lines.append("")
    for result in all_results:
        round_name = Path(result['round_dir']).name
        lines.append(f"### {round_name}")
        for arm_label in ['LL', 'CX']:
            arm = result['arms'].get(arm_label)
            if arm is None:
                continue
            lines.append(f"#### {arm_label} arm")
            lines.append("| Epoch | Job A slowdown | Job A c_i | Job A prio | Job B slowdown | Job B c_i | Job B prio |")
            lines.append("|-------|---------------|-----------|------------|---------------|-----------|------------|")
            max_len = max(len(arm['trajectory']['jobA']), len(arm['trajectory']['jobB']))
            for i in range(max_len):
                ea = arm['trajectory']['jobA'][i] if i < len(arm['trajectory']['jobA']) else (None,)*4
                eb = arm['trajectory']['jobB'][i] if i < len(arm['trajectory']['jobB']) else (None,)*4
                ea_s = f"{ea[1]:.3f}" if ea[0] is not None else "—"
                ea_ci = f"{ea[2]:.1f}" if ea[0] is not None else "—"
                ea_p = str(ea[3]) if ea[0] is not None else "—"
                eb_s = f"{eb[1]:.3f}" if eb[0] is not None else "—"
                eb_ci = f"{eb[2]:.1f}" if eb[0] is not None else "—"
                eb_p = str(eb[3]) if eb[0] is not None else "—"
                epoch = ea[0] if ea[0] is not None else eb[0]
                lines.append(f"| {epoch} | {ea_s} | {ea_ci} | {ea_p} | {eb_s} | {eb_ci} | {eb_p} |")
            lines.append("")

    report_path = output_dir / "expB_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved to: {report_path}")
    return report_path


# ============================================================
# Trajectory plot
# ============================================================
def plot_trajectories(all_results, output_dir):
    """Generate trajectory plots for all rounds."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return None

    n_rounds = len(all_results)
    if n_rounds == 0:
        return None

    fig, axes = plt.subplots(n_rounds, 2, figsize=(14, 4 * n_rounds), squeeze=False)
    if n_rounds == 1:
        axes = axes.reshape(1, -1)

    for row, result in enumerate(all_results):
        round_name = Path(result['round_dir']).name
        for col, arm_label in enumerate(['LL', 'CX']):
            ax = axes[row][col]
            arm = result['arms'].get(arm_label)
            if arm is None:
                ax.set_title(f"{round_name} — {arm_label} (no data)")
                continue

            for job_label, color in [('A', 'blue'), ('B', 'red')]:
                traj = arm['trajectory'][f'job{job_label}']
                if not traj:
                    continue
                epochs = [t[0] for t in traj]
                slowdowns = [t[1] for t in traj]
                ax.plot(epochs, slowdowns, f'-o{color[0]}', label=f'Job {job_label}',
                        markersize=4, linewidth=1.5)

            # Mark swap epoch
            ax.axvline(x=REVERSE_EPOCH, color='green', linestyle='--', alpha=0.7,
                       label=f'Swap (epoch {REVERSE_EPOCH})')
            # Mark windows
            for w_name, w_range, w_color in [('W1', W1_RANGE, 'yellow'), ('W2', W2_RANGE, 'orange'), ('W3', W3_RANGE, 'red')]:
                ax.axvspan(min(w_range) - 0.5, max(w_range) + 0.5, alpha=0.1, color=w_color, label=w_name)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Slowdown')
            ax.set_title(f'{round_name} — {arm_label} arm')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 25.5)

    plt.tight_layout()
    plot_path = output_dir / "expB_trajectory.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved to: {plot_path}")
    return plot_path


# ============================================================
# Summary CSV
# ============================================================
def save_summary_csv(all_results, agg, output_dir):
    """Save summary CSV for easy import into paper tables."""
    csv_path = output_dir / "expB_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'arm', 'window', 'tight_job',
                         'tight_slowdown_mean', 'tight_slowdown_std',
                         'tight_slowdown_min', 'tight_slowdown_max',
                         'loose_slowdown_mean', 'loose_slowdown_std', 'n_epochs'])

        for result in all_results:
            round_name = Path(result['round_dir']).name
            for arm_label in ['LL', 'CX']:
                arm = result['arms'].get(arm_label)
                if arm is None:
                    continue
                for w_name in ['w1', 'w2', 'w3']:
                    w = arm[w_name]
                    ts = w['tight_slowdown']
                    ls = w['loose_slowdown']
                    writer.writerow([
                        round_name, arm_label, w_name.upper(), w['tight_job'],
                        f"{ts['mean']:.4f}", f"{ts['std']:.4f}",
                        f"{ts['min']:.4f}", f"{ts['max']:.4f}",
                        f"{ls['mean']:.4f}", f"{ls['std']:.4f}", ts['n'],
                    ])

        # Aggregated rows
        writer.writerow([])
        writer.writerow(['round', 'arm', 'window', 'tight_job',
                         'tight_slowdown_mean', 'tight_slowdown_std',
                         'tight_slowdown_min', 'tight_slowdown_max',
                         'loose_slowdown_mean', 'loose_slowdown_std', 'n_rounds'])
        for arm_label in ['LL', 'CX']:
            if agg.get(arm_label) is None:
                continue
            for w_name in ['w1', 'w2', 'w3']:
                w = agg[arm_label][w_name]
                writer.writerow([
                    'AGGREGATED', arm_label, w_name.upper(), '',
                    f"{w['tight_mean_of_means']:.4f}", f"{w['tight_std_of_means']:.4f}",
                    '', '', f"{w['loose_mean_of_means']:.4f}",
                    f"{w['loose_std_of_means']:.4f}", w['tight_n_rounds'],
                ])

    print(f"Summary CSV saved to: {csv_path}")
    return csv_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Experiment B Window Analysis')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Directory containing round data subdirectories')
    parser.add_argument('--rounds', type=str, default='1,2,3,4',
                        help='Comma-separated round numbers to analyze (default: 1,2,3,4)')
    parser.add_argument('--output-dir', type=str, default=str(ANALYSIS_DIR),
                        help='Output directory for reports and plots')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find round directories
    round_nums = [int(x) for x in args.rounds.split(',')]
    round_dirs = []
    for rn in round_nums:
        # Match round<N>_* pattern
        for d in data_dir.iterdir():
            if d.is_dir() and d.name.startswith(f'round{rn}_'):
                round_dirs.append(d)
                break

    if not round_dirs:
        print(f"ERROR: No round directories found in {data_dir}")
        print(f"  Expected: round<N>_<joborder>_<armorder>/")
        print(f"  Run run_expB_all.sh first to generate data.")
        sys.exit(1)

    round_dirs.sort()
    print(f"Found {len(round_dirs)} round directories:")
    for d in round_dirs:
        print(f"  {d.name}")

    # Analyze each round
    all_results = []
    for rd in round_dirs:
        print(f"\n--- Analyzing {rd.name} ---")
        result = analyze_round(rd)
        all_results.append(result)

        # Print per-round summary
        for arm_label in ['LL', 'CX']:
            arm = result['arms'].get(arm_label)
            if arm is None:
                print(f"  {arm_label}: no data")
                continue
            w1 = arm['w1']['tight_slowdown']['mean']
            w2 = arm['w2']['tight_slowdown']['mean']
            w3 = arm['w3']['tight_slowdown']['mean']
            ratio = w3 / w1 if w1 == w1 and w3 == w3 and w1 > 0 else float('nan')
            print(f"  {arm_label}: W1={w1:.3f} W2={w2:.3f} W3={w3:.3f} (W3/W1={ratio:.2f}×)")

    # Aggregate
    agg = aggregate_rounds(all_results)

    # Generate outputs
    print("\n=== Generating reports ===")
    generate_report(all_results, agg, output_dir)
    save_summary_csv(all_results, agg, output_dir)
    plot_trajectories(all_results, output_dir)

    print("\n=== Done ===")
    print(f"All outputs in: {output_dir}/")


if __name__ == '__main__':
    main()
