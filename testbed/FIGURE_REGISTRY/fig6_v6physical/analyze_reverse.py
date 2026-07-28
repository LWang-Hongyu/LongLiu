#!/usr/bin/env python3
"""
P4-1 Role-Reversal Experiment: Analysis & Comparison
=====================================================
Compare LongLiu v1(π) vs CRUX-static under role reversal.
Generate per-epoch comparison data for plotting.

Output:
  - reverse_comparison_epoch.csv  (per-epoch summary, both modes, both jobs)
  - reverse_comparison_iter.csv   (per-iter comm time, both modes, both jobs)
  - Console summary with key metrics
"""

import csv
import os
import sys
from collections import defaultdict

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

def load_epoch_csv(job, mode):
    """Load per-epoch CSV for a given job and mode."""
    path = os.path.join(EXP_DIR, f'p4_job{job}_reverse_{mode}_rank0_epoch.csv')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return []
    with open(path, 'r') as f:
        return list(csv.DictReader(f))

def load_iter_csv(job, mode):
    """Load per-iter CSV for a given job and mode (rank 0)."""
    path = os.path.join(EXP_DIR, f'p4_job{job}_reverse_{mode}_rank0_iter.csv')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return []
    with open(path, 'r') as f:
        return list(csv.DictReader(f))

def analyze():
    print("=" * 70)
    print("P4-1 Role-Reversal Experiment: LongLiu v1(π) vs CRUX-static (SP queue)")
    print("=" * 70)

    reverse_epoch = 7
    modes = ['longliu', 'crux']
    jobs = ['A', 'B']

    # Load all data
    data = {}
    for mode in modes:
        for job in jobs:
            data[(mode, job)] = {
                'epoch': load_epoch_csv(job, mode),
                'iter': load_iter_csv(job, mode),
            }

    # Per-epoch comparison table
    print("\n### Per-Epoch Summary ###\n")
    print(f"{'Mode':<8} {'Job':<4} {'Epoch':<6} {'Phase':<7} {'Sleep':<6} "
          f"{'AvgComm(ms)':<12} {'AvgBW(Gbps)':<12} {'π':<10} {'Prio':<5} {'DSCP':<5}")
    print("-" * 90)

    for mode in modes:
        for job in jobs:
            for row in data[(mode, job)]['epoch']:
                pi_str = f"{float(row['pi']):+.4f}" if row['pi'] != 'nan' else 'nan'
                print(f"{mode:<8} {job:<4} {row['epoch']:<6} {row['phase']:<7} "
                      f"{row['sleep_us']:<6} "
                      f"{float(row['avg_comm_s'])*1000:<12.1f} "
                      f"{float(row['avg_bw_gbps']):<12.2f} "
                      f"{pi_str:<10} "
                      f"{row['priority']:<5} {row['dscp']:<5}")
        print("-" * 90)

    # Phase-aggregated comparison
    print("\n### Phase-Aggregated Comparison ###\n")
    print(f"{'Mode':<8} {'Job':<4} {'Phase':<7} {'AvgComm(ms)':<14} "
          f"{'AvgBW(Gbps)':<14} {'Avgπ':<10} {'AvgPrio':<8}")
    print("-" * 70)

    for mode in modes:
        for job in jobs:
            epoch_data = data[(mode, job)]['epoch']
            for phase in ['phase1', 'phase2']:
                phase_rows = [r for r in epoch_data if r['phase'] == phase]
                if not phase_rows:
                    continue
                avg_comm = sum(float(r['avg_comm_s']) for r in phase_rows) / len(phase_rows)
                avg_bw = sum(float(r['avg_bw_gbps']) for r in phase_rows) / len(phase_rows)
                pi_vals = [float(r['pi']) for r in phase_rows if r['pi'] != 'nan']
                avg_pi = sum(pi_vals) / len(pi_vals) if pi_vals else float('nan')
                prio_vals = [int(r['priority']) for r in phase_rows]
                avg_prio = sum(prio_vals) / len(prio_vals) if prio_vals else 0
                print(f"{mode:<8} {job:<4} {phase:<7} "
                      f"{avg_comm*1000:<14.1f} {avg_bw:<14.2f} "
                      f"{avg_pi:<+10.4f} {avg_prio:<8.1f}")
        print("-" * 70)

    # Key comparison: does LongLiu respond to role reversal?
    print("\n### Key Comparison: Role-Reversal Response ###\n")

    for job in jobs:
        print(f"--- Job {job} ---")
        for mode in modes:
            epoch_data = data[(mode, job)]['epoch']
            if not epoch_data:
                continue
            # Phase 1 vs Phase 2 priority
            p1_prios = [int(r['priority']) for r in epoch_data if r['phase'] == 'phase1']
            p2_prios = [int(r['priority']) for r in epoch_data if r['phase'] == 'phase2']
            p1_avg = sum(p1_prios) / len(p1_prios) if p1_prios else 0
            p2_avg = sum(p2_prios) / len(p2_prios) if p2_prios else 0

            # Phase 1 vs Phase 2 bandwidth
            p1_bws = [float(r['avg_bw_gbps']) for r in epoch_data if r['phase'] == 'phase1']
            p2_bws = [float(r['avg_bw_gbps']) for r in epoch_data if r['phase'] == 'phase2']
            p1_bw = sum(p1_bws) / len(p1_bws) if p1_bws else 0
            p2_bw = sum(p2_bws) / len(p2_bws) if p2_bws else 0

            mode_label = 'LongLiu v1(π)' if mode == 'longliu' else 'CRUX-static'
            print(f"  {mode_label}:")
            print(f"    Phase 1: avg_prio=P{p1_avg:.1f}, avg_bw={p1_bw:.2f} Gbps")
            print(f"    Phase 2: avg_prio=P{p2_avg:.1f}, avg_bw={p2_bw:.2f} Gbps")
            print(f"    Δ prio: {p2_avg - p1_avg:+.1f}  Δ bw: {p2_bw - p1_bw:+.2f} Gbps")
        print()

    # Write combined CSV for plotting
    combined_epoch_path = os.path.join(EXP_DIR, 'reverse_comparison_epoch.csv')
    with open(combined_epoch_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'job', 'epoch', 'phase', 'sleep_us',
                         'avg_comm_ms', 'avg_bw_gbps', 'pi', 'priority', 'dscp'])
        for mode in modes:
            for job in jobs:
                for row in data[(mode, job)]['epoch']:
                    writer.writerow([
                        mode, job, row['epoch'], row['phase'], row['sleep_us'],
                        f"{float(row['avg_comm_s'])*1000:.2f}",
                        f"{float(row['avg_bw_gbps']):.4f}",
                        row['pi'], row['priority'], row['dscp']
                    ])
    print(f"\nCombined per-epoch CSV: {combined_epoch_path}")

    # Combined per-iter CSV
    combined_iter_path = os.path.join(EXP_DIR, 'reverse_comparison_iter.csv')
    with open(combined_iter_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'job', 'iter', 'epoch', 'sleep_us',
                         'comm_dur_ms', 'bw_gbps', 'phase'])
        for mode in modes:
            for job in jobs:
                for row in data[(mode, job)]['iter']:
                    writer.writerow([
                        mode, job, row['iter'], row['epoch'], row['sleep_us'],
                        f"{float(row['comm_dur_s'])*1000:.3f}",
                        f"{float(row['bw_gbps']):.4f}",
                        row['phase']
                    ])
    print(f"Combined per-iter CSV: {combined_iter_path}")

if __name__ == '__main__':
    analyze()
