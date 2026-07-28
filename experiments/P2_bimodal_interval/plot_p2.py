#!/usr/bin/env python3
"""
P2: Bimodal Interval Distribution - Plotting Script
Generates histogram showing intra-iteration vs inter-iteration gaps.
"""

import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.use('Agg')

def load_timestamps(csv_path):
    """Load timestamps from CSV, return list of (iter, start_ns, end_ns, duration_ns)"""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                int(row['iteration']),
                int(row['start_ns']),
                int(row['end_ns']),
                int(row['duration_ns'])
            ))
    return rows

def compute_intervals(rows):
    """Compute intervals between consecutive collective end times (in microseconds)"""
    intervals_us = []
    for i in range(1, len(rows)):
        delta_ns = rows[i][2] - rows[i-1][2]  # end_i - end_{i-1}
        intervals_us.append(delta_ns / 1000.0)  # convert to μs
    return intervals_us

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load data from both ranks
    all_intervals = {}
    for rank in [0, 1]:
        csv_path = os.path.join(base_dir, f'p2_timestamps_rank{rank}.csv')
        if not os.path.exists(csv_path):
            # Try generic name
            csv_path = os.path.join(base_dir, 'p2_timestamps.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: No data found for rank {rank}")
            continue

        rows = load_timestamps(csv_path)
        intervals = compute_intervals(rows)
        all_intervals[rank] = {
            'intervals': intervals,
            'durations': [r[3] / 1000.0 for r in rows]  # μs
        }
        print(f"Rank {rank}: {len(intervals)} intervals, "
              f"range [{min(intervals):.1f}, {max(intervals):.1f}] μs")

    if not all_intervals:
        print("Error: No data files found. Run p2_collect_data.py first.")
        return

    # Use rank 0 data (or merge)
    rank = 0 if 0 in all_intervals else list(all_intervals.keys())[0]
    intervals = all_intervals[rank]['intervals']
    durations = all_intervals[rank]['durations']

    # ============================================================
    # Figure 1: Bimodal Interval Distribution (Main Plot)
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    intervals_arr = np.array(intervals)

    # Use log-scale bins for better visualization of bimodal distribution
    log_intervals = np.log10(intervals_arr[intervals_arr > 0])
    bins = np.linspace(np.floor(log_intervals.min()), np.ceil(log_intervals.max()), 60)

    ax1.hist(log_intervals, bins=bins, color='#1565C0', alpha=0.75, edgecolor='black', linewidth=0.3)

    # Mark the two modes
    # Find peaks by binning
    hist_vals, bin_edges = np.histogram(log_intervals, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the two highest peaks
    peak_indices = np.argsort(hist_vals)[-2:]
    peak_centers = sorted(bin_centers[peak_indices])

    for i, peak in enumerate(peak_centers):
        label = 'Intra-iteration' if i == 0 else 'Inter-iteration'
        color = '#C62828' if i == 0 else '#2E7D32'
        ax1.axvline(x=peak, color=color, linestyle='--', linewidth=2, label=f'{label} peak')

    ax1.set_xlabel('log₁₀(Interval between collectives) [μs]', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('P2: Bimodal Interval Distribution\n'
                  'Intra-iteration (~μs) vs Inter-iteration (~ms)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add secondary x-axis with actual μs values
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    tick_positions = ax1.get_xticks()
    ax1_top.set_xticks(tick_positions)
    ax1_top.set_xticklabels([f'{10**x:.1f}' for x in tick_positions], fontsize=8)
    ax1_top.set_xlabel('Interval [μs]', fontsize=10)

    plt.tight_layout()
    fig1.savefig(os.path.join(base_dir, 'fig_p2_bimodal_histogram.png'), dpi=150)
    print("Saved fig_p2_bimodal_histogram.png")

    # ============================================================
    # Figure 2: Interval Time Series
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    x = np.arange(len(intervals))
    intervals_ms = intervals_arr / 1000.0  # Convert to ms

    # Color by type (intra vs inter)
    threshold = np.sqrt(intervals_arr.min() * intervals_arr.max())  # geometric mean
    colors = ['#C62828' if v < threshold else '#2E7D32' for v in intervals_arr]

    ax2.scatter(x, intervals_ms, c=colors, s=15, alpha=0.6)
    ax2.axhline(y=threshold/1000, color='gray', linestyle=':', alpha=0.5,
                label=f'Threshold = {threshold:.0f} μs')
    ax2.set_xlabel('Collective Pair Index', fontsize=12)
    ax2.set_ylabel('Interval [ms]', fontsize=12)
    ax2.set_title('P2: Interval Between Consecutive Collectives', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(base_dir, 'fig_p2_interval_timeseries.png'), dpi=150)
    print("Saved fig_p2_interval_timeseries.png")

    # ============================================================
    # Figure 3: Per-Iteration Duration Distribution
    # ============================================================
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))

    durations_ms = np.array(durations) / 1000.0  # Convert to ms
    ax3.hist(durations_ms, bins=30, color='#FF8F00', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax3.axvline(x=np.mean(durations_ms), color='#C62828', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(durations_ms):.2f} ms')
    ax3.axvline(x=np.percentile(durations_ms, 95), color='#2E7D32', linestyle='--', linewidth=2,
                label=f'P95 = {np.percentile(durations_ms, 95):.2f} ms')

    ax3.set_xlabel('AllReduce Duration [ms]', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('P2: Per-Iteration AllReduce Duration', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(os.path.join(base_dir, 'fig_p2_iteration_duration.png'), dpi=150)
    print("Saved fig_p2_iteration_duration.png")

    # ============================================================
    # Summary Statistics
    # ============================================================
    print("\n=== P2 Summary Statistics ===")
    print(f"Total iterations: {len(durations)}")
    print(f"Total intervals: {len(intervals)}")
    print(f"\nIteration duration (ms):")
    print(f"  Mean:   {np.mean(durations_ms):.3f}")
    print(f"  Std:    {np.std(durations_ms):.3f}")
    print(f"  Min:    {np.min(durations_ms):.3f}")
    print(f"  Max:    {np.max(durations_ms):.3f}")
    print(f"  P95:    {np.percentile(durations_ms, 95):.3f}")

    print(f"\nInterval between collectives (μs):")
    print(f"  Mean:   {np.mean(intervals_arr):.1f}")
    print(f"  Min:    {np.min(intervals_arr):.1f}")
    print(f"  Max:    {np.max(intervals_arr):.1f}")

    # Bimodal analysis
    intra = intervals_arr[intervals_arr < threshold]
    inter = intervals_arr[intervals_arr >= threshold]
    print(f"\nBimodal analysis (threshold = {threshold:.0f} μs):")
    print(f"  Intra-iteration: {len(intra)} intervals, mean = {np.mean(intra):.1f} μs")
    print(f"  Inter-iteration: {len(inter)} intervals, mean = {np.mean(inter):.1f} μs")
    print(f"  Separation ratio: {np.mean(inter) / np.mean(intra):.1f}x")

    print("\nAll P2 figures generated successfully.")

if __name__ == '__main__':
    main()
