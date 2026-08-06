#!/usr/bin/env python3
"""
P4: Dumbbell SLO Verification — Plot all 5 baselines.
Generates:
  1. Epoch timeline (iter time per epoch, both jobs)
  2. Bandwidth comparison (contested only)
  3. SLO attainment bar chart
  4. Ui comparison (SLO violation ratio)
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODES = ['fair', 'longliu', 'crux', 'srpt', 'mltcp']
MODE_LABELS = {'fair': 'Fair', 'longliu': 'LongLiu', 'crux': 'CRUX-like',
               'srpt': 'SRPT-like', 'mltcp': 'MLTCP-like'}
COLORS = {'fair': '#2196F3', 'longliu': '#4CAF50', 'crux': '#FF9800',
          'srpt': '#E91E63', 'mltcp': '#9C27B0'}
MARKERS = {'fair': 'o', 'longliu': 's', 'crux': '^',
           'srpt': 'D', 'mltcp': 'v'}

# SLO thresholds
SLO_J1 = 1.5  # Tight SLO
SLO_J2 = 2.5  # Loose SLO


def load_csv(mode):
    """Load CSV for a given mode, return (j1_data, j2_data) as lists of dicts."""
    j1_data = []
    j2_data = []
    for job in ['job1', 'job2']:
        path = os.path.join(SCRIPT_DIR, f'p4_{job}_{mode}_rank0.csv')
        if not os.path.exists(path):
            print(f"WARNING: {path} not found")
            continue
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['epoch'] = int(row['epoch'])
                row['comm_dur_s'] = float(row['comm_dur_s'])
                row['bw_obs_gbps'] = float(row['bw_obs_gbps'])
                row['total_bytes'] = float(row['total_bytes'])
                if job == 'job1':
                    j1_data.append(row)
                else:
                    j2_data.append(row)
    return j1_data, j2_data


def plot_iter_time_timeline(all_data):
    """Plot epoch vs iter time for all modes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for mode in MODES:
        j1, j2 = all_data[mode]
        if not j1 or not j2:
            continue
        c = COLORS[mode]
        m = MARKERS[mode]
        lbl = MODE_LABELS[mode]

        # Job 1
        epochs_j1 = [d['epoch'] for d in j1 if d['phase'] == 'contested']
        times_j1 = [d['comm_dur_s'] * 1000 for d in j1 if d['phase'] == 'contested']
        if epochs_j1:
            axes[0].plot(epochs_j1, times_j1, color=c, marker=m, label=lbl, linewidth=2, markersize=6)

        # Job 2
        epochs_j2 = [d['epoch'] for d in j2 if d['phase'] == 'contested']
        times_j2 = [d['comm_dur_s'] * 1000 for d in j2 if d['phase'] == 'contested']
        if epochs_j2:
            axes[1].plot(epochs_j2, times_j2, color=c, marker=m, label=lbl, linewidth=2, markersize=6)

    for i, (ax, title) in enumerate(zip(axes, ['Job 1 (512 MB/iter, tight SLO)',
                                                 'Job 2 (128 MB/iter, loose SLO)'])):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Epoch Iter Time (ms)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(3, 13))

    plt.suptitle('P4: Dumbbell SLO — Epoch Iteration Time (Contested Phase)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'p4_iter_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: p4_iter_time.png")


def plot_bandwidth_bars(all_data):
    """Bar chart: contested average bandwidth per mode, both jobs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(MODES))
    width = 0.35

    j1_bw = []
    j2_bw = []
    j1_solo_bw = []
    for mode in MODES:
        j1, j2 = all_data[mode]
        contested_j1 = [d['bw_obs_gbps'] for d in j1 if d['phase'] == 'contested']
        contested_j2 = [d['bw_obs_gbps'] for d in j2 if d['phase'] == 'contested']
        solo_j1 = [d['bw_obs_gbps'] for d in j1 if d['phase'] == 'solo']
        j1_bw.append(np.mean(contested_j1) if contested_j1 else 0)
        j2_bw.append(np.mean(contested_j2) if contested_j2 else 0)
        j1_solo_bw.append(np.mean(solo_j1) if solo_j1 else 0)

    bars1 = ax.bar(x - width/2, j1_bw, width, label='Job 1 (512MB)', color='#2196F3', edgecolor='black')
    bars2 = ax.bar(x + width/2, j2_bw, width, label='Job 2 (128MB)', color='#FF9800', edgecolor='black')

    # Add solo reference line
    for i, (mode, solo) in enumerate(zip(MODES, j1_solo_bw)):
        if solo:
            ax.axhline(y=solo, xmin=(i - 0.4)/len(MODES), xmax=(i + 0.4)/len(MODES),
                       color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Observed Bandwidth (Gbps)', fontsize=12)
    ax.set_title('P4: Contested Bandwidth — 5 Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'p4_bandwidth.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: p4_bandwidth.png")


def compute_ui(solo_avg, contested_data, num_iters=20):
    """Compute Ui = T_contested_per_iter / T_solo_per_iter."""
    if solo_avg <= 0:
        return None
    solo_per_iter = solo_avg / num_iters
    contested_per_iter = np.mean([d['comm_dur_s'] for d in contested_data]) / num_iters
    return contested_per_iter / solo_per_iter if solo_per_iter > 0 else None


def plot_slo_ui(all_data):
    """Plot Ui values as bar chart with SLO thresholds."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(MODES))
    width = 0.35

    j1_ui = []
    j2_ui = []
    for mode in MODES:
        j1, j2 = all_data[mode]
        solo_j1 = [d for d in j1 if d['phase'] == 'solo']
        contested_j1 = [d for d in j1 if d['phase'] == 'contested']
        contested_j2 = [d for d in j2 if d['phase'] == 'contested']

        solo_avg_j1 = np.mean([d['comm_dur_s'] for d in solo_j1]) if solo_j1 else 0
        ui1 = compute_ui(solo_avg_j1, contested_j1) if contested_j1 else None
        j1_ui.append(ui1)

        # Job2 has no solo phase; use Job1's solo as reference or estimate
        # For Ui=2.5 SLO, estimate solo bw ~ 17 Gbps
        if contested_j2:
            avg_contested_j2 = np.mean([d['comm_dur_s'] for d in contested_j2])
            # Estimate solo: assume Job2 solo would get similar bw to solo Job1 but proportionally faster
            if solo_j1 and contested_j1:
                j1_contested_avg = np.mean([d['comm_dur_s'] for d in contested_j1])
                j1_solo_avg = np.mean([d['comm_dur_s'] for d in solo_j1])
                j2_solo_est = avg_contested_j2 * (j1_solo_avg / j1_contested_avg)
            else:
                j2_solo_est = avg_contested_j2 * 1.5  # rough estimate
            ui2 = avg_contested_j2 / j2_solo_est if j2_solo_est > 0 else None
        else:
            ui2 = None
        j2_ui.append(ui2)

    bars1 = ax.bar(x - width/2, j1_ui, width, label='Job 1 (c_i=1.5)', color='#2196F3', edgecolor='black')
    bars2 = ax.bar(x + width/2, j2_ui, width, label='Job 2 (c_i=2.5)', color='#FF9800', edgecolor='black')

    # SLO threshold lines
    ax.axhline(y=SLO_J1, color='#2196F3', linestyle='--', linewidth=2, alpha=0.7, label=f'Job 1 SLO (c_i={SLO_J1})')
    ax.axhline(y=SLO_J2, color='#FF9800', linestyle='--', linewidth=2, alpha=0.7, label=f'Job 2 SLO (c_i={SLO_J2})')

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        if height:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        if height:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Ui (Contested / Solo)', fontsize=12)
    ax.set_title('P4: SLO Violation Index (Ui) — 5 Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0, top=3.0)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'p4_slo_ui.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: p4_slo_ui.png")


def main():
    # Load all data
    all_data = {}
    for mode in MODES:
        all_data[mode] = load_csv(mode)

    # Check all have data
    missing = [m for m in MODES if not all_data[m][0] or not all_data[m][1]]
    if missing:
        print(f"Missing data for modes: {missing}")
        return

    # Generate plots
    plot_iter_time_timeline(all_data)
    plot_bandwidth_bars(all_data)
    plot_slo_ui(all_data)

    # Print summary table
    print("\n" + "="*80)
    print("P4: Dumbbell SLO — Summary Table")
    print("="*80)
    print(f"{'Mode':<12} {'J1 Solo Gbps':<14} {'J1 Contested Gbps':<18} {'J2 Contested Gbps':<18} {'J1 Ui':<8}")
    print("-"*80)
    for mode in MODES:
        j1, j2 = all_data[mode]
        solo_j1 = np.mean([d['bw_obs_gbps'] for d in j1 if d['phase'] == 'solo'])
        contested_j1 = np.mean([d['bw_obs_gbps'] for d in j1 if d['phase'] == 'contested'])
        contested_j2 = np.mean([d['bw_obs_gbps'] for d in j2 if d['phase'] == 'contested'])
        ui1 = solo_j1 / contested_j1 if contested_j1 > 0 else 0
        print(f"{MODE_LABELS[mode]:<12} {solo_j1:<14.2f} {contested_j1:<18.2f} {contested_j2:<18.2f} {ui1:<8.2f}")
    print("="*80)

    print("\nDone! Generated: p4_iter_time.png, p4_bandwidth.png, p4_slo_ui.png")


if __name__ == '__main__':
    main()
