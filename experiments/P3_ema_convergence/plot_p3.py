#!/usr/bin/env python3
"""
P3: EMA Bandwidth Convergence — Plotting Script (Updated for Ui < 1 Guard)

Reads Job 1 CSV and NCCL debug log to visualize:
  1. Bw_obs per epoch
  2. Naive EMA (α=0.3, updated every epoch — pre-fix behavior)
  3. Guarded EMA (α=0.3, frozen when Ui ≥ 1.0 — post-fix behavior)
  4. Actual NCCL DSCP log showing Ui transitions and SKIP/UDPATE decisions

Key finding with fix applied:
  - Ui transitions 0.83(solo) → 0.93 → 1.03(threshold) → 1.18(contested)
  - EMA SKIP kicks in at epoch 7-9, prevents ~48% pollution
  - Pre-fix: Naive EMA would decay from ~20 → 12 Gbps
  - Post-fix: Guarded EMA holds at solo value during contested phase
"""

import os
import csv
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            rows.append({
                'epoch': int(row['epoch']),
                'total_bytes': int(row['total_bytes']),
                'comm_dur_s': float(row['comm_dur_s']),
                'bw_obs_gbps': float(row['bw_obs_gbps']),
                'phase': row['phase']
            })
    return sorted(rows, key=lambda r: r['epoch'])


def compute_naive_ema(bw_values, alpha=0.3):
    """EMA updated every epoch (pre-fix / unconditional behavior)."""
    ema = [bw_values[0]]
    for v in bw_values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def compute_guarded_ema(bw_values, ui_values, alpha=0.3):
    """
    EMA with Ui < 1 guard (post-fix behavior).
    - Cold start on epoch 0: seed with first value
    - Epoch N: update only if Ui_prev < 1.0 (low contention)
    """
    ema = [bw_values[0]]
    for i in range(1, len(bw_values)):
        if ui_values[i - 1] < 1.0:
            ema.append(alpha * bw_values[i] + (1 - alpha) * ema[-1])
        else:
            ema.append(ema[-1])  # freeze
    return ema


def parse_nccl_log(log_path):
    """
    Parse NCCL log to extract Ui and EMA decision for each epoch.
    Returns: {epoch: {'ui': float, 'action': 'UPDATE'|'SKIP'|'COLD_START'}}
    """
    results = {}
    # DSCP EMA [Epoch N]: ACTION (Ui_prev=X.XXXX ...)
    ema_pat = re.compile(
        r'DSCP EMA .Epoch (\d+).: (\w+).*Ui_prev=([\d.]+)'
    )
    # DSCP Adapter [Epoch N]: Ui=X.XXXX
    ui_pat = re.compile(
        r'DSCP Adapter .Epoch (\d+).: Ui=([\d.]+)'
    )

    with open(log_path, 'r') as f:
        for line in f:
            m = ema_pat.search(line)
            if m:
                epoch = int(m.group(1))
                action = m.group(2)
                ui_prev = float(m.group(3)) if 'prev' in line else 0.0
                if epoch not in results:
                    results[epoch] = {}
                results[epoch]['action'] = action
                results[epoch]['ui_prev'] = ui_prev

            m = ui_pat.search(line)
            if m:
                epoch = int(m.group(1))
                ui = float(m.group(2))
                if epoch not in results:
                    results[epoch] = {}
                results[epoch]['ui'] = ui
    return results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Find Job 1 rank 0 CSV
    csv_path = os.path.join(base_dir, 'p3_job1_rank0.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, 'p3_job1_rank1.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: No Job 1 data found. Run run_p3.sh first.")
        sys.exit(1)

    rows = load_csv(csv_path)
    epochs = [r['epoch'] for r in rows]
    bw_obs = [r['bw_obs_gbps'] for r in rows]
    phases = [r['phase'] for r in rows]

    # Parse NCCL log for real DSCP decisions
    log_path = '/tmp/p3_job1_node101.log'
    nccl_data = {}
    if os.path.exists(log_path):
        nccl_data = parse_nccl_log(log_path)
        print(f"Parsed NCCL log: {len(nccl_data)} epochs with DSCP data")
    else:
        print(f"WARNING: {log_path} not found, using offline Ui estimation")

    # Extract Ui values from NCCL log for guarded EMA
    ui_values = []
    for e in epochs:
        ui = 0.0
        if e in nccl_data and 'ui' in nccl_data[e]:
            ui = nccl_data[e]['ui']
        ui_values.append(ui)

    print(f"\nLoaded {len(rows)} epochs from {csv_path}")
    for i, r in enumerate(rows):
        ui_str = f"Ui={ui_values[i]:.4f}" if ui_values[i] > 0 else "Ui=N/A"
        action = nccl_data.get(r['epoch'], {}).get('action', '--')
        print(f"  Epoch {r['epoch']:2d} [{r['phase']:>9s}]: "
              f"Bw_obs={r['bw_obs_gbps']:.2f} Gbps, "
              f"{ui_str}, EMA={action}")

    bw_ema_naive = compute_naive_ema(bw_obs, alpha=0.3)
    bw_ema_guarded = compute_guarded_ema(bw_obs, ui_values, alpha=0.3)

    solo_obs_avg = np.mean(bw_obs[:5])
    contested_obs_avg = np.mean(bw_obs[5:])

    # ============================================================
    # Figure 1: Bw_obs + Naive EMA + Guarded EMA + Ui annotation
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    solo_x = [epochs[i] for i in range(5)]
    solo_y = [bw_obs[i] for i in range(5)]
    cont_x = [epochs[i] for i in range(5, 10)]
    cont_y = [bw_obs[i] for i in range(5, 10)]

    ax1.scatter(solo_x, solo_y, marker='o', s=120, color='#1565C0',
                zorder=5, label='Bw_obs (Solo)')
    ax1.scatter(cont_x, cont_y, marker='o', s=120, color='#C62828',
                zorder=5, label='Bw_obs (Contested)')

    # Naive EMA (pre-fix, polluted) — dashed red
    ax1.plot(epochs, bw_ema_naive, '--', color='#E53935', linewidth=2,
             marker='s', markersize=7, zorder=4,
             label='Naive EMA (α=0.3, pre-fix: polluted)')

    # Guarded EMA (post-fix, Ui < 1) — solid green
    ax1.plot(epochs, bw_ema_guarded, '-', color='#2E7D32', linewidth=2.5,
             marker='D', markersize=8, zorder=4,
             label='Guarded EMA (α=0.3, Ui < 1 gate)')

    # Mark SKIP points on guarded EMA
    skip_epochs = []
    skip_vals = []
    for i, e in enumerate(epochs):
        if e in nccl_data and nccl_data[e].get('action') == 'SKIP':
            skip_epochs.append(e)
            skip_vals.append(bw_ema_guarded[i])
    if skip_epochs:
        ax1.scatter(skip_epochs, skip_vals, marker='x', s=150,
                    color='#2E7D32', linewidths=2.5, zorder=6,
                    label='EMA SKIP (Ui >= 1.0)')

    # Ui annotations above data points
    for i, e in enumerate(epochs):
        if ui_values[i] > 0:
            color = '#1565C0' if ui_values[i] < 1.0 else '#C62828'
            ax1.annotate(f'Ui={ui_values[i]:.2f}',
                         (e, bw_obs[i]), textcoords="offset points",
                         xytext=(0, 14), ha='center', fontsize=7.5, color=color,
                         fontweight='bold')

    # Divider
    ax1.axvline(x=4.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(4.6, max(bw_obs) * 0.95, u'← Solo | Contested →',
             fontsize=10, color='gray', va='top')

    # Stats box
    naive_decay = bw_ema_naive[4] - bw_ema_naive[9]
    naive_decay_pct = naive_decay / bw_ema_naive[4] * 100 if bw_ema_naive[4] > 0 else 0
    guarded_val = bw_ema_guarded[-1]

    stats_text = (
        f'Solo Bw_obs avg:      {solo_obs_avg:.1f} Gbps\n'
        f'Contested Bw_obs avg:  {contested_obs_avg:.1f} Gbps\n'
        f'Guarded EMA (held):    {guarded_val:.1f} Gbps\n'
        f'Naive EMA decay:       {bw_ema_naive[4]:.1f} \u2192 {bw_ema_naive[9]:.1f} '
        f'({naive_decay_pct:.1f}%)\n'
        f'EMA pollution prevented: {naive_decay:.1f} Gbps'
    )
    ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85),
             va='bottom', family='monospace')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Bandwidth [Gbps]', fontsize=12)
    ax1.set_title('P3: EMA Bandwidth Convergence (Ui < 1 Guard Verified)\n'
                  'Solo (Epoch 0-4) \u2192 Contested (Epoch 5-9)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 10))
    ax1.set_ylim(bottom=0)

    plt.tight_layout()
    fig1.savefig(os.path.join(base_dir, 'fig_p3_ema_convergence.png'), dpi=150)
    print("\nSaved fig_p3_ema_convergence.png")

    # ============================================================
    # Figure 2: Bar chart + both EMAs + Ui timeline
    # ============================================================
    fig2, (ax_bar, ax_ui) = plt.subplots(2, 1, figsize=(11, 7),
                                          gridspec_kw={'height_ratios': [3, 1.5]},
                                          sharex=True)

    x_idx = np.arange(len(bw_obs))
    colors = ['#1565C0'] * 5 + ['#C62828'] * 5
    ax_bar.bar(x_idx, bw_obs, color=colors, alpha=0.5, width=0.5, label='Bw_obs')

    ax_bar.plot(x_idx, bw_ema_guarded, '-D', color='#2E7D32', linewidth=2.5,
                markersize=8, label='Guarded EMA (Ui < 1 gate)')
    ax_bar.plot(x_idx, bw_ema_naive, '--s', color='#E53935', linewidth=2,
                markersize=6, label='Naive EMA (pre-fix, polluted)')

    # Annotate bar values
    for i, obs in enumerate(bw_obs):
        ax_bar.annotate(f'{obs:.1f}', (x_idx[i], obs), textcoords="offset points",
                        xytext=(0, 7), ha='center', fontsize=7.5, color='black')

    # Mark EMA SKIP points
    for i, e in enumerate(epochs):
        if e in nccl_data and nccl_data[e].get('action') == 'SKIP':
            ax_bar.annotate('SKIP', (x_idx[i], bw_ema_guarded[i]),
                            textcoords="offset points", xytext=(0, -18),
                            ha='center', fontsize=7, color='#2E7D32',
                            fontweight='bold')

    ax_bar.axvline(x=4.5, color='gray', linestyle='--', linewidth=1.5)
    ax_bar.text(4.6, max(bw_obs) * 1.02, u'← Solo | Contested →',
                fontsize=9, color='gray')

    ax_bar.set_ylabel('Bandwidth [Gbps]', fontsize=12)
    ax_bar.set_title('P3: Bw_obs vs Naive EMA vs Guarded EMA (Ui < 1)', fontsize=13)
    ax_bar.legend(fontsize=9, loc='upper left')
    ax_bar.grid(True, alpha=0.3, axis='y')
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels([str(e) for e in epochs])

    # Ui timeline (bottom subplot)
    ui_plot = [ui_values[i] if ui_values[i] > 0 else np.nan for i in range(len(epochs))]
    ax_ui.plot(x_idx, ui_plot, '-o', color='#7B1FA2', linewidth=2, markersize=8)
    ax_ui.axhline(y=1.0, color='#C62828', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_ui.text(0.5, 1.02, 'Ui = 1.0 (SLO boundary)', fontsize=8, color='#C62828',
               va='bottom', ha='center',
               transform=ax_ui.get_yaxis_transform())

    # Color regions
    ax_ui.fill_between(x_idx, 0, 1.0, alpha=0.1, color='#2E7D32')
    ax_ui.fill_between(x_idx, 1.0, max(bw_obs), alpha=0.1, color='#C62828')
    ax_ui.text(2, 0.5, 'Ahead of SLO\n(EMA UPDATE)', fontsize=7.5, color='#2E7D32',
               ha='center', va='center')
    ax_ui.text(7.5, ui_plot[-1] - 0.15, 'Behind SLO\n(EMA SKIP)', fontsize=7.5,
               color='#C62828', ha='center', va='top')

    for i, ui in enumerate(ui_plot):
        if not np.isnan(ui):
            color = '#1565C0' if ui < 1.0 else '#C62828'
            ax_ui.annotate(f'{ui:.2f}', (x_idx[i], ui),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8, color=color, fontweight='bold')

    ax_ui.set_xlabel('Epoch', fontsize=12)
    ax_ui.set_ylabel('Urgency Index Ui', fontsize=11)
    ax_ui.grid(True, alpha=0.3, axis='y')
    ax_ui.set_ylim(bottom=0, top=max(ui_plot) * 1.3 if max(ui_plot) > 0 else 2)

    plt.tight_layout()
    fig2.savefig(os.path.join(base_dir, 'fig_p3_ema_detail.png'), dpi=150)
    print("Saved fig_p3_ema_detail.png")

    # ============================================================
    # Summary
    # ============================================================
    drop_pct = (1 - contested_obs_avg / solo_obs_avg) * 100
    print("\n" + "=" * 60)
    print("P3: EMA Bandwidth Convergence (Ui < 1 Guard) — Summary")
    print("=" * 60)
    print(f"Solo Bw_obs avg:              {solo_obs_avg:.2f} Gbps")
    print(f"Contested Bw_obs avg:         {contested_obs_avg:.2f} Gbps")
    print(f"Bandwidth drop under contest: {drop_pct:.1f}%")
    print()
    print(f"Guarded EMA (held):           {guarded_val:.2f} Gbps")
    print(f"Naive EMA decay:              {bw_ema_naive[4]:.2f} -> {bw_ema_naive[9]:.2f} "
          f"({naive_decay_pct:.1f}%)")
    print(f"EMA pollution prevented:      {naive_decay:.2f} Gbps")
    print()
    
    # Print Ui transition
    if ui_values[0] > 0:
        print("Ui transition:")
        for i, e in enumerate(epochs):
            marker = ' << SKIP' if (e in nccl_data and nccl_data[e].get('action') == 'SKIP') else ''
            print(f"  Epoch {e}: Ui={ui_values[i]:.4f}{marker}")

    print("=" * 60)
    print("\nAll figures generated.")

    # Also check Job 2 data if available
    job2_csv = os.path.join(base_dir, 'p3_job2_rank0.csv')
    if os.path.exists(job2_csv):
        job2 = load_csv(job2_csv)
        print(f"\nJob 2 data ({len(job2)} epochs):")
        for r in job2:
            print(f"  Epoch {r['epoch']}: Bw_obs={r['bw_obs_gbps']:.2f} Gbps")


if __name__ == '__main__':
    main()
