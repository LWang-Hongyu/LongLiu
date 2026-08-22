#!/usr/bin/env python3
"""
Experiment C v2 — IEEE-style Figure Generation
================================================
Generates two figures for the paper:
  1. S1 P-attn across regimes (bar chart)
  2. S2 per-job slowdown (grouped bar chart)

Uses actual data from data/expC_v2_per_round.csv
(本文件已统一迁移至 results/figures_unified/fig_expC/ 管理；
 输入数据在 ../data/expC_v2_per_round.csv，输出在 ../figures/)
IEEE style: serif fonts, clear labels, black/white/grayscale or colorblind-friendly palette
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# IEEE style settings
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.4,
    'lines.markersize': 5,
    'patch.linewidth': 0.6,
})

# Colorblind-friendly palette (from Wong 2011)
COLORS = {
    'longliu': '#0173B2',   # Blue
    'static': '#DE8F05',    # Orange  
    'fair': '#02790F',      # Green
    'longliu_light': '#A0C4E2',
    'static_light': '#F5D5A0',
    'fair_light': '#A0D5A0',
}

SCRIPT_DIR = Path(__file__).resolve().parent   # .../fig_expC/scripts
FIG_ROOT = SCRIPT_DIR.parent                    # .../fig_expC
DATA_DIR = FIG_ROOT / "data"                    # 输入数据
FIGURES_DIR = FIG_ROOT / "figures"              # 输出图片


def load_data():
    """Load per-round data from CSV."""
    csv_file = DATA_DIR / "expC_v2_per_round.csv"
    if not csv_file.exists():
        print(f"[ERROR] Data file not found: {csv_file}")
        print("Run analyze_expC_v2.py first, then copy the CSV into ../data/.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    return df


def plot_s1_pattn(df, output_dir):
    """Plot S1 P-attn across regimes as bar chart."""
    # Filter S1 data
    s1_df = df[df['regime'].str.startswith('S1_')]
    
    # Aggregate by regime and arm
    agg = s1_df.groupby(['regime', 'arm']).agg({
        'slowdown_mean': ['mean', 'std']
    }).reset_index()
    agg.columns = ['regime', 'arm', 'sd_mean', 'sd_std']
    
    # Compute P-attn per regime-arm combination
    # P-attn = sum of max(0, SD-1) for premium jobs
    pattn_data = []
    regimes = ['S1_ample', 'S1_moderate', 'S1_deep', 'S1_very_deep']
    arms = ['longliu', 'static', 'fair']
    
    for regime in regimes:
        regime_df = s1_df[s1_df['regime'] == regime]
        for arm in arms:
            arm_df = regime_df[regime_df['arm'] == arm]
            prem_df = arm_df[arm_df['tier'] == 'premium']
            
            # Per-round P-attn calculation
            by_round = prem_df.groupby('round')
            pattn_values = []
            for rnd, rnd_df in by_round:
                pattn = sum(max(0, row['slowdown_mean'] - 1.0) for _, row in rnd_df.iterrows())
                pattn_values.append(pattn)
            
            arm_display = {'longliu': 'LongLiu', 'static': 'Static', 'fair': 'Fair'}[arm]
            pattn_data.append({
                'regime': regime.replace('S1_', '').replace('_', ' ').title(),
                'arm': arm_display,
                'pattn_mean': np.mean(pattn_values),
                'pattn_std': np.std(pattn_values),
            })
    
    pattn_df = pd.DataFrame(pattn_data)
    
    # Plot — wider canvas, legend below
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    
    regimes_order = ['Ample', 'Moderate', 'Deep', 'Very Deep']
    arms_order = ['LongLiu', 'Static', 'Fair']
    arm_colors = [COLORS['longliu'], COLORS['static'], COLORS['fair']]
    x = np.arange(len(regimes_order))
    width = 0.25
    
    for i, (arm, color) in enumerate(zip(arms_order, arm_colors)):
        arm_data = pattn_df[pattn_df['arm'] == arm]
        arm_data = arm_data.set_index('regime').loc[regimes_order].reset_index()
        
        bars = ax.bar(x + i * width, arm_data['pattn_mean'], width,
                      label=arm, color=color, edgecolor='black', linewidth=0.6)
        
        # Add error bars
        ax.errorbar(x + i * width, arm_data['pattn_mean'], yerr=arm_data['pattn_std'],
                    fmt='none', ecolor='black', elinewidth=1.0, capsize=3)
    
    ax.set_ylabel('P-attn')
    ax.set_xlabel('Regime')
    ax.set_xticks(x + width)
    ax.set_xticklabels(regimes_order)
    # Legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.38),
              frameon=True, framealpha=0.95, ncol=3, edgecolor='#CCCCCC')
    ax.set_ylim(0, 0.75)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    fig.subplots_adjust(bottom=0.34)
    output_file = output_dir / "fig_expC_s1_pattn.pdf"
    plt.savefig(output_file)
    plt.savefig(output_file.with_suffix('.png'))
    print(f"[plot] Saved {output_file}")
    
    return pattn_df


def plot_s2_perjob(df, output_dir):
    """Plot S2 per-job slowdown as grouped bar chart."""
    # Filter S2 data
    s2_df = df[df['regime'] == 'S2_starvation']
    
    # Aggregate by job and arm
    agg = s2_df.groupby(['job_id', 'arm']).agg({
        'slowdown_mean': ['mean', 'std'],
        'tier': 'first',
        'label': 'first'
    }).reset_index()
    agg.columns = ['job_id', 'arm', 'sd_mean', 'sd_std', 'tier', 'label']
    
    # Plot — wider canvas, legend below
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    
    jobs = sorted(agg['job_id'].unique())
    arms = ['fair', 'longliu', 'static']
    arm_labels = ['Fair', 'LongLiu', 'Static']
    arm_colors = [COLORS['fair'], COLORS['longliu'], COLORS['static']]
    x = np.arange(len(jobs))
    width = 0.25
    
    # Background color blocks for Premium / Standard tiers
    # Bars: 3 arms at offsets 0, 0.25, 0.5, width=0.25
    #   J0 spans [-0.125, 0.625], J5 spans [4.875, 5.625]
    # Color blocks tightly wrap the bar groups, split at x=2.75
    ax.set_xlim(-0.125, 5.625)
    ax.axvspan(-0.125, 2.75, alpha=0.06, color='#1565C0', zorder=0)
    ax.axvspan(2.75, 5.625, alpha=0.06, color='#E65100', zorder=0)
    
    for i, (arm, label, color) in enumerate(zip(arms, arm_labels, arm_colors)):
        arm_data = agg[agg['arm'] == arm].sort_values('job_id')
        
        bars = ax.bar(x + i * width, arm_data['sd_mean'], width,
                      label=label, color=color, edgecolor='black', linewidth=0.6)
        
        # Add error bars
        ax.errorbar(x + i * width, arm_data['sd_mean'], yerr=arm_data['sd_std'],
                    fmt='none', ecolor='black', elinewidth=1.0, capsize=3)
    
    # Tier labels inside the colored blocks
    ax.text(1.0, 3.55, 'Premium', ha='center', fontsize=11, fontweight='bold',
            color='#1565C0')
    ax.text(4.0, 3.55, 'Standard', ha='center', fontsize=11, fontweight='bold',
            color='#E65100')
    
    # Job labels with tier indicator
    job_labels = ['J0\n(P)', 'J1\n(P)', 'J2\n(P)', 'J3\n(S)', 'J4\n(S)', 'J5\n(S)']
    ax.set_xticks(x + width)
    ax.set_xticklabels(job_labels)
    
    ax.set_ylabel('Slowdown')
    ax.set_xlabel('Job')
    # Legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
              frameon=True, framealpha=0.95, ncol=3, edgecolor='#CCCCCC')
    ax.set_ylim(0, 3.9)
    
    # Add c_eval reference line — label inside plot area
    ax.axhline(y=1.5, color='red', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.text(5.65, 1.55, r'$c_{eval}$', color='red', fontsize=10, ha='right',
            fontweight='bold')
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    fig.subplots_adjust(bottom=0.24)
    output_file = output_dir / "fig_expC_s2_perjob.pdf"
    plt.savefig(output_file)
    plt.savefig(output_file.with_suffix('.png'))
    print(f"[plot] Saved {output_file}")
    
    return agg


def plot_s1_premium_sd(df, output_dir):
    """Plot S1 premium slowdown across regimes (line chart for comparison with E1)."""
    # Filter S1 premium data
    s1_df = df[(df['regime'].str.startswith('S1_')) & (df['tier'] == 'premium')]
    
    # Aggregate by regime and arm
    agg = s1_df.groupby(['regime', 'arm']).agg({
        'slowdown_mean': ['mean', 'std']
    }).reset_index()
    agg.columns = ['regime', 'arm', 'sd_mean', 'sd_std']
    
    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    regimes_order = ['S1_ample', 'S1_moderate', 'S1_deep', 'S1_very_deep']
    regime_labels = ['Ample', 'Moderate', 'Deep', 'Very Deep']
    arms = ['longliu', 'static', 'fair']
    arm_labels = ['LongLiu', 'Static', 'Fair']
    
    for arm, label in zip(arms, arm_labels):
        arm_data = agg[agg['arm'] == arm]
        arm_data = arm_data.set_index('regime').loc[regimes_order].reset_index()
        
        ax.errorbar(regime_labels, arm_data['sd_mean'], yerr=arm_data['sd_std'],
                    marker='o', label=label, color=COLORS[arm], linewidth=1.5,
                    capsize=3)
    
    ax.set_ylabel('Premium Slowdown')
    ax.set_xlabel('Regime')
    ax.legend(loc='upper left', frameon=False)
    ax.set_ylim(0.9, 1.5)
    ax.axhline(y=1.5, color='red', linestyle=':', linewidth=0.8, alpha=0.7, label='$c_{eval}$')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_file = output_dir / "fig_expC_s1_premium_sd.pdf"
    plt.savefig(output_file)
    plt.savefig(output_file.with_suffix('.png'))
    print(f"[plot] Saved {output_file}")


def main():
    print("=" * 60)
    print("Experiment C v2 — IEEE-style Figure Generation")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"[load] Loaded {len(df)} rows from expC_v2_per_round.csv")
    
    # Generate figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[plot] Generating S1 P-attn bar chart...")
    plot_s1_pattn(df, FIGURES_DIR)
    
    print("\n[plot] Generating S2 per-job slowdown chart...")
    plot_s2_perjob(df, FIGURES_DIR)
    
    print("\n[plot] Generating S1 premium slowdown line chart...")
    plot_s1_premium_sd(df, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
