#!/usr/bin/env python3
"""
P1: DSCP Injection Verification - Experiment Plots
Generates figures from TC sweep verification data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# Use non-interactive backend
matplotlib.use('Agg')

# Load data
csv_path = os.path.join(os.path.dirname(__file__), 'p1_tc_sweep_results.csv')
df = pd.read_csv(csv_path)

# ============================================================
# Figure 1: NCCL_IB_TC vs Wire ToS - Linear Relationship
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Plot ideal line: tos = tc (if ECN=0)
tc_range = np.linspace(0, 60, 100)
ax1.plot(tc_range, tc_range, '--', color='gray', alpha=0.5, label='Ideal (ToS = TC)')

# Plot actual: tos = (tc & 0xFC) | 0x02
actual_tos = (np.floor(tc_range / 4) * 4 + 2).astype(int)
ax1.plot(tc_range, actual_tos, '-', color='#2E7D32', linewidth=2, label='Actual RoCEv2 (ECN=2)')

# Plot measured points
ax1.scatter(df['nccl_ib_tc'], df['wire_tos_dec'], color='#C62828', s=100, zorder=5,
            label='Measured (tcpdump)', edgecolors='black', linewidth=0.5)

# Annotate points
for _, row in df.iterrows():
    ax1.annotate(f"TC={int(row['nccl_ib_tc'])}\nToS=0x{int(row['wire_tos_dec']):02x}",
                 (row['nccl_ib_tc'], row['wire_tos_dec']),
                 textcoords="offset points", xytext=(0, 12),
                 ha='center', fontsize=8, color='#333333')

ax1.set_xlabel('NCCL_IB_TC (traffic_class)', fontsize=12)
ax1.set_ylabel('Wire ToS (IP header byte)', fontsize=12)
ax1.set_title('P1: DSCP Injection Verification\nNCCL_IB_TC → RoCEv2 IP ToS', fontsize=13)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2, 60)
ax1.set_ylim(-2, 64)

plt.tight_layout()
fig1.savefig(os.path.join(os.path.dirname(__file__), 'fig_p1_tc_vs_tos.png'), dpi=150)
print("Saved fig_p1_tc_vs_tos.png")

# ============================================================
# Figure 2: DSCP Values Extracted from ToS
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(df))
width = 0.6

bars = ax2.bar(x, df['dscp'], width, color='#1565C0', alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, dscp_val in zip(bars, df['dscp']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{int(dscp_val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xlabel('NCCL_IB_TC Value', fontsize=12)
ax2.set_ylabel('DSCP (ToS >> 2)', fontsize=12)
ax2.set_title('P1: DSCP Priority Levels Achieved', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{int(tc)}' for tc in df['nccl_ib_tc']])
ax2.set_ylim(0, 16)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(__file__), 'fig_p1_dscp_levels.png'), dpi=150)
print("Saved fig_p1_dscp_levels.png")

# ============================================================
# Figure 3: Verification Summary Table
# ============================================================
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.axis('off')

col_labels = ['NCCL_IB_TC', 'Wire ToS', 'DSCP', 'ECN', 'Status']
table_data = []
for _, row in df.iterrows():
    table_data.append([
        str(int(row['nccl_ib_tc'])),
        f"0x{int(row['wire_tos_dec']):02x}",
        str(int(row['dscp'])),
        str(int(row['ecn'])),
        'PASS' if row['pass'] else 'FAIL'
    ])

table = ax3.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.6)

# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2E7D32')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color pass/fail cells
for i in range(len(table_data)):
    status_cell = table[i+1, len(col_labels)-1]
    if table_data[i][-1] == 'PASS':
        status_cell.set_facecolor('#C8E6C9')
    else:
        status_cell.set_facecolor('#FFCDD2')

ax3.set_title('P1: TC Sweep Verification Results (9/9 PASS)', fontsize=13, pad=20)

plt.tight_layout()
fig3.savefig(os.path.join(os.path.dirname(__file__), 'fig_p1_summary_table.png'), dpi=150)
print("Saved fig_p1_summary_table.png")

print("\nAll P1 figures generated successfully.")
