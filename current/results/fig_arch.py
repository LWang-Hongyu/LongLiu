#!/usr/bin/env python3
"""
LongLiu Architecture Diagram — IEEE Publication Quality
========================================================
Two-panel architecture figure:
  (a) System deployment: GPU nodes -> NCCL Shim -> Switch (SPQ)
  (b) Per-job control loop: Measure -> pi -> DSCP -> Enforce -> Feedback
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# ── IEEE style ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'DejaVu Serif'],
    'font.size': 8,
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})

# ── Colors (Wong 2011 colorblind-friendly) ───────────────────
C = {
    'bg_node':     '#E8EEF4',
    'bg_shim':     '#FFF3E0',
    'bg_switch':   '#E8F5E9',
    'bg_queue':    ['#1565C0', '#1976D2', '#64B5F6', '#90CAF9'],
    'border_node': '#546E7A',
    'border_shim': '#E65100',
    'border_sw':   '#2E7D32',
    'arrow_dscp':  '#D32F2F',
    'arrow_fb':    '#1565C0',
    'text_dark':   '#212121',
    'text_mid':    '#424242',
    'text_light':  '#757575',
    'accent':      '#FF6F00',
    'nic':         '#E3F2FD',
    'pi_box':      '#FFF3E0',
}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR


def rbox(ax, xy, w, h, fc, ec, lw=0.8, zorder=2):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


# ════════════════════════════════════════════════════════════
# Panel (a): System Deployment View
# ════════════════════════════════════════════════════════════
def draw_panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5, 6.78, '(a) System Deployment', ha='center', va='top',
            fontsize=9, fontweight='bold', color=C['text_dark'])

    # ── GPU Node 1 (Premium) ──
    n1_x, n1_y, n1_w, n1_h = 0.3, 3.3, 4.2, 3.1
    rbox(ax, (n1_x, n1_y), n1_w, n1_h, C['bg_node'], C['border_node'], lw=1.0)
    ax.text(n1_x + n1_w/2, n1_y + n1_h - 0.12, 'GPU Node 1  (Premium, $c_i{=}1.2$)',
            ha='center', va='top', fontsize=7, fontweight='bold', color=C['text_dark'])

    # PyTorch
    rbox(ax, (n1_x+0.3, n1_y+1.7), 3.6, 0.9, '#F5F5F5', '#9E9E9E', lw=0.6)
    ax.text(n1_x+2.1, n1_y+2.35, 'PyTorch DDP Training', ha='center', va='center',
            fontsize=6.5, color=C['text_dark'])

    # Shim (highlighted)
    rbox(ax, (n1_x+0.3, n1_y+0.2), 3.6, 1.3, C['bg_shim'], C['border_shim'], lw=1.2)
    ax.text(n1_x+2.1, n1_y+1.2, 'LongLiu NCCL Shim', ha='center', va='center',
            fontsize=7, fontweight='bold', color=C['border_shim'])
    # pi result
    ax.text(n1_x+0.6, n1_y+0.65, r'$\pi{=}+0.35$', ha='left', va='center',
            fontsize=7, color=C['accent'], fontweight='bold')
    ax.text(n1_x+2.2, n1_y+0.65, r'$\rightarrow$ P6 (DSCP 48)', ha='left', va='center',
            fontsize=6, color=C['border_shim'])

    # ── GPU Node 2 (Standard) ──
    n2_x, n2_y, n2_w, n2_h = 5.5, 3.3, 4.2, 3.1
    rbox(ax, (n2_x, n2_y), n2_w, n2_h, C['bg_node'], C['border_node'], lw=1.0)
    ax.text(n2_x + n2_w/2, n2_y + n2_h - 0.12, 'GPU Node 2  (Standard, $c_i{=}2.0$)',
            ha='center', va='top', fontsize=7, fontweight='bold', color=C['text_dark'])

    rbox(ax, (n2_x+0.3, n2_y+1.7), 3.6, 0.9, '#F5F5F5', '#9E9E9E', lw=0.6)
    ax.text(n2_x+2.1, n2_y+2.35, 'PyTorch DDP Training', ha='center', va='center',
            fontsize=6.5, color=C['text_dark'])

    rbox(ax, (n2_x+0.3, n2_y+0.2), 3.6, 1.3, C['bg_shim'], C['border_shim'], lw=1.2)
    ax.text(n2_x+2.1, n2_y+1.2, 'LongLiu NCCL Shim', ha='center', va='center',
            fontsize=7, fontweight='bold', color=C['border_shim'])
    ax.text(n2_x+0.6, n2_y+0.65, r'$\pi{=}{-}0.12$', ha='left', va='center',
            fontsize=7, color='#1565C0', fontweight='bold')
    ax.text(n2_x+2.2, n2_y+0.65, r'$\rightarrow$ P2 (DSCP 16)', ha='left', va='center',
            fontsize=6, color=C['border_shim'])

    # ── Arrows: Node -> Switch ──
    ax.annotate('', xy=(3.5, 2.95), xytext=(3.5, 3.25),
                arrowprops=dict(arrowstyle='->', color=C['arrow_dscp'], lw=1.8))
    ax.text(4.3, 3.1, 'DSCP 48', ha='left', va='center', fontsize=5.5,
            color=C['arrow_dscp'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.08', fc='white', ec=C['arrow_dscp'], lw=0.4))

    ax.annotate('', xy=(7.0, 2.95), xytext=(7.0, 3.25),
                arrowprops=dict(arrowstyle='->', color=C['arrow_dscp'], lw=1.2))
    ax.text(7.8, 3.1, 'DSCP 16', ha='left', va='center', fontsize=5.5,
            color=C['arrow_dscp'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.08', fc='white', ec=C['arrow_dscp'], lw=0.4))

    ax.text(5.0, 3.2, 'RoCEv2', ha='center', va='bottom', fontsize=5.5,
            color=C['text_light'], style='italic')

    # ── Switch with Priority Queues ──
    sw_x, sw_y, sw_w, sw_h = 1.0, 0.2, 8.0, 2.5
    rbox(ax, (sw_x, sw_y), sw_w, sw_h, C['bg_switch'], C['border_sw'], lw=1.2)
    ax.text(sw_x+sw_w/2, sw_y+sw_h-0.1,
            'Commodity Switch \u2014 Strict Priority Queuing (SPQ)',
            ha='center', va='top', fontsize=7, fontweight='bold', color=C['border_sw'])

    # Queues
    q_labels = ['P6', 'P4', 'P2', 'P1']
    q_dscp  = ['DSCP 48', 'DSCP 32', 'DSCP 16', 'DSCP 8']
    q_h     = [1.5, 1.1, 0.8, 0.55]
    for i, (lbl, dscp, qh) in enumerate(zip(q_labels, q_dscp, q_h)):
        qx = sw_x + 0.5 + i * 2.0
        qy = sw_y + 0.15
        qw = 1.5
        qb = FancyBboxPatch((qx, qy), qw, qh, boxstyle="round,pad=0.04",
                            facecolor=C['bg_queue'][i], edgecolor='#37474F',
                            linewidth=0.6, alpha=0.85, zorder=3)
        ax.add_patch(qb)
        ax.text(qx+qw/2, qy+qh/2+0.12, lbl, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white')
        ax.text(qx+qw/2, qy+qh/2-0.12, dscp, ha='center', va='center',
                fontsize=5, color='#E0E0E0')
        # traffic dots
        if lbl == 'P6':
            for dx in [-0.25, 0, 0.25]:
                ax.plot(qx+qw/2+dx, qy+0.18, 'o', color='white', ms=2.5, alpha=0.8)
        elif lbl == 'P2':
            ax.plot(qx+qw/2, qy+0.15, 'o', color='white', ms=2, alpha=0.8)

    # SP ordering
    ax.annotate('', xy=(8.3, 1.85), xytext=(1.7, 1.85),
                arrowprops=dict(arrowstyle='->', color=C['border_sw'], lw=0.7, linestyle='--'))
    ax.text(5.0, 1.95, 'strict priority order', ha='center', va='bottom',
            fontsize=5, color=C['border_sw'], style='italic')

    # BW allocation
    ax.annotate('', xy=(2.3, 0.15), xytext=(2.3, sw_y),
                arrowprops=dict(arrowstyle='->', color=C['accent'], lw=1.3))
    ax.text(2.3, 0.08, 'More BW', ha='center', va='top', fontsize=5, color=C['accent'], fontweight='bold')

    ax.annotate('', xy=(6.3, 0.15), xytext=(6.3, sw_y),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=0.8))
    ax.text(6.3, 0.08, 'Less BW', ha='center', va='top', fontsize=5, color='#1565C0')

    # Feedback
    ax.annotate('', xy=(9.5, 4.6), xytext=(9.5, 2.0),
                arrowprops=dict(arrowstyle='->', color=C['arrow_fb'], lw=1.0))
    ax.text(9.65, 3.6, 'Feedback:', ha='left', va='center', fontsize=5, color=C['arrow_fb'], fontweight='bold')
    ax.text(9.65, 3.25, 'BW change', ha='left', va='center', fontsize=4.5, color=C['arrow_fb'])
    ax.text(9.65, 2.95, r'$\rightarrow$ iter time', ha='left', va='center', fontsize=4.5, color=C['arrow_fb'])
    ax.text(9.65, 2.65, r'$\rightarrow$ $\pi$ update', ha='left', va='center', fontsize=4.5, color=C['arrow_fb'])

    # No coordination
    ax.text(5.0, 2.72, 'No cross-job coordination', ha='center', va='center',
            fontsize=5.5, color=C['text_light'], style='italic',
            bbox=dict(boxstyle='round,pad=0.1', fc='#FFF9C4', ec='#F9A825', lw=0.4, alpha=0.9))


# ════════════════════════════════════════════════════════════
# Panel (b): Per-Job Control Loop
# ════════════════════════════════════════════════════════════
def draw_panel_b(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5, 5.05, '(b) Per-Job Closed-Loop Control', ha='center', va='top',
            fontsize=9, fontweight='bold', color=C['text_dark'])

    # ── Step boxes ──
    # 1. Measure
    bx1, by1 = 0.3, 3.0
    rbox(ax, (bx1, by1), 1.9, 1.5, C['nic'], '#1565C0', lw=0.9)
    ax.text(bx1+0.95, by1+1.3, '1. Measure', ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='#1565C0')
    ax.text(bx1+0.95, by1+0.85, 'Epoch end:', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(bx1+0.95, by1+0.55, 'aggregate comm', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(bx1+0.95, by1+0.3, 'time + bytes', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])

    # 2. Compute pi
    bx2, by2 = 2.7, 3.0
    rbox(ax, (bx2, by2), 2.1, 1.5, C['pi_box'], C['border_shim'], lw=1.2)
    ax.text(bx2+1.05, by2+1.3, '2. Compute', ha='center', va='center',
            fontsize=6.5, fontweight='bold', color=C['border_shim'])
    # pi formula (pure math, no mixing)
    ax.text(bx2+1.05, by2+0.8, r'$\pi = \frac{A_i}{c_i \cdot T^{\mathrm{tgt}} \cdot k_i} - 1$',
            ha='center', va='center', fontsize=7.5, color=C['accent'])
    ax.text(bx2+1.05, by2+0.35, r'$\pi{>}0$: under-fed  |  $\pi{<}0$: over-fed',
            ha='center', va='center', fontsize=5, color=C['text_mid'])

    # 3. Map priority
    bx3, by3 = 5.3, 3.0
    rbox(ax, (bx3, by3), 1.9, 1.5, C['bg_shim'], '#7B1FA2', lw=0.9)
    ax.text(bx3+0.95, by3+1.3, '3. Map', ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='#7B1FA2')
    ax.text(bx3+0.95, by3+0.95, r'$\pi{>}0.3$ $\rightarrow$ P6', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(bx3+0.95, by3+0.7, r'$[-0.1, 0.3]$ $\rightarrow$ P4', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(bx3+0.95, by3+0.45, r'$[-0.5, -0.1)$ $\rightarrow$ P2', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(bx3+0.95, by3+0.2, r'$\pi{\leq}{-}0.5$ $\rightarrow$ P1', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])

    # 4. Inject DSCP
    bx4, by4 = 7.7, 3.0
    rbox(ax, (bx4, by4), 2.0, 1.5, '#F3E5F5', '#6A1B9A', lw=0.9)
    ax.text(bx4+1.0, by4+1.3, '4. Inject', ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='#6A1B9A')
    ax.text(bx4+1.0, by4+0.85, 'Set DSCP on', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(bx4+1.0, by4+0.6, 'QP traffic_class', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(bx4+1.0, by4+0.3, '= DSCP $\ll$ 2', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])

    # ── Arrows between steps ──
    ap = dict(arrowstyle='->', color=C['text_mid'], lw=0.9)
    ax.annotate('', xy=(2.65, 3.75), xytext=(2.25, 3.75), arrowprops=ap)
    ax.annotate('', xy=(5.25, 3.75), xytext=(4.85, 3.75), arrowprops=ap)
    ax.annotate('', xy=(7.65, 3.75), xytext=(7.25, 3.75), arrowprops=ap)

    ax.text(2.45, 3.95, 'stats', ha='center', va='bottom', fontsize=5, color=C['text_light'])
    ax.text(5.05, 3.95, r'$\pi$', ha='center', va='bottom', fontsize=6.5, color=C['accent'], fontweight='bold')
    ax.text(7.45, 3.95, 'priority', ha='center', va='bottom', fontsize=5, color=C['text_light'])

    # ── Bottom: Switch enforcement ──
    rbox(ax, (2.5, 0.3), 5.0, 1.4, C['bg_switch'], C['border_sw'], lw=1.0)
    ax.text(5.0, 1.5, '5. Enforce (Switch SPQ)', ha='center', va='top',
            fontsize=6.5, fontweight='bold', color=C['border_sw'])
    ax.text(5.0, 1.15, 'P6 queue drains first', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(5.0, 0.9, r'$\rightarrow$ premium gets more BW $\rightarrow$ $\pi$ decreases',
            ha='center', va='center', fontsize=5, color=C['accent'])
    ax.text(5.0, 0.55, 'P2 waits', ha='center', va='center',
            fontsize=5.5, color=C['text_dark'])
    ax.text(5.0, 0.35, r'$\rightarrow$ standard gets less BW $\rightarrow$ $\pi$ increases',
            ha='center', va='center', fontsize=5, color='#1565C0')

    # Arrow: Inject -> Switch
    ax.annotate('', xy=(7.5, 1.75), xytext=(8.7, 2.95),
                arrowprops=dict(arrowstyle='->', color=C['arrow_dscp'], lw=1.0))
    ax.text(8.55, 2.35, 'RoCEv2', ha='center', va='center',
            fontsize=5, color=C['arrow_dscp'], fontweight='bold')

    # Feedback loop
    ax.annotate('',
                xy=(1.3, 2.95),
                xytext=(2.5, 1.0),
                arrowprops=dict(arrowstyle='->', color=C['arrow_fb'], lw=1.3,
                                connectionstyle='arc3,rad=0.3', linestyle='--'))
    ax.text(0.9, 1.9, 'Feedback', ha='center', va='center',
            fontsize=5.5, color=C['arrow_fb'], fontweight='bold')
    ax.text(0.9, 1.55, 'BW change', ha='center', va='center',
            fontsize=4.5, color=C['arrow_fb'])
    ax.text(0.9, 1.25, r'$\rightarrow$ next window', ha='center', va='center',
            fontsize=4.5, color=C['arrow_fb'])

    # ── Key design badges ──
    for i, (txt, clr) in enumerate([
        ('Sender-side only', C['accent']),
        ('No coordinator', '#1565C0'),
        ('Closed-form', '#2E7D32'),
    ]):
        ax.text(9.5, 4.7 - i*0.55, txt, ha='center', va='center',
                fontsize=5.5, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.15', fc=clr, ec='none'))

    # T_target annotation
    ax.text(2.5, 2.8, r'$T^{\mathrm{tgt}}$: EMA calibrated from solo windows',
            ha='center', va='top', fontsize=5, color=C['text_light'], style='italic',
            bbox=dict(boxstyle='round,pad=0.06', fc='#FFFDE7', ec='#F9A825', lw=0.3, alpha=0.85))


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    print("Generating LongLiu Architecture Diagram...")

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.5, 5.5),
        gridspec_kw={'height_ratios': [1, 0.85], 'hspace': 0.12})

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    out_pdf = OUTPUT_DIR / "fig_arch_longliu.pdf"
    out_png = OUTPUT_DIR / "fig_arch_longliu.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
