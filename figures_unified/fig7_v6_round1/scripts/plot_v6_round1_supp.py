#!/usr/bin/env python3
"""
Fig-7 (Evaluation Supplement): V6 round-1 contested deployment.
IEEEtran full-width (7.16 in), 2×2 subplots (15/30 Gbps bg × Job A/B).
Style consistent with fig6_v6physical/plot_fig6.py:
  - LongLiu blue #0072B2 solid + square  |  CRUX orange #D55E00 dashed + circle
  - serif font, stix mathtext, tight-window grey shade, dotted grid
  - 600 dpi PNG + PDF, bottom shared legend

Panel fig7a (trajectory): window-by-window slowdown, LongLiu vs CRUX,
  SLO-tight window range shaded.
Panel fig7b (summary): per (bg × SLO-tightness) mean slowdown bars
  (mean of the two jobs, error bar = job range; window 0 excluded).

Data:
  P4_dumbbell_slo/data_v6_bg{15,30}_round1/
    p4_job{A,B}_v6_round1_LLthenCX_{longliu,crux}_rank0_window.csv
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'TeX Gyre Termes', 'Times'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.30,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
})
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(FIG_ROOT, 'figures')
DATA = '/home/why/LongLiu_rebuild/current/experiments_evaluation/P4_dumbbell_slo'

FIG_W, FIG_H = 7.16, 4.2

LL_COLOR = '#0072B2'
CX_COLOR = '#D55E00'
SHADE_ALPHA = 0.12
SHADE_COLOR = '#666666'

LL_STYLE = {'color': LL_COLOR, 'linewidth': 1.5, 'linestyle': '-',
            'marker': 's', 'markersize': 3.5, 'markerfacecolor': LL_COLOR,
            'markeredgewidth': 0.4, 'markeredgecolor': '#004080', 'zorder': 4}
CX_STYLE = {'color': CX_COLOR, 'linewidth': 1.5, 'linestyle': '--',
            'marker': 'o', 'markersize': 3.5, 'markerfacecolor': CX_COLOR,
            'markeredgewidth': 0.4, 'markeredgecolor': '#993300', 'zorder': 3}

# Job A: tight = phase1 (windows 0-6), loose = phase2 (7-14)
# Job B: tight = phase2 (windows 7-14), loose = phase1 (0-6)
TIGHT_A = range(0, 7)
TIGHT_B = range(7, 15)


def load_window(bg, job, mode):
    """Return (windows, slowdown) arrays for one CSV."""
    f = os.path.join(DATA, f'data_v6_bg{bg}_round1',
                     f'p4_job{job}_v6_round1_LLthenCX_{mode}_rank0_window.csv')
    ws, ss = [], []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            ws.append(int(r['window']))
            ss.append(float(r['slowdown']))
    return np.array(ws), np.array(ss)


def job_phase_mean(bg, job, mode, tight):
    """Mean slowdown over the tight/loose windows of one job, excluding window 0."""
    w, s = load_window(bg, job, mode)
    tight_w = TIGHT_A if job == 'A' else TIGHT_B
    sel = [v for i, v in zip(w, s) if (i in tight_w) == tight and i != 0]
    return float(np.mean(sel)) if sel else np.nan


def draw_panel(ax, bg, job, title, xlabel, ylabel):
    w_ll, s_ll = load_window(bg, job, 'longliu')
    w_cx, s_cx = load_window(bg, job, 'crux')
    tight_w = TIGHT_A if job == 'A' else TIGHT_B
    ax.axvspan(min(tight_w) - 0.4, max(tight_w) + 0.4,
               alpha=SHADE_ALPHA, color=SHADE_COLOR, zorder=0)
    ax.plot(w_cx, s_cx, **CX_STYLE)
    ax.plot(w_ll, s_ll, **LL_STYLE)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    if xlabel:
        ax.set_xlabel('Window', fontsize=10)
    if ylabel:
        ax.set_ylabel('Slowdown (×)', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.grid(True, alpha=0.30, linewidth=0.4, linestyle=':')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Fig 7a: window-level trajectories (2×2) ────────────────
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.90,
                        wspace=0.10, hspace=0.32)
    titles = ['(a) 15 Gbps bg · Job A (tight w0–6)',
              '(b) 15 Gbps bg · Job B (tight w7–14)',
              '(c) 30 Gbps bg · Job A (tight w0–6)',
              '(d) 30 Gbps bg · Job B (tight w7–14)']
    specs = [(15, 'A'), (15, 'B'), (30, 'A'), (30, 'B')]
    ymax = 0
    for bg, job in specs:
        for mode in ('longliu', 'crux'):
            _, s = load_window(bg, job, mode)
            ymax = max(ymax, s.max())
    for idx, (bg, job) in enumerate(specs):
        ax = axes[idx // 2, idx % 2]
        draw_panel(ax, bg, job, titles[idx],
                   xlabel=(idx // 2 == 1), ylabel=(idx % 2 == 0))
        ax.set_xlim(-0.5, 14.5)
        ax.set_ylim(0, ymax * 1.15)
    handles = [
        plt.Line2D([], [], color=LL_COLOR, lw=1.5, ls='-', marker='s', ms=3.5,
                   markerfacecolor=LL_COLOR, label='LongLiu (P6)'),
        plt.Line2D([], [], color=CX_COLOR, lw=1.5, ls='--', marker='o', ms=3.5,
                   markerfacecolor=CX_COLOR, label='CRUX (P3)'),
        Rectangle((0, 0), 1, 1, alpha=SHADE_ALPHA, color=SHADE_COLOR,
                  label='SLO-tight window (c$_i$ = 1.2)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.005),
               bbox_transform=fig.transFigure)
    fig.savefig(os.path.join(OUT_DIR, 'fig7_v6_round1_trajectory.pdf'),
                bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(OUT_DIR, 'fig7_v6_round1_trajectory_600.png'),
                dpi=600, bbox_inches='tight', pad_inches=0.02)
    print('[OK] fig7_v6_round1_trajectory')

    # ── Fig 7b: phase-summary bars ─────────────────────────────
    groups = [('15 Gbps bg', 15, True), ('15 Gbps bg', 15, False),
              ('30 Gbps bg', 30, True), ('30 Gbps bg', 30, False)]
    x = np.arange(len(groups))
    wbar = 0.34
    ll_means, ll_lo, ll_hi = [], [], []
    cx_means, cx_lo, cx_hi = [], [], []
    for _, bg, tight in groups:
        v_ll = [job_phase_mean(bg, j, 'longliu', tight) for j in 'AB']
        v_cx = [job_phase_mean(bg, j, 'crux', tight) for j in 'AB']
        ll_means.append(np.mean(v_ll)); ll_lo.append(np.min(v_ll)); ll_hi.append(np.max(v_ll))
        cx_means.append(np.mean(v_cx)); cx_lo.append(np.min(v_cx)); cx_hi.append(np.max(v_cx))

    fig2, ax2 = plt.subplots(figsize=(FIG_W, 2.55))
    fig2.subplots_adjust(left=0.075, right=0.985, bottom=0.26, top=0.80)
    ax2.bar(x - wbar / 2, ll_means, wbar, color=LL_COLOR, alpha=0.85,
            yerr=[np.array(ll_means) - np.array(ll_lo),
                  np.array(ll_hi) - np.array(ll_means)],
            capsize=3, error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor='#333333'),
            label='LongLiu (P6)')
    ax2.bar(x + wbar / 2, cx_means, wbar, color=CX_COLOR, alpha=0.85,
            yerr=[np.array(cx_means) - np.array(cx_lo),
                  np.array(cx_hi) - np.array(cx_means)],
            capsize=3, error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor='#333333'),
            label='CRUX (P3)')
    ax2.axhline(1.0, color='#BBBBBB', lw=0.8, ls=':')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{g}\nSLO-{"tight" if t else "loose"}' for g, _, t in groups],
                        fontsize=9)
    ax2.set_ylabel('Mean slowdown (×)', fontsize=10)
    ax2.set_ylim(0, max(max(ll_hi), max(cx_hi)) * 1.15)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.grid(True, axis='y', alpha=0.30, linewidth=0.4, linestyle=':')
    ax2.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    fig2.savefig(os.path.join(OUT_DIR, 'fig7_v6_round1_summary.pdf'),
                 bbox_inches='tight', pad_inches=0.02)
    fig2.savefig(os.path.join(OUT_DIR, 'fig7_v6_round1_summary_600.png'),
                 dpi=600, bbox_inches='tight', pad_inches=0.02)
    print('[OK] fig7_v6_round1_summary')

    # ── console summary (cross-check numbers for the supplement) ──
    print('\n  phase-summary means (LL / CX):')
    for (g, bg, tight), lm, cm in zip(groups, ll_means, cx_means):
        tag = 'tight' if tight else 'loose'
        print(f'    {g} {tag:5s}: LL={lm:.2f}  CX={cm:.2f}  '
              f'delta={(cm - lm) / cm * 100:+.1f}% (CX-relative)')


if __name__ == '__main__':
    main()
