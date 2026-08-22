#!/usr/bin/env python3
"""
Fig-6s: V6 statistical replications — Phase 2 tight Job B slowdown.
IEEEtran full-width (7.16 in × 3.2 in), 1×2 subplots.
Style consistent with fig6_v6physical/plot_fig6.py:
  - LongLiu blue #0072B2 solid + square  |  CRUX orange #D55E00 dashed + circle
  - serif font, stix mathtext, stable-window grey shade, dotted grid
  - 600 dpi PNG + PDF, bottom shared legend

Panel 1 (left): epoch-by-epoch mean ± SEM across the 6 new-config runs
                 (replication 3/4/5 × both orders), stable window shaded.
Panel 2 (right): front-window (epochs 7-11) paired comparison per run,
                 with mean ± 95% CI columns.

Data (copied into ../data/):
  v6_replication_{3,4,5}/p4_jobB_v6_round{N}_{order}_{mode}_rank0_epoch.csv

Output: ../figures/fig6_v6stats_testbed.{pdf,png}
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

# ── paths ──────────────────────────────────────────────────────
# .../fig6_v6stats/{scripts,data,figures}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_ROOT = os.path.dirname(SCRIPT_DIR)
BASE = os.path.join(FIG_ROOT, 'data')
OUT_DIR = os.path.join(FIG_ROOT, 'figures')
NEW_REPS = ['v6_replication_3', 'v6_replication_4', 'v6_replication_5']
ROUNDS = [('round1_LLthenCX', 'R1 LL→CX'), ('round2_CXthenLL', 'R2 CX→LL')]
EPOCHS = list(range(7, 15))
FRONT = list(range(7, 12))

OUT_PDF = os.path.join(OUT_DIR, 'fig6_v6stats_testbed.pdf')
OUT_PNG = os.path.join(OUT_DIR, 'fig6_v6stats_testbed_600.png')

FIG_W, FIG_H = 7.16, 3.25

LL_COLOR = '#0072B2'
CX_COLOR = '#D55E00'
SHADE_ALPHA = 0.12
SHADE_COLOR = '#666666'

LL_LINE = dict(color=LL_COLOR, linewidth=1.6, linestyle='-', zorder=5)
CX_LINE = dict(color=CX_COLOR, linewidth=1.6, linestyle='--', zorder=4)
LL_MK = dict(marker='s', markersize=4.5, markerfacecolor=LL_COLOR,
             markeredgecolor='#004080', markeredgewidth=0.4)
CX_MK = dict(marker='o', markersize=4.5, markerfacecolor=CX_COLOR,
             markeredgecolor='#993300', markeredgewidth=0.4)


# t-distribution two-tailed 0.975 quantile (lookup, no scipy dependency)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
         6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_ppf(alpha, df):
    if df in _T975:
        return _T975[df]
    lo = min(_T975.keys(), key=lambda k: abs(k - df))
    hi = lo + 1 if lo + 1 in _T975 else lo
    return (_T975[lo] + _T975[hi]) / 2 if hi != lo else _T975[lo]


def load(rep, rnd, mode, epoch):
    f = os.path.join(BASE, rep, f'p4_jobB_v6_{rnd}_{mode}_rank0_epoch.csv')
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if int(r['epoch']) == epoch and r['phase'] == 'phase2':
                return float(r['slowdown'])
    return np.nan


def main():
    # ── collect: new-config (6 runs) epoch-by-epoch + front-window ──
    ll_ep = {e: [] for e in EPOCHS}
    cx_ep = {e: [] for e in EPOCHS}
    front_ll, front_cx = [], []
    for rep in NEW_REPS:
        for rnd, _ in ROUNDS:
            fl, fc = [], []
            for e in EPOCHS:
                v_ll = load(rep, rnd, 'longliu', e)
                v_cx = load(rep, rnd, 'crux', e)
                ll_ep[e].append(v_ll)
                cx_ep[e].append(v_cx)
                if e in FRONT:
                    fl.append(v_ll)
                    fc.append(v_cx)
            front_ll.append(np.nanmean(fl))
            front_cx.append(np.nanmean(fc))
    front_ll = np.array(front_ll)
    front_cx = np.array(front_cx)

    # ── figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.25, top=0.82,
                        wspace=0.22)

    # ── Panel 1: epoch-by-epoch mean ± SEM ─────────────────────
    ax1 = axes[0]
    x = np.array(EPOCHS)
    ll_m = np.array([np.nanmean(ll_ep[e]) for e in EPOCHS])
    cx_m = np.array([np.nanmean(cx_ep[e]) for e in EPOCHS])
    ll_s = np.array([np.nanstd(ll_ep[e], ddof=1) / np.sqrt(len(ll_ep[e])) for e in EPOCHS])
    cx_s = np.array([np.nanstd(cx_ep[e], ddof=1) / np.sqrt(len(cx_ep[e])) for e in EPOCHS])

    ax1.axvspan(FRONT[0] - 0.4, FRONT[-1] + 0.4,
                alpha=SHADE_ALPHA, color=SHADE_COLOR, zorder=0)
    ax1.fill_between(x, cx_m - cx_s, cx_m + cx_s, color=CX_COLOR, alpha=0.15, zorder=2)
    ax1.fill_between(x, ll_m - ll_s, ll_m + ll_s, color=LL_COLOR, alpha=0.15, zorder=2)
    ax1.plot(x, cx_m, **CX_LINE, **CX_MK, label='CRUX (static P3)')
    ax1.plot(x, ll_m, **LL_LINE, **LL_MK, label='LongLiu (dynamic)')

    ax1.set_xlim(6.5, 14.5)
    ax1.set_ylim(0.7, 1.7)
    ax1.set_title('(a) Epoch-by-epoch (mean ± SEM, n=6)', fontsize=11, fontweight='bold', pad=5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Slowdown (×)', fontsize=11)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax1.grid(True, alpha=0.30, linewidth=0.4, linestyle=':')

    # ── Panel 2: front-window advantage (%) per run ────────────
    ax2 = axes[1]
    adv = (front_cx - front_ll) / front_cx * 100.0
    runs = np.arange(1, len(adv) + 1)

    ax2.plot(runs, adv, linestyle='', **LL_MK, zorder=4)

    # mean ± 95% CI at x = n+1
    m = adv.mean()
    se = adv.std(ddof=1) / np.sqrt(len(adv))
    h = t_ppf(0.975, len(adv) - 1) * se
    ax2.errorbar(len(runs) + 1, m, yerr=h, fmt='s', color=LL_COLOR, ms=6,
                 capsize=3, elinewidth=1.2, markeredgecolor='#333333', zorder=5)
    ax2.axvline(len(runs) + 1, color='#BBBBBB', lw=0.8, ls=':')
    ax2.set_xlim(0.4, len(runs) + 2.4)
    ax2.set_ylim(20, 40)
    ax2.set_title('(b) Front-window advantage per run (epochs 7-11)',
                  fontsize=11, fontweight='bold', pad=5)
    ax2.set_xlabel('Replication run', fontsize=11)
    ax2.set_ylabel('Advantage over CRUX (%)', fontsize=11)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.set_xticks(np.concatenate([runs, [len(runs) + 1]]))
    ax2.set_xticklabels([f'{i}' for i in runs] + ['mean'])
    ax2.grid(True, alpha=0.30, linewidth=0.4, linestyle=':')

    # ── shared legend (bottom, outside) ────────────────────────
    handles = [
        plt.Line2D([], [], color=LL_COLOR, lw=1.6, ls='-', marker='s', ms=4.5,
                   markerfacecolor=LL_COLOR, label='LongLiu (dynamic)'),
        plt.Line2D([], [], color=CX_COLOR, lw=1.6, ls='--', marker='o', ms=4.5,
                   markerfacecolor=CX_COLOR, label='CRUX (static P3)'),
        Rectangle((0, 0), 1, 1, alpha=SHADE_ALPHA, color=SHADE_COLOR,
                  label='Stable window (epochs 7–11, primary)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02),
               bbox_transform=fig.transFigure)

    # ── save ───────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=600, format='png', bbox_inches='tight', pad_inches=0.02)
    print(f'[OK] {OUT_PDF}')
    print(f'[OK] {OUT_PNG}')
    print(f'  front LL mean={front_ll.mean():.3f}  CX mean={front_cx.mean():.3f}  '
          f'adv={((front_cx.mean() - front_ll.mean()) / front_cx.mean() * 100):.1f}%  '
          f'per-run adv: {", ".join(f"{a:.1f}" for a in (front_cx - front_ll) / front_cx * 100)}%')


if __name__ == '__main__':
    main()
