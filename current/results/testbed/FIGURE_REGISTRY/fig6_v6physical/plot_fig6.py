#!/usr/bin/env python3
"""
Fig-6: Physical testbed epoch-level slowdown comparison.
IEEEtran full-width (7.16 in × ≤3.2 in), 2×2 subplots.
LongLiu blue solid #0072B2 □ | CRUX orange dashed #D55E00 ○
Style: INFOCOM 2026 v1.0 — print-friendly, no AI-isms.
"""
import csv, os, re, sys, math
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'fig6_data.csv')
OUT_PDF = os.path.join(SCRIPT_DIR, 'fig6_testbed.pdf')
OUT_PNG = os.path.join(SCRIPT_DIR, 'fig6_testbed_600.png')
OUT_CHECK = os.path.join(SCRIPT_DIR, 'fig6_self_check.csv')

# ── IEEEtran full-width canvas ─────────────────────────────────
FIG_W = 7.16          # inches, full-width
FIG_H = 3.15          # inches, ≤3.2 spec

# ── Okabe-Ito palette (print-friendly) ─────────────────────────
LL_COLOR = '#0072B2'   # blue (LongLiu)
CX_COLOR = '#D55E00'   # orange (CRUX)

# ── styles (line type + marker = redundant encoding) ───────────
LL_STYLE = {
    'color': LL_COLOR, 'linewidth': 1.5, 'linestyle': '-',
    'marker': 's', 'markersize': 4.0, 'markevery': 1,
    'markerfacecolor': LL_COLOR, 'markeredgewidth': 0.4,
    'markeredgecolor': '#004080', 'zorder': 4,
}
CX_STYLE = {
    'color': CX_COLOR, 'linewidth': 1.5, 'linestyle': '--',
    'marker': 'o', 'markersize': 4.0, 'markevery': 1,
    'markerfacecolor': CX_COLOR, 'markeredgewidth': 0.4,
    'markeredgecolor': '#993300', 'zorder': 3,
}
SHADE_ALPHA = 0.12
SHADE_COLOR = '#666666'

# ── round metadata: display names for natural-language labels ──
ROUNDS = [
    ('orig_r1', 'Round 1 (orig, LongLiu→CRUX)'),
    ('orig_r2', 'Round 2 (orig, CRUX→LongLiu)'),
    ('rep2_r1', 'Round 1 (rep2, LongLiu→CRUX)'),
    ('rep2_r2', 'Round 2 (rep2, CRUX→LongLiu)'),
]
STABLE = (7, 11)  # inclusive


def parse_epoch_detail(detail_str):
    """Parse 'e7:1.1441;e8:1.1220;...' into sorted list of (epoch, value)."""
    pairs = []
    for token in detail_str.strip().split(';'):
        token = token.strip()
        if not token:
            continue
        m = re.match(r'e(\d+):([\d.]+)', token)
        if m:
            pairs.append((int(m.group(1)), float(m.group(2))))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_round_data():
    """Load fig6_data.csv, return dict keyed by (round_id, scheduler, phase)."""
    data = {}
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row['round_id'].strip()
            sched = row['scheduler'].strip()
            phase = row['phase'].strip()
            detail = parse_epoch_detail(row['epoch_detail'])
            data[(rid, sched, phase)] = {
                'mean': float(row['slowdown_mean']),
                'min': float(row['slowdown_min']),
                'max': float(row['slowdown_max']),
                'detail': detail,
            }
    return data


def build_epoch_arrays(data, rid):
    """Build x (epochs) and y_ll, y_cx arrays (full window)."""
    ll_key = (rid, 'longliu', 'phase2_tight_full')
    cx_key = (rid, 'crux', 'phase2_tight_full')
    ll_detail = data.get(ll_key, {}).get('detail', [])
    cx_detail = data.get(cx_key, {}).get('detail', [])
    epochs = [p[0] for p in ll_detail]
    ll_vals = [p[1] for p in ll_detail]
    cx_map = {p[0]: p[1] for p in cx_detail}
    cx_vals = [cx_map.get(e, float('nan')) for e in epochs]
    return np.array(epochs), np.array(ll_vals), np.array(cx_vals)


def plot_subpanel(ax, epochs, ll_vals, cx_vals, title, xlabel=False, ylabel=False):
    """Draw one subpanel (one round)."""
    # ── stable-window shade ────────────────────────────────────
    ax.axvspan(STABLE[0] - 0.4, STABLE[1] + 0.4,
               alpha=SHADE_ALPHA, color=SHADE_COLOR, zorder=0)

    # ── data lines ─────────────────────────────────────────────
    ax.plot(epochs, cx_vals, **CX_STYLE)
    ax.plot(epochs, ll_vals, **LL_STYLE)

    # ── axes ───────────────────────────────────────────────────
    ax.set_xlim(6.5, 14.5)
    ax.set_ylim(0.6, 1.5)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    if xlabel:
        ax.set_xlabel('Epoch', fontsize=9)
    if ylabel:
        ax.set_ylabel('Slowdown (×)', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.grid(True, alpha=0.30, linewidth=0.4, linestyle=':')


def compute_self_check(data):
    """Compute plotted mean (stable window) vs CSV reference."""
    rows = []
    for rid, _ in ROUNDS:
        for sched, label in [('longliu', 'LongLiu'), ('crux', 'CRUX')]:
            full_key = (rid, sched, 'phase2_tight_full')
            detail = data.get(full_key, {}).get('detail', [])
            stable_vals = [v for e, v in detail if STABLE[0] <= e <= STABLE[1]]
            plotted_mean = np.mean(stable_vals) if stable_vals else float('nan')
            csv_mean = data.get((rid, sched, 'phase2_tight_stable'), {}).get('mean', float('nan'))
            diff = plotted_mean - csv_mean if not (math.isnan(plotted_mean) or math.isnan(csv_mean)) else float('nan')
            rows.append({
                'round': rid, 'scheduler': label,
                'plotted_mean': f'{plotted_mean:.6f}',
                'csv_mean': f'{csv_mean:.6f}',
                'diff': f'{diff:.2e}' if not math.isnan(diff) else 'N/A',
            })
    return rows


def main():
    data = load_round_data()

    # ── 2×2 subplot grid ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.21, top=0.86,
                        wspace=0.12, hspace=0.28)

    for idx, (rid, title) in enumerate(ROUNDS):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        epochs, ll_vals, cx_vals = build_epoch_arrays(data, rid)
        xlabel = (row == 1)
        ylabel = (col == 0)
        plot_subpanel(ax, epochs, ll_vals, cx_vals, title,
                      xlabel=xlabel, ylabel=ylabel)

    # ── shared legend (bottom, outside) ────────────────────────
    handles = [
        plt.Line2D([], [], **{k: LL_STYLE[k] for k in ['color', 'linewidth', 'linestyle', 'marker', 'markersize']},
                    markerfacecolor=LL_COLOR, label='LongLiu (dynamic)'),
        plt.Line2D([], [], **{k: CX_STYLE[k] for k in ['color', 'linewidth', 'linestyle', 'marker', 'markersize']},
                    markerfacecolor=CX_COLOR, label='CRUX (static P3)'),
        Rectangle((0,0), 1, 1, alpha=SHADE_ALPHA, color=SHADE_COLOR,
                  label='Stable window (epochs 7–11, primary)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01), bbox_transform=fig.transFigure)

    # ── save ────────────────────────────────────────────────────
    fig.savefig(OUT_PDF, dpi=300, format='pdf',
                bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=600, format='png',
                bbox_inches='tight', pad_inches=0.02)
    print(f'[OK] Saved: {OUT_PDF}')
    print(f'[OK] Saved: {OUT_PNG}')

    # ── self-check ──────────────────────────────────────────────
    check_rows = compute_self_check(data)
    with open(OUT_CHECK, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['round', 'scheduler', 'plotted_mean', 'csv_mean', 'diff'])
        w.writeheader()
        w.writerows(check_rows)
    print(f'[OK] Self-check: {OUT_CHECK}')
    print()
    all_ok = True
    for r in check_rows:
        d = float(r['diff']) if r['diff'] != 'N/A' else 1.0
        ok = abs(d) < 1e-4
        all_ok &= ok
        print(f'  {"✓" if ok else "✗"} {r["round"]} {r["scheduler"]:>8}: '
              f'plotted={r["plotted_mean"]}  csv={r["csv_mean"]}  diff={r["diff"]}')
    print()
    print(f'[DONE] Fig-6 complete.  All diffs ≤ 1e-4? {"YES ✓" if all_ok else "CHECK ✗"}')


if __name__ == '__main__':
    main()
