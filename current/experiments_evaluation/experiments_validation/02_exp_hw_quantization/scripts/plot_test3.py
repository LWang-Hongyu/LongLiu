#!/usr/bin/env python3
"""
plot_test3.py — Exp2 测试3（SP 严格性判定·受控双流）绘图

图1 (fig_test3_timeseries.pdf)：时间序列
  选最接近均值的一轮，画 P3/P6 带宽 vs 相对时间，
  用背景色块标注真实并发窗口（双向重叠），叠加 P6 solo 参考线。
  直观展示：P3 前 3s solo → 并发窗口内 P6:P3≈6:4 → P6 后半段恢复 solo。

图2 (fig_test3_compare.pdf)：test1 vs test3 三指标对比
  抢占度 / P6/solo / P3 饿死度，两组柱 + 误差棒。
  结论可视化：三组柱几乎重合 → 流量形态（突发/连续）不改变并发窗口内分配。

口径：与 analyze_sp_strict.py 一致，双向重叠窗口 [max(A首,B首), min(A末,B末)]。
输出：analysis/fig_test3_timeseries.pdf, analysis/fig_test3_compare.pdf
"""
import os
import csv
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
ANA_DIR = os.path.join(BASE, 'analysis')
os.makedirs(ANA_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
})

C_P6 = '#d62728'   # 红：P6
C_P3 = '#1f77b4'   # 蓝：P3
C_SOLO = '#2ca02c'  # 绿：solo 参考
C_WIN = '#fff3cd'   # 浅黄：并发窗口


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mean_bw(rows, key='bw_gbps'):
    bws = [float(r[key]) for r in rows if r.get(key) not in ('', 'NA', None)]
    return sum(bws) / len(bws) if bws else 0.0


def parse_solo_bw(round_dir):
    csv_path = os.path.join(round_dir, 'exp2_soloB_rank0_iter.csv')
    if os.path.exists(csv_path):
        rows = load_csv(csv_path)
        if rows:
            return mean_bw(rows)
    log_path = os.path.join(round_dir, 'exp2_soloB_rank0_solocalib.log')
    if os.path.exists(log_path):
        with open(log_path) as f:
            m = re.search(r'平均带宽\s*=\s*([\d.]+)\s*Gbps', f.read())
        if m:
            return float(m.group(1))
    return 0.0


def find_success_attempts(round_dir):
    out = []
    for name in sorted(glob.glob(os.path.join(round_dir, 'attempt*'))):
        if (os.path.exists(os.path.join(name, 'exp2_jobA_rank0_iter.csv')) and
                os.path.exists(os.path.join(name, 'exp2_jobB_rank0_iter.csv'))):
            out.append(name)
    return out


def contest_window(jobA_rows, jobB_rows):
    """双向重叠窗口 = [max(A首,B首), min(A末,B末)]；A、B 均只取窗口内迭代。"""
    a_ts = sorted(float(r['ts']) for r in jobA_rows if r.get('ts'))
    b_ts = sorted(float(r['ts']) for r in jobB_rows if r.get('ts'))
    if not a_ts or not b_ts:
        return [], [], 0.0, 0.0
    t0 = max(a_ts[0], b_ts[0])
    t1 = min(a_ts[-1], b_ts[-1])
    if t1 < t0:
        return [], [], 0.0, 0.0
    win_a = [r for r in jobA_rows if r.get('ts') and t0 <= float(r['ts']) <= t1]
    win_b = [r for r in jobB_rows if r.get('ts') and t0 <= float(r['ts']) <= t1]
    return win_a, win_b, t0, t1


def load_round(round_dir):
    """解析一轮 test1/test3：返回统计 dict 与原始时间序列。"""
    solo = parse_solo_bw(round_dir)
    atts = find_success_attempts(round_dir)
    if not atts or solo == 0.0:
        return None
    att = max(atts, key=lambda a: len(load_csv(
        os.path.join(a, 'exp2_jobB_rank0_iter.csv'))))
    jobA_all = load_csv(os.path.join(att, 'exp2_jobA_rank0_iter.csv'))
    jobB_all = load_csv(os.path.join(att, 'exp2_jobB_rank0_iter.csv'))
    win_a, win_b, t0, t1 = contest_window(jobA_all, jobB_all)
    if not win_a or not win_b:
        return None
    bw_a = mean_bw(win_a)
    bw_b = mean_bw(win_b)
    return {
        'name': os.path.basename(round_dir), 'attempt': os.path.basename(att),
        'solo': solo, 'p3_bw': bw_a, 'p6_bw': bw_b,
        'preempt': bw_b / (bw_b + bw_a) if (bw_a + bw_b) > 0 else 0.0,
        'starve': bw_a / solo,
        'p6_ratio': bw_b / solo,
        'A_all': jobA_all, 'B_all': jobB_all,
        'win': (t0, t1),
    }


# ---------------------------------------------------------------------------
# 图1：时间序列（选最接近均值的一轮）
# ---------------------------------------------------------------------------
def plot_timeseries(rounds):
    data = [d for d in (load_round(r) for r in rounds) if d]
    if not data:
        print('  [图1] 无有效轮次，跳过')
        return
    p6r = np.array([d['p6_ratio'] for d in data])
    target = p6r.mean()
    rep = min(data, key=lambda d: abs(d['p6_ratio'] - target))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    A, B = rep['A_all'], rep['B_all']
    t_min = min(float(r['ts']) for r in A)
    t0, t1 = rep['win']

    # 并发窗口背景色块
    ax.axvspan(t0 - t_min, t1 - t_min, color=C_WIN, zorder=0)

    ax.plot([float(r['ts']) - t_min for r in A],
            [float(r['bw_gbps']) for r in A],
            color=C_P3, lw=1.0, alpha=0.85, label='P3 (DSCP16, tc:2)')
    ax.plot([float(r['ts']) - t_min for r in B],
            [float(r['bw_gbps']) for r in B],
            color=C_P6, lw=1.2, alpha=0.9, label='P6 (DSCP8, tc:0)')
    ax.axhline(rep['solo'], color=C_SOLO, ls='--', lw=1.2,
               label=f"P6 solo = {rep['solo']:.1f} Gbps")
    ax.set_xlabel('Time since P3 start (s)')
    ax.set_ylabel('NCCL AllReduce bandwidth (Gbps)')
    ax.set_title(f"Test3: continuous dual-flow on SP queues "
                 f"({rep['name'].split('_')[-2]})")
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)
    # 并发窗口标注需在 ylim 确定后
    ax.text((t0 - t_min + t1 - t_min) / 2, ax.get_ylim()[1],
            'concurrent window', ha='center', va='top',
            fontsize=9, color='#8a6d1a')
    fig.tight_layout()
    out = os.path.join(ANA_DIR, 'fig_test3_timeseries.pdf')
    fig.savefig(out)
    print(f'图1 已保存: {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图2：test1 vs test3 三指标对比
# ---------------------------------------------------------------------------
def plot_compare(t1_rounds, t3_rounds):
    d1 = [d for d in (load_round(r) for r in t1_rounds) if d]
    d3 = [d for d in (load_round(r) for r in t3_rounds) if d]
    if not d1 or not d3:
        print(f'  [图2] test1={len(d1)} 轮, test3={len(d3)} 轮，跳过')
        return

    def stat(ds, key):
        v = np.array([d[key] for d in ds])
        return v.mean() * 100, v.std() * 100

    labels = ['Preemption\nP6/(P6+P3)', 'P6 vs solo', 'P3 starve\n(P3/solo)']
    t1_vals = [stat(d1, 'preempt'), stat(d1, 'p6_ratio'), stat(d1, 'starve')]
    t3_vals = [stat(d3, 'preempt'), stat(d3, 'p6_ratio'), stat(d3, 'starve')]

    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    b1 = ax.bar(x - w / 2, [v[0] for v in t1_vals], w, yerr=[v[1] for v in t1_vals],
                label=f'Test1 burst (sleep 10ms, n={len(d1)})',
                color='#9ecae1', capsize=4, edgecolor='black', lw=0.6)
    b2 = ax.bar(x + w / 2, [v[0] for v in t3_vals], w, yerr=[v[1] for v in t3_vals],
                label=f'Test3 continuous (sleep 0, n={len(d3)})',
                color='#dda8a8', capsize=4, edgecolor='black', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Ratio to solo (%)')
    ax.set_ylim(0, 110)
    ax.set_title('SP queue behavior: burst vs continuous traffic')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    for bars, vals in ((b1, t1_vals), (b2, t3_vals)):
        for bar, (m, s) in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    f'{m:.1f}', ha='center', fontsize=9)

    ax.text(0.02, 0.03,
            'All three metrics coincide within std. dev.\n'
            'Traffic shape does not change per-contention allocation.',
            transform=ax.transAxes, fontsize=9, style='italic', color='#555555')
    fig.tight_layout()
    out = os.path.join(ANA_DIR, 'fig_test3_compare.pdf')
    fig.savefig(out)
    print(f'图2 已保存: {out}')
    plt.close(fig)
    for i, lab in enumerate(labels):
        print(f'  {lab}: test1={t1_vals[i][0]:.1f}±{t1_vals[i][1]:.1f}, '
              f'test3={t3_vals[i][0]:.1f}±{t3_vals[i][1]:.1f}')


def main():
    t3_rounds = sorted(glob.glob(os.path.join(DATA_DIR, 'exp2_test3_r*')))
    t1_rounds = [r for r in sorted(glob.glob(os.path.join(DATA_DIR, 'exp2_test1_r*')))
                 if '_'.join(os.path.basename(r).split('_')[-2:]) >= '20260808_204437']
    print(f'test3 轮次: {len(t3_rounds)}, test1 修正后轮次: {len(t1_rounds)}')
    plot_timeseries(t3_rounds)
    plot_compare(t1_rounds, t3_rounds)


if __name__ == '__main__':
    main()
