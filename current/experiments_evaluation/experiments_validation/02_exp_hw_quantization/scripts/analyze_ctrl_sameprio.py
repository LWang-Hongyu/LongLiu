#!/usr/bin/env python3
"""
analyze_ctrl_sameprio.py — 对照实验（两流同为 DSCP16）分析

目的：甄别 test3 中 P6:P3 = 6:4 带宽分配的成因。
  * jobB 份额 ≈ 50% → 6:4 由优先级造成（SP 有效但非严格 per-packet）
  * jobB 份额仍 ≈ 59% → 6:4 与优先级无关（拥塞控制等固有机制）

数据布局（run_test3_dscp16_ctrl.sh 产物）：
  exp2_ctrl_dscp16_r<round>_<ts>/
    exp2_soloB_rank0_iter.csv          # P3 solo 校准（连续模式）
    attemptN/exp2_jobA_rank0_iter.csv  # jobA（P3，先启动）
    attemptN/exp2_jobB_rank0_iter.csv  # jobB（P3，后启动 3s）

输出 analysis/exp2_ctrl_dscp16_report.md
"""
import os
import csv
import glob
import re
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
ANA_DIR = os.path.join(BASE, 'analysis')
os.makedirs(ANA_DIR, exist_ok=True)

PREFIX = 'exp2_ctrl_dscp16_r*'


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
            return mean_bw(rows), 'csv'
    log_path = os.path.join(round_dir, 'exp2_soloB_rank0_solocalib.log')
    if os.path.exists(log_path):
        with open(log_path) as f:
            m = re.search(r'平均带宽\s*=\s*([\d.]+)\s*Gbps', f.read())
        if m:
            return float(m.group(1)), 'log'
    return 0.0, None


def find_success_attempts(round_dir):
    out = []
    for name in sorted(glob.glob(os.path.join(round_dir, 'attempt*'))):
        if (os.path.exists(os.path.join(name, 'exp2_jobA_rank0_iter.csv')) and
                os.path.exists(os.path.join(name, 'exp2_jobB_rank0_iter.csv'))):
            out.append(name)
    return out


def contest_window(jobA_rows, jobB_rows):
    """双向重叠窗口 = [max(A首,B首), min(A末,B末)]；A、B 均只取窗口内迭代。
    （与 analyze_sp_strict.py 同一口径修正，保证可比。）"""
    a_ts = sorted(float(r['ts']) for r in jobA_rows if r.get('ts'))
    b_ts = sorted(float(r['ts']) for r in jobB_rows if r.get('ts'))
    if not a_ts or not b_ts:
        return jobA_rows, jobB_rows
    t0 = max(a_ts[0], b_ts[0])
    t1 = min(a_ts[-1], b_ts[-1])
    if t1 < t0:
        return [], []
    win_a = [r for r in jobA_rows
             if r.get('ts') and t0 <= float(r['ts']) <= t1]
    win_b = [r for r in jobB_rows
             if r.get('ts') and t0 <= float(r['ts']) <= t1]
    return win_a, win_b


def main():
    rounds = sorted(glob.glob(os.path.join(DATA_DIR, PREFIX)))
    print(f'=== 对照实验: 两流同为 DSCP16（同优先级），{len(rounds)} 轮 ===')

    summary = []
    for r in rounds:
        solo, solo_src = parse_solo_bw(r)
        atts = find_success_attempts(r)
        if not atts:
            print(f'  [{os.path.basename(r)}] 无成功 attempt，跳过')
            continue
        att = max(atts, key=lambda a: len(load_csv(
            os.path.join(a, 'exp2_jobB_rank0_iter.csv'))))
        jobA_all = load_csv(os.path.join(att, 'exp2_jobA_rank0_iter.csv'))
        jobB_all = load_csv(os.path.join(att, 'exp2_jobB_rank0_iter.csv'))
        win_a, win_b = contest_window(jobA_all, jobB_all)
        if not win_a or not win_b:
            print(f'  [{os.path.basename(r)}] 争抢窗口为空，跳过')
            continue
        bw_a = mean_bw(win_a)
        bw_b = mean_bw(win_b)
        share_b = bw_b / (bw_a + bw_b) if (bw_a + bw_b) > 0 else 0.0
        agg = (bw_a + bw_b) / solo if solo > 0 else 0.0  # 并发聚合 vs solo
        summary.append({'name': os.path.basename(r), 'attempt': os.path.basename(att),
                        'solo': solo, 'solo_src': solo_src, 'bw_a': bw_a,
                        'bw_b': bw_b, 'n_a': len(win_a), 'n_b': len(win_b),
                        'share_b': share_b, 'agg': agg})
        print(f"  [{os.path.basename(r)}/{os.path.basename(att)}] "
              f"solo={solo:.2f}G({solo_src}), 窗口jobA={bw_a:.2f}G(n={len(win_a)}), "
              f"窗口jobB={bw_b:.2f}G(n={len(win_b)}), "
              f"jobB份额={share_b*100:.1f}%, 并发聚合/solo={agg*100:.1f}%")

    md = ['# 对照实验: 两流同为 DSCP16 — 同优先级带宽分配', '']
    md += ['> 方法：与 test3 完全同机制（fixed_prio_job，连续通信 sleep=0），'
           '唯一差异为 **jobA 与 jobB 都标记为 P3(DSCP16/tc:2)**，'
           '即同入一个 TC 队列、无优先级差。', '']
    md += ['> 判定标准：',
           '- jobB 份额 ≈ 50% → 6:4 由优先级造成（SP 有效但非严格 per-packet）',
           '- jobB 份额仍 ≈ 59% → 6:4 与优先级无关（拥塞控制等固有机制）', '',
           '| Round | solo(G) | 窗口jobA(G) | 窗口jobB(G) | jobB份额(%) | 聚合/solo(%) |',
           '|-------|---------|-------------|-------------|-------------|--------------|']
    for s in summary:
        md.append(f"| {s['name']} | {s['solo']:.2f} | {s['bw_a']:.2f} | "
                  f"{s['bw_b']:.2f} | {s['share_b']*100:.1f} | {s['agg']*100:.1f} |")

    if summary:
        share = np.array([s['share_b'] for s in summary])
        agg = np.array([s['agg'] for s in summary])
        md += ['', f"- 平均 jobB 份额 **{share.mean()*100:.1f}% ± {share.std()*100:.1f}**（n={len(summary)}）"]
        md += [f"- 并发聚合/solo {agg.mean()*100:.1f}% ± {agg.std()*100:.1f}"]
        md += ['', '### 对照结论（对照 test3 的 P6:P3 = 6:4 / jobB 份额 ~59%）', '']
        if share.mean() < 0.52:
            md += [f'- **同优先级下两流近似均分（{share.mean()*100:.1f}%）** → '
                   'test3 的 6:4 确由优先级差造成：SP/优先级机制**有效但非严格**'
                   '（非 per-packet 饿死，只是把共享瓶颈内的带宽按 6:4 倾斜）。']
        elif share.mean() < 0.60:
            md += [f'- **同优先级下份额 {share.mean()*100:.1f}% 仍接近 test3 的 ~59%** → '
                   '6:4 基本与优先级无关，由拥塞控制/多流共享机制固有决定；'
                   '优先级在并发分配上的作用有限。']
        else:
            md += [f'- **同优先级下份额 {share.mean()*100:.1f}%** → 需结合原始数据复查。']

    with open(os.path.join(ANA_DIR, 'exp2_ctrl_dscp16_report.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f'\n报告已保存: {ANA_DIR}/exp2_ctrl_dscp16_report.md')


if __name__ == '__main__':
    main()
