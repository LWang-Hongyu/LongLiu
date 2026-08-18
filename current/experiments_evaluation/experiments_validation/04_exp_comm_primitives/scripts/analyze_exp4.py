#!/usr/bin/env python3
"""
Exp4 — 通信原语多样性（AllGather）分析
=======================================
输入（每个轮次目录 data/exp4_r<N>_<ts>/）：
  exp4_jobA_rank0_iter.csv       per-iter priority/dscp/pi/bw
  exp4_jobA_rank0_epoch.csv      per-epoch 汇总
  ttarget.json                   校准锚点（T_target / solo_bw）
  nic_10.csv, nic_226.csv         NIC prio0-7 计数器
  gpu_10.csv, gpu_226.csv         GPU 状态（供稳定性参考）

分析指标：
  1. DSCP 切换准确性 —— 逐轮提取 priority→dscp 轨迹，统计切换次数、
     各级别驻留 iter 比例；对照 NIC prio 计数器验证流量确实按
     dscp 对应的队列行走（该点依赖监控时间窗口，作定性对照）。
  2. 锚点测量精度 —— calib 的 solo_bw 与 main 全段带宽均值偏差、
     T_target 与 main 实际每 epoch 通信时间的偏差（反映锚点可复现性）。

输出（写入 analysis/）：
  exp4_dscp_trajectory.png   多轮 priority/dscp 轨迹（分面）
  exp4_anchor_accuracy.png   锚点 vs 观测带宽对比
  exp4_report.md             结构化报告
"""
import os
import sys
import csv
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
ANALYSIS = os.path.join(BASE, 'analysis')
os.makedirs(ANALYSIS, exist_ok=True)

DSCP_LABEL = {8: 'P6/DSCP8', 0: 'P4/DSCP0', 16: 'P3/DSCP16',
              24: 'P2/DSCP24', 32: 'P1/DSCP32', 40: 'P0/DSCP40'}


def load_rounds():
    """收集所有 exp4_r* 轮次目录，返回排序后的目录列表。"""
    dirs = sorted(glob.glob(os.path.join(DATA, 'exp4_r*')))
    return dirs


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def iter_priority_stats(iter_rows):
    """从 per-iter 记录统计优先级驻留。"""
    prio_seq = [int(r['priority']) for r in iter_rows]
    uniq = []
    for p in prio_seq:
        if not uniq or uniq[-1] != p:
            uniq.append(p)
    n_switch = max(len(uniq) - 1, 0)
    # 驻留 iter 比例
    counts = {}
    for p in prio_seq:
        counts[p] = counts.get(p, 0) + 1
    total = len(prio_seq)
    frac = {p: c / total for p, c in counts.items()} if total else {}
    return prio_seq, n_switch, frac


def nic_prio_budget(nic_path, iface_key=None):
    """读取 NIC prio 计数器：返回 {prio: bytes}（最近一行增量差）。"""
    if not os.path.exists(nic_path):
        return None
    rows = read_csv(nic_path)
    if len(rows) < 2:
        return None
    first, last = rows[0], rows[-1]
    out = {}
    for p in range(8):
        k = f'tx_prio{p}_bytes'
        if k in first and k in last:
            try:
                out[p] = max(float(last[k]) - float(first[k]), 0.0)
            except (ValueError, KeyError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=str, default=None,
                    help='逗号分隔的轮次名（默认扫描全部 data/exp4_r*）')
    args = ap.parse_args()

    dirs = load_rounds()
    if args.rounds:
        dirs = [os.path.join(DATA, r) for r in args.rounds.split(',')]
        dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print(f'[Exp4] 未找到轮次数据: {DATA}')
        sys.exit(1)

    round_names = [os.path.basename(d) for d in dirs]
    print(f'[Exp4] 分析 {len(dirs)} 轮: {round_names}')

    # ------------------------------------------------------------------
    # 图1: DSCP 切换轨迹（多轮分面）
    # ------------------------------------------------------------------
    n_rounds = len(dirs)
    fig, axes = plt.subplots(n_rounds, 1, figsize=(10, 3.2 * n_rounds),
                             sharex=True)
    if n_rounds == 1:
        axes = [axes]

    for ax, d in zip(axes, dirs):
        rname = os.path.basename(d)
        iter_csv = os.path.join(d, 'exp4_jobA_rank0_iter.csv')
        rows = read_csv(iter_csv)
        iters = [int(r['iter']) for r in rows]
        dscps = [int(r['dscp']) for r in rows]
        prios = [int(r['priority']) for r in rows]
        ax.step(iters, prios, where='post', color='tab:blue', lw=1.6,
                label='priority')
        ax.set_ylabel(f'{rname}\npriority')
        ax.set_yticks(sorted(set(prios)))
        ax.grid(alpha=0.3)
        # 标出切换点
        switch_pts = [i for i in range(1, len(prios))
                      if prios[i] != prios[i - 1]]
        if switch_pts:
            ax.scatter(switch_pts, [prios[i] for i in switch_pts],
                       marker='v', s=40, color='red', zorder=5,
                       label='switch')
        # 标注 DSCP 文本（在右侧）
        seen = set()
        for p, dsc in zip(prios, dscps):
            if p not in seen:
                seen.add(p)
                ax.text(len(prios) - 1, p, f'  DSCP{dsc}',
                        va='center', fontsize=9, color='gray')
        ax.legend(loc='upper right', ncol=2, framealpha=0.9)

    axes[0].set_title('Exp4: DSCP 切换轨迹（AllGather, LongLiu π 调度）')
    axes[-1].set_xlabel('global iteration')
    axes[-1].set_xticks(np.arange(0, len(iters) + 1, 25))
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, 'exp4_dscp_trajectory.png'))
    plt.close(fig)

    # ------------------------------------------------------------------
    # 图2: 锚点精度（solo_bw vs main 观测带宽；T_target vs 实际）
    # ------------------------------------------------------------------
    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.5))

    solo_bws, main_bws, tt_ms_list, obs_ms_list = [], [], [], []
    for d in dirs:
        ttarget = os.path.join(d, 'ttarget.json')
        if not os.path.exists(ttarget):
            continue
        tdata = read_json(ttarget)
        solo_bw = tdata.get('solo_bw_gbps', 0.0)
        tt_ms = tdata.get('target_comm_time_ms', 0.0)
        epoch_csv = os.path.join(d, 'exp4_jobA_rank0_epoch.csv')
        rows = read_csv(epoch_csv)
        if not rows:
            continue
        main_bw = np.mean([float(r['avg_bw_gbps']) for r in rows])
        obs_ms = np.mean([float(r['avg_comm_s']) for r in rows]) * 1000.0
        solo_bws.append(solo_bw)
        main_bws.append(main_bw)
        tt_ms_list.append(tt_ms)
        obs_ms_list.append(obs_ms)

    if solo_bws:
        x = np.arange(len(solo_bws))
        w = 0.35
        axa.bar(x - w / 2, solo_bws, w, label='solo (calib)', color='#4C72B0')
        axa.bar(x + w / 2, main_bws, w, label='main (contended)',
                color='#DD8452')
        axa.set_xticks(x)
        axa.set_xticklabels([os.path.basename(d) for d in dirs],
                            rotation=30, ha='right')
        axa.set_ylabel('Bandwidth (Gbps)')
        axa.set_title('锚点测量精度: solo vs main 带宽')
        axa.legend()
        axa.grid(alpha=0.3, axis='y')

        axb.bar(x - w / 2, tt_ms_list, w, label='T_target (calib)',
                color='#55A868')
        axb.bar(x + w / 2, obs_ms_list, w, label='actual (main)',
                color='#C44E52')
        axb.set_xticks(x)
        axb.set_xticklabels([os.path.basename(d) for d in dirs],
                            rotation=30, ha='right')
        axb.set_ylabel('Comm time (ms/epoch)')
        axb.set_title('锚点测量精度: T_target vs 实际 epoch 通信时间')
        axb.legend()
        axb.grid(alpha=0.3, axis='y')

    fig2.tight_layout()
    fig2.savefig(os.path.join(ANALYSIS, 'exp4_anchor_accuracy.png'))
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    rep = []
    rep.append('# Exp4 通信原语多样性（AllGather）报告\n')
    rep.append(f'- 生成时间: {__import__("time").strftime("%Y-%m-%d %H:%M:%S")}')
    rep.append(f'- 分析轮次: {", ".join(round_names)}\n')
    rep.append('## 1. DSCP 切换准确性\n')
    rep.append('| 轮次 | 优先级切换次数 | 各级别驻留 iter 比例 | 说明 |')
    rep.append('|---|---|---|---|')
    for d in dirs:
        rname = os.path.basename(d)
        iter_csv = os.path.join(d, 'exp4_jobA_rank0_iter.csv')
        rows = read_csv(iter_csv)
        prio_seq, n_switch, frac = iter_priority_stats(rows)
        frac_str = ', '.join(f'P{p}:{v*100:.0f}%'
                             for p, v in sorted(frac.items()))
        note = '有切换' if n_switch > 0 else '始终驻留单一优先级'
        rep.append(f'| {rname} | {n_switch} | {frac_str} | {note} |')
    rep.append('')
    rep.append('> 说明：DSCP 由 priority 查表映射（P6→8, P4→0, P3→16, ...）。'
               '若流量确实按该 DSCP 行走，NIC prio 计数器应出现相应队列增长；'
               '该定量对照见分析脚本日志中的 nic_prio_budget 输出。\n')

    # NIC 对照（定性）
    rep.append('## 2. NIC prio 计数器对照（轮次 → {prio: bytes}）\n')
    for d in dirs:
        rname = os.path.basename(d)
        for suf in ('10', '226'):
            nicp = os.path.join(d, f'nic_{suf}.csv')
            budget = nic_prio_budget(nicp)
            if budget is not None:
                total = sum(budget.values())
                top = sorted(budget.items(), key=lambda kv: -kv[1])[:3]
                top_str = ', '.join(f'prio{p}: {v/total*100:.0f}%'
                                    for p, v in top if total > 0)
                rep.append(f'- {rname} (node {suf}): {top_str} '
                           f'[prio0={budget.get(0,0)/1e9:.1f}GB, '
                           f'prio1={budget.get(1,0)/1e9:.1f}GB, '
                           f'prio2={budget.get(2,0)/1e9:.1f}GB]')
    rep.append('')

    rep.append('## 3. 锚点测量精度\n')
    if solo_bws:
        rep.append('| 轮次 | solo_bw (Gbps) | main 观测带宽 (Gbps) | '
                   '偏差 % | T_target (ms) | main 实际 (ms) | 偏差 % |')
        rep.append('|---|---|---|---|---|---|')
        for i, d in enumerate(dirs):
            rname = os.path.basename(d)
            if i >= len(solo_bws):
                continue
            dev_bw = (main_bws[i] - solo_bws[i]) / solo_bws[i] * 100
            dev_tt = (obs_ms_list[i] - tt_ms_list[i]) / tt_ms_list[i] * 100
            rep.append(f'| {rname} | {solo_bws[i]:.2f} | {main_bws[i]:.2f} | '
                       f'{dev_bw:+.1f}% | {tt_ms_list[i]:.1f} | '
                       f'{obs_ms_list[i]:.1f} | {dev_tt:+.1f}% |')
        if len(solo_bws) >= 2:
            cv_bw = np.std(solo_bws) / np.mean(solo_bws) * 100
            cv_tt = np.std(tt_ms_list) / np.mean(tt_ms_list) * 100
            rep.append(f'\n- 跨轮 solo_bw 变异系数 CV={cv_bw:.1f}% '
                       f'（<5% 视为锚点测量稳定）')
            rep.append(f'- 跨轮 T_target 变异系数 CV={cv_tt:.1f}%')
    rep.append('')

    report_path = os.path.join(ANALYSIS, 'exp4_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(rep))
    print(f'[Exp4] 报告已写入 {report_path}')
    print(f'[Exp4] 图: {ANALYSIS}/exp4_dscp_trajectory.png, '
          f'{ANALYSIS}/exp4_anchor_accuracy.png')


if __name__ == '__main__':
    main()
