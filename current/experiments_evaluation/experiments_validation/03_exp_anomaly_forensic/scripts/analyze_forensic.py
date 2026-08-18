#!/usr/bin/env python3
"""
Exp3 Forensic 分析 — 图11异常点（rep2 r1）底层原因定位

流程：
  1. 加载重跑数据（最新 exp3_rerun_* 目录）
  2. 从带时间戳的作业日志提取 epoch → 墙钟时间映射
  3. 对齐 NIC/GPU 硬件计数器
  4. 定位异常窗口：LongLiu 紧作业 slowdown 异常升高时段
  5. 检查异常窗口内：GPU 降频（thermal/power）/ RoCE 重传 / out_of_buffer（PFC 下游效应）/
     中断率 / PCIe 吞吐异常
  6. 输出时间线图 + 结论

输出 analysis/：exp3_timeline.png, exp3_report.md
"""
import os
import re
import csv
import glob
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
ANA_DIR = os.path.join(BASE, 'analysis')
os.makedirs(ANA_DIR, exist_ok=True)

SLOWDOWN_ANOMALY_THRESH = 1.20   # LL 紧作业 slowdown 超过该值视为异常升高


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_epoch_times(log_path, run_date):
    """从带时间戳日志提取每个 epoch 的开始墙钟时间（epoch → unix epoch 秒）。
    日志行格式: [HH:MM:SS] [JobX-...-MAIN] iter <g> (epoch <e>): ..."""
    rows = {}
    base = datetime.datetime.combine(run_date, datetime.time(0, 0, 0))
    with open(log_path) as f:
        for line in f:
            m = re.match(r'\[(\d{2}:\d{2}:\d{2})\].*iter \d+ \(epoch (\d+)\)', line)
            if not m:
                continue
            hms = m.group(1)
            epoch = int(m.group(2))
            h, mi, s = map(int, hms.split(':'))
            ts = (base + datetime.timedelta(hours=h, minutes=mi, seconds=s)).timestamp()
            if epoch not in rows:
                rows[epoch] = ts
    return rows


def load_nic(path):
    """返回 (ts[], dict[col][i])。"""
    rows = load_csv(path)
    ts = []
    cols = {}
    for r in rows:
        try:
            ts.append(float(r['ts']))
        except (ValueError, KeyError):
            continue
    for key in rows[0]:
        if key == 'ts':
            continue
        cols[key] = [float(r[key]) if r[key] not in ('', 'NA') else np.nan for r in rows]
    return np.array(ts), cols


def load_gpu(path):
    """nvidia-smi csv → (datetime[], cols)。"""
    rows = load_csv(path)
    ts = []
    cols = {}
    keys = [k.strip() for k in rows[0].keys()]
    for r in rows:
        try:
            dt = datetime.datetime.strptime(r['timestamp'].strip(), '%Y/%m/%d %H:%M:%S.%f')
            ts.append(dt.timestamp())
        except Exception:
            try:
                dt = datetime.datetime.strptime(r['timestamp'].strip(), '%Y/%m/%d %H:%M:%S')
                ts.append(dt.timestamp())
            except Exception:
                ts.append(np.nan)
    for k in keys:
        if k == 'timestamp':
            continue
        cols[k] = [float(r[k]) if r[k].strip() not in ('', 'NA', '[N/A]') else np.nan
                   for r in rows]
    return np.array(ts), cols


def window_stats(ts, cols, t0, t1):
    """在 [t0,t1) 窗口内对每列求均值/速率。"""
    mask = (ts >= t0) & (ts < t1)
    stats = {}
    for k, v in cols.items():
        sel = np.array(v)[mask]
        sel = sel[~np.isnan(sel)]
        if len(sel) == 0:
            stats[k] = np.nan
        elif 'retrans' in k or 'nak' in k or 'out_of' in k or 'naks' in k or 'irq' in k or 'data' in k:
            # 累计型计数器 → 窗口内速率（每秒）
            stats[k] = (sel[-1] - sel[0]) / max(t1 - t0, 1e-6)
        else:
            stats[k] = np.mean(sel)
    return stats


def main():
    runs = sorted(glob.glob(os.path.join(DATA_DIR, 'exp3_rerun_*')))
    if not runs:
        print('未找到 exp3 重跑数据'); return
    rundir = runs[-1]
    print(f'分析目录: {rundir}')

    with open(os.path.join(rundir, 'run_start.epoch')) as f:
        run_start = int(f.read().strip())
    run_date = datetime.datetime.fromtimestamp(run_start).date()

    # ---------- 1) epoch CSV（LL 紧作业 = Job B 在 phase2 为 tight）----------
    ll_epoch = os.path.join(rundir, 'exp3_jobB_longliu_rank0_epoch.csv')
    llA_epoch = os.path.join(rundir, 'exp3_jobA_longliu_rank0_epoch.csv')
    cx_epoch = os.path.join(rundir, 'exp3_jobB_crux_rank0_epoch.csv')
    cxA_epoch = os.path.join(rundir, 'exp3_jobA_crux_rank0_epoch.csv')

    ll_rows = load_csv(ll_epoch)
    llA_rows = load_csv(llA_epoch)
    cx_rows = load_csv(cx_epoch) if os.path.exists(cx_epoch) else []
    cxA_rows = load_csv(cxA_epoch) if os.path.exists(cxA_epoch) else []

    if not ll_rows:
        print('缺少 LongLiu 紧作业 epoch CSV'); return

    # ---------- 2) epoch → 墙钟（LL 模式日志）----------
    ll_log = os.path.join(rundir, 'exp3_jobB_longliu_node101.log')
    epoch_times = parse_epoch_times(ll_log, run_date)
    print(f'  LL 紧作业 epoch 时间映射: {len(epoch_times)} 个 epoch')

    # ---------- 3) 硬件计数器 ----------
    nic10_ts, nic10 = load_nic(os.path.join(rundir, 'nic_10.csv'))
    nic26_ts, nic26 = load_nic(os.path.join(rundir, 'nic_226.csv'))
    gpu10_ts, gpu10 = load_gpu(os.path.join(rundir, 'gpu_10.csv'))
    gpu26_ts, gpu26 = load_gpu(os.path.join(rundir, 'gpu_226.csv'))

    # ---------- 4) 定位异常窗口 ----------
    # LL 紧作业（Job B）在 phase2（epoch 7+）的 slowdown
    anomaly_epochs = [r['epoch'] for r in ll_rows
                      if int(r['epoch']) >= 7 and float(r['slowdown']) > SLOWDOWN_ANOMALY_THRESH]
    baseline_epochs = [r['epoch'] for r in ll_rows if 0 <= int(r['epoch']) < 7]
    print(f'  LL 紧作业异常升高 epoch: {anomaly_epochs}（阈值 {SLOWDOWN_ANOMALY_THRESH}）')
    print(f'  对照基线 epoch: {baseline_epochs}')

    # 汇总每个 epoch 的硬件指标
    def epoch_hw(epoch):
        if epoch not in epoch_times:
            return None
        t0 = epoch_times[epoch]
        t1 = epoch_times.get(epoch + 1, t0 + 30)
        s26 = window_stats(nic26_ts, nic26, t0, t1)
        s10 = window_stats(nic10_ts, nic10, t0, t1)
        g26 = window_stats(gpu26_ts, gpu26, t0, t1)
        g10 = window_stats(gpu10_ts, gpu10, t0, t1)
        xd, rd = s26.get('port_xmit_data', np.nan), s26.get('port_rcv_data', np.nan)
        pcie = ((xd + rd) * 8 / 1e9) if (xd == xd and rd == rd) else np.nan
        return {'epoch': epoch, 't0': t0, 't1': t1,
                'roce_retrans_226': s26.get('roce_adp_retrans', np.nan),
                'rnr_nak_226': s26.get('rnr_nak_retry_err', np.nan),
                'oob_226': s26.get('out_of_buffer', np.nan),
                'seq_err_226': s26.get('packet_seq_err', np.nan),
                'irq_226': s26.get('irq_count', np.nan),
                'pcie_tput_226': pcie,
                'sm_clk_226': g26.get('clocks.sm', np.nan),
                'gtemp_226': g26.get('temperature.gpu', np.nan),
                'thr_active_226': g26.get('clocks_throttle_reasons.active', np.nan),
                'util_226': g26.get('utilization.gpu', np.nan),
                'sm_clk_10': g10.get('clocks.sm', np.nan),
                'gtemp_10': g10.get('temperature.gpu', np.nan),
                }

    hw_anom = [epoch_hw(e) for e in anomaly_epochs]
    hw_base = [epoch_hw(e) for e in baseline_epochs]
    hw_anom = [h for h in hw_anom if h]
    hw_base = [h for h in hw_base if h]

    # ---------- 5) 结论判定 ----------
    def avg(key, lst):
        vals = [h[key] for h in lst if not (h[key] is np.nan or h[key] is None)]
        vals = [v for v in vals if v == v]
        return np.mean(vals) if vals else np.nan

    conclusions = []
    if hw_anom and hw_base:
        # GPU 降频检查（226 = BF-3 宿主 GPU RTX 5000）
        sm_a, sm_b = avg('sm_clk_226', hw_anom), avg('sm_clk_226', hw_base)
        tmp_a, tmp_b = avg('gtemp_226', hw_anom), avg('gtemp_226', hw_base)
        thr_a = avg('thr_active_226', hw_anom)
        if sm_a == sm_a and sm_b == sm_b and sm_b > 0:
            if sm_a < 0.9 * sm_b:
                conclusions.append(f'GPU 降频: 异常窗 SM 时钟 {sm_a:.0f}MHz vs 基线 {sm_b:.0f}MHz '
                                   f'（降幅 {100*(1-sm_a/sm_b):.1f}%）→ 疑似 thermal throttling')
            else:
                conclusions.append(f'GPU 无显著降频（异常窗 {sm_a:.0f}MHz vs 基线 {sm_b:.0f}MHz），'
                                   f'温度异常窗 {tmp_a:.0f}C vs 基线 {tmp_b:.0f}C，throttle 事件均值 {thr_a:.2f}')
        else:
            conclusions.append('GPU 数据不足，无法判定降频')

        # RoCE 重传 / PFC 效应
        r_a, r_b = avg('roce_retrans_226', hw_anom), avg('roce_retrans_226', hw_base)
        oob_a, oob_b = avg('oob_226', hw_anom), avg('oob_226', hw_base)
        rnr_a, rnr_b = avg('rnr_nak_226', hw_anom), avg('rnr_nak_226', hw_base)
        if (r_a == r_a and r_b == r_b and r_a > r_b * 5 and r_b >= 0 and r_a > 0) or \
           (oob_a == oob_a and oob_b == oob_b and oob_a > oob_b * 5 and oob_a > 0):
            conclusions.append(f'RoCE 拥塞异常: 异常窗重传 {r_a:.1f}/s vs 基线 {r_b:.1f}/s; '
                               f'out_of_buffer {oob_a:.1f}/s vs {oob_b:.1f}/s; '
                               f'RNR {rnr_a:.1f}/s vs {rnr_b:.1f}/s → 疑似 PFC/队列溢出')
        else:
            conclusions.append(f'RoCE 无显著异常（重传 {r_a:.1f}/s vs 基线 {r_b:.1f}/s; '
                               f'OOB {oob_a:.1f}/s vs {oob_b:.1f}/s; RNR {rnr_a:.1f}/s vs {rnr_b:.1f}/s）')

        # 优先级行为（LL 是否到达 P6）
        prio_anom = [int(r['priority']) for r in ll_rows
                     if int(r['epoch']) in [int(h['epoch']) for h in hw_anom]]
        prio_all = [int(r['priority']) for r in ll_rows]
        conclusions.append(f'LL 紧作业优先级轨迹: 异常窗平均 P{np.mean(prio_anom):.1f}，'
                           f'全程平均 P{np.mean(prio_all):.1f}（P6 占比 '
                           f'{100*sum(1 for p in prio_all if p==6)/max(len(prio_all),1):.0f}%）')
    else:
        conclusions.append('异常窗口或基线数据不足')

    # ---------- 6) 图 ----------
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ep_ll = [int(r['epoch']) for r in ll_rows]
    sd_ll = [float(r['slowdown']) for r in ll_rows]
    axes[0].plot(ep_ll, sd_ll, 'o-', label='LongLiu (tight job B)')
    if cx_rows:
        ep_cx = [int(r['epoch']) for r in cx_rows]
        sd_cx = [float(r['slowdown']) for r in cx_rows]
        axes[0].plot(ep_cx, sd_cx, 's--', label='CRUX (tight job B)')
    axes[0].axhline(SLOWDOWN_ANOMALY_THRESH, c='r', ls=':', lw=0.8)
    axes[0].set_ylabel('slowdown')
    axes[0].set_title(f'Exp3 Forensic: {os.path.basename(rundir)}')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # RoCE 重传率时间线（226）
    ts_r = nic26_ts
    r_ser = np.array([float(v) if v == v else np.nan for v in nic26.get('roce_adp_retrans', [])])
    if len(r_ser) > 1:
        dr = np.diff(np.nan_to_num(r_ser))
        axes[1].plot(ts_r[1:], np.maximum(dr, 0) / np.maximum(np.diff(ts_r), 1e-6), color='C1')
        axes[1].set_ylabel('RoCE retrans/s (226)')
        axes[1].grid(alpha=0.3)

    # GPU SM clock（226）
    if len(gpu26_ts) and 'clocks.sm' in gpu26:
        gts = gpu26_ts
        gsm = np.array([float(v) if v == v else np.nan for v in gpu26['clocks.sm']])
        axes[2].plot(gts, gsm, color='C2')
        axes[2].set_ylabel('GPU SM clk (MHz)')
        axes[2].grid(alpha=0.3)

    # 优先级
    pr = [int(r['priority']) for r in ll_rows]
    ep_pri = [int(r['epoch']) for r in ll_rows]
    axes[3].plot(ep_pri, pr, 'o-', color='C3', label='LL priority')
    axes[3].set_ylabel('priority')
    axes[3].set_xlabel('epoch')
    axes[3].set_ylim(0, 7)
    axes[3].grid(alpha=0.3)

    # 标注异常窗
    for h in hw_anom:
        for ax in axes:
            ax.axvspan(h['epoch'] - 0.5, h['epoch'] + 0.5, color='red', alpha=0.08)

    fig.tight_layout()
    fig.savefig(os.path.join(ANA_DIR, 'exp3_timeline.png'), dpi=150)
    print(f'图已保存: {ANA_DIR}/exp3_timeline.png')

    # ---------- 7) 报告 ----------
    md = ['# Exp3 图11异常点（rep2 r1）Forensic 分析报告', '']
    md.append(f'> 数据: {os.path.basename(rundir)}（重跑 rep2 r1, LL→CX）')
    md.append(f'> 分析时间: {datetime.datetime.now().isoformat(timespec="seconds")}')
    md.append('')
    md.append('## 1. 异常窗口定位')
    md.append('')
    md.append(f'- LL 紧作业 slowdown 异常升高（>{SLOWDOWN_ANOMALY_THRESH}）epoch: '
              f'{anomaly_epochs}')
    md.append(f'- 基线（epoch 0-6）slowdown: '
              f'{np.mean([float(r["slowdown"]) for r in ll_rows if int(r["epoch"])<7] or [0]):.3f}')
    if cx_rows:
        md.append(f'- 同窗 CRUX 紧作业 slowdown: '
                  f'{np.mean([float(r["slowdown"]) for r in cx_rows if int(r["epoch"])>=7] or [0]):.3f}')
    md.append('')
    md.append('## 2. 硬件计数器对比（异常窗 vs 基线，226/BF-3 侧）')
    md.append('')
    md.append('| 指标 | 异常窗均值 | 基线均值 |')
    md.append('|------|-----------|----------|')
    for key, label in [('sm_clk_226', 'GPU SM 时钟 (MHz)'),
                       ('gtemp_226', 'GPU 温度 (C)'),
                       ('roce_retrans_226', 'RoCE 重传 (/s)'),
                       ('oob_226', 'out_of_buffer (/s)'),
                       ('rnr_nak_226', 'RNR NAK (/s)'),
                       ('irq_226', 'IRQ 数 (样本差/窗)')]:
        a, b = avg(key, hw_anom), avg(key, hw_base)
        md.append(f'| {label} | {a:.2f} | {b:.2f} |')
    md.append('')
    md.append('## 3. 结论')
    md.append('')
    for c in conclusions:
        md.append(f'- {c}')
    md.append('')
    md.append('## 4. 备注')
    md.append('- PCIe 吞吐以 226 IB port 计数器速率近似（BF-3 所有宿主流量经 PCIe 到达 NIC）。')
    md.append('- 交换机 PFC 计数器需交换机凭据（本实验环境无免密访问），PFC 效应以 NIC 侧 '
              'out_of_buffer/RNR/重传增量作为下游证据。')

    with open(os.path.join(ANA_DIR, 'exp3_report.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f'报告已保存: {ANA_DIR}/exp3_report.md')


if __name__ == '__main__':
    main()
