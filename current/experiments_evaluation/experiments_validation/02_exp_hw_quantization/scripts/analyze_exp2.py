#!/usr/bin/env python3
"""
Exp2 分析脚本 — 硬件量化误差与 P6 抢占微观验证

测试1（P6 抢占）：
  - P6 流 vs P3 流带宽时间序列（争抢窗口内）
  - 抢占度 = P6_bw / (P6_bw + P3_bw)（应接近 1）
  - P3 饿死度 = P3_bw / solo（应接近 0）
  - 争抢窗口用 ts 时间戳对齐：P6(jobB) 全程都在窗口内，
    P3(jobA) 只取与 jobB 时间重叠的迭代

数据布局（run_test1_preempt.sh 产物）：
  <round>/
    exp2_soloB_rank0_iter.csv          # solo 校准（P6，20 iters）
    exp2_soloB_rank0_solocalib.log     # solo 校准日志
    attemptN/                          # 每次并发尝试（N=1,2,...）
      exp2_jobA_rank0_iter.csv         # P3（长作业，150 iters）
      exp2_jobB_rank0_iter.csv         # P6（短作业，60 iters，后启动 3s）
      exp2_jobA_rank0_main.log         # 含 `平均带宽` 行
      ...

测试2（P3 内部共享）：
  - 3×P3 各流带宽时间序列
  - 公平份额 = solo/3；Jain 公平指数
  - 聚合利用率 = sum(bw_i)/solo
  - 性能慢度 = 各流相对公平份额的偏离（FIFO 不均匀导致的慢度）

输出 analysis/：exp2_test1_preempt.png, exp2_test2_p3share.png, exp2_report.md
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


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mean_bw(rows, key='bw_gbps'):
    bws = [float(r[key]) for r in rows if r.get(key) not in ('', 'NA', None)]
    return sum(bws) / len(bws) if bws else 0.0


def jain_index(values):
    """Jain 公平指数，输入为各流平均带宽（正数）。"""
    vals = np.array([v for v in values if v > 0])
    if len(vals) == 0:
        return 0.0
    return (vals.sum() ** 2) / (len(vals) * (vals ** 2).sum())


def parse_solo_bw(round_dir):
    """solo 校准带宽：优先读 iter.csv，失败则解析 .log 中 '平均带宽 = X Gbps'。"""
    csv_path = os.path.join(round_dir, 'exp2_soloB_rank0_iter.csv')
    if os.path.exists(csv_path):
        rows = load_csv(csv_path)
        if rows:
            return mean_bw(rows), 'csv'
    for name in ('exp2_soloB_rank0_solocalib.log',):
        log_path = os.path.join(round_dir, name)
        if os.path.exists(log_path):
            with open(log_path) as f:
                m = re.search(r'平均带宽\s*=\s*([\d.]+)\s*Gbps', f.read())
            if m:
                return float(m.group(1)), 'log'
    return 0.0, None


def find_success_attempts(round_dir):
    """返回所有 jobA+jobB 数据齐全的 attempt 目录列表。"""
    out = []
    for name in sorted(glob.glob(os.path.join(round_dir, 'attempt*'))):
        if (os.path.exists(os.path.join(name, 'exp2_jobA_rank0_iter.csv')) and
                os.path.exists(os.path.join(name, 'exp2_jobB_rank0_iter.csv'))):
            out.append(name)
    return out


def contest_window(jobA_rows, jobB_rows):
    """按 ts 对齐争抢窗口：窗口 = [jobB 首迭代, jobB 末迭代]。
    jobA 只保留落在窗口内的迭代；jobB 全部保留。"""
    if not jobB_rows:
        return [], []
    b_ts = sorted(float(r['ts']) for r in jobB_rows if r.get('ts'))
    if not b_ts:
        return jobA_rows, jobB_rows
    t0, t1 = b_ts[0], b_ts[-1]
    win_a = [r for r in jobA_rows
             if r.get('ts') and t0 <= float(r['ts']) <= t1]
    return win_a, jobB_rows


def analyze_test1(rounds):
    print('\n=== 测试1: P6 抢占 P3 ===')
    summary = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in rounds:
        solo, solo_src = parse_solo_bw(r)
        atts = find_success_attempts(r)
        if not atts:
            print(f'  [{os.path.basename(r)}] 无成功 attempt（jobA+jobB 数据齐全），跳过')
            continue
        # 取数据最完整的 attempt（iter 数最多）
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
        preempt = bw_b / (bw_a + bw_b) if (bw_a + bw_b) > 0 else 0.0
        starve = bw_a / solo if solo > 0 else 1.0
        summary.append({'name': os.path.basename(r), 'attempt': os.path.basename(att),
                        'solo': solo, 'solo_src': solo_src, 'p3_bw': bw_a,
                        'p6_bw': bw_b, 'n_a': len(win_a), 'n_b': len(win_b),
                        'preempt': preempt, 'starve': starve})
        print(f"  [{os.path.basename(r)}/{os.path.basename(att)}] "
              f"solo={solo:.2f}G({solo_src}), 窗口P3={bw_a:.2f}G(n={len(win_a)}), "
              f"窗口P6={bw_b:.2f}G(n={len(win_b)}), "
              f"抢占度={preempt*100:.1f}%, P3饿死度={starve*100:.1f}%")

        for rows, ls, label in ((win_a, '-', f"{os.path.basename(r)} P3(window)"),
                                (win_b, '--', f"{os.path.basename(r)} P6")):
            x = [int(row['iter']) for row in rows]
            y = [float(row['bw_gbps']) for row in rows]
            ax.plot(x, y, alpha=0.7, ls=ls, label=label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Bandwidth (Gbps)')
    ax.set_title('Test1: P6 preempts P3 (SP queue, contest window)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANA_DIR, 'exp2_test1_preempt.png'), dpi=200)
    print(f'图已保存: {ANA_DIR}/exp2_test1_preempt.png')
    return summary


# ---------------------------------------------------------------------------
def analyze_test2(rounds):
    print('\n=== 测试2: 3×P3 内部共享 ===')
    summary = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['C0', 'C1', 'C2']
    for r in rounds:
        solo, _ = parse_solo_bw(r)
        if solo == 0.0:
            csv_path = os.path.join(r, 'exp2_solo_rank0_iter.csv')
            if os.path.exists(csv_path):
                solo = mean_bw(load_csv(csv_path))
        flows = []
        for i in range(1, 4):
            rows = load_csv(os.path.join(r, f'exp2_p3flow{i}_rank0_iter.csv'))
            if rows:
                flows.append({'idx': i, 'bw': mean_bw(rows), 'rows': rows})
        if not flows:
            print(f'  [{os.path.basename(r)}] 数据缺失，跳过')
            continue
        bws = [f['bw'] for f in flows]
        fair = solo / 3 if solo > 0 else 0.0
        jain = jain_index(bws)
        agg_util = sum(bws) / solo if solo > 0 else 0.0
        worst_rel = min(bws) / fair if fair > 0 else 1.0
        slowdown_pct = (1 - worst_rel) * 100
        summary.append({'name': os.path.basename(r), 'solo': solo, 'flows': bws,
                        'fair': fair, 'jain': jain, 'agg_util': agg_util,
                        'slowdown_pct': slowdown_pct})
        print(f"  [{os.path.basename(r)}] solo={solo:.2f}G, fair={fair:.2f}G, "
              f"flows={[f'{b:.2f}' for b in bws]}, Jain={jain:.3f}, "
              f"聚合利用率={agg_util*100:.1f}%, 最差流慢度={slowdown_pct:.1f}%")

        for f, c in zip(flows, colors):
            x = [int(row['iter']) for row in f['rows']]
            y = [float(row['bw_gbps']) for row in f['rows']]
            ax.plot(x, y, alpha=0.7, color=c,
                    label=f"{os.path.basename(r)} flow{f['idx']}")
    ax.axhline(0, c='gray', ls='--', lw=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Bandwidth (Gbps)')
    ax.set_title('Test2: 3×P3 share one TC queue (FIFO)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANA_DIR, 'exp2_test2_p3share.png'), dpi=200)
    print(f'图已保存: {ANA_DIR}/exp2_test2_p3share.png')
    return summary


def main():
    t1_rounds = sorted(glob.glob(os.path.join(DATA_DIR, 'exp2_test1_r*')))
    t2_rounds = sorted(glob.glob(os.path.join(DATA_DIR, 'exp2_test2_r*')))

    s1 = analyze_test1(t1_rounds)
    s2 = analyze_test2(t2_rounds)

    # 修复前/后分界：2026-08-08 20:44（DSCP 映射修正后的首轮）
    CUTOFF = '20260808_204437'
    def ts_of(s):
        return '_'.join(os.path.basename(s['name']).split('_')[-2:])
    s1_fixed = [s for s in s1 if ts_of(s) >= CUTOFF]
    s1_old = [s for s in s1 if ts_of(s) < CUTOFF]

    md = ['# Exp2 硬件量化误差与 P6 抢占 — 分析报告', '']
    md += ['> 统计口径：抢占度/饿死度在 **争抢窗口内** 计算'
           '（jobA 仅取与 jobB ts 重叠的迭代），'
           '避免 P6 短作业退出后 P3 独占链路拉高平均。', '']
    md += ['> **调度约束（2026-08-10 更新）**：两端 NIC 均按 DSCP 分类（226 出口'
           '实测 DSCP8→tx_prio1/DSCP16→tx_prio2/DSCP0→tx_prio0，见 '
           'HANDOFF_physical_evidence.md §d）。58% 而非完全饿死的归因（v6）见文末'
           '附录：SP 严格 per-packet（perftest 实锤），58% 系 NCCL 包级突发间隙所致。', '',
           '## 测试1: P6 抢占（验证严格优先）', '',
           '| Round | attempt | solo(G) | 窗口P3(G) | 窗口P6(G) | 抢占度(%) | P3饿死度(%) | 映射 |',
           '|-------|---------|---------|-----------|-----------|-----------|-------------|------|']
    for s in s1:
        fixed = os.path.basename(s['name']) >= CUTOFF
        md.append(f"| {s['name']} | {s['attempt']} | {s['solo']:.2f} | "
                  f"{s['p3_bw']:.2f} | {s['p6_bw']:.2f} | "
                  f"{s['preempt']*100:.1f} | {s['starve']*100:.1f} | "
                  f"{'修正后' if fixed else '修复前'} |")
    if s1_fixed:
        avg_p = np.mean([s['preempt'] for s in s1_fixed]) * 100
        avg_s = np.mean([s['starve'] for s in s1_fixed]) * 100
        sd_p = np.std([s['preempt'] for s in s1_fixed]) * 100
        md.append(f"\n- 修正后平均抢占度 {avg_p:.1f}%±{sd_p:.1f} "
                  f"（n={len(s1_fixed)}；物理上限说明见上）")
        md.append(f"- 修正后平均 P3 饿死度 {avg_s:.1f}%（~50%，SP 未完全饿死，归因见文末附录）")
    if s1_old:
        avg_p_old = np.mean([s['preempt'] for s in s1_old]) * 100
        md.append(f"- 修复前平均抢占度 {avg_p_old:.1f}%（n={len(s1_old)}，P6 反被抢占）")
        md.append(f"- **修复前后对比**：抢占度 "
                  f"{avg_p_old:.1f}% → {avg_p:.1f}%，方向反转，机制有效")

    md += ['', '## 测试2: P3 内部共享（FIFO 退化为 Fair 的程度）', '',
           '| Round | solo(G) | fair=1/3(G) | 各流 bw(G) | Jain | 聚合利用率(%) | 最差流慢度(%) |',
           '|-------|---------|------------|------------|------|---------------|--------------|']
    for s in s2:
        md.append(f"| {s['name']} | {s['solo']:.2f} | {s['fair']:.2f} | "
                  f"{'/'.join(f'{b:.2f}' for b in s['flows'])} | {s['jain']:.3f} | "
                  f"{s['agg_util']*100:.1f} | {s['slowdown_pct']:.1f} |")
    if s2:
        avg_j = np.mean([s['jain'] for s in s2])
        avg_sd = np.mean([s['slowdown_pct'] for s in s2])
        avg_util = np.mean([s['agg_util'] for s in s2])
        md.append(f"\n- 平均 Jain 指数 {avg_j:.3f}（1=完全公平；阈值 ≥0.9 → "
                  f"{'通过' if avg_j >= 0.9 else '未通过'}）")
        md.append(f"- 平均聚合利用率 {avg_util*100:.1f}%（接近 100% 说明链路无空闲）")
        md.append(f"- 平均最差流慢度 {avg_sd:.1f}%（量化 P3 内部无法细分优先级导致的性能慢度）")

    # ------------------------------------------------------------------
    # 附录：58% 归因（v6）— SP 严格性三组受控实验（2026-08-10）
    # ------------------------------------------------------------------
    md += ['', '## 附录: 58% 抢占度归因（v6，SP 严格性判定）', '',
           'test1 抢占度 58.2%±0.2%（而非 95%+ 完全饿死）的成因，由三组受控实验'
           '（各 3 轮）判定（详见 `exp2_test3_record.md`）：', '',
           '| 实验 | 流量形态 | 优先级差 | 结果 |',
           '|------|---------|---------|------|',
           '| perftest 双流（`exp2_perftest_report.md`） | 持续饱和 | 有 | 高优先级'
           '45.92±0.10 Gb/s，低优先级 **饿死 99.9%**（0.06 Gb/s） |',
           '| NCCL test3（`exp2_test3_report.md`） | 包级突发 | 有 | 58.7%±0.0（6:4） |',
           '| NCCL 同优先级对照（`exp2_ctrl_dscp16_report.md`） | 包级突发 | 无 | **50:50** |',
           '',
           '**判定（v6）**：SP 队列**严格 per-packet**——持续饱和流下低优先级被完全'
           '饿死，实现正确、无配置错配。NCCL 场景的 58% 系 **NCCL 流量包级突发间隙**'
           '（chunk 注入 + DCQCN 暂停）所致：高优先级流 tc:0 队列的微秒级空闲被'
           '低优先级利用。证据链自洽：perftest 99.9% 饿死 → NCCL 59:41 → 同优先级'
           '50:50，逐级由"流量形态"与"优先级差"两个因素解释。',
           '早期 v5 判定（"SP 非严格/物理上限"）已被 perftest 证据推翻。']

    with open(os.path.join(ANA_DIR, 'exp2_report.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f'\n报告已保存: {ANA_DIR}/exp2_report.md')


if __name__ == '__main__':
    main()
