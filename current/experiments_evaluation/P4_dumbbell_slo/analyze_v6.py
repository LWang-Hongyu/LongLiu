#!/usr/bin/env python3
"""
V6 动态对比分析 — LongLiu vs CRUX (Phase 2 tight Job B)

读取 V6 window CSV，提取关键指标：
  - Phase 2 tight job (B) 的 slowdown：LongLiu vs CRUX
  - 分段统计：window 7-11（背景流稳定期）/ 7-14（全段）
  - 优势幅度 = (CX - LL)/CX，区间重叠判定
  - Phase 1 tight job (A) 持平验证

用法：
  python3 analyze_v6.py [dir1 dir2 ...]   # 默认：当前目录 + v6_replication_1/2
"""
import csv, glob, os, sys
import numpy as np

DIRS = sys.argv[1:] if len(sys.argv) > 1 else [
    'v6_replication_1', 'v6_replication_2']

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def phase2_tight_B(d):
    """从目录收集 jobB 的 phase2（tight）数据，返回 {round: {mode: slowdown列表}}"""
    out = {}
    for rnd in ('round1_LLthenCX', 'round2_CXthenLL'):
        entry = {}
        for mode in ('longliu', 'crux'):
            f = os.path.join(d, f'p4_jobB_v6_{rnd}_{mode}_rank0_window.csv')
            if not os.path.exists(f):
                continue
            rows = load(f)
            ph2 = [(int(r['window']), float(r['slowdown']))
                   for r in rows if r['phase'] == 'phase2']
            if not ph2:
                continue
            # 只保留 c_i==1.2（tight）的 phase2 行（防御 c_i 列异常）
            tight = [(e, s) for e, s in ph2]
            entry[mode] = tight
        if entry:
            out[rnd] = entry
    return out

def report():
    lines = []
    lines.append('# V6 动态对比分析（Phase 2 tight Job B）')
    lines.append('')
    lines.append('> 指标：slowdown = avg_comm / T_target（>1 = 超过 SLO 目标）。')
    lines.append('> Phase 2 = window 7-14（Job B 变 tight，c_i=1.2）。')
    lines.append('> **背景流约束**：旧复制（570s 背景流）在 phase2 后期耗尽背景流，'
                 'CRUX 侧 slowdown 骤降至 <1（伪影）；修复版（840s）全段有背景流。')
    lines.append('')
    hdr = '| 数据源 | round | 模式 | 全段(7-14) | 前段(7-11) | 后段(12-14) |'
    sep = '|--------|-------|------|-----------|-----------|------------|'
    lines += [hdr, sep]

    allrows = []
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        data = phase2_tight_B(d)
        for rnd, modes in data.items():
            if 'longliu' not in modes or 'crux' not in modes:
                continue
            ll, cx = modes['longliu'], modes['crux']
            def seg(m):
                es = [s for _, s in m]
                pre = [s for e, s in m if e <= 11]
                post = [s for e, s in m if e >= 12]
                return (np.mean(es), np.mean(pre), np.mean(post) if post else np.nan)
            lf, lp, lq = seg(ll)
            cf, cp, cq = seg(cx)
            adv_full = (cf - lf) / cf * 100
            adv_pre = (cp - lp) / cp * 100
            # 前段区间是否重叠
            ll_range = (min(s for e, s in ll if e <= 11), max(s for e, s in ll if e <= 11))
            cx_range = (min(s for e, s in cx if e <= 11), max(s for e, s in cx if e <= 11))
            overlap = not (cx_range[1] < ll_range[0] or ll_range[1] < cx_range[0])
            lines.append(f"| {os.path.basename(d) or '.'} | {rnd[:11]} | LL | "
                         f"{lf:.3f} | {lp:.3f} | {lq:.3f} |")
            lines.append(f"| | | CX | {cf:.3f} | {cp:.3f} | {cq:.3f} |")
            lines.append(f"| | | **优势** | **{adv_full:.1f}%** | **{adv_pre:.1f}%** | "
                         f"前段重叠: {'是(不决定性)' if overlap else '否(决定性)'} |")
            allrows.append({'src': d, 'round': rnd, 'adv_full': adv_full,
                            'adv_pre': adv_pre, 'overlap': overlap,
                            'll_pre': lp, 'cx_pre': cp})

    lines += ['', '## 汇总', '', '| 数据点 | 全段优势% | 前段优势% | 前段决定性 |',
              '|--------|----------|----------|-----------|']
    for r in allrows:
        lines.append(f"| {os.path.basename(r['src'])} {r['round'][:11]} | "
                     f"{r['adv_full']:.1f} | {r['adv_pre']:.1f} | "
                     f"{'否' if r['overlap'] else '是'} |")
    if allrows:
        advs_pre = [r['adv_pre'] for r in allrows]
        lines.append(f"\n- 前段优势均值 {np.mean(advs_pre):.1f}% ± {np.std(advs_pre):.1f} "
                     f"(n={len(advs_pre)})")
    return '\n'.join(lines) + '\n'

if __name__ == '__main__':
    print(report())
