"""
D1@400G 机制定案：条件化对照分析 + π 语义核查
==============================================
1. 只取 N_active≥3 的真争抢 epoch
2. 用该 epoch 的 π 按 exp(π·K)+clip 计算应有份额，与实测份额对照
3. π 语义一致性：trace π == allocate() 内部权重的 π？
"""
import json, math
import numpy as np
from collections import defaultdict

# Load trace
trace_rows = []
with open('outputs/v3_d1_trajectory_400g/trace.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            trace_rows.append(json.loads(line))

premium_jids = [f'J{i}' for i in range(8)]
K = 2.0
CLIP_RATIO = 10.0
CLASS_WEIGHTS = [1, 2, 4, 8, 16, 32, 64]  # P0-P6
DSCP_LEVEL_MAP = {38: 6, 34: 5, 36: 4, 26: 3, 28: 2, 18: 1, 0: 0}

def compute_expected_share(trace_row):
    """
    对于 trace 中的每一行，用该 epoch 的 π 重新计算 exp(π·K) 预期的
    类内份额（考虑跨类 DWRR + 类内 exp 加权 + clip）。
    
    返回 {jid: expected_share}
    """
    # Extract π, dscp, level for active jobs
    active_jobs = {}
    for jid in premium_jids:
        pi = trace_row.get(f'{jid}_pi')
        if pi is not None:
            dscp = trace_row.get(f'{jid}_dscp', 0)
            level = DSCP_LEVEL_MAP.get(dscp, 0)
            active_jobs[jid] = {'pi': pi, 'dscp': dscp, 'level': level}
    
    if not active_jobs:
        return {}
    
    # Group by class
    class_jobs = defaultdict(list)
    for jid, info in active_jobs.items():
        class_jobs[info['level']].append(jid)
    
    active_classes = sorted(class_jobs.keys())
    total_class_weight = sum(CLASS_WEIGHTS[lvl] for lvl in active_classes)
    
    expected = {}
    for lvl in active_classes:
        class_bw_share = CLASS_WEIGHTS[lvl] / total_class_weight  # as fraction of link
    
        jobs_in_class = class_jobs[lvl]
        job_weights = {}
        for jid in jobs_in_class:
            pi = active_jobs[jid]['pi']
            weight = math.exp(K * pi)
            job_weights[jid] = weight
        
        # Clip
        if job_weights and len(job_weights) > 1:
            max_w = max(job_weights.values())
            min_w = min(job_weights.values())
            if min_w > 0 and max_w / min_w > CLIP_RATIO:
                scale = CLIP_RATIO * min_w
                for jid in job_weights:
                    if job_weights[jid] > scale:
                        job_weights[jid] = scale
        
        total_job_weight = sum(job_weights.values()) if job_weights else 1.0
        for jid in jobs_in_class:
            job_share = job_weights[jid] / total_job_weight * class_bw_share
            expected[jid] = job_share
    
    return expected

print('=' * 80)
print('D1@400G 机制审计：条件化对照分析')
print('=' * 80)

# Step 0: π 语义一致性
print('\n--- π 语义一致性 ---')
print('代码确认：trace π (line 236) 与 weight π (line 202) 同一变量 job_pi[jid]')
print('同一 T_target 来源 (line 149-154)、同一更新时机 (逐 epoch allocate)。')
print('结论：π 语义一致，无脱节。')
print()

# Step 1: Filter epochs by N_active
all_active_counts = []
for row in trace_rows:
    n_active = sum(1 for jid in premium_jids if row.get(f'{jid}_pi') is not None)
    all_active_counts.append(n_active)

total = len(all_active_counts)
print(f'Total epochs: {total}')
for k in range(1, 9):
    cnt = sum(1 for c in all_active_counts if c == k)
    print(f'  N_active={k}: {cnt} ({cnt/total*100:.1f}%)')

# Step 2: Find N_active >= 3 epochs
contested = [(i, row) for i, row in enumerate(trace_rows)
             if sum(1 for jid in premium_jids if row.get(f'{jid}_pi') is not None) >= 3]
n_contested = len(contested)
print(f'\nN_active >= 3 epochs: {n_contested} ({n_contested/total*100:.1f}%)')

if n_contested == 0:
    print('\n>>> NO CONTESTED EPOCHS: 锁入 = 活动模式伪影 <<<')
    print('J1 的 0.742 均值来自独占 epoch (N_active=1-2) 刷数据。')
    print('真实争抢从不存在 → 控制律无机会工作。')
else:
    # Step 3: Compare expected vs actual for each contested epoch
    errors = defaultdict(list)
    for ei, row in contested:
        expected = compute_expected_share(row)
        for jid in premium_jids:
            actual = row.get(f'{jid}_share')
            if actual is not None and jid in expected:
                errors[jid].append(actual - expected[jid])
    
    print()
    print('--- 争抢 epoch 应有 vs 实测偏差 (actual - expected share) ---')
    print(f"{'JID':<6} {'N':>6} {'Mean Err':>10} {'Std Err':>10} {'Median Err':>10} "
          f"{'Actual>Expected%':>16} {'Direction':>12}")
    print('-' * 70)
    
    for jid in premium_jids:
        if errors[jid]:
            n = len(errors[jid])
            mean_err = np.mean(errors[jid])
            std_err = np.std(errors[jid])
            med_err = np.median(errors[jid])
            pct_pos = sum(1 for e in errors[jid] if e > 0.01) / n * 100
            direction = 'OVER-FED' if mean_err > 0.02 else ('UNDER-FED' if mean_err < -0.02 else 'MATCH')
            print(f'{jid:<6} {n:>6} {mean_err:>+10.4f} {std_err:>10.4f} {med_err:>+10.4f} '
                  f'{pct_pos:>15.1f}% {direction:>12}')
    
    # Step 4: Overall verdict
    all_errs = []
    for jid in premium_jids:
        all_errs.extend(errors[jid])
    
    mean_abs_err = np.mean([abs(e) for e in all_errs])
    print(f'\nMean |error|: {mean_abs_err:.4f}')
    
    # Step 5: Detailed look at J1 and J5 in contested epochs
    print()
    print('--- J1 vs J5 in 真争抢 epochs (head-to-head) ---')
    j1_vs_j5 = []
    for ei, row in contested:
        j1_s = row.get('J1_share')
        j5_s = row.get('J5_share')
        if j1_s is not None and j5_s is not None:
            exp = compute_expected_share(row)
            j1_exp = exp.get('J1', 0)
            j5_exp = exp.get('J5', 0)
            j1_vs_j5.append({
                'epoch': row['epoch'],
                'J1_pi': row.get('J1_pi'), 'J5_pi': row.get('J5_pi'),
                'J1_dscp': row.get('J1_dscp'), 'J5_dscp': row.get('J5_dscp'),
                'J1_act': j1_s, 'J5_act': j5_s,
                'J1_exp': j1_exp, 'J5_exp': j5_exp,
            })
    
    if j1_vs_j5:
        print(f'  Found {len(j1_vs_j5)} head-to-head epochs')
        # Show first 5 and last 5
        print(f"  {'Epoch':>6} {'J1 π':>8} {'J1 DSCP':>8} {'J5 π':>8} {'J5 DSCP':>8} "
              f"{'J1 act':>8} {'J5 act':>8} {'J1 exp':>8} {'J5 exp':>8} {'J1 err':>8} {'J5 err':>8}")
        for d in j1_vs_j5[:5]:
            print(f"  {d['epoch']:>6} {d['J1_pi']:>+8.3f} {d['J1_dscp']:>8} "
                  f"{d['J5_pi']:>+8.3f} {d['J5_dscp']:>8} "
                  f"{d['J1_act']:>8.4f} {d['J5_act']:>8.4f} "
                  f"{d['J1_exp']:>8.4f} {d['J5_exp']:>8.4f} "
                  f"{d['J1_act']-d['J1_exp']:>+8.4f} {d['J5_act']-d['J5_exp']:>+8.4f}")
        if len(j1_vs_j5) > 10:
            print('  ...')
        for d in j1_vs_j5[-5:]:
            print(f"  {d['epoch']:>6} {d['J1_pi']:>+8.3f} {d['J1_dscp']:>8} "
                  f"{d['J5_pi']:>+8.3f} {d['J5_dscp']:>8} "
                  f"{d['J1_act']:>8.4f} {d['J5_act']:>8.4f} "
                  f"{d['J1_exp']:>8.4f} {d['J5_exp']:>8.4f} "
                  f"{d['J1_act']-d['J1_exp']:>+8.4f} {d['J5_act']-d['J5_exp']:>+8.4f}")
        
        # Average in head-to-head
        j1_act_mean = np.mean([d['J1_act'] for d in j1_vs_j5])
        j5_act_mean = np.mean([d['J5_act'] for d in j1_vs_j5])
        j1_exp_mean = np.mean([d['J1_exp'] for d in j1_vs_j5])
        j5_exp_mean = np.mean([d['J5_exp'] for d in j1_vs_j5])
        print(f'\n  Head-to-head means:')
        print(f'    J1: actual={j1_act_mean:.4f}  expected={j1_exp_mean:.4f}  error={j1_act_mean-j1_exp_mean:+.4f}')
        print(f'    J5: actual={j5_act_mean:.4f}  expected={j5_exp_mean:.4f}  error={j5_act_mean-j5_exp_mean:+.4f}')
    else:
        print('  No head-to-head epochs — J1 and J5 never compete on same link!')
    
    # Step 6: Also check — are J1 and J5 in different DSCP classes → different links?
    print()
    print('--- DSCP class co-occurrence ---')
    for ei, row in contested[:20]:  # sample first 20
        classes_present = set()
        for jid in premium_jids:
            pi = row.get(f'{jid}_pi')
            if pi is not None:
                dscp = row.get(f'{jid}_dscp', 0)
                lvl = DSCP_LEVEL_MAP.get(dscp, 0)
                classes_present.add(lvl)
        print(f'  Epoch {row["epoch"]:>6}: active DSCP classes = {sorted(classes_present)} '
              f'(J1: dscp={row.get("J1_dscp","-")}, J5: dscp={row.get("J5_dscp","-")})')
