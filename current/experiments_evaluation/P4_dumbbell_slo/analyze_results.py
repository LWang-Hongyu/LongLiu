#!/usr/bin/env python3
"""分析非对称工作负载实验结果"""

import csv
import sys

def analyze_csv(filename, job_name):
    """分析单个 CSV 文件"""
    solo_comm = []
    solo_bw = []
    contested_comm = []
    contested_bw = []
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row['phase']
            comm_dur = float(row['comm_dur_s'])
            bw = float(row['bw_gbps'])
            
            if phase == 'solo_rampup':
                solo_comm.append(comm_dur)
                solo_bw.append(bw)
            elif phase == 'contested':
                contested_comm.append(comm_dur)
                contested_bw.append(bw)
    
    print(f"\n{job_name}:")
    print(f"  Solo 阶段:")
    if solo_comm:
        avg_comm = sum(solo_comm) / len(solo_comm)
        avg_bw = sum(solo_bw) / len(solo_bw)
        print(f"    迭代数: {len(solo_comm)}")
        print(f"    平均通信时间: {avg_comm*1000:.1f} ms")
        print(f"    平均带宽: {avg_bw:.2f} Gbps")
    else:
        print(f"    无 solo 阶段数据")
    
    if contested_comm:
        avg_comm = sum(contested_comm) / len(contested_comm)
        avg_bw = sum(contested_bw) / len(contested_bw)
        print(f"  Contested 阶段:")
        print(f"    迭代数: {len(contested_comm)}")
        print(f"    平均通信时间: {avg_comm*1000:.1f} ms")
        print(f"    平均带宽: {avg_bw:.2f} Gbps")
        
        # 计算 slowdown
        if solo_comm:
            solo_avg = sum(solo_comm) / len(solo_comm)
            slowdown = avg_comm / solo_avg
            print(f"    Slowdown: {slowdown:.2f}x")
    
    return {
        'solo_comm': solo_comm,
        'solo_bw': solo_bw,
        'contested_comm': contested_comm,
        'contested_bw': contested_bw
    }

if __name__ == '__main__':
    print("=" * 60)
    print("CRUX 模式实验结果")
    print("=" * 60)
    
    crux_job1 = analyze_csv('p4_job1_asym_crux_rank0.csv', 'Job1 (P3, 通信密集, c_i=1.2)')
    crux_job2 = analyze_csv('p4_job2_asym_crux_rank0.csv', 'Job2 (P4, 计算密集, c_i=2.0)')
    
    print("\n" + "=" * 60)
    print("LongLiu 模式实验结果")
    print("=" * 60)
    
    ll_job1 = analyze_csv('p4_job1_asym_longliu_rank0.csv', 'Job1 (动态优先级, 通信密集, c_i=1.2)')
    ll_job2 = analyze_csv('p4_job2_asym_longliu_rank0.csv', 'Job2 (动态优先级, 计算密集, c_i=2.0)')
    
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)
    
    # 计算 contested 阶段的平均值
    if crux_job1['contested_comm'] and ll_job1['contested_comm']:
        crux_j1_avg = sum(crux_job1['contested_comm']) / len(crux_job1['contested_comm'])
        ll_j1_avg = sum(ll_job1['contested_comm']) / len(ll_job1['contested_comm'])
        
        crux_j1_solo = sum(crux_job1['solo_comm']) / len(crux_job1['solo_comm']) if crux_job1['solo_comm'] else 0
        ll_j1_solo = sum(ll_job1['solo_comm']) / len(ll_job1['solo_comm']) if ll_job1['solo_comm'] else 0
        
        crux_j1_slowdown = crux_j1_avg / crux_j1_solo if crux_j1_solo > 0 else 0
        ll_j1_slowdown = ll_j1_avg / ll_j1_solo if ll_j1_solo > 0 else 0
        
        print(f"\nJob1 (严格 SLO, c_i=1.2):")
        print(f"  CRUX:    slowdown = {crux_j1_slowdown:.2f}x (avg comm = {crux_j1_avg*1000:.1f} ms)")
        print(f"  LongLiu: slowdown = {ll_j1_slowdown:.2f}x (avg comm = {ll_j1_avg*1000:.1f} ms)")
        
        if crux_j1_slowdown < ll_j1_slowdown:
            print(f"  → CRUX 表现更好 (slowdown 更低)")
        elif ll_j1_slowdown < crux_j1_slowdown:
            print(f"  → LongLiu 表现更好 (slowdown 更低)")
        else:
            print(f"  → 两者表现相当")
    
    if crux_job2['contested_comm'] and ll_job2['contested_comm']:
        crux_j2_avg = sum(crux_job2['contested_comm']) / len(crux_job2['contested_comm'])
        ll_j2_avg = sum(ll_job2['contested_comm']) / len(ll_job2['contested_comm'])
        
        print(f"\nJob2 (宽松 SLO, c_i=2.0):")
        print(f"  CRUX:    avg comm = {crux_j2_avg*1000:.1f} ms, avg bw = {sum(crux_job2['contested_bw'])/len(crux_job2['contested_bw']):.2f} Gbps")
        print(f"  LongLiu: avg comm = {ll_j2_avg*1000:.1f} ms, avg bw = {sum(ll_job2['contested_bw'])/len(ll_job2['contested_bw']):.2f} Gbps")
