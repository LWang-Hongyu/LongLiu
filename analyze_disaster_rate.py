#!/usr/bin/env python3
"""分析灾难性违约率：SAS < 阈值的任务比例。"""

import json
import sys
from pathlib import Path


def analyze_disaster_rates(result_dir: str, thresholds: list = [0.2, 0.5]):
    """
    分析每个策略的灾难性违约率。
    
    参数：
        result_dir: 结果目录路径
        thresholds: SAS 阈值列表（默认 0.2 和 0.5）
    """
    result_path = Path(result_dir)
    per_job_file = result_path / "per_job.json"
    
    if not per_job_file.exists():
        print(f"Error: {per_job_file} not found")
        return
    
    with open(per_job_file) as f:
        data = json.load(f)
    
    # 数据已经按策略分组
    # 格式: {"Fair": [{...}, {...}], "SRPT": [...], ...}
    from collections import defaultdict
    by_policy = defaultdict(list)
    
    for policy, jobs in data.items():
        by_policy[policy] = jobs
    
    # 统计每个策略的灾难性违约率
    print("\n" + "=" * 80)
    print("灾难性违约率分析 (SAS Distribution)")
    print("=" * 80)
    
    for policy in ["Fair", "SRPT", "CRUX", "LongLiu"]:
        if policy not in by_policy:
            continue
        
        print(f"\n{policy}:")
        print("-" * 60)
        
        # 收集所有 SAS 值
        all_sas = []
        large_sas = []  # 大模型 SAS
        medium_sas = []  # 中模型 SAS
        small_sas = []  # 小模型 SAS
        
        for job in by_policy[policy]:
            all_sas.append(job["sas"])
            if job["ci"] == 1.5:
                large_sas.append(job["sas"])
            elif job["ci"] == 2.0:
                medium_sas.append(job["sas"])
            else:
                small_sas.append(job["sas"])
        
        # 计算各阈值的违约率
        for thresh in thresholds:
            disaster_count = sum(1 for s in all_sas if s < thresh)
            disaster_rate = disaster_count / len(all_sas) if all_sas else 0
            print(f"  SAS < {thresh:.1f}: {disaster_rate*100:.1f}% ({disaster_count}/{len(all_sas)})")
        
        # 分层统计
        print(f"\n  按模型规模:")
        for label, sas_list in [("Large (ci=1.5)", large_sas), 
                                 ("Medium (ci=2.0)", medium_sas),
                                 ("Small (ci=3.0)", small_sas)]:
            if not sas_list:
                continue
            mean_sas = sum(sas_list) / len(sas_list)
            min_sas = min(sas_list)
            max_sas = max(sas_list)
            disaster_02 = sum(1 for s in sas_list if s < 0.2) / len(sas_list)
            disaster_05 = sum(1 for s in sas_list if s < 0.5) / len(sas_list)
            print(f"    {label}: Mean={mean_sas:.3f}, Min={min_sas:.3f}, "
                  f"SAS<0.2={disaster_02*100:.1f}%, SAS<0.5={disaster_05*100:.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_disaster_rate.py <result_dir>")
        sys.exit(1)
    
    analyze_disaster_rates(sys.argv[1])