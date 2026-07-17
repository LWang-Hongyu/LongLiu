#!/usr/bin/env python3
"""
诊断四问：从现有 per_job.json 拉数，不跑新仿真。

使用：
    python3 diagnose.py

输出：分层崩溃率、GPU小时加权崩溃率、no_weighted 消融崩溃率、生命周期曲线
"""

import json
import os
import sys
from collections import defaultdict

# ---- 模型分层（基于 model_params.py 的 ci 映射逻辑） ----
# ci=1.5: >=7B params (large)
# ci=2.0: >=340M params (medium) 
# ci=3.0: <340M params (small)
LARGE_MODELS = {
    "LLaMA-2-13B", "LLaMA-2-7B", "LLaMA-3-8B", "Mixtral-8x7B",
    "LLaMA-3-70B", "GPT-3-175B",
}
MEDIUM_MODELS = {
    "ViT-Large", "BERT-Large-fp16", "BERT-Large-fp32",
    "ResNet-152", "GPT-2-XL", "T5-Large",
}
SMALL_MODELS = {
    "ResNet-50", "MobileNet", "EfficientNet", "BERT-Base",
    "DistilBERT", "TinyLLaMA",
}

def get_tier(model: str) -> str:
    if model in LARGE_MODELS:
        return "large"
    elif model in MEDIUM_MODELS:
        return "medium"
    elif model in SMALL_MODELS:
        return "small"
    else:
        return f"unknown({model})"

def compute_catastrophic(jobs: list) -> float:
    """SAS < 0.2 比例"""
    if not jobs:
        return 0.0
    return sum(1 for j in jobs if j["sas"] < 0.2) / len(jobs)

def compute_gpu_hour_catastrophic(jobs: list) -> float:
    """按 GPU 数加权的灾难性违约率"""
    total_gpu = sum(j["dp"] for j in jobs)
    if total_gpu == 0:
        return 0.0
    crash_gpu = sum(j["dp"] for j in jobs if j["sas"] < 0.2)
    return crash_gpu / total_gpu

def analyze(data: dict, label: str):
    """对一份 per_job.json 做诊断四问。"""
    print(f"{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    # Q1: 分层崩溃率
    print(f"\n  --- Q1: 分层崩溃率 ---")
    per_tier = defaultdict(list)
    for policy_name, jobs in data.items():
        for j in jobs:
            tier = get_tier(j["model"])
            per_tier[(policy_name, tier)].append(j)
    
    tiers = ["large", "medium", "small"]
    print(f"  {'Policy':<14} {'Tier':<8} {'Count':<6} {'Crash%':<8} {'Avg SAS':<8}")
    print(f"  {'-'*46}")
    for policy_name in data:
        for tier in tiers:
            tier_jobs = per_tier.get((policy_name, tier), [])
            cnt = len(tier_jobs)
            crash = compute_catastrophic(tier_jobs) * 100
            avg_sas = sum(j["sas"] for j in tier_jobs) / cnt if cnt else 0
            print(f"  {policy_name:<14} {tier:<8} {cnt:<6} {crash:<8.1f} {avg_sas:<8.3f}")
    
    # Q2: GPU 小时加权崩溃率
    print(f"\n  --- Q2: GPU 小时加权崩溃率 ---")
    print(f"  {'Policy':<14} {'Raw Crash%':<12} {'GPU-hr Crash%':<15} {'Diff':<8}")
    print(f"  {'-'*49}")
    for policy_name in data:
        jobs = data[policy_name]
        raw = compute_catastrophic(jobs) * 100
        gpu = compute_gpu_hour_catastrophic(jobs) * 100
        print(f"  {policy_name:<14} {raw:<12.1f} {gpu:<15.1f} {gpu-raw:<+8.1f}")
    
    # 附加：崩溃 job 的 GPU 分布
    print(f"\n  --- 崩溃 job 的 GPU 分布 ---")
    for policy_name in data:
        jobs = data[policy_name]
        crashed = [j for j in jobs if j["sas"] < 0.2]
        if crashed:
            gpu_dps = [j["dp"] for j in crashed]
            avg_dp = sum(gpu_dps) / len(gpu_dps)
            print(f"  {policy_name:<14}: {len(crashed):<4} crashed jobs, avg dp={avg_dp:.1f}, "
                  f"dp分布={sorted(gpu_dps)[:10]}")
        else:
            print(f"  {policy_name:<14}: 0 crashed jobs")
    
    # Q3: 崩溃 job 的到达时间（需要从 jid 反推，如果 jid 按到达序号）
    print(f"\n  --- Q4: 崩溃 job 的 jid 分布（≈到达顺序） ---")
    for policy_name in data:
        jobs = data[policy_name]
        crashed = [j for j in jobs if j["sas"] < 0.2]
        if crashed:
            # jid 格式：J0, J1, ..., J23
            jid_nums = sorted([int(j["jid"][1:]) for j in crashed])
            print(f"  {policy_name:<14}: crashed jid={jid_nums}")
        else:
            print(f"  {policy_name:<14}: no crashes")


def analyze_no_weighted(data: dict, no_weighted_data: dict):
    """Q3: 对比 no_weighted 消融的崩溃率"""
    print(f"\n{'='*60}")
    print(f"  Q3: no_weighted 消融 vs 基线 LongLiu")
    print(f"{'='*60}")
    
    ll_baseline = data.get("LongLiu", [])
    ll_noweight = no_weighted_data.get("LongLiu", [])
    
    print(f"\n  {'Metric':<30} {'LongLiu':<12} {'LongLiu(no_w)':<15}")
    print(f"  {'-'*57}")
    
    for metric_name, fn in [
        ("Overall Crash%", lambda jobs: compute_catastrophic(jobs) * 100),
        ("Large Crash%", lambda jobs: compute_catastrophic([j for j in jobs if get_tier(j["model"])=="large"]) * 100),
        ("Medium Crash%", lambda jobs: compute_catastrophic([j for j in jobs if get_tier(j["model"])=="medium"]) * 100),
        ("Small Crash%", lambda jobs: compute_catastrophic([j for j in jobs if get_tier(j["model"])=="small"]) * 100),
        ("GPU-hr Crash%", lambda jobs: compute_gpu_hour_catastrophic(jobs) * 100),
        ("Avg SAS", lambda jobs: sum(j["sas"] for j in jobs) / len(jobs) if jobs else 0),
    ]:
        v1 = fn(ll_baseline)
        v2 = fn(ll_noweight)
        print(f"  {metric_name:<30} {v1:<12.2f} {v2:<15.2f}")


if __name__ == "__main__":
    # 主实验
    main_path = "/home/why/LongLiu_rebuild/sim-nextgen/outputs/final_v3_10seeds/per_job.json"
    # no_weighted 消融
    no_weighted_path = "/home/why/LongLiu_rebuild/sim-nextgen/outputs/ablation_v3/no_weighted/per_job.json"
    
    if not os.path.exists(main_path):
        print(f"ERROR: {main_path} not found")
        sys.exit(1)
    
    with open(main_path) as f:
        main_data = json.load(f)
    
    analyze(main_data, "主实验 10 seeds")
    
    if os.path.exists(no_weighted_path):
        with open(no_weighted_path) as f:
            nw_data = json.load(f)
        analyze_no_weighted(main_data, nw_data)
    else:
        print(f"\nWARNING: {no_weighted_path} not found")
