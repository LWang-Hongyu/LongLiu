#!/usr/bin/env python3
"""
任务1：分层 SLO 达成率 + SAS 分位数
"""

import json
import numpy as np
from collections import defaultdict

LARGE_MODELS = {
    "LLaMA-2-13B", "LLaMA-2-7B", "LLaMA-3-8B", "Mixtral-8x7B",
    "LLaMA-3-70B", "GPT-3-175B",
}

MEDIUM_MODELS = {
    "ViT-Large", "BERT-Large-fp16", "BERT-Large-fp32",
    "ResNet-152", "GPT-2-XL", "T5-Large", "T5-11B-fp16",
}

def get_tier(model: str) -> str:
    if model in LARGE_MODELS:
        return "large"
    elif model in MEDIUM_MODELS:
        return "medium"
    else:
        return "small"

def percentile(arr, p):
    """计算分位数，p 为 0-100"""
    if not arr:
        return None
    arr = sorted(arr)
    k = (len(arr) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(arr) else f
    return arr[f] + (k - f) * (arr[c] - arr[f]) if f != c else arr[f]

def main():
    path = "/home/why/LongLiu_rebuild/sim-nextgen/outputs/final_v3_10seeds/per_job.json"
    with open(path) as f:
        data = json.load(f)

    print("=" * 80)
    print("  任务1: 分层 SLO 达成率 + SAS 分位数 (final_v3_10seeds)")
    print("=" * 80)

    # 按 policy × tier 分组
    per_tier = defaultdict(list)
    for policy in data:
        for j in data[policy]:
            tier = get_tier(j["model"])
            per_tier[(policy, tier)].append(j["sas"])

    policies = ["Fair", "SRPT", "CRUX", "LongLiu"]
    tiers = ["large", "medium"]

    # 表头
    print("\n表1: SLO 达成率 (SAS >= 1.0 的 job 占比)")
    print("-" * 80)
    print(f"{'Policy':<12} {'Tier':<8} {'Total':<6} {'SLO≥1':<6} {'Rate%':<8} {'p10':<8} {'p50':<8} {'p90':<8}")
    print("-" * 80)

    for policy in policies:
        for tier in tiers:
            sas_vals = per_tier.get((policy, tier), [])
            if not sas_vals:
                continue
            total = len(sas_vals)
            slo_met = sum(1 for s in sas_vals if s >= 1.0)
            rate = slo_met / total * 100
            p10 = percentile(sas_vals, 10)
            p50 = percentile(sas_vals, 50)
            p90 = percentile(sas_vals, 90)
            print(f"{policy:<12} {tier:<8} {total:<6} {slo_met:<6} {rate:<8.1f} {p10:<8.3f} {p50:<8.3f} {p90:<8.3f}")
        print()

    # 补充：每个 tier 的 job 数量（跨 policy 去重）
    print("\n表2: 各 tier 的 job 数量统计")
    print("-" * 40)
    for tier in tiers:
        # 统计该 tier 有多少个不同的 jid（按 seed=0 去重）
        jids = set()
        for policy in data:
            for j in data[policy]:
                if j["seed"] == 0 and get_tier(j["model"]) == tier:
                    jids.add(j["jid"])
        print(f"  {tier}: {len(jids)} unique jobs (per seed)")
    
    # 补充：总 job 数
    total_jobs = set()
    for policy in data:
        for j in data[policy]:
            if j["seed"] == 0:
                total_jobs.add(j["jid"])
    print(f"  total: {len(total_jobs)} unique jobs (per seed)")

    # 表3: 分层崩溃率
    print("\n表3: 分层崩溃率 (SAS < 0.2)")
    print("-" * 60)
    print(f"{'Policy':<12} {'Tier':<8} {'Total':<6} {'Crash':<6} {'Crash%':<8}")
    print("-" * 60)
    for policy in policies:
        for tier in tiers:
            sas_vals = per_tier.get((policy, tier), [])
            if not sas_vals:
                continue
            total = len(sas_vals)
            crash = sum(1 for s in sas_vals if s < 0.2)
            rate = crash / total * 100
            print(f"{policy:<12} {tier:<8} {total:<6} {crash:<6} {rate:<8.1f}")
        print()

if __name__ == "__main__":
    main()