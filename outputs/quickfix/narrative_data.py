#!/usr/bin/env python3
"""
叙事路线数据支撑：从 per_job.json 提取"有序降级"叙事所需的数据对比。

输出：
1. 大模型 SAS 对比 (LongLiu vs CRUX)
2. 底部 20% job 的 SAS 分布（最差情况对比）
3. 各策略下同一 job 的 SAS 差值
"""

import json
import os
from collections import defaultdict

LARGE_MODELS = {
    "LLaMA-2-13B", "LLaMA-2-7B", "LLaMA-3-8B", "Mixtral-8x7B",
    "LLaMA-3-70B", "GPT-3-175B",
}

def get_sas_across_seeds(data, policy, jid):
    """获取一个 job 在所有 seeds 下的 SAS 列表"""
    vals = []
    for j in data[policy]:
        if j["jid"] == jid:
            vals.append(j["sas"])
    return vals

def main():
    main_path = "/home/why/LongLiu_rebuild/sim-nextgen/outputs/final_v3_10seeds/per_job.json"
    
    with open(main_path) as f:
        data = json.load(f)

    print("=" * 70)
    print("  叙事支撑数据：有序降级 (Ordered Degradation)")
    print("=" * 70)

    # 1. 大模型 SAS 对比（所有 seeds 聚合）
    print("\n--- 1. Large model SAS aggregated across 10 seeds ---")
    large_sas = {}
    for policy in ["Fair", "SRPT", "CRUX", "LongLiu"]:
        vals = [j["sas"] for j in data[policy] if j["model"] in LARGE_MODELS]
        large_sas[policy] = vals
        bottom5 = sorted(vals)[:max(1, len(vals)//10)]
        print(f"  {policy:<12}: count={len(vals):<4} mean={sum(vals)/len(vals):.3f}  "
              f"min={min(vals):.3f}  bottom10%_avg={sum(bottom5)/len(bottom5):.3f}")

    # 2. 底部 20% job 的 SAS：LongLiu 的"最差情况"对比
    print("\n--- 2. Bottom 20% jobs (by SAS) per policy ---")
    for policy in ["CRUX", "LongLiu"]:
        sorted_jobs = sorted(data[policy], key=lambda j: j["sas"])
        n_bottom = max(1, len(sorted_jobs) // 5)
        bottom = sorted_jobs[:n_bottom]
        avg_bottom = sum(j["sas"] for j in bottom) / n_bottom
        min_sas = min(j["sas"] for j in bottom)
        max_sas = max(j["sas"] for j in bottom)
        print(f"  {policy:<12}: bottom {n_bottom:>2} jobs: "
              f"avg={avg_bottom:.3f}  min={min_sas:.3f}  max={max_sas:.3f}")
        for j in bottom:
            tier = "L" if j["model"] in LARGE_MODELS else "M"
            print(f"    {j['jid']:<6} {j['model']:<18} dp={j['dp']:<2} SAS={j['sas']:.3f}  tier={tier}")

    # 3. 同一 job 的跨策略 SAS 对比（找 LongLiu 最差 vs CRUX 最差的同一个 job）
    print("\n--- 3. Per-job SAS: LongLiu vs CRUX (all jobs, one seed=0) ---")
    seed0_ll = {j["jid"]: j for j in data["LongLiu"] if j["seed"] == 0}
    seed0_crux = {j["jid"]: j for j in data["CRUX"] if j["seed"] == 0}
    diffs = []
    for jid in sorted(seed0_ll, key=lambda x: int(x[1:])):
        ll_sas = seed0_ll[jid]["sas"]
        crux_sas = seed0_crux[jid]["sas"]
        diff = ll_sas - crux_sas
        diffs.append((jid, ll_sas, crux_sas, diff, seed0_ll[jid]["model"]))
    
    print(f"  {'JID':<6} {'Model':<18} {'LongLiu':<10} {'CRUX':<10} {'Diff':<8}")
    print(f"  {'-'*52}")
    for jid, ll_sas, crux_sas, diff, model in diffs:
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {jid:<6} {model:<18} {ll_sas:<10.3f} {crux_sas:<10.3f} {diff:<+7.3f} {arrow}")

    # 4. 关键叙事对比：LongLiu 最差 vs CRUX 最差
    print("\n--- 4. Narrative Key Numbers ---")
    # 4a. Large model mean SAS
    for policy in ["CRUX", "LongLiu"]:
        vals = [j["sas"] for j in data[policy] if j["model"] in LARGE_MODELS]
        print(f"  Large SAS {policy}: {sum(vals)/len(vals):.3f}")
    
    # 4b. Bottom 10% comparison
    for policy in ["CRUX", "LongLiu"]:
        sorted_jobs = sorted(data[policy], key=lambda j: j["sas"])
        n = max(1, len(sorted_jobs) // 10)
        bottom = sorted_jobs[:n]
        avg = sum(j["sas"] for j in bottom) / n
        print(f"  Bottom10% SAS {policy}: {avg:.3f} (n={n})")
    
    # 4c. Catastrophic job SAS distribution
    print()
    for policy in ["CRUX", "LongLiu"]:
        crashed = [j for j in data[policy] if j["sas"] < 0.2]
        if crashed:
            print(f"  {policy} crashed jobs SAS: min={min(j['sas'] for j in crashed):.4f} "
                  f"max={max(j['sas'] for j in crashed):.4f} avg={sum(j['sas'] for j in crashed)/len(crashed):.4f}")
            for j in crashed:
                print(f"    {j['jid']:<6} {j['model']:<18} dp={j['dp']:<2} SAS={j['sas']:.4f}")
        else:
            print(f"  {policy}: no crashed jobs in this sample")

    # 5. 新增：LongLiu vs CRUX 的 SAS 差异汇总（统计上有多少 job 是 LongLiu 更好的）
    print("\n--- 5. Win/Loss analysis (LongLiu vs CRUX per-job SAS) ---")
    ll_wins = 0
    crux_wins = 0
    total_jobs = 0
    ll_win_sum = 0.0
    crux_win_sum = 0.0
    for jid in seed0_ll:
        if jid in seed0_crux:
            total_jobs += 1
            diff = seed0_ll[jid]["sas"] - seed0_crux[jid]["sas"]
            if diff > 0:
                ll_wins += 1
                ll_win_sum += diff
            else:
                crux_wins += 1
                crux_win_sum -= diff  # positive value for CRUX wins
    print(f"  LongLiu better: {ll_wins}/{total_jobs} jobs (avg margin: {ll_win_sum/max(1,ll_wins):.3f})")
    print(f"  CRUX better:    {crux_wins}/{total_jobs} jobs (avg margin: {crux_win_sum/max(1,crux_wins):.3f})")

if __name__ == "__main__":
    main()
