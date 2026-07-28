#!/usr/bin/env python3
"""sas_eval 七策略全表（修正 per-job 基准）。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def estimate_model_from_avg_iter(avg_iter_ms: float) -> tuple:
    """从 avg_iter_ms 推断模型类型和 solo 时间。"""
    # LLaMA-2-13B/8: solo ≈ 162 ms，实际竞争下 ≈ 400-500+ ms
    # LLaMA-2-7B/4: solo ≈ 128 ms，实际竞争下 ≈ 200-300+ ms

    if avg_iter_ms > 350:  # 推断为 13B
        return "13B", 162.0, 1.2
    elif avg_iter_ms > 100:  # 推断为 7B
        return "7B", 128.0, 1.2
    else:  # 零迭代或异常
        return "unknown", 145.0, 1.2  # 使用平均值


def compute_sas_eval_corrected(avg_iter_ms: float) -> float:
    """计算 sas_eval（修正 per-job 基准）。"""
    if avg_iter_ms <= 0:
        return 0.0

    model, iter_solo, ci = estimate_model_from_avg_iter(avg_iter_ms)
    sas_eval = (ci * iter_solo) / avg_iter_ms

    return sas_eval


def main():
    print("=" * 160)
    print("sas_eval 七策略全表（修正 per-job 基准，94% 负载）")
    print("=" * 160)

    # 读取 DWRR 实验结果
    with open("outputs/dwrr_ablation_v4/dwrr_results.json") as f:
        data = json.load(f)

    # 读取配置
    with open("outputs/dwrr_ablation_v4/run_meta.json") as f:
        meta = json.load(f)

    overhead_factor = meta["config"]["overhead_factor"]  # 1.3
    overlap_factor = meta["config"]["overlap_factor"]    # 0.85

    print(f"配置：overhead={overhead_factor}, overlap={overlap_factor}, spine={meta['config']['topology']['spine_bw_bps']/1e9:.0f}G")
    print()

    # 分析各策略
    policies = ["Fair", "SRPT", "CRUX", "LongLiu", "D1", "D2", "D3"]

    print(f"{'策略':<12} {'Premium Mean':>13} {'Premium Cap':>13} {'Premium 崩溃':>13} {'Premium Starv':>14} {'Premium 达成':>13} "
          f"{'Standard Mean':>14} {'Standard Cap':>14} {'Standard 崩溃':>14} {'Standard Starv':>15} {'Standard 达成':>14}")
    print("-" * 160)

    for policy in policies:
        results = data[policy]

        # 统计 premium 档
        all_sas_eval = []
        all_starved = 0  # 零迭代 job
        total_premium = 0

        for r in results:
            for job in r["tier_stats"]["premium"]:
                total_premium += 1
                avg_iter_ms = job["avg_iter_ms"]

                if avg_iter_ms == 0.0:  # 零迭代
                    all_starved += 1
                    all_sas_eval.append(0.0)
                else:
                    sas_eval = compute_sas_eval_corrected(avg_iter_ms)
                    all_sas_eval.append(sas_eval)

        # Premium 统计
        mean_sas_eval = sum(all_sas_eval) / len(all_sas_eval) if all_sas_eval else 0.0
        capped_sas_eval = sum(min(s, 1.0) for s in all_sas_eval) / len(all_sas_eval) if all_sas_eval else 0.0
        collapsed_eval = sum(1 for s in all_sas_eval if s < 0.2)
        collapse_rate_eval = collapsed_eval / total_premium * 100 if total_premium > 0 else 0.0
        starve_rate = all_starved / total_premium * 100 if total_premium > 0 else 0.0
        achieved_eval = sum(1 for s in all_sas_eval if s >= 1.0)
        achieve_rate_eval = achieved_eval / total_premium * 100 if total_premium > 0 else 0.0

        # Standard 档统计
        all_sas_eval_std = []
        all_starved_std = 0
        total_standard = 0

        for r in results:
            for job in r["tier_stats"]["standard"]:
                total_standard += 1
                avg_iter_ms = job["avg_iter_ms"]

                if avg_iter_ms == 0.0:
                    all_starved_std += 1
                    all_sas_eval_std.append(0.0)
                else:
                    # Standard job: ci = 2.0, iter_solo ≈ 143 ms (BERT-Large-fp16/2)
                    sas_eval = (2.0 * 143.0) / avg_iter_ms
                    all_sas_eval_std.append(sas_eval)

        # Standard 统计
        mean_sas_eval_std = sum(all_sas_eval_std) / len(all_sas_eval_std) if all_sas_eval_std else 0.0
        capped_sas_eval_std = sum(min(s, 1.0) for s in all_sas_eval_std) / len(all_sas_eval_std) if all_sas_eval_std else 0.0
        collapsed_eval_std = sum(1 for s in all_sas_eval_std if s < 0.2)
        collapse_rate_eval_std = collapsed_eval_std / total_standard * 100 if total_standard > 0 else 0.0
        starve_rate_std = all_starved_std / total_standard * 100 if total_standard > 0 else 0.0
        achieve_rate_eval_std = sum(1 for s in all_sas_eval_std if s >= 1.0) / total_standard * 100 if total_standard > 0 else 0.0

        print(f"{policy:<12} {mean_sas_eval:>13.3f} {capped_sas_eval:>13.3f} {collapse_rate_eval:>13.1f}% {starve_rate:>14.1f}% {achieve_rate_eval:>13.1f}% "
              f"{mean_sas_eval_std:>14.3f} {capped_sas_eval_std:>14.3f} {collapse_rate_eval_std:>14.1f}% {starve_rate_std:>15.1f}% {achieve_rate_eval_std:>14.1f}%")

    print()

    # 关键对比
    print("=" * 160)
    print("关键发现（sas_eval 口径）")
    print("=" * 160)

    # 读取详细数据
    fair_results = data["Fair"]
    d1_results = data["D1"]
    sp_results = data["LongLiu"]

    # Premium 档
    fair_sas_eval = []
    d1_sas_eval = []
    sp_sas_eval = []

    fair_starved = 0
    d1_starved = 0
    sp_starved = 0

    for r in fair_results:
        for job in r["tier_stats"]["premium"]:
            if job["avg_iter_ms"] == 0:
                fair_starved += 1
            else:
                sas_eval = compute_sas_eval_corrected(job["avg_iter_ms"])
                fair_sas_eval.append(sas_eval)

    for r in d1_results:
        for job in r["tier_stats"]["premium"]:
            if job["avg_iter_ms"] == 0:
                d1_starved += 1
            else:
                sas_eval = compute_sas_eval_corrected(job["avg_iter_ms"])
                d1_sas_eval.append(sas_eval)

    for r in sp_results:
        for job in r["tier_stats"]["premium"]:
            if job["avg_iter_ms"] == 0:
                sp_starved += 1
            else:
                sas_eval = compute_sas_eval_corrected(job["avg_iter_ms"])
                sp_sas_eval.append(sas_eval)

    print("Premium 档关键指标：")
    print(f"  Fair: mean={sum(fair_sas_eval)/len(fair_sas_eval) if fair_sas_eval else 0:.3f}, starvation={fair_starved}, collapse={sum(1 for s in fair_sas_eval if s < 0.2)}")
    print(f"  D1: mean={sum(d1_sas_eval)/len(d1_sas_eval) if d1_sas_eval else 0:.3f}, starvation={d1_starved}, collapse={sum(1 for s in d1_sas_eval if s < 0.2)}")
    print(f"  LongLiu-SP: mean={sum(sp_sas_eval)/len(sp_sas_eval) if sp_sas_eval else 0:.3f}, starvation={sp_starved}, collapse={sum(1 for s in sp_sas_eval if s < 0.2)}")
    print()


if __name__ == "__main__":
    main()