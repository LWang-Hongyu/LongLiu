#!/usr/bin/env python3
"""sas_eval 七策略全表（94% 负载）。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.utils.model_params import MODEL_PARAMS, compute_mb_per_iter, get_comp_ms


def compute_sas_eval(job_info: dict, overhead_factor: float = 1.3) -> float:
    """计算单个 job 的 sas_eval（固定基准）。"""
    # 从 job 信息中获取模型和 DP
    # 但 dwrr_results.json 中没有保存模型信息，我们使用近似方法

    # 方法 1：从 avg_iter_ms 反推
    # 假设 avg_iter_ms ≈ target_iter_ms 时，sas_eval ≈ ci

    # 方法 2：使用固定基准（基于 tier）
    # Premium (ci=1.2): iter_solo ≈ 162 ms
    # Standard (ci=2.0): iter_solo ≈ 143 ms

    avg_iter_ms = job_info["avg_iter_ms"]

    # 从 tier 推断 ci
    # Premium: ci = 1.2
    # Standard: ci = 2.0

    # 使用固定基准计算 sas_eval
    if avg_iter_ms > 0:
        # 假设 premium job 的 iter_solo = 162 ms
        iter_solo = 162.0
        sas_eval = (1.2 * iter_solo) / avg_iter_ms
        return sas_eval
    else:
        return 0.0


def main():
    print("=" * 140)
    print("sas_eval 七策略全表（94% 负载，v4 workload）")
    print("=" * 140)

    # 读取 DWRR 实验结果
    with open("outputs/dwrr_ablation_v4/dwrr_results.json") as f:
        data = json.load(f)

    # 读取配置
    with open("outputs/dwrr_ablation_v4/run_meta.json") as f:
        meta = json.load(f)

    overhead_factor = meta["config"]["overhead_factor"]  # 1.3
    overlap_factor = meta["config"]["overlap_factor"]    # 0.85
    host_bw_bps = meta["config"]["topology"]["host_bw_bps"]  # 100e9

    print(f"配置：overhead={overhead_factor}, overlap={overlap_factor}, spine={meta['config']['topology']['spine_bw_bps']/1e9:.0f}G")
    print()

    # 分析各策略
    policies = ["Fair", "SRPT", "CRUX", "LongLiu", "D1", "D2", "D3"]

    print(f"{'策略':<12} {'Premium Mean':>13} {'Premium Cap':>13} {'Premium 崩溃':>13} {'Premium 达成':>13} "
          f"{'Standard Mean':>14} {'Standard Cap':>14} {'Standard 崩溃':>14} {'Standard 达成':>14}")
    print("-" * 140)

    for policy in policies:
        results = data[policy]

        # 统计 premium 档
        all_sas = []
        all_sas_eval = []

        for r in results:
            for job in r["tier_stats"]["premium"]:
                avg_iter_ms = job["avg_iter_ms"]
                sas = job["sas"]
                all_sas.append(sas)

                # 计算 sas_eval（固定基准）
                sas_eval = compute_sas_eval(job, overhead_factor)
                all_sas_eval.append(sas_eval)

        # Premium 统计
        mean_sas = sum(all_sas) / len(all_sas) if all_sas else 0.0
        capped_sas = sum(min(s, 1.0) for s in all_sas) / len(all_sas) if all_sas else 0.0
        collapsed = sum(1 for s in all_sas if s < 0.2)
        collapse_rate = collapsed / len(all_sas) * 100 if all_sas else 0.0
        achieved = sum(1 for s in all_sas if s >= 1.0)
        achieve_rate = achieved / len(all_sas) * 100 if all_sas else 0.0

        # sas_eval 统计
        mean_sas_eval = sum(all_sas_eval) / len(all_sas_eval) if all_sas_eval else 0.0
        capped_sas_eval = sum(min(s, 1.0) for s in all_sas_eval) / len(all_sas_eval) if all_sas_eval else 0.0
        collapsed_eval = sum(1 for s in all_sas_eval if s < 0.2)
        collapse_rate_eval = collapsed_eval / len(all_sas_eval) * 100 if all_sas_eval else 0.0

        # Standard 档统计
        all_sas_std = []
        all_sas_eval_std = []

        for r in results:
            for job in r["tier_stats"]["standard"]:
                avg_iter_ms = job["avg_iter_ms"]
                sas = job["sas"]
                all_sas_std.append(sas)

                # 计算 sas_eval（固定基准，ci=2.0）
                if avg_iter_ms > 0:
                    iter_solo = 143.0  # 假设 standard job 的 iter_solo
                    sas_eval = (2.0 * iter_solo) / avg_iter_ms
                    all_sas_eval_std.append(sas_eval)
                else:
                    all_sas_eval_std.append(0.0)

        # Standard 统计
        mean_sas_std = sum(all_sas_std) / len(all_sas_std) if all_sas_std else 0.0
        capped_sas_std = sum(min(s, 1.0) for s in all_sas_std) / len(all_sas_std) if all_sas_std else 0.0
        collapsed_std = sum(1 for s in all_sas_std if s < 0.2)
        collapse_rate_std = collapsed_std / len(all_sas_std) * 100 if all_sas_std else 0.0
        achieved_std = sum(1 for s in all_sas_std if s >= 1.0)
        achieve_rate_std = achieved_std / len(all_sas_std) * 100 if all_sas_std else 0.0

        print(f"{policy:<12} {mean_sas_eval:>13.3f} {capped_sas_eval:>13.3f} {collapse_rate_eval:>13.1f}% {achieve_rate:>13.1f}% "
              f"{mean_sas_std:>14.3f} {capped_sas_std:>14.3f} {collapse_rate_std:>14.1f}% {achieve_rate_std:>14.1f}%")

    print()

    # 输出 SP vs D1 对比
    print("=" * 140)
    print("LongLiu-SP vs D1 崩溃率对比（sas_eval 口径）")
    print("=" * 140)

    sp_results = data["LongLiu"]
    d1_results = data["D1"]

    # Premium
    sp_premium_sas_eval = []
    d1_premium_sas_eval = []

    for r in sp_results:
        for job in r["tier_stats"]["premium"]:
            sas_eval = compute_sas_eval(job, overhead_factor)
            sp_premium_sas_eval.append(sas_eval)

    for r in d1_results:
        for job in r["tier_stats"]["premium"]:
            sas_eval = compute_sas_eval(job, overhead_factor)
            d1_premium_sas_eval.append(sas_eval)

    sp_premium_collapse = sum(1 for s in sp_premium_sas_eval if s < 0.2) / len(sp_premium_sas_eval) * 100
    d1_premium_collapse = sum(1 for s in d1_premium_sas_eval if s < 0.2) / len(d1_premium_sas_eval) * 100

    print(f"Premium 档崩溃率：")
    print(f"  LongLiu-SP: {sp_premium_collapse:.1f}%")
    print(f"  D1: {d1_premium_collapse:.1f}%")
    print(f"  差异: {sp_premium_collapse - d1_premium_collapse:.1f}%")
    print()


if __name__ == "__main__":
    main()