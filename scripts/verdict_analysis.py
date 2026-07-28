#!/usr/bin/env python3
"""真实路由需求-容量核算（基于现有数据的推断分析）。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS


def main():
    print("=" * 120)
    print("真实路由需求-容量核算（推断分析）")
    print("=" * 120)

    # 读取实验结果
    with open("outputs/dwrr_ablation_v4/dwrr_results.json") as f:
        data = json.load(f)

    # 分析 Fair 的 avg_iter_ms
    print("Fair 策略的 avg_iter_ms 分布：")
    print("-" * 120)

    fair_results = data["Fair"]

    # Premium 档
    premium_avg_iters = []
    for r in fair_results:
        for job in r["tier_stats"]["premium"]:
            premium_avg_iters.append(job["avg_iter_ms"])

    # Standard 档
    standard_avg_iters = []
    for r in fair_results:
        for job in r["tier_stats"]["standard"]:
            standard_avg_iters.append(job["avg_iter_ms"])

    print(f"Premium 档：")
    print(f"  平均 avg_iter_ms: {sum(premium_avg_iters)/len(premium_avg_iters):.1f} ms")
    print(f"  最小值: {min(premium_avg_iters):.1f} ms")
    print(f"  最大值: {max(premium_avg_iters):.1f} ms")

    print(f"\nStandard 档：")
    print(f"  平均 avg_iter_ms: {sum(standard_avg_iters)/len(standard_avg_iters):.1f} ms")
    print(f"  最小值: {min(standard_avg_iters):.1f} ms")
    print(f"  最大值: {max(standard_avg_iters):.1f} ms")

    print()

    # 推断真实超订倍数
    print("=" * 120)
    print("推断分析：真实超订倍数")
    print("=" * 120)

    # 从 workload 获取配置
    workload = DEFAULT_TIERED_WORKLOAD

    # 计算 premium job 的 solo 时间
    premium_iter_solos = []
    for i, (model, dp, ci) in enumerate(workload[:12]):  # 前 12 个是 premium/standard
        params = MODEL_PARAMS[model]
        comp_ms = params.get("comp_ms", 50.0)
        bpp = 2 if params.get("fp16", True) else 4
        mb_per_iter = 2 * params["params"] * bpp / dp / 1e6

        host_bw_gbps = 100.0
        comm_solo_ms = mb_per_iter * 8 / host_bw_gbps

        overlap_factor = 0.85
        iter_solo = comp_ms + comm_solo_ms * (1 - overlap_factor)

        premium_iter_solos.append(iter_solo)

    avg_solo = sum(premium_iter_solos) / len(premium_iter_solos)
    avg_actual = sum(premium_avg_iters) / len(premium_avg_iters)

    print(f"Premium job 平均 solo 时间: {avg_solo:.1f} ms")
    print(f"Fair 平均实际时间: {avg_actual:.1f} ms")
    print(f"时间倍数: {avg_actual / avg_solo:.1f}×")
    print()

    # 判断瓶颈
    print("=" * 120)
    print("瓶颈判断")
    print("=" * 120)

    # 如果 avg_actual ≈ avg_solo，说明没有严重竞争
    # 如果 avg_actual ≫ avg_solo，说明有严重竞争

    time_ratio = avg_actual / avg_solo

    if time_ratio < 2:
        print(f"时间倍数 {time_ratio:.1f}× < 2×：竞争轻微，瓶颈可能不在 spine")
        print(f"推断：大部分流量在 ToR 本地消化，spine 压力远小于静态估算")
    elif time_ratio < 5:
        print(f"时间倍数 {time_ratio:.1f}× ∈ [2, 5)：竞争中等")
        print(f"推断：部分流量过 spine，存在一定竞争")
    else:
        print(f"时间倍数 {time_ratio:.1f}× ≥ 5：竞争严重")
        print(f"推断：大量流量过 spine，spine 是瓶颈")

    print()

    # 核心矛盾解释
    print("=" * 120)
    print("核心矛盾解释：Fair 原 SAS≈0.9 vs 静态估算 10× 超订")
    print("=" * 120)

    print("可能解释：")
    print("1. 流量路由假设错误：大部分流量不过 spine（ToR 本地消化）")
    print("2. Overlap 因素：85% 计算通信重叠，暴露通信需求远小于静态估算")
    print("3. 动态 T_target 通胀：原 SAS 口径下 Fair 用静态目标，实际性能被低估")

    print()
    print("基于当前数据推断：")
    print(f"  时间倍数 {time_ratio:.1f}× 表明竞争程度为 {'轻微' if time_ratio < 2 else ('中等' if time_ratio < 5 else '严重')}")
    print(f"  真实超订倍数可能远小于静态估算的 10×")
    print()

    # 建议
    print("=" * 120)
    print("下一步建议")
    print("=" * 120)
    print("1. 运行逐链路利用率采样，确认真实瓶颈位置")
    print("2. 从 topology/placement 提取实际路由路径，精确核算超订倍数")
    print("3. 基于核算结果设计新主场景（可行性边界）")


if __name__ == "__main__":
    main()