#!/usr/bin/env python3
"""需求-容量核算：计算真实超订倍数。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.utils.model_params import MODEL_PARAMS, compute_mb_per_iter


def compute_demand_gbps(mb_per_iter: float, iter_interval_ms: float = 162.0) -> float:
    """计算单个 job 的带宽需求（Gbps）。

    假设流量均匀分布在所有 spine links 上（简化模型）。
    """
    # 需求 = 数据量 / 时间
    # mb_per_iter: MB
    # iter_interval_ms: ms (solo 时间，无竞争)

    # 正确计算：
    # mb_per_iter (MB) × 8 (bits/byte) / (iter_interval_ms / 1000) (seconds) / 1000 (Gbps)
    # = mb_per_iter × 8 × 1000 / iter_interval_ms / 1000
    # = mb_per_iter × 8 / iter_interval_ms × 1000

    demand_gbps = mb_per_iter * 8 / (iter_interval_ms / 1000.0) / 1000.0

    return demand_gbps


def main():
    print("=" * 120)
    print("需求-容量核算：计算真实超订倍数")
    print("=" * 120)

    # 拓扑配置
    host_bw_gbps = 100.0  # 100 Gbps
    spine_bw_gbps = 400.0  # 400 Gbps
    num_spine_links = 8    # k=4 FatTree 有 8 条 spine links
    total_spine_bw = spine_bw_gbps * num_spine_links

    print(f"拓扑：host_bw={host_bw_gbps}G, spine_bw={spine_bw_gbps}G, {num_spine_links} spine links, 总 spine 容量={total_spine_bw}G")
    print()

    # 分析 v4 workload 中的 premium job
    print("Premium job 需求分析：")
    print("-" * 120)

    # 从 DEFAULT_TIERED_WORKLOAD 获取配置
    from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD

    premium_jobs = [j for j in DEFAULT_TIERED_WORKLOAD if j[2] == 1.5 or j[2] == 1.2]  # ci=1.5 or 1.2

    total_demand = 0.0
    job_demands = []

    for i, (model, dp, ci) in enumerate(premium_jobs[:12]):  # 前 12 个是 premium
        params = MODEL_PARAMS[model]
        mb_per_iter = compute_mb_per_iter(model, dp)

        # Solo 迭代时间（无竞争）
        # 修正：iter_solo = comp + comm_solo（无重叠）
        # comm_solo_ms = mb_per_iter × 8 / host_bw_gbps × 1000
        comm_solo_ms = mb_per_iter * 8 / host_bw_gbps * 1000

        # 从 MODEL_PARAMS 获取 comp_ms
        comp_ms = params.get("comp_ms", 50.0)

        # Solo 迭代时间（无重叠）
        iter_solo_ms = comp_ms + comm_solo_ms

        # 带宽需求（假设所有流量过 spine，worst-case）
        demand_gbps = compute_demand_gbps(mb_per_iter, iter_solo_ms)

        job_demands.append({
            "model": model,
            "dp": dp,
            "ci": ci,
            "mb_per_iter": mb_per_iter,
            "iter_solo_ms": iter_solo_ms,
            "demand_gbps": demand_gbps,
        })

        total_demand += demand_gbps

        print(f"{i+1:>2}. {model:<20} DP={dp}, ci={ci}, mb={mb_per_iter:>6.0f}MB, solo={iter_solo_ms:>6.0f}ms, demand={demand_gbps:>6.1f}Gbps")

    print("-" * 120)
    print(f"总需求（worst-case，全部过 spine）：{total_demand:.1f} Gbps")
    print(f"总 spine 容量：{total_spine_bw:.0f} Gbps")
    print(f"超订倍数：{total_demand / total_spine_bw:.1f}×")
    print()

    # 更精确的分析：考虑路由
    print("=" * 120)
    print("路由分析（简化模型）")
    print("=" * 120)

    # 简化：假设所有 job 都是 cross-pod（需要通过 spine）
    # 实际超订倍数 = total_demand / total_spine_bw

    # 但如果 job 是 intra-pod（同一 ToR 下），则不占用 spine
    # v4 workload 的 job 分布需要从 trace 中获取

    print("假设：所有 premium job 都是 cross-pod（worst-case）")
    print(f"  超订倍数 = {total_demand / total_spine_bw:.1f}×")
    print()

    print("如果 50% 是 intra-pod（best-case）：")
    print(f"  超订倍数 = {(total_demand * 0.5) / total_spine_bw:.1f}×")
    print()

    # 可行性边界计算
    print("=" * 120)
    print("可行性边界计算")
    print("=" * 120)

    # 假设每个 premium job 的平均需求
    avg_demand = total_demand / len(job_demands)

    # 系统达到可行性边界（需求 = 容量）时的最大并发数
    max_feasible_jobs = int(total_spine_bw / avg_demand)

    print(f"单个 premium job 平均需求：{avg_demand:.1f} Gbps")
    print(f"系统达到可行性边界（需求 = 容量）时的最大并发数：{max_feasible_jobs} 个")
    print()

    print(f"当前 workload：{len(job_demands)} 个 premium job")
    print(f"超订倍数：{len(job_demands) / max_feasible_jobs:.1f}×")
    print()

    # 建议的 workload 配置
    print("=" * 120)
    print("建议的 workload 配置（可行性边界附近）")
    print("=" * 120)

    # 目标：Σ需求 ∈ [1.0, 1.3] × 瓶颈容量
    target_ratios = [1.0, 1.2, 1.3]

    for ratio in target_ratios:
        target_demand = total_spine_bw * ratio
        target_jobs = int(target_demand / avg_demand)

        print(f"目标比例 {ratio:.1f}×：需求 = {target_demand:.0f} Gbps，约 {target_jobs} 个 premium job")


if __name__ == "__main__":
    main()