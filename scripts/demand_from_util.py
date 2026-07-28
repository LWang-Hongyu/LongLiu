#!/usr/bin/env python3
"""从瓶颈链路利用率反推需求，验证计算逻辑。

对比：
1. 从 bits_per_iter 正向计算（有风险）
2. 从瓶颈链路利用率反推（正确）
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD, MODEL_PARAMS


def main():
    # 配置（与 exp_link_util_single.py 一致）
    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 400e9,
        },
        "duration_ms": 600000,
        "overhead_factor": 1.3,
        "overlap_factor": 0.85,
        "num_hosts": 16,
    }

    # Workload
    loader = SyntheticTraceLoader(
        model_types=[
            "ResNet-18", "ResNet-50-fp16", "BERT-Base", "BERT-Large-fp16",
            "ViT-Base", "ViT-Large", "LLaMA-2-1B", "LLaMA-2-7B", "T5-1B",
        ],
        gpu_distribution={1: 0.2, 2: 0.2, 4: 0.3, 8: 0.3},
        ci_distribution={1.5: 0.3, 2.0: 0.35, 3.0: 0.35},
        job_count=24,
        duration_ms=cfg["duration_ms"],
        seed=0,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=cfg["num_hosts"],
        workload_profile=DEFAULT_TIERED_WORKLOAD,
    )

    jobs = loader.load()

    # 方法 1：从 bits_per_iter 正向计算（风险：假设迭代间隔）
    print("=" * 80)
    print("方法 1：从 bits_per_iter 正向计算（假设迭代间隔 = duration / target_iters）")
    print("=" * 80)

    total_demand_1 = 0.0
    for job in jobs:
        bits_per_iter = job.bits_per_iter
        target_iters = job.target_iters
        total_bits = bits_per_iter * target_iters

        # 假设迭代间隔 = duration / target_iters
        iter_interval_ms = cfg["duration_ms"] / target_iters if target_iters > 0 else cfg["duration_ms"]

        # 通信速率（Gbps）= total_bits / duration_ms × 1000 / 1e9
        demand_gbps = total_bits / cfg["duration_ms"] * 1000 / 1e9

        total_demand_1 += demand_gbps

    print(f"总需求: {total_demand_1:.2f} Gbps")
    print(f"静态超订倍数: {total_demand_1 / 400:.2f}×")

    # 方法 2：从瓶颈链路利用率反推（正确）
    print("\n" + "=" * 80)
    print("方法 2：从瓶颈链路利用率反推（实测值）")
    print("=" * 80)

    # 从 link_stats.json 读取
    stats_path = "outputs/link_util_d1_seed0/link_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)

        total_demand_2 = 0.0
        for link_stat in stats["links"]:
            avg_util = link_stat["avg_util"]
            capacity_gbps = 200.0  # 每条 spine link 容量
            demand_gbps = avg_util * capacity_gbps
            total_demand_2 += demand_gbps

            print(f"{link_stat['link_id']}: 利用率 {avg_util*100:.1f}% → 需求 {demand_gbps:.2f} Gbps")

        print(f"\n总需求: {total_demand_2:.2f} Gbps")
        print(f"真实超订倍数: {total_demand_2 / 400:.2f}×")

    # 对比
    print("\n" + "=" * 80)
    print("对比")
    print("=" * 80)
    print(f"方法 1（正向计算）: {total_demand_1:.2f} Gbps")
    if os.path.exists(stats_path):
        print(f"方法 2（利用率反推）: {total_demand_2:.2f} Gbps")
        print(f"误差: {abs(total_demand_1 - total_demand_2) / total_demand_2 * 100:.1f}%")

    # 方法 3：检查 bits_per_iter 的正确性
    print("\n" + "=" * 80)
    print("方法 3：验证 bits_per_iter 的计算")
    print("=" * 80)

    for job in jobs[:3]:  # 只打印前 3 个
        print(f"\n{job.jid} ({job.model}, dp={job.num_workers}):")
        print(f"  mb_per_iter: {job.mb_per_iter:.2f} MB")
        print(f"  bits_per_iter: {job.bits_per_iter / 1e6:.2f} Mb")
        print(f"  target_iters: {job.target_iters}")
        print(f"  total_bits: {job.bits_per_iter * job.target_iters / 1e9:.2f} Gb")


if __name__ == "__main__":
    main()