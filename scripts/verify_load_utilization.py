#!/usr/bin/env python3
"""验证负载扫描的瓶颈链路利用率。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import LongLiuDWRR
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD


def run_single_load(load_factor: float, seed: int = 0) -> dict:
    """运行单个负载点的仿真。"""
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
    }

    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )

    policy = LongLiuDWRR(
        K=2.0,
        use_soft_weights=False,  # D1: 标准 DWRR 权重
        intra_class_fair=False,  # D1: 类内 exp(pi·K) 加权
        clip_ratio=10.0,
    )

    sim = Simulator(
        topo,
        policy,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

    # 调整到达间隔以匹配负载因子
    base_interval = 2.0 * cfg["duration_ms"] / 24
    adjusted_interval = base_interval / load_factor

    loader = SyntheticTraceLoader(
        model_types=[
            "ResNet-18", "ResNet-50-fp16", "BERT-Base", "BERT-Large-fp16",
            "ViT-Base", "ViT-Large", "LLaMA-2-1B", "LLaMA-2-7B", "T5-1B",
        ],
        gpu_distribution={1: 0.2, 2: 0.2, 4: 0.3, 8: 0.3},
        ci_distribution={1.5: 0.3, 2.0: 0.35, 3.0: 0.35},
        job_count=24,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=DEFAULT_TIERED_WORKLOAD,
    )
    jobs = loader.load()

    # 调整 job 到达时间
    import random
    rng = random.Random(seed)
    current_time = 0.0
    for job in jobs:
        interval = rng.expovariate(1.0 / adjusted_interval)
        current_time += interval
        job.start_time_ms = min(current_time, cfg["duration_ms"] * 0.9)

    for j in jobs:
        sim.submit(j)

    result = sim.run()

    # 调试：检查是否有活跃 flow
    print(f"    调试：total_active_flows = {len(sim.active_flows)}")
    print(f"    调试：link_utilization_history = {len(sim.link_utilization_history)}")

    # 提取瓶颈链路利用率
    link_util = result.link_utilization
    spine_utils = {k: v for k, v in link_util.items() if k.startswith("spine-")}

    bottleneck_util = 0.0
    if spine_utils:
        bottleneck_util = max(v["mean"] for v in spine_utils.values())

    return {
        "load_factor": load_factor,
        "seed": seed,
        "bottleneck_utilization": bottleneck_util,
        "spine_link_stats": spine_utils,
    }


def main():
    print("验证负载扫描的瓶颈链路利用率")
    print("="*60)

    load_factors = [0.6, 0.75, 0.9, 1.1]

    for load_factor in load_factors:
        print(f"\n负载 {load_factor:.0%}:")
        result = run_single_load(load_factor, seed=0)
        print(f"  瓶颈链路利用率: {result['bottleneck_utilization']*100:.1f}%")
        print(f"  Spine links 统计:")
        for link_id, stats in result["spine_link_stats"].items():
            print(f"    {link_id}: mean={stats['mean']*100:.1f}%, max={stats['max']*100:.1f}%")


if __name__ == "__main__":
    main()