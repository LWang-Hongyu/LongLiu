#!/usr/bin/env python3
"""运行 D1 × 1 seed，输出全链路利用率时间序列。"""

import sys
import os
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import LongLiuDWRR
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/link_util_d1")
    args = parser.parse_args()

    # 配置
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

    # D1 策略
    policy = LongLiuDWRR(
        K=2.0,
        use_soft_weights=False,  # D1 使用标准权重表
        intra_class_fair=False,  # D1 使用 exp(pi·K) 加权
        clip_ratio=10.0,
    )

    # 拓扑
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )

    # 仿真器
    sim = Simulator(
        topo,
        policy,
        duration_ms=cfg["duration_ms"],
        seed=args.seed,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

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
        seed=args.seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=DEFAULT_TIERED_WORKLOAD,
    )

    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    # 运行
    print(f"Running D1 × seed {args.seed}...")
    result = sim.run()

    # 输出利用率时间序列
    os.makedirs(args.out, exist_ok=True)

    # 保存 run_meta
    meta = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "config": cfg,
    }
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 1. 输出利用率时间序列 CSV
    csv_path = os.path.join(args.out, "link_util_timeseries.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头
        link_ids = sorted(sim.link_utilization_history.keys())
        writer.writerow(["time_idx"] + link_ids)

        # 时间序列
        if link_ids:
            max_len = max(len(sim.link_utilization_history[lid]) for lid in link_ids)
            for t in range(max_len):
                row = [t]
                for lid in link_ids:
                    hist = sim.link_utilization_history[lid]
                    if t < len(hist):
                        row.append(f"{hist[t]:.6f}")
                    else:
                        row.append("")
                writer.writerow(row)

    print(f"✓ 时间序列已保存: {csv_path}")

    # 2. 输出详细统计
    print("\n" + "=" * 80)
    print("Spine 链路利用率统计（D1 × seed 0）")
    print("=" * 80)

    stats = []
    for link_id in sorted(sim.link_utilization_history.keys()):
        util_history = sim.link_utilization_history[link_id]
        if not util_history:
            continue

        avg_util = sum(util_history) / len(util_history)
        max_util = max(util_history)
        min_util = min(util_history)

        # 利用率 ≥95% 的时间占比
        high_util_count = sum(1 for u in util_history if u >= 0.95)
        high_util_ratio = high_util_count / len(util_history) if util_history else 0.0

        stats.append({
            "link_id": link_id,
            "avg_util": avg_util,
            "max_util": max_util,
            "min_util": min_util,
            "high_util_ratio": high_util_ratio,
        })

        print(f"\n{link_id}:")
        print(f"  平均利用率: {avg_util*100:.1f}%")
        print(f"  峰值利用率: {max_util*100:.1f}%")
        print(f"  最低利用率: {min_util*100:.1f}%")
        print(f"  利用率≥95% 时间占比: {high_util_ratio*100:.1f}%")

    # 3. 检查两条 spine 的负载均衡性
    if len(stats) >= 2:
        util_diff = abs(stats[0]["avg_util"] - stats[1]["avg_util"])
        print(f"\n负载均衡性:")
        print(f"  Spine-0 vs Spine-1 利用率差: {util_diff*100:.2f}%")
        if util_diff < 0.01:
            print(f"  ✓ ECMP 分布完全均衡")
        elif util_diff < 0.05:
            print(f"  ~ ECMP 分布基本均衡（差值<5%）")
        else:
            print(f"  ✗ ECMP 分布偏斜")

    # 4. 保存统计摘要
    summary = {
        "num_links": len(stats),
        "links": stats,
        "load_balance": {
            "util_diff": util_diff if len(stats) >= 2 else None,
            "balanced": util_diff < 0.05 if len(stats) >= 2 else None,
        }
    }
    with open(os.path.join(args.out, "link_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ 统计摘要已保存: {args.out}/link_stats.json")


if __name__ == "__main__":
    main()