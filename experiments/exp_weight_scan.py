"""权重展宽扫描：验证"集中度-稳定性"权衡。

目标：
1. 测试 D3(12:1)/D1(64:1)/D5(256:1) 三档权重展宽
2. 验证达成率随展宽上升、median 下降、崩溃保持 ~0
3. 定位运营商可根据 SLA 罚则选择的权衡点
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import Fair, CRUX, LongLiu
from longliu_sim.policy.dwrr import LongLiuDWRR
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD


def get_git_info():
    """获取 git 信息。"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        modified_files = [line[3:] for line in dirty_output.split("\n") if line.startswith(" M")]
        return {
            "commit": commit,
            "dirty": bool(modified_files),
            "dirty_files": modified_files
        }
    except:
        return {"commit": "unknown", "dirty": True, "dirty_files": []}


def run_single(cfg: dict, policy, seed: int) -> dict:
    """运行单个 seed 的仿真。"""
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
    sim = Simulator(
        topo,
        policy,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

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
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    stats = result.per_job_stats()

    # 分层统计
    tier_stats = {
        "premium": [],
        "standard": [],
        "medium": [],
        "small": [],
    }
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci

        if ci == 1.2:
            tier_stats["premium"].append({
                "jid": jid,
                "sas": s["sas"],
                "avg_iter_ms": s["avg_iter_ms"],
            })
        elif ci == 2.0:
            if "LLaMA" in job.model or "T5-11B" in job.model:
                tier_stats["standard"].append({
                    "jid": jid,
                    "sas": s["sas"],
                    "avg_iter_ms": s["avg_iter_ms"],
                })
            else:
                tier_stats["medium"].append({
                    "jid": jid,
                    "sas": s["sas"],
                    "avg_iter_ms": s["avg_iter_ms"],
                })
        elif ci == 3.0:
            tier_stats["small"].append({
                "jid": jid,
                "sas": s["sas"],
                "avg_iter_ms": s["avg_iter_ms"],
            })

    return {
        "seed": seed,
        "tier_stats": tier_stats,
        "overall_mean_sas": sum(s["sas"] for s in stats.values()) / len(stats),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="outputs/weight_scan")
    args = parser.parse_args()

    # 配置（90% 负载）
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

    # 权重展宽档位
    # D3: 12:1 (soft weights)
    # D1: 64:1 (standard weights)
    # D5: 256:1 (steep weights, 需要自定义权重表)

    # Git 信息
    git_info = get_git_info()
    if git_info["dirty"]:
        print("❌ Git 工作区有未提交的改动，拒绝运行。")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # 保存 run_meta.json
    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "cmdline": " ".join(sys.argv),
        "seeds": args.seeds,
        "config": cfg,
    }
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("="*80)
    print("Weight Scan: 3 档权重展宽 × 3 seeds")
    print("="*80)
    print(f"Git commit: {git_info['commit']}")
    print(f"Output: {args.out}")
    print()

    # D3 (12:1 soft weights)
    print("运行 D3 (12:1 soft weights)...")
    d3_results = []
    policy_d3 = LongLiuDWRR(K=2.0, use_soft_weights=True, intra_class_fair=False, clip_ratio=10.0)
    for seed in range(args.seeds):
        r = run_single(cfg, policy_d3, seed)
        d3_results.append(r)
        print(f"  Seed {seed}: Overall Mean SAS {r['overall_mean_sas']:.3f}")

    # D1 (64:1 standard weights)
    print("运行 D1 (64:1 standard weights)...")
    d1_results = []
    policy_d1 = LongLiuDWRR(K=2.0, use_soft_weights=False, intra_class_fair=False, clip_ratio=10.0)
    for seed in range(args.seeds):
        r = run_single(cfg, policy_d1, seed)
        d1_results.append(r)
        print(f"  Seed {seed}: Overall Mean SAS {r['overall_mean_sas']:.3f}")

    # D5 (256:1 steep weights)
    print("运行 D5 (256:1 steep weights)...")
    d5_results = []
    policy_d5 = LongLiuDWRR(K=2.0, use_soft_weights=False, intra_class_fair=False, clip_ratio=10.0)
    # 手动覆盖权重表
    policy_d5.class_weights = [1, 2, 4, 16, 64, 256, 1024]  # 1024:1 max ratio
    for seed in range(args.seeds):
        r = run_single(cfg, policy_d5, seed)
        d5_results.append(r)
        print(f"  Seed {seed}: Overall Mean SAS {r['overall_mean_sas']:.3f}")

    all_results = {
        "D3_12:1": d3_results,
        "D1_64:1": d1_results,
        "D5_1024:1": d5_results,
    }

    # 保存结果
    with open(os.path.join(args.out, "weight_scan_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print(f"结果已保存至: {args.out}")


if __name__ == "__main__":
    main()