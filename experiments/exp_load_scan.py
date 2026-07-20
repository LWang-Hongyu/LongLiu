"""负载扫描实验：验证 DWRR 在可行域内的优势。

目标：
1. 扫描 60%/75%/90%/110% 四档负载
2. 验证 DWRR 在可行域内达成率追上 SP 且崩溃恒 0
3. 定位"加冕 vs 普惠"的交叉点
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import Fair, CRUX, LongLiu, LongLiuDWRR
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


def run_single(cfg: dict, policy, seed: int, load_factor: float) -> dict:
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

    # 根据负载因子调整到达率
    # load_factor = actual_load / target_load
    # 当前默认 interval = 2.0 * duration / job_count
    # 需要缩放：interval / load_factor
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

    # 调整到达时间（通过修改 start_time_ms）
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
        "load_factor": load_factor,
        "tier_stats": tier_stats,
        "overall_mean_sas": sum(s["sas"] for s in stats.values()) / len(stats),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="outputs/load_scan")
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

    # 负载档位
    load_factors = [0.6, 0.75, 0.9, 1.1]

    # 策略
    policies = {
        "Fair": Fair(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu-SP": LongLiu(K=2.0, use_dynamic_T_target=True),
        "D1": LongLiuDWRR(K=2.0, use_soft_weights=False, intra_class_fair=False, clip_ratio=10.0),
    }

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
        "load_factors": load_factors,
    }
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("="*80)
    print("Load Scan: 4 策略 × 4 档负载 × 3 seeds")
    print("="*80)
    print(f"Git commit: {git_info['commit']}")
    print(f"Output: {args.out}")
    print()

    all_results = {}
    for load_factor in load_factors:
        load_results = {}
        for pname, policy in policies.items():
            print(f"负载 {load_factor:.0%} | 运行 {pname}...")
            results = []
            for seed in range(args.seeds):
                r = run_single(cfg, policy, seed, load_factor)
                results.append(r)
            load_results[pname] = results
        all_results[f"load_{int(load_factor*100)}"] = load_results

    # 保存结果
    with open(os.path.join(args.out, "load_scan_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print(f"结果已保存至: {args.out}")


if __name__ == "__main__":
    main()