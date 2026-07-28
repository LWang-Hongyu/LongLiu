"""
Pilot: 4 策略 × 3 seeds，验证工作点

检查项：
1. 瓶颈链路利用率（70-90% 区间）
2. 四策略分层 SAS 排序是否可解释
3. 崩溃率分布是否有区分度
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu
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
        # 只检查修改的文件（M），不检查未跟踪的文件（??）
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
    tier_stats = {"large": [], "medium": [], "small": []}
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        tier = "large" if ci == 1.5 else ("medium" if ci == 2.0 else "small")
        tier_stats[tier].append({
            "jid": jid,
            "sas": s["sas"],
            "avg_iter_ms": s["avg_iter_ms"],
        })

    # 链路利用率（暂时跳过，Simulator 未提供此属性）
    link_util = {}

    return {
        "seed": seed,
        "tier_stats": tier_stats,
        "link_utilization": link_util,
        "overall_mean_sas": sum(s["sas"] for s in stats.values()) / len(stats),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="outputs/paper_baseline_v2/pilot")
    args = parser.parse_args()

    # 配置（与 paper-baseline-v2 一致）
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

    # 四策略
    policies = {
        "Fair": Fair(),
        "SRPT": SRPT(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu": LongLiu(K=2.0, use_dynamic_T_target=True),
    }

    # Git 信息
    git_info = get_git_info()
    if git_info["dirty"]:
        print("❌ Git 工作区有未提交的改动，拒绝运行主实验。")
        print("未提交文件:")
        for f in git_info["dirty_files"]:
            print(f"  {f}")
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
    print("Pilot: 4 策略 × 3 seeds")
    print("="*80)
    print(f"Git commit: {git_info['commit']}")
    print(f"Output: {args.out}")
    print()

    all_results = {}
    for pname, policy in policies.items():
        print(f"运行 {pname}...")
        results = []
        for seed in range(args.seeds):
            r = run_single(cfg, policy, seed)
            results.append(r)
            print(f"  Seed {seed}: Overall Mean SAS {r['overall_mean_sas']:.3f}")
        all_results[pname] = results

    # 汇总报告
    print()
    print("="*80)
    print("Pilot 报告")
    print("="*80)

    # 1. 链路利用率（暂时跳过）
    print("1. 链路利用率:")
    print("   （暂未实现链路利用率统计）")
    print()

    # 2. 分层 SAS
    print("2. 分层 Mean SAS:")
    print(f"{'Policy':<12} {'Large':>10} {'Medium':>10} {'Small':>10} {'Overall':>10}")
    print("-"*52)
    for pname, results in all_results.items():
        large_sas = [s["sas"] for r in results for s in r["tier_stats"]["large"]]
        medium_sas = [s["sas"] for r in results for s in r["tier_stats"]["medium"]]
        small_sas = [s["sas"] for r in results for s in r["tier_stats"]["small"]]
        overall_sas = [r["overall_mean_sas"] for r in results]

        print(f"{pname:<12} {sum(large_sas)/len(large_sas):>10.3f} "
              f"{sum(medium_sas)/len(medium_sas):>10.3f} "
              f"{sum(small_sas)/len(small_sas):>10.3f} "
              f"{sum(overall_sas)/len(overall_sas):>10.3f}")
    print()

    # 3. 崩溃率
    print("3. 分层崩溃率（SAS < 0.2）:")
    print(f"{'Policy':<12} {'Large':>10} {'Medium':>10} {'Small':>10} {'Overall':>10}")
    print("-"*52)
    for pname, results in all_results.items():
        large_collapse = sum(1 for r in results for s in r["tier_stats"]["large"] if s["sas"] < 0.2) / \
                         sum(len(r["tier_stats"]["large"]) for r in results) * 100
        medium_collapse = sum(1 for r in results for s in r["tier_stats"]["medium"] if s["sas"] < 0.2) / \
                          sum(len(r["tier_stats"]["medium"]) for r in results) * 100
        small_collapse = sum(1 for r in results for s in r["tier_stats"]["small"] if s["sas"] < 0.2) / \
                         sum(len(r["tier_stats"]["small"]) for r in results) * 100
        overall_collapse = sum(1 for r in results for s in r["tier_stats"]["large"] + r["tier_stats"]["medium"] + r["tier_stats"]["small"] if s["sas"] < 0.2) / \
                           sum(len(r["tier_stats"]["large"]) + len(r["tier_stats"]["medium"]) + len(r["tier_stats"]["small"]) for r in results) * 100

        print(f"{pname:<12} {large_collapse:>9.1f}% "
              f"{medium_collapse:>9.1f}% "
              f"{small_collapse:>9.1f}% "
              f"{overall_collapse:>9.1f}%")
    print()

    # 4. 工作点判断
    print("4. 工作点判断:")
    longliu_results = all_results["LongLiu"]
    longliu_collapse = sum(1 for r in longliu_results for s in r["tier_stats"]["large"] if s["sas"] < 0.2) / \
                       sum(len(r["tier_stats"]["large"]) for r in longliu_results) * 100

    if longliu_collapse > 40:
        print(f"   ⚠️ Large 崩溃率 {longliu_collapse:.1f}% > 40%，系统可能过载")
        print(f"   建议：暂停并报告，等待人工决策调整工作点")
    else:
        print(f"   ✅ Large 崩溃率 {longliu_collapse:.1f}% ≤ 40%，工作点合理")
        print(f"   可以继续运行主批次（20 seeds）")

    # 保存结果
    with open(os.path.join(args.out, "pilot_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print(f"结果已保存至: {args.out}")


if __name__ == "__main__":
    main()