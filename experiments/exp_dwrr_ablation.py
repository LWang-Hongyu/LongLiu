"""
DWRR 消融实验：D1/D2/D3 + 对照组

目的：验证 DWRR 有界加权是否能解决 SP 的饿死问题

D1: LongLiuDWRR（权重表 1:2:4:8:16:32:64，类内限幅 ≤10×）
D2: LongLiuDWRR + 类内公平分配（isolate 类内加权）
D3: LongLiuDWRR 软权重表 1:2:3:4:6:8:12（权重展宽敏感性）
对照组：Fair, SRPT, CRUX, LongLiu-SP
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu, LongLiuDWRR, LongLiuDWRRFair
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS


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


def run_single(cfg: dict, policy, seed: int, policy_name: str) -> dict:
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

    # 分层统计（新增 premium/standard 细分）
    tier_stats = {
        "premium": [],   # ci=1.2（紧租户）
        "standard": [],  # ci=2.0（松租户）
        "medium": [],    # 中模型
        "small": [],     # 小模型
    }
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci

        # 大模型细分：premium (ci=1.2) vs standard (ci=2.0)
        if ci == 1.2:
            tier_stats["premium"].append({
                "jid": jid,
                "sas": s["sas"],
                "avg_iter_ms": s["avg_iter_ms"],
            })
        elif ci == 2.0:
            # 区分大模型 standard（从模型名判断）和中模型
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
    parser.add_argument("--out", type=str, default="outputs/dwrr_ablation")
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

    # 策略：D1/D2/D3 + 对照组
    policies = {
        "Fair": Fair(),
        "SRPT": SRPT(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu": LongLiu(K=2.0, use_dynamic_T_target=True),
        "D1": LongLiuDWRR(K=2.0, use_soft_weights=False, intra_class_fair=False, clip_ratio=10.0),
        "D2": LongLiuDWRRFair(K=2.0),
        "D3": LongLiuDWRR(K=2.0, use_soft_weights=True, intra_class_fair=False, clip_ratio=10.0),
    }

    # Git 信息
    git_info = get_git_info()
    if git_info["dirty"]:
        print("❌ Git 工作区有未提交的改动，拒绝运行。")
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
    print("DWRR Ablation: 7 策略 × 3 seeds")
    print("="*80)
    print(f"Git commit: {git_info['commit']}")
    print(f"Output: {args.out}")
    print()

    all_results = {}
    for pname, policy in policies.items():
        print(f"运行 {pname}...")
        results = []
        for seed in range(args.seeds):
            r = run_single(cfg, policy, seed, pname)
            results.append(r)
            print(f"  Seed {seed}: Overall Mean SAS {r['overall_mean_sas']:.3f}")
        all_results[pname] = results

    # 汇总报告
    print()
    print("="*80)
    print("DWRR Ablation 报告（异构 workload）")
    print("="*80)

    # 1. 分层 Mean/Median/Capped SAS
    print("1. 分层 Mean/Median/Capped SAS:")
    print(f"{'Policy':<12} {'Premium Mean':>11} {'Premium Med':>11} {'Premium Cap':>11} | "
          f"{'Standard Mean':>12} {'Standard Med':>12} {'Standard Cap':>12}")
    print("-"*90)
    for pname, results in all_results.items():
        # Premium 档
        premium_sas = [s["sas"] for r in results for s in r["tier_stats"]["premium"]]
        premium_mean = sum(premium_sas) / len(premium_sas) if premium_sas else 0.0
        premium_median = sorted(premium_sas)[len(premium_sas)//2] if premium_sas else 0.0
        premium_capped = sum(min(s, 1.0) for s in premium_sas) / len(premium_sas) if premium_sas else 0.0

        # Standard 档
        standard_sas = [s["sas"] for r in results for s in r["tier_stats"]["standard"]]
        standard_mean = sum(standard_sas) / len(standard_sas) if standard_sas else 0.0
        standard_median = sorted(standard_sas)[len(standard_sas)//2] if standard_sas else 0.0
        standard_capped = sum(min(s, 1.0) for s in standard_sas) / len(standard_sas) if standard_sas else 0.0

        print(f"{pname:<12} {premium_mean:>11.3f} {premium_median:>11.3f} {premium_capped:>11.3f} | "
              f"{standard_mean:>12.3f} {standard_median:>12.3f} {standard_capped:>12.3f}")
    print()

    # 2. 分层崩溃率
    print("2. 分层崩溃率（SAS < 0.2）:")
    print(f"{'Policy':<12} {'Premium':>9} {'Standard':>9} {'Medium':>9} {'Small':>9} {'Overall':>9}")
    print("-"*60)
    for pname, results in all_results.items():
        premium_collapse = sum(1 for r in results for s in r["tier_stats"]["premium"] if s["sas"] < 0.2) / \
                           sum(len(r["tier_stats"]["premium"]) for r in results) * 100 if any(r["tier_stats"]["premium"] for r in results) else 0.0
        standard_collapse = sum(1 for r in results for s in r["tier_stats"]["standard"] if s["sas"] < 0.2) / \
                            sum(len(r["tier_stats"]["standard"]) for r in results) * 100 if any(r["tier_stats"]["standard"] for r in results) else 0.0
        medium_collapse = sum(1 for r in results for s in r["tier_stats"]["medium"] if s["sas"] < 0.2) / \
                          sum(len(r["tier_stats"]["medium"]) for r in results) * 100 if any(r["tier_stats"]["medium"] for r in results) else 0.0
        small_collapse = sum(1 for r in results for s in r["tier_stats"]["small"] if s["sas"] < 0.2) / \
                         sum(len(r["tier_stats"]["small"]) for r in results) * 100 if any(r["tier_stats"]["small"] for r in results) else 0.0

        all_sas = [s["sas"] for r in results for tier in ["premium", "standard", "medium", "small"] for s in r["tier_stats"][tier]]
        overall_collapse = sum(1 for s in all_sas if s < 0.2) / len(all_sas) * 100 if all_sas else 0.0

        print(f"{pname:<12} {premium_collapse:>8.1f}% {standard_collapse:>8.1f}% "
              f"{medium_collapse:>8.1f}% {small_collapse:>8.1f}% {overall_collapse:>8.1f}%")
    print()

    # 3. SLO 达成率
    print("3. 分层 SLO 达成率:")
    print(f"{'Policy':<12} {'Premium':>9} {'Standard':>9} {'Medium':>9} {'Small':>9} {'Overall':>9}")
    print("-"*60)
    for pname, results in all_results.items():
        premium_slo = sum(1 for r in results for s in r["tier_stats"]["premium"] if s["sas"] >= 1.0) / \
                      sum(len(r["tier_stats"]["premium"]) for r in results) * 100 if any(r["tier_stats"]["premium"] for r in results) else 0.0
        standard_slo = sum(1 for r in results for s in r["tier_stats"]["standard"] if s["sas"] >= 1.0) / \
                       sum(len(r["tier_stats"]["standard"]) for r in results) * 100 if any(r["tier_stats"]["standard"] for r in results) else 0.0
        medium_slo = sum(1 for r in results for s in r["tier_stats"]["medium"] if s["sas"] >= 1.0) / \
                     sum(len(r["tier_stats"]["medium"]) for r in results) * 100 if any(r["tier_stats"]["medium"] for r in results) else 0.0
        small_slo = sum(1 for r in results for s in r["tier_stats"]["small"] if s["sas"] >= 1.0) / \
                    sum(len(r["tier_stats"]["small"]) for r in results) * 100 if any(r["tier_stats"]["small"] for r in results) else 0.0

        all_sas = [s["sas"] for r in results for tier in ["premium", "standard", "medium", "small"] for s in r["tier_stats"][tier]]
        overall_slo = sum(1 for s in all_sas if s >= 1.0) / len(all_sas) * 100 if all_sas else 0.0

        print(f"{pname:<12} {premium_slo:>8.1f}% {standard_slo:>8.1f}% "
              f"{medium_slo:>8.1f}% {small_slo:>8.1f}% {overall_slo:>8.1f}%")
    print()

    # 保存结果
    with open(os.path.join(args.out, "dwrr_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"结果已保存至: {args.out}")


if __name__ == "__main__":
    main()


def dump_topology_ground_truth(topo, jobs):
    """输出拓扑 ground truth。"""
    print("\n" + "="*80)
    print("拓扑 Ground Truth")
    print("="*80)
    
    print(f"\nFat-Tree 参数：")
    print(f"  k = {topo.k}")
    print(f"  hosts = k³/4 = {topo.k ** 3 // 4}")
    print(f"  spine switches = (k/2)² = {(topo.k // 2) ** 2}")
    
    print(f"\n带宽配置：")
    print(f"  host_bw_bps = {topo.host_bw_bps / 1e9:.0f} Gbps")
    print(f"  spine_bw_bps = {topo.spine_bw_bps / 1e9:.0f} Gbps")
    print(f"  num_spine_links = {topo.num_spine_links}")
    print(f"  per_spine_link_bw = {topo.spine_bw_bps / topo.num_spine_links / 1e9:.0f} Gbps")
    
    print(f"\n链路清单：")
    for i, link in enumerate(topo.spine_links):
        print(f"  spine-{i}: {link.bw_bps / 1e9:.0f} Gbps")
    
    print(f"\n24 个 job 的实际放置表：")
    for i, job in enumerate(jobs[:5]):  # 只显示前 5 个
        print(f"  {job.jid}: model={job.model}, dp={job.num_workers}, hosts={job.worker_hosts}")
    print(f"  ... (共 {len(jobs)} 个 job)")


def dump_traffic_table():
    """输出通信量数值表。"""
    print("\n" + "="*80)
    print("通信量数值表（D_i 作为 workload 参数）")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Params':>12} {'FP16':>8} {'MB/iter':>12}")
    print("-"*60)
    
    for model, params in sorted(MODEL_PARAMS.items()):
        param_count = params.get("params", 0)
        fp16 = params.get("fp16", True)
        bpp = 2 if fp16 else 4
        mb_per_iter = 2 * param_count * bpp / (1024 * 1024)
        
        print(f"{model:<20} {param_count/1e9:>10.2f}B {str(fp16):>8} {mb_per_iter:>12.0f}")


def run_single(cfg: dict, policy, seed: int, policy_name: str) -> dict:
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
        model_types=["ResNet-18", "ResNet-50-fp16", "BERT-Base", "BERT-Large-fp16",
                     "ViT-Base", "ViT-Large", "LLaMA-2-1B", "LLaMA-2-7B", "T5-1B"],
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

    # 拓扑 ground truth（只输出一次）
    if seed == 0 and policy_name == "Fair":
        dump_topology_ground_truth(topo, jobs)
        dump_traffic_table()

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

    return {
        "seed": seed,
        "policy": policy_name,
        "tier_stats": tier_stats,
        "overall_mean_sas": sum(s["sas"] for s in stats.values()) / len(stats),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="outputs/paper_baseline_v2/dwrr_ablation")
    args = parser.parse_args()

    # 配置（保持当前工作点）
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

    # Git 信息
    git_info = get_git_info()
    
    # 只检查 .py 文件的修改，忽略 __pycache__
    dirty_py_files = [f for f in git_info["dirty_files"] if f.endswith(".py")]
    if dirty_py_files:
        print("❌ Git 工作区有未提交的改动，拒绝运行。")
        print("未提交的 Python 文件:")
        for f in dirty_py_files:
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
    print("DWRR 消融实验：D1/D2/D3 + 对照组")
    print("="*80)
    print(f"Git commit: {git_info['commit']}")
    print(f"工作点: 400G spine / 600s / 24 jobs")
    print(f"Output: {args.out}")
    print()

    # 定义策略
    policies = {
        # 对照组
        "Fair": Fair(),
        "SRPT": SRPT(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu-SP": LongLiu(K=2.0, use_dynamic_T_target=True, max_weight_ratio=10.0),
        
        # DWRR 消融臂
        "D1_DWRR_1:2:4:8:16:32:64": LongLiuDWRR(
            K=2.0, 
            use_dynamic_T_target=True,
            dwrr_weights={0:1, 18:2, 28:4, 26:8, 36:16, 34:32, 38:64}
        ),
        "D2_DWRR_fair_intra": LongLiuDWRRFair(
            K=2.0,
            use_dynamic_T_target=True,
            dwrr_weights={0:1, 18:2, 28:4, 26:8, 36:16, 34:32, 38:64}
        ),
        "D3_DWRR_soft_1:2:3:4:6:8:12": LongLiuDWRR(
            K=2.0,
            use_dynamic_T_target=True,
            dwrr_weights={0:1, 18:2, 28:3, 26:4, 36:6, 34:8, 38:12}
        ),
    }

    all_results = {}
    for policy_name, policy in policies.items():
        print(f"运行 {policy_name}...")
        results = []
        for seed in range(args.seeds):
            r = run_single(cfg, policy, seed, policy_name)
            results.append(r)
            print(f"  Seed {seed}: Overall Mean SAS {r['overall_mean_sas']:.3f}")
        all_results[policy_name] = results

    # 汇总报告
    print()
    print("="*80)
    print("DWRR 消融结果报告")
    print("="*80)

    # 分层崩溃率
    print("\n分层崩溃率（SAS < 0.2）:")
    print(f"{'Policy':<30} {'Large':>10} {'Medium':>10} {'Small':>10} {'Overall':>10}")
    print("-"*80)
    
    for policy_name, results in all_results.items():
        large_collapse = sum(1 for r in results for s in r["tier_stats"]["large"] if s["sas"] < 0.2) / \
                         sum(len(r["tier_stats"]["large"]) for r in results) * 100
        medium_collapse = sum(1 for r in results for s in r["tier_stats"]["medium"] if s["sas"] < 0.2) / \
                          sum(len(r["tier_stats"]["medium"]) for r in results) * 100
        small_collapse = sum(1 for r in results for s in r["tier_stats"]["small"] if s["sas"] < 0.2) / \
                         sum(len(r["tier_stats"]["small"]) for r in results) * 100
        overall_collapse = sum(1 for r in results for s in r["tier_stats"]["large"] + r["tier_stats"]["medium"] + r["tier_stats"]["small"] if s["sas"] < 0.2) / \
                           sum(len(r["tier_stats"]["large"]) + len(r["tier_stats"]["medium"]) + len(r["tier_stats"]["small"]) for r in results) * 100
        
        print(f"{policy_name:<30} {large_collapse:>9.1f}% {medium_collapse:>9.1f}% {small_collapse:>9.1f}% {overall_collapse:>9.1f}%")

    # Mean SAS
    print("\n分层 Mean SAS:")
    print(f"{'Policy':<30} {'Large':>10} {'Medium':>10} {'Small':>10} {'Overall':>10}")
    print("-"*80)
    
    for policy_name, results in all_results.items():
        large_sas = [s["sas"] for r in results for s in r["tier_stats"]["large"]]
        medium_sas = [s["sas"] for r in results for s in r["tier_stats"]["medium"]]
        small_sas = [s["sas"] for r in results for s in r["tier_stats"]["small"]]
        overall_sas = [r["overall_mean_sas"] for r in results]
        
        print(f"{policy_name:<30} {sum(large_sas)/len(large_sas):>10.3f} "
              f"{sum(medium_sas)/len(medium_sas):>10.3f} "
              f"{sum(small_sas)/len(small_sas):>10.3f} "
              f"{sum(overall_sas)/len(overall_sas):>10.3f}")

    # Median SAS
    print("\n分层 Median SAS:")
    print(f"{'Policy':<30} {'Large':>10} {'Medium':>10} {'Small':>10} {'Overall':>10}")
    print("-"*80)
    
    for policy_name, results in all_results.items():
        large_sas = [s["sas"] for r in results for s in r["tier_stats"]["large"]]
        medium_sas = [s["sas"] for r in results for s in r["tier_stats"]["medium"]]
        small_sas = [s["sas"] for r in results for s in r["tier_stats"]["small"]]
        all_sas = [s["sas"] for r in results for s in r["tier_stats"]["large"] + r["tier_stats"]["medium"] + r["tier_stats"]["small"]]
        
        large_sorted = sorted(large_sas)
        medium_sorted = sorted(medium_sas)
        small_sorted = sorted(small_sas)
        all_sorted = sorted(all_sas)
        
        large_median = large_sorted[len(large_sorted)//2] if large_sorted else 0.0
        medium_median = medium_sorted[len(medium_sorted)//2] if medium_sorted else 0.0
        small_median = small_sorted[len(small_sorted)//2] if small_sorted else 0.0
        overall_median = all_sorted[len(all_sorted)//2] if all_sorted else 0.0
        
        print(f"{policy_name:<30} {large_median:>10.3f} "
              f"{medium_median:>10.3f} "
              f"{small_median:>10.3f} "
              f"{overall_median:>10.3f}")

    # Capped SAS
    print("\n分层 Capped SAS (min(SAS,1)):")
    print(f"{'Policy':<30} {'Large':>10} {'Medium':>10} {'Small':>10} {'Overall':>10}")
    print("-"*80)
    
    for policy_name, results in all_results.items():
        large_sas = [min(s["sas"], 1.0) for r in results for s in r["tier_stats"]["large"]]
        medium_sas = [min(s["sas"], 1.0) for r in results for s in r["tier_stats"]["medium"]]
        small_sas = [min(s["sas"], 1.0) for r in results for s in r["tier_stats"]["small"]]
        all_sas = [min(s["sas"], 1.0) for r in results for s in r["tier_stats"]["large"] + r["tier_stats"]["medium"] + r["tier_stats"]["small"]]
        
        print(f"{policy_name:<30} {sum(large_sas)/len(large_sas):>10.3f} "
              f"{sum(medium_sas)/len(medium_sas):>10.3f} "
              f"{sum(small_sas)/len(small_sas):>10.3f} "
              f"{sum(all_sas)/len(all_sas):>10.3f}")

    # SLO 达成率（SAS >= 1.0）
    print("\n分层 SLO 达成率（SAS >= 1.0）:")
    print(f"{'Policy':<30} {'Large':>10} {'Medium':>10} {'Small':>10} {'Overall':>10}")
    print("-"*80)
    
    for policy_name, results in all_results.items():
        large_met = sum(1 for r in results for s in r["tier_stats"]["large"] if s["sas"] >= 1.0) / \
                    sum(len(r["tier_stats"]["large"]) for r in results) * 100
        medium_met = sum(1 for r in results for s in r["tier_stats"]["medium"] if s["sas"] >= 1.0) / \
                     sum(len(r["tier_stats"]["medium"]) for r in results) * 100
        small_met = sum(1 for r in results for s in r["tier_stats"]["small"] if s["sas"] >= 1.0) / \
                    sum(len(r["tier_stats"]["small"]) for r in results) * 100
        overall_met = sum(1 for r in results for s in r["tier_stats"]["large"] + r["tier_stats"]["medium"] + r["tier_stats"]["small"] if s["sas"] >= 1.0) / \
                      sum(len(r["tier_stats"]["large"]) + len(r["tier_stats"]["medium"]) + len(r["tier_stats"]["small"]) for r in results) * 100
        
        print(f"{policy_name:<30} {large_met:>9.1f}% {medium_met:>9.1f}% {small_met:>9.1f}% {overall_met:>9.1f}%")

    # 判读
    print("\n" + "="*80)
    print("判读标准：")
    print("="*80)
    print("1. Large 达成率 > Fair/SRPT/CRUX/LongLiu-SP")
    print("2. Overall capped SAS ≥ SRPT (0.848)")
    print("3. Large 崩溃率 ≤ CRUX (27.8%)")
    print("="*80)

    # 保存结果
    with open(os.path.join(args.out, "dwrr_ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n结果已保存至: {args.out}")


if __name__ == "__main__":
    main()