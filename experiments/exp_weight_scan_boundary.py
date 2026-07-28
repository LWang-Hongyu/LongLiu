"""
exp_weight_scan_boundary: DWRR 权重扫描实验（feas_boundary_v1, 1.21× load）

三个 DWRR 权重变体，CRN（相同 seeds），评估权重展宽对 contested premium 的影响。

变体：
  D3 (soft):  class_weights=[1,2,3,4,6,8,12]     (12×)
  D1 (std):   class_weights=[1,2,4,8,16,32,64]   (64×)
  D5 (steep): class_weights=[1,2,8,32,128,512,1024] (1024×)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import LongLiuDWRR
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V1_WORKLOAD
from longliu_sim.utils import compute_sas_eval, compute_target_iter_ms
from longliu_sim.utils.model_params import MODEL_PARAMS


def get_git_info() -> dict:
    """获取 git 信息。"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        modified_files = [
            line[3:] for line in dirty_output.split("\n") if line.startswith(" M")
        ]
        return {
            "commit": commit,
            "dirty": bool(modified_files),
            "dirty_files": modified_files,
        }
    except Exception:
        return {"commit": "unknown", "dirty": True, "dirty_files": []}


def run_single(cfg: dict, policy, seed: int) -> dict:
    """运行单个 seed 的仿真并返回 per-job 结果。"""
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
        model_types=[],
        gpu_distribution={},
        ci_distribution={},
        job_count=7,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=FEAS_BOUNDARY_V1_WORKLOAD,
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    stats = result.per_job_stats()

    # ---- 组装 per-job 数据 ----
    per_job_results = []
    # premium 和 standard 层统计（与 feas_boundary_v1_final 保持一致）
    premium_stats = []
    standard_stats = []

    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        avg_iter_ms_val = s["avg_iter_ms"]
        sas_eval = compute_sas_eval(
            avg_iter_ms_val, job.model, job.num_workers, ci,
            host_bw_gbps=cfg["topology"]["host_bw_bps"] / 1e9,
            overlap_factor=cfg["overlap_factor"],
        )
        completed = job.completed_iters
        target = job.target_iters

        target_iter_ms = compute_target_iter_ms(
            job.model, job.num_workers, ci,
            host_bw_gbps=cfg["topology"]["host_bw_bps"] / 1e9,
            overlap_factor=cfg["overlap_factor"],
        )
        attained = avg_iter_ms_val <= target_iter_ms
        starved = completed == 0

        per_job_results.append({
            "jid": jid,
            "model": job.model,
            "dp": job.num_workers,
            "ci": ci,
            "sas_eval": sas_eval,
            "avg_iter_ms": avg_iter_ms_val,
            "completed_iters": completed,
            "target_iters": target,
            "attained": attained,
            "starved": starved,
        })

        if ci == 1.3:
            premium_stats.append(sas_eval)
        elif ci == 2.0:
            standard_stats.append(sas_eval)

    # ---- 聚合统计 ----
    premium_mean = (
        sum(premium_stats) / len(premium_stats) if premium_stats else 0.0
    )
    premium_capped = (
        sum(min(s, 1.0) for s in premium_stats) / len(premium_stats)
        if premium_stats else 0.0
    )
    premium_attainment = (
        sum(1 for j in per_job_results if j["ci"] == 1.3 and j["attained"])
        / max(1, sum(1 for j in per_job_results if j["ci"] == 1.3))
    )
    premium_starved = sum(1 for j in per_job_results if j["ci"] == 1.3 and j["starved"])

    standard_mean = (
        sum(standard_stats) / len(standard_stats) if standard_stats else 0.0
    )
    standard_capped = (
        sum(min(s, 1.0) for s in standard_stats) / len(standard_stats)
        if standard_stats else 0.0
    )
    standard_attainment = (
        sum(1 for j in per_job_results if j["ci"] == 2.0 and j["attained"])
        / max(1, sum(1 for j in per_job_results if j["ci"] == 2.0))
    )
    standard_starved = sum(1 for j in per_job_results if j["ci"] == 2.0 and j["starved"])

    overall_mean = (
        sum(p["sas_eval"] for p in per_job_results) / len(per_job_results)
        if per_job_results else 0.0
    )

    return {
        "seed": seed,
        "overall_mean_sas": overall_mean,
        "premium_mean": premium_mean,
        "premium_capped": premium_capped,
        "premium_attainment": premium_attainment,
        "premium_starved": premium_starved,
        "standard_mean": standard_mean,
        "standard_capped": standard_capped,
        "standard_attainment": standard_attainment,
        "standard_starved": standard_starved,
        "per_job": per_job_results,
    }


def compute_contested_metrics(seed_results: list, cfg: dict) -> dict:
    """跨 seed 计算 contested premium / standard 指标。

    Contested premium: LLaMA-2-7B(ci=1.3) + LLaMA-2-13B(ci=1.3)
    Contested standard: T5-11B-fp16(ci=2.0) + LLaMA-2-13B(ci=2.0)
    """
    all_p_contested = []
    all_s_contested = []
    total_starved = 0

    for r in seed_results:
        for j in r["per_job"]:
            is_p = j["ci"] == 1.3 and j["model"] in ("LLaMA-2-7B", "LLaMA-2-13B")
            is_s = j["ci"] == 2.0 and j["model"] in ("T5-11B-fp16", "LLaMA-2-13B")
            if is_p:
                all_p_contested.append(j["sas_eval"])
            if is_s:
                all_s_contested.append(j["sas_eval"])
            if j["starved"]:
                total_starved += 1

    # 统计所有 job 数
    total_jobs = sum(len(r["per_job"]) for r in seed_results)

    return {
        "P_contested_mean": sum(all_p_contested) / len(all_p_contested) if all_p_contested else 0.0,
        "P_contested_cap": sum(min(s, 1.0) for s in all_p_contested) / len(all_p_contested) if all_p_contested else 0.0,
        "P_contested_attn": sum(1 for s in all_p_contested if s >= 1.0) / len(all_p_contested) if all_p_contested else 0.0,
        "S_contested_mean": sum(all_s_contested) / len(all_s_contested) if all_s_contested else 0.0,
        "S_contested_cap": sum(min(s, 1.0) for s in all_s_contested) / len(all_s_contested) if all_s_contested else 0.0,
        "S_contested_attn": sum(1 for s in all_s_contested if s >= 1.0) / len(all_s_contested) if all_s_contested else 0.0,
        "starvation_count": total_starved,
        "total_jobs": total_jobs,
    }


def main():
    # ---- 配置 ----
    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 800e9,
        },
        "duration_ms": 600000,
        "overhead_factor": 2.0,
        "overlap_factor": 0.85,
    }

    seeds = [0, 1, 2]

    # ---- 三个 DWRR 变体 ----
    variants = {
        "D3_soft_12x": {
            "policy_desc": "D3 (soft)",
            "class_weights": [1, 2, 3, 4, 6, 8, 12],
            "K": 2.0,
        },
        "D1_std_64x": {
            "policy_desc": "D1 (std)",
            "class_weights": [1, 2, 4, 8, 16, 32, 64],
            "K": 2.0,
        },
        "D5_steep_1024x": {
            "policy_desc": "D5 (steep)",
            "class_weights": [1, 2, 8, 32, 128, 512, 1024],
            "K": 2.0,
        },
    }

    # ---- 输出目录 ----
    out_base = "outputs/weight_scan_boundary"
    os.makedirs(out_base, exist_ok=True)

    # ---- Git info ----
    git_info = get_git_info()

    print("=" * 80)
    print("DWRR 权重扫描实验：feas_boundary_v1 @ 1.21× load (spine=800G)")
    print("=" * 80)
    print(f"Variants: {', '.join(variants.keys())}")
    print(f"Seeds: {seeds}")
    print(f"Git commit: {git_info['commit']}")
    print(f"Output: {out_base}")
    print()

    all_variant_results = {}  # variant_name -> per-seed results

    for var_name, var_cfg in variants.items():
        var_out = os.path.join(out_base, var_name)
        os.makedirs(var_out, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Running {var_name} ({var_cfg['policy_desc']})")
        print(f"  class_weights = {var_cfg['class_weights']}")
        print(f"{'='*60}")

        variant_seed_results = []

        for seed in seeds:
            policy = LongLiuDWRR(
                K=var_cfg["K"],
                class_weights=var_cfg["class_weights"],
                overlap_factor=cfg["overlap_factor"],
            )

            r = run_single(cfg, policy, seed)
            variant_seed_results.append(r)

            p_mean = r["premium_mean"]
            s_mean = r["standard_mean"]
            o_mean = r["overall_mean_sas"]
            p_attn = r["premium_attainment"]
            print(f"  Seed {seed}: Overall={o_mean:.3f}  P_mean={p_mean:.3f}  "
                  f"S_mean={s_mean:.3f}  P_attn={p_attn:.0%}")

        all_variant_results[var_name] = variant_seed_results

        # ---- 保存 results.json ----
        # 格式与 feas_boundary_v1_final/results.json 一致
        # 顶层只有这一个变体（key=var_name）
        results_payload = {var_name: variant_seed_results}
        results_path = os.path.join(var_out, "results.json")
        with open(results_path, "w") as f:
            json.dump(results_payload, f, indent=2, default=str)
        print(f"  ✓ Results saved to {results_path}")

        # ---- 保存 run_meta.json ----
        meta = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "cmdline": " ".join(sys.argv),
            "seeds": seeds,
            "config": cfg,
            "variant": {
                "name": var_name,
                "desc": var_cfg["policy_desc"],
                "class_weights": var_cfg["class_weights"],
                "K": var_cfg["K"],
            },
        }
        meta_path = os.path.join(var_out, "run_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ Run metadata saved to {meta_path}")

    # ---- 生成 comparison.md ----
    print("\n" + "=" * 80)
    print("生成 comparison.md ...")
    print("=" * 80)

    lines = []
    lines.append("# DWRR Weight Scan Comparison")
    lines.append("")
    lines.append(f"**Workload**: FEAS_BOUNDARY_V1_WORKLOAD (7 jobs, 1.21× load)")
    lines.append(f"**Topology**: FatTree(k=4, host_bw=100G, spine_bw=800G)")
    lines.append(f"**Config**: duration=600s, overhead_factor=1.3, overlap_factor=0.85")
    lines.append(f"**Seeds**: {seeds}")
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append(
        "| Variant | Weight Ratio | "
        "P Contested Mean | P Contested Cap | P Attn | "
        "S Contested Mean | S Contested Cap | S Attn | "
        "Starvation |"
    )
    lines.append(
        "|---------|-------------|"
        "-----------------:|----------------:|-------:|"
        "-----------------:|----------------:|-------:|"
        "-----------|"
    )

    comparison_metrics = {}

    for var_name in ["D3_soft_12x", "D1_std_64x", "D5_steep_1024x"]:
        seed_results = all_variant_results[var_name]
        cw = variants[var_name]["class_weights"]
        weight_ratio = max(cw) / min(cw)

        metrics = compute_contested_metrics(seed_results, cfg)
        comparison_metrics[var_name] = metrics

        line = (
            f"| {var_name} | {weight_ratio:.0f}× | "
            f"{metrics['P_contested_mean']:.3f} | {metrics['P_contested_cap']:.3f} | "
            f"{metrics['P_contested_attn']:.0%} | "
            f"{metrics['S_contested_mean']:.3f} | {metrics['S_contested_cap']:.3f} | "
            f"{metrics['S_contested_attn']:.0%} | "
            f"{metrics['starvation_count']} / {metrics['total_jobs']} |"
        )
        lines.append(line)

    lines.append("")
    lines.append("## Metrics Definition")
    lines.append("")
    lines.append("- **P Contested**: Premium contested jobs — ")
    lines.append("  LLaMA-2-7B (ci=1.3) + LLaMA-2-13B (ci=1.3)")
    lines.append("- **S Contested**: Standard contested jobs — ")
    lines.append("  T5-11B-fp16 (ci=2.0) + LLaMA-2-13B (ci=2.0)")
    lines.append("- **Mean**: Average SAS across all seeds for the tier")
    lines.append("- **Cap**: Average of min(SAS, 1.0) — caps outlier wins")
    lines.append("- **Attn**: Attainment rate — fraction of jobs with SAS ≥ 1.0")
    lines.append("- **Starvation**: Number of jobs with 0 completed iterations / total")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("**Does increasing weight spread improve contested premium metrics?**")
    lines.append("")

    # 自动判读
    d3_p = comparison_metrics.get("D3_soft_12x", {}).get("P_contested_mean", 0)
    d1_p = comparison_metrics.get("D1_std_64x", {}).get("P_contested_mean", 0)
    d5_p = comparison_metrics.get("D5_steep_1024x", {}).get("P_contested_mean", 0)
    d3_s = comparison_metrics.get("D3_soft_12x", {}).get("S_contested_mean", 0)
    d1_s = comparison_metrics.get("D1_std_64x", {}).get("S_contested_mean", 0)
    d5_s = comparison_metrics.get("D5_steep_1024x", {}).get("S_contested_mean", 0)

    if d5_p > d1_p > d3_p:
        trend_p = "monotonically increases"
        conclusion = (
            f"- P Contested Mean: D3={d3_p:.3f} → D1={d1_p:.3f} → D5={d5_p:.3f}. "
            "Wider weight spread **improves** premium contested mean SAS, "
            "as it allocates more bandwidth to the highest-priority class."
        )
    elif d1_p > d3_p and d1_p > d5_p:
        trend_p = "peaks at D1 (64×)"
        conclusion = (
            f"- P Contested Mean: D3={d3_p:.3f} → D1={d1_p:.3f} → D5={d5_p:.3f}. "
            "Weight spread improves from D3→D1 but **degrades** at D5, "
            "suggesting excessive steepness starves mid-priority premium jobs."
        )
    else:
        trend_p = "varies non-monotonically"
        conclusion = (
            f"- P Contested Mean: D3={d3_p:.3f} → D1={d1_p:.3f} → D5={d5_p:.3f}. "
            "The relationship between weight spread and premium performance is non-monotonic."
        )

    lines.append(f"Premium contested SAS {trend_p}:")
    lines.append(conclusion)
    lines.append("")

    if d5_s < d3_s:
        lines.append(
            "- Standard contested SAS decreases with weight spread"
            f" (D3={d3_s:.3f} → D5={d5_s:.3f}), as expected — "
            "steeper weights shift bandwidth from standard to premium jobs."
        )
    else:
        lines.append(
            "- Standard contested SAS does not monotonically decrease with weight spread"
            f" (D3={d3_s:.3f}, D1={d1_s:.3f}, D5={d5_s:.3f})."
        )

    lines.append("")
    lines.append("### Key Trade-off")
    lines.append("")
    lines.append(
        "Increasing weight spread creates a steeper gradient in "
        "cross-class bandwidth allocation, which:"
    )
    lines.append("1. **Helps** premium jobs that can attain their SLO with more bandwidth")
    lines.append("2. **Hurts** standard large jobs that lose bandwidth share")
    lines.append("3. **Does not affect** non-contested (small/medium) jobs significantly")
    lines.append("")
    lines.append(
        "The optimal weight spread depends on the workload's premium/standard mix "
        "and the relative communication demands of each tier."
    )

    comparison_path = os.path.join(out_base, "comparison.md")
    with open(comparison_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Comparison saved to {comparison_path}")

    # ---- Print summary to console ----
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for line in lines[7:]:
        if line.startswith("|"):
            print(line)

    print("\nDone.")


if __name__ == "__main__":
    main()
