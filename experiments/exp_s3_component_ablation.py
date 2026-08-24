"""
S3 组件消融实验（3 组）

目标：拆解 LongLiu 各核心组件对系统性能的贡献。

3 组消融方向（CRUX 和 LongLiu 都使用严格优先级，区别仅在 key）：
1. Priority Key 对比：CRUX (GPU intensity, 静态) vs LongLiu (progress deficit, 动态反馈)
   → 核心差异：静态属性 vs 动态反馈作为优先级依据
2. T_target 校准对比：动态 T_target (EMA+RTT) vs 静态 T_target (ci*comm_solo)
   → 核心差异：在线校准 vs 离线固定
3. synchronized vs. staggered phases：同步通信 vs CASSINI time-shift
   → 核心差异：通信阶段对齐 vs 错开

输出：outputs/s3_component_ablation/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from longliu_sim.network import SingleLinkTopology, FatTreeTopology
from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu, CASSINI
from longliu_sim.policy.base import Policy, Allocation
from longliu_sim.core import Simulator
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD
from longliu_sim.trace.synthetic_128 import TABLE4_TIERED_WORKLOAD_128
from longliu_sim.network.flow import Flow
from longliu_sim.network.link import Link
from longliu_sim.utils.model_params import MODEL_PARAMS

# 复用 exp_ablation.py 的辅助函数
from experiments.exp_ablation import (
    load_config,
    build_topology,
    generate_jobs,
    compute_fairness_metrics,
    _validate_tiers,
    run_experiment,
    _HAS_SCIPY,
)


class LongLiuStaticT(Policy):
    """
    消融方向 2：LongLiu with static T_target

    - Key：progress deficit (pi)，与 LongLiu 相同
    - 但 T_target 使用静态值 ci * comm_solo_ms，不进行动态校准
    - 用于验证动态 T_target 校准的贡献
    """

    def __init__(self, **kwargs):
        super().__init__("LongLiu_StaticT")
        # 强制 use_dynamic_T_target=False
        kwargs.pop("use_dynamic_T_target", None)
        self.longliu = LongLiu(use_dynamic_T_target=False, **kwargs)

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        return self.longliu.allocate(flows, links, time_ms, job_stats)


class LongLiuStaggered(Policy):
    """
    消融方向 3：LongLiu + CASSINI staggered phases

    - Key/Allocator：完整 LongLiu
    - Phase 安排：CASSINI time-shift（staggered phases）
    通过修改 job.comm_offset_ms 在实验前实现；本策略 allocate 与 LongLiu 相同。
    """

    def __init__(self, **kwargs):
        super().__init__("LongLiu_Staggered")
        self.longliu = LongLiu(**kwargs)

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        return self.longliu.allocate(flows, links, time_ms, job_stats)


def apply_cassini_offsets(jobs: list) -> None:
    """为所有 job 静态计算并设置 CASSINI time-shift 偏移。"""
    intervals = [j.iter_interval_ms for j in jobs]
    offsets = CASSINI.compute_offsets(intervals)
    for j, off in zip(jobs, offsets):
        j.comm_offset_ms = off


def main():
    parser = argparse.ArgumentParser(
        description="S3 component-swap ablation (CRUX/LongLiu/CASSINI hybrid)"
    )
    parser.add_argument("--config", default="configs/fatree_16host.yaml")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="outputs/s3_component_ablation")
    parser.add_argument("--overhead", type=float, default=1.3,
                        help="NCCL/PCIe overhead factor (paper default 1.3)")
    parser.add_argument("--overlap", type=float, default=0.85,
                        help="Compute-comm overlap factor (paper default 0.85)")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    cfg = load_config(cfg_path)
    if args.duration is not None:
        cfg["duration_ms"] = args.duration

    longliu_kwargs = {"K": 2.0, "use_dynamic_T_target": True}

    # 注意：LongLiuStaggered 需要配合 apply_cassini_offsets 才能生效
    POLICIES = {
        # 基线
        "Fair": Fair(),
        # 消融方向 1：Priority Key 对比（都用 SRPT）
        "CRUX": CRUX(alpha=1.0),               # intensity key + SRPT
        "LongLiu": LongLiu(**longliu_kwargs),   # pi key + SRPT + 动态校准
        # 消融方向 2：T_target 校准对比（都用 pi key + SRPT）
        "LongLiu_StaticT": LongLiuStaticT(K=2.0),  # pi key + SRPT + 静态校准
        # 消融方向 3：Phase 安排对比
        "CASSINI": CASSINI(),
        "LongLiu_Staggered": LongLiuStaggered(**longliu_kwargs),
    }

    print(f"  Config: {args.config}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Overhead factor: {args.overhead}")
    print(f"  Overlap factor: {args.overlap}")

    results: dict[str, list[dict]] = {name: [] for name in POLICIES}
    per_seed_sas: dict[str, list[float]] = {name: [] for name in POLICIES}

    for name, policy in POLICIES.items():
        print(f"  Running {name} ...")
        for seed in range(args.seeds):
            print(f"    seed {seed}/{args.seeds} ...", flush=True)
            # CASSINI 和 LongLiu_Staggered 需要重置 comm_offset_ms
            if name in ("CASSINI", "LongLiu_Staggered"):
                jobs = generate_jobs(cfg, seed, args.overhead)
                apply_cassini_offsets(jobs)
                topo = build_topology(cfg)
                sim = Simulator(topo, policy, duration_ms=cfg["duration_ms"],
                                seed=seed, overhead_factor=args.overhead,
                                overlap_factor=args.overlap)
                for j in jobs:
                    sim.submit(j)
                result = sim.run()
                host_bw_gbps = cfg["topology"].get("host_bw_bps", 100e9) / 1e9
                stats = result.per_job_stats(host_bw_gbps=host_bw_gbps)

                actual_tiers = _validate_tiers(DEFAULT_TIERED_WORKLOAD)
                tier_meets = {c: [] for c in actual_tiers}
                tier_sas = {c: [] for c in actual_tiers}
                all_sas = []
                for jid, s in stats.items():
                    job = sim.jobs[jid]
                    ci = job.slo_ci
                    tier_meets[ci].append(s["meets_slo"])
                    tier_sas[ci].append(s["sas"])
                    all_sas.append(s["sas"])

                r = {
                    "total_iters": result.total_iterations(),
                    "slo_attainment_overall": sum(1 for s in stats.values() if s["meets_slo"]) / len(stats) if stats else 0.0,
                    "sas_mean_overall": sum(all_sas) / len(all_sas) if all_sas else 0.0,
                }
                for ci in actual_tiers:
                    r[f"slo_attainment_ci{ci}"] = sum(tier_meets[ci]) / len(tier_meets[ci]) if tier_meets[ci] else 0.0
                    r[f"sas_mean_ci{ci}"] = sum(tier_sas[ci]) / len(tier_sas[ci]) if tier_sas[ci] else 0.0

                fairness = compute_fairness_metrics(all_sas)
                r.update(fairness)

                r["per_job"] = []
                for jid, s in stats.items():
                    job = sim.jobs[jid]
                    r["per_job"].append({
                        "jid": jid,
                        "model": job.model,
                        "dp": job.num_workers,
                        "ci": job.slo_ci,
                        "iter_solo_ms": job.iter_solo_ms,
                        "comp_ms": job.comp_ms,
                        "comm_solo_ms": job.comm_solo_ms,
                        "avg_iter_ms": s["avg_iter_ms"],
                        "avg_comm_ms": s["avg_comm_ms"],
                        "target_iter_ms": s["target_iter_ms"],
                        "meets_slo": s["meets_slo"],
                        "sas": s["sas"],
                    })
            else:
                r = run_experiment(cfg, policy, seed, args.overhead, args.overlap)

            results[name].append(r)
            per_seed_sas[name].append(r["sas_mean_overall"])

    # 动态 metrics
    actual_tiers = sorted({float(k.split("ci")[1]) for k in results[list(POLICIES.keys())[0]][0].keys()
                           if k.startswith("slo_attainment_ci")})
    metrics = ["total_iters"] + [f"slo_attainment_ci{ci}" for ci in actual_tiers] + ["slo_attainment_overall"]
    sas_metrics = ["sas_mean_overall"] + [f"sas_mean_ci{ci}" for ci in actual_tiers]
    fairness_metrics = ["jain_index", "gini_coeff", "handoff_rate"]
    all_metric_names = metrics + sas_metrics + fairness_metrics

    summary: dict[str, dict] = {}
    for name in POLICIES:
        summary[name] = {}
        for m in all_metric_names:
            vals = [r[m] for r in results[name]]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            summary[name][m] = mean
            summary[name][f"{m}_std"] = std
            if len(vals) > 1:
                ci95 = 1.96 * std / (len(vals) ** 0.5)
                summary[name][f"{m}_ci_lower"] = mean - ci95
                summary[name][f"{m}_ci_upper"] = mean + ci95
            else:
                summary[name][f"{m}_ci_lower"] = mean
                summary[name][f"{m}_ci_upper"] = mean

    # paired t-test vs LongLiu
    baseline_name = "LongLiu"
    if baseline_name in per_seed_sas and _HAS_SCIPY:
        for name in POLICIES:
            if name != baseline_name:
                baseline_vals = per_seed_sas[baseline_name]
                target_vals = per_seed_sas[name]
                if len(baseline_vals) == len(target_vals) and len(baseline_vals) >= 2:
                    try:
                        from scipy import stats as scipy_stats
                        _, p_value = scipy_stats.ttest_rel(target_vals, baseline_vals)
                        summary[name]["p_vs_longliu"] = p_value
                    except Exception:
                        summary[name]["p_vs_longliu"] = 1.0

    out_dir = os.path.join(os.path.dirname(__file__), "..", args.output)
    os.makedirs(out_dir, exist_ok=True)

    # CSV 汇总
    csv_path = os.path.join(out_dir, "s3_component_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Policy"] + [f"{m}(mean±std)" for m in all_metric_names] + ["p_vs_longliu"]
        w.writerow(header)
        for name in POLICIES:
            row = [name]
            for m in all_metric_names:
                mean = summary[name][m]
                std = summary[name].get(f"{m}_std", 0.0)
                row.append(f"{mean:.4f}±{std:.4f}")
            p_val = summary[name].get("p_vs_longliu", "")
            row.append(f"{p_val:.4e}" if isinstance(p_val, float) else "")
            w.writerow(row)
    print(f"  CSV → {csv_path}")

    # JSON 原始数据
    raw_path = os.path.join(out_dir, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump({name: [r for r in results[name]] for name in POLICIES}, f, indent=2)
    print(f"  Raw results → {raw_path}")

    # 打印汇总
    print("\n  Summary (Binary Attainment):")
    tier_headers = " ".join(f"ci={ci}%".ljust(10) for ci in actual_tiers)
    print(f"  {'Policy':<25} {'Total(K)':<10} {tier_headers} {'Overall%':<10}")
    for name in POLICIES:
        s = summary[name]
        tier_vals = " ".join(f"{s[f'slo_attainment_ci{ci}']*100:<10.1f}" for ci in actual_tiers)
        print(f"  {name:<25} {s['total_iters']/10000:<10.2f} {tier_vals} {s['slo_attainment_overall']*100:<10.1f}")

    print("\n  Summary (SAS):")
    print(f"  {'Policy':<25} {'Mean SAS':<10} {'Jain':<10} {'HANDOFF%':<10} {'p_vs_LL':<12}")
    for name in POLICIES:
        s = summary[name]
        p_str = f"{s.get('p_vs_longliu', 1.0):.4e}" if "p_vs_longliu" in s else "N/A"
        print(f"  {name:<25} {s['sas_mean_overall']:<10.3f} {s.get('jain_index', 0):<10.4f} "
              f"{s.get('handoff_rate', 0)*100:<10.1f} {p_str:<12}")


if __name__ == "__main__":
    main()
