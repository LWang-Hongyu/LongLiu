"""
复现论文 Table 3: SLO attainment comparison (16 hosts, 24 jobs).

运行 Fair, SRPT, CRUX, LongLiu 四种策略，输出：
- Tight SLO (ci=1.5) attainment %
- Medium SLO (ci=2.0) attainment %
- Loose SLO (ci=3.0) attainment %
- Overall attainment %
- Total iterations (×10^4)

用法：
    python experiments/exp_ablation.py --seeds 10
    python experiments/exp_ablation.py --config configs/fatree_16host.yaml --seeds 10
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
from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu
from longliu_sim.core import Simulator
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD
from longliu_sim.trace.synthetic_128 import TABLE4_TIERED_WORKLOAD_128
from longliu_sim.utils.model_params import MODEL_PARAMS

# 可选依赖：scipy.stats.ttest_rel（无 scipy 时跳过 p-value 计算）
try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# CI_TIERS 必须在运行时从 workload 推断；此处仅保留为向后兼容的兜底。
# 实际 tier 集合由 _validate_tiers() 从 workload_profile 提取并断言。
CI_TIERS = [1.2, 2.0, 3.0]

DEFAULT_MODEL_TYPES = [
    "ResNet-18", "ResNet-50-fp16", "BERT-Base", "BERT-Large-fp16",
    "ViT-Base", "ViT-Large", "LLaMA-2-1B", "LLaMA-2-7B", "T5-1B",
]
DEFAULT_GPU_DIST = {1: 0.2, 2: 0.2, 4: 0.3, 8: 0.3}
DEFAULT_CI_DIST = {1.5: 0.3, 2.0: 0.35, 3.0: 0.35}
# DEFAULT_OVERHEAD 仅用于 SyntheticTraceLoader 内部计算 iter_interval_ms，
# 不再参与 SAS/T_target 计算；SAS 统一使用 sas_eval 口径（overlap_factor）。
DEFAULT_OVERHEAD = 2.0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_topology(cfg: dict):
    topo_cfg = cfg["topology"]
    t = topo_cfg["type"]
    if t == "single_bottleneck":
        return SingleLinkTopology(
            num_hosts=topo_cfg.get("num_hosts", 2),
            bw_bps=topo_cfg["bandwidth_bps"],
        )
    elif t == "fatree":
        return FatTreeTopology(
            k=topo_cfg["k"],
            host_bw_bps=topo_cfg.get("host_bw_bps", topo_cfg.get("tor_bw_bps", 100e9)),
            spine_bw_bps=topo_cfg.get("spine_bw_bps", 100e9),
        )
    else:
        raise ValueError(f"Unknown topology type: {t}")


def generate_jobs(cfg: dict, seed: int, overhead_factor: float,
                  static_arrival: bool = False) -> list:
    """用 SyntheticTraceLoader 生成 workload。"""
    jobs_cfg = cfg.get("jobs", {})
    count = jobs_cfg.get("sample_count", jobs_cfg.get("count", 24))
    topo_cfg = cfg["topology"]
    host_bw = topo_cfg.get("host_bw_bps", topo_cfg.get("bandwidth_bps", 40e9))

    # 从拓扑配置推导主机数量，用于 worker placement（确保跨 rack 通信）
    t = topo_cfg["type"]
    if t == "fatree":
        num_hosts = topo_cfg["k"] ** 2 // 2  # k=4 → 16 hosts
    else:
        num_hosts = topo_cfg.get("num_hosts", 2)

    # 根据 job 数量选择对应的 workload profile
    if "workload_profile" in jobs_cfg:
        profile_name = jobs_cfg["workload_profile"]
        if profile_name == "TABLE4_TIERED_WORKLOAD_128":
            workload_profile = TABLE4_TIERED_WORKLOAD_128
        else:
            raise ValueError(f"Unknown workload_profile: {profile_name}")
    else:
        if count == 128:
            workload_profile = TABLE4_TIERED_WORKLOAD_128
        else:
            workload_profile = DEFAULT_TIERED_WORKLOAD

    loader = SyntheticTraceLoader(
        model_types=jobs_cfg.get("model_types", DEFAULT_MODEL_TYPES),
        gpu_distribution=jobs_cfg.get("gpu_distribution", DEFAULT_GPU_DIST),
        ci_distribution=jobs_cfg.get("ci_distribution", DEFAULT_CI_DIST),
        job_count=count,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=overhead_factor,
        target_bw_bps=host_bw,
        num_hosts=num_hosts,
        workload_profile=workload_profile,
    )
    jobs = loader.load()
    # 消融模式：所有 job 在 time=0 同时启动
    if static_arrival:
        for j in jobs:
            j.start_time_ms = 0.0
    return jobs


def compute_fairness_metrics(sas_values: list[float]) -> dict:
    """计算公平性指标。"""
    n = len(sas_values)
    if n == 0:
        return {"jain_index": 1.0, "gini_coeff": 0.0, "catastrophic_rate": 0.0}

    # Jain 公平指数
    sum_sas = sum(sas_values)
    sum_sas_sq = sum(v * v for v in sas_values)
    jain = (sum_sas ** 2) / (n * sum_sas_sq) if sum_sas_sq > 0 else 1.0

    # Gini 系数
    sorted_sas = sorted(sas_values)
    total = sum_sas
    gini_sum = 0.0
    for i, v in enumerate(sorted_sas):
        gini_sum += (i + 1) * v
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n if total > 0 else 0.0

    # HANDOFF 指标：零迭代 OR SAS < 0.2（零迭代已在 SAS=0 时自然落入此区间）
    handoff = sum(1 for v in sas_values if v < 0.2) / n

    return {"jain_index": jain, "gini_coeff": gini, "handoff_rate": handoff}


def _validate_tiers(workload_profile: list):
    """从 workload profile 提取实际 ci 集合，并断言无空 tier。"""
    actual_tiers = sorted({ci for _, _, ci in workload_profile})
    if not actual_tiers:
        raise ValueError("workload_profile 为空")
    # 禁止 CI_TIERS 与实际 tier 不一致导致空集合输出 0.0
    for ci in actual_tiers:
        if ci not in CI_TIERS:
            raise ValueError(
                f"workload 包含未在 CI_TIERS 中声明的 tier ci={ci}"
            )
    return actual_tiers


def run_experiment(cfg: dict, policy, seed: int, overhead_factor: float,
                   overlap_factor: float = 1.0,
                   static_arrival: bool = False) -> dict:
    """运行一次仿真，返回 SLO tier 指标。"""
    topo = build_topology(cfg)
    sim = Simulator(topo, policy, duration_ms=cfg["duration_ms"],
                    seed=seed, overhead_factor=overhead_factor,
                    overlap_factor=overlap_factor)
    jobs = generate_jobs(cfg, seed, overhead_factor, static_arrival=static_arrival)

    # 推断实际 tier 集合（从使用的 workload_profile）
    count = cfg.get("jobs", {}).get("sample_count", cfg.get("jobs", {}).get("count", 24))
    if count == 128:
        workload_profile = TABLE4_TIERED_WORKLOAD_128
    else:
        workload_profile = cfg.get("jobs", {}).get("workload_profile", DEFAULT_TIERED_WORKLOAD)
        if workload_profile == "TABLE4_TIERED_WORKLOAD_128":
            workload_profile = TABLE4_TIERED_WORKLOAD_128
    actual_tiers = _validate_tiers(workload_profile)

    for j in jobs:
        sim.submit(j)
    result = sim.run()
    host_bw_gbps = cfg["topology"].get("host_bw_bps", 100e9) / 1e9
    stats = result.per_job_stats(host_bw_gbps=host_bw_gbps)

    tier_meets: dict[float, list[bool]] = {c: [] for c in actual_tiers}
    tier_sas: dict[float, list[float]] = {c: [] for c in actual_tiers}
    all_sas: list[float] = []
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        if ci not in tier_meets:
            raise ValueError(
                f"job {jid} 的 ci={ci} 不在 actual_tiers={actual_tiers} 中"
            )
        tier_meets[ci].append(s["meets_slo"])
        tier_sas[ci].append(s["sas"])
        all_sas.append(s["sas"])

    out = {"total_iters": result.total_iterations()}
    for ci in actual_tiers:
        if not tier_meets[ci]:
            raise ValueError(f"tier ci={ci} 为空，禁止输出 0.0")
        attainment = sum(tier_meets[ci]) / len(tier_meets[ci])
        out[f"slo_attainment_ci{ci}"] = attainment
        mean_sas = sum(tier_sas[ci]) / len(tier_sas[ci])
        median_sas = sorted(tier_sas[ci])[len(tier_sas[ci]) // 2]
        out[f"sas_mean_ci{ci}"] = mean_sas
        out[f"sas_median_ci{ci}"] = median_sas

    all_ok = sum(1 for s in stats.values() if s["meets_slo"])
    out["slo_attainment_overall"] = all_ok / len(stats) if stats else 0.0

    if all_sas:
        out["sas_mean_overall"] = sum(all_sas) / len(all_sas)
        out["sas_median_overall"] = sorted(all_sas)[len(all_sas) // 2]
        out["sas_min_overall"] = min(all_sas)
        out["sas_max_overall"] = max(all_sas)
    else:
        out["sas_mean_overall"] = 0.0
        out["sas_median_overall"] = 0.0
        out["sas_min_overall"] = 0.0
        out["sas_max_overall"] = 0.0

    # 计算公平性指标
    fairness = compute_fairness_metrics(all_sas)
    out.update(fairness)

    # 收集 LongLiu 插桩数据（启动期占比）
    if hasattr(policy, "get_instrumentation"):
        instr = policy.get_instrumentation()
        total = instr["total_alloc_calls"]
        if total > 0:
            out["startup_epoch_pct"] = instr["startup_alloc_calls"] / total * 100
            out["startup_job_avg"] = instr["startup_job_total"] / total
        else:
            out["startup_epoch_pct"] = 0.0
            out["startup_job_avg"] = 0.0

    out["per_job"] = []
    for jid, s in stats.items():
        job = sim.jobs[jid]
        out["per_job"].append({
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

    return out


def main():
    parser = argparse.ArgumentParser(description="Table 3: SLO attainment ablation")
    parser.add_argument("--config", default="configs/fatree_16host.yaml")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="outputs/table3")
    parser.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD,
                        help=f"NCCL/PCIe overhead factor (default {DEFAULT_OVERHEAD})")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override simulation duration_ms from config")
    parser.add_argument("--overlap", type=float, default=0.85,
                        help="Compute-comm overlap factor (0=serial, 1=full overlap, default 0.85)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_startup", "no_weighted", "4level_dscp", "static_arrival", "no_ema"],
                        help="P1 ablation mode")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["v2_robust"],
                        help="Variant mode: v2_robust enables all four control-theoretic mechanisms")
    parser.add_argument("--dwrr_floor", action="store_true",
                        help="Enable DWRR floor bandwidth protection (5/5/8/10/12/20/40%)")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    cfg = load_config(cfg_path)
    if args.duration is not None:
        cfg["duration_ms"] = args.duration
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "outputs"), exist_ok=True)

    if args.ablation is not None:
        # P1 消融实验：只有 LongLiu，和对应的消融变种
        ablation_params = {}
        if args.ablation == "no_startup":
            ablation_params["no_startup"] = True
        elif args.ablation == "no_weighted":
            ablation_params["no_weighted"] = True
        elif args.ablation == "4level_dscp":
            ablation_params["n_dscp_levels"] = 4
        elif args.ablation == "no_ema":
            ablation_params["use_dynamic_T_target"] = False

        default_params = {"K": 2.0, "use_dynamic_T_target": True}
        # 避免重复传 use_dynamic_T_target（no_ema 消融会覆盖它）
        variant_params = {**default_params, **ablation_params}
        POLICIES = {
            "LongLiu_default": LongLiu(**default_params),
            f"LongLiu_{args.ablation}": LongLiu(**variant_params),
        }
        ablation_label = args.ablation
    else:
        POLICIES = {
            "Fair": Fair(),
            "SRPT": SRPT(),
            "CRUX": CRUX(alpha=1.0),
            "LongLiu": LongLiu(K=2.0, use_dynamic_T_target=True),
        }
        ablation_label = None

    if args.variant == "v2_robust":
        # 控制论四件套全开版本（vs 原生 LongLiu）
        POLICIES["LongLiu_v2"] = LongLiu(
            K=2.0, use_dynamic_T_target=True,
            dead_zone_delta=0.1,
            window_size=8,
            aging_L=5,
            hysteresis_h=0.05,
        )

    if args.dwrr_floor:
        # DWRR 地板版本：P0-P6 各保底 5/5/8/10/12/20/40%
        POLICIES["LongLiu_DWRR"] = LongLiu(
            K=2.0, use_dynamic_T_target=True,
            dwrr_floor=True,
        )

    print(f"  Config: {args.config}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Overhead factor: {args.overhead}")
    print(f"  Overlap factor: {args.overlap}")

    results: dict[str, list[dict]] = {name: [] for name in POLICIES}
    per_seed_sas: dict[str, list[float]] = {name: [] for name in POLICIES}
    static_arrival = (args.ablation == "static_arrival")

    for name, policy in POLICIES.items():
        print(f"  Running {name} ...")
        for seed in range(args.seeds):
            r = run_experiment(cfg, policy, seed, args.overhead, args.overlap,
                               static_arrival=static_arrival)
            results[name].append(r)
            per_seed_sas[name].append(r["sas_mean_overall"])

    # 动态 metrics：基于 actual_tiers（从第一个结果推断，假设所有 seed tier 一致）
    actual_tiers = sorted({float(k.split("ci")[1]) for k in results[list(POLICIES.keys())[0]][0].keys()
                           if k.startswith("slo_attainment_ci")})
    metrics = ["total_iters"] + [f"slo_attainment_ci{ci}" for ci in actual_tiers] + ["slo_attainment_overall"]
    sas_metrics = ["sas_mean_overall", "sas_median_overall", "sas_min_overall", "sas_max_overall"] + \
                  [f"sas_mean_ci{ci}" for ci in actual_tiers]

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
                ci = 1.96 * std / (len(vals) ** 0.5)
                summary[name][f"{m}_ci_lower"] = mean - ci
                summary[name][f"{m}_ci_upper"] = mean + ci
            else:
                summary[name][f"{m}_ci_lower"] = mean
                summary[name][f"{m}_ci_upper"] = mean

    # Paired t-test: 各策略 vs CRUX
    baseline_name = "CRUX"
    if baseline_name in per_seed_sas and _HAS_SCIPY:
        for name in POLICIES:
            if name != baseline_name:
                baseline_vals = per_seed_sas[baseline_name]
                target_vals = per_seed_sas[name]
                if len(baseline_vals) == len(target_vals) and len(baseline_vals) >= 2:
                    try:
                        _, p_value = scipy_stats.ttest_rel(target_vals, baseline_vals)
                        summary[name]["p_vs_crux"] = p_value
                    except Exception:
                        summary[name]["p_vs_crux"] = 1.0

    out_dir = os.path.join(os.path.dirname(__file__), "..", args.output)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "table3.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Policy"] + [f"{m}(mean±std)" for m in all_metric_names] + ["p_vs_crux"]
        w.writerow(header)
        for name in POLICIES:
            row = [name]
            for m in all_metric_names:
                mean = summary[name][m]
                std = summary[name].get(f"{m}_std", 0.0)
                row.append(f"{mean:.4f}±{std:.4f}")
            p_val = summary[name].get("p_vs_crux", "")
            row.append(f"{p_val:.4e}" if isinstance(p_val, float) else "")
            w.writerow(row)
    print(f"  CSV → {csv_path}")

    tex_path = os.path.join(out_dir, "table3.tex")
    with open(tex_path, "w") as f:
        f.write("% Table 3: SLO Attainment & Achievement Comparison (16 hosts, 24 jobs)\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{SLO attainment and achievement comparison (16 hosts, 24 jobs)}\n")
        f.write("\\label{tab:slo_16host}\n")
        f.write("\\begin{tabular}{l|r|r|r|r|r|r}\n")
        f.write("\\hline\n")
        # LaTeX 列头动态对齐实际 tier
        tier_cols = " & ".join(f"ci={ci}" for ci in actual_tiers)
        f.write(f"Policy & Total Iters ($\\times 10^4$) & {tier_cols} & Overall & SAS \\\\\n")
        f.write("\\hline\n")
        for name in POLICIES:
            s = summary[name]
            total_k = s["total_iters"] / 10000
            tier_vals = " & ".join(f"{s[f'slo_attainment_ci{ci}']*100:.1f}\\%" for ci in actual_tiers)
            overall = s["slo_attainment_overall"] * 100
            sas_mean = s["sas_mean_overall"]
            f.write(f"{name} & {total_k:.2f} & {tier_vals} & {overall:.1f}\\% & {sas_mean:.3f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  LaTeX → {tex_path}")

    print("\n  Summary (Binary Attainment):")
    tier_headers = " ".join(f"ci={ci}%".ljust(10) for ci in actual_tiers)
    print(f"  {'Policy':<12} {'Total(K)':<10} {tier_headers} {'Overall%':<10}")
    for name in POLICIES:
        s = summary[name]
        tier_vals = " ".join(f"{s[f'slo_attainment_ci{ci}']*100:<10.1f}" for ci in actual_tiers)
        print(f"  {name:<12} {s['total_iters']/10000:<10.2f} {tier_vals} {s['slo_attainment_overall']*100:<10.1f}")

    print("\n  Summary (SAS - SLO Achievement Score):")
    print(f"  {'Policy':<12} {'Mean SAS':<10} {'Median SAS':<12} {'Min SAS':<10} {'Max SAS':<10} {'p_vs_CRUX':<12}")
    for name in POLICIES:
        s = summary[name]
        p_str = f"{s.get('p_vs_crux', 1.0):.4e}" if "p_vs_crux" in s else "N/A"
        print(f"  {name:<12} {s['sas_mean_overall']:<10.3f} {s['sas_median_overall']:<12.3f} "
              f"{s['sas_min_overall']:<10.3f} {s['sas_max_overall']:<10.3f} {p_str:<12}")

    print("\n  Summary (Fairness Metrics):")
    print(f"  {'Policy':<12} {'Jain Index':<12} {'Gini Coeff':<12} {'HANDOFF Rate':<20}")
    for name in POLICIES:
        s = summary[name]
        print(f"  {name:<12} {s.get('jain_index', 0):<12.4f} {s.get('gini_coeff', 0):<12.4f} "
              f"{s.get('handoff_rate', 0):<20.4f}")

    per_job_path = os.path.join(out_dir, "per_job.json")
    per_job_data = {}
    for name in POLICIES:
        per_job_data[name] = []
        for seed_idx, r in enumerate(results[name]):
            for job in r["per_job"]:
                job_copy = job.copy()
                job_copy["seed"] = seed_idx
                per_job_data[name].append(job_copy)
    with open(per_job_path, "w") as f:
        json.dump(per_job_data, f, indent=2)
    print(f"\n  Per-job details → {per_job_path}")


if __name__ == "__main__":
    main()