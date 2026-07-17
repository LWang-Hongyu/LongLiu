#!/usr/bin/env python3
"""
SAS (SLO Achievement Score) 诊断脚本。

分析各策略下每个 job 的 SAS 分布，验证 LongLiu 的 SAS 是否 > CRUX。
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from longliu_sim.core import Simulator
from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_topology(cfg: dict):
    from longliu_sim.network import SingleLinkTopology
    topo_cfg = cfg["topology"]
    t = topo_cfg["type"]
    if t == "single_bottleneck":
        return SingleLinkTopology(
            num_hosts=topo_cfg.get("num_hosts", 2),
            bw_bps=topo_cfg["bandwidth_bps"],
        )
    elif t == "fatree":
        bw = topo_cfg.get("spine_bw_bps", topo_cfg.get("tor_bw_bps", 100e9))
        num_hosts = topo_cfg["k"] ** 2 // 2
        return SingleLinkTopology(num_hosts=num_hosts, bw_bps=bw)
    else:
        raise ValueError(f"Unknown topology type: {t}")


def generate_jobs(cfg: dict, seed: int, overhead_factor: float) -> list:
    jobs_cfg = cfg.get("jobs", {})
    count = jobs_cfg.get("sample_count", jobs_cfg.get("count", 24))
    topo_cfg = cfg["topology"]
    host_bw = topo_cfg.get("host_bw_bps", topo_cfg.get("bandwidth_bps", 40e9))

    workload_profile = DEFAULT_TIERED_WORKLOAD

    loader = SyntheticTraceLoader(
        model_types=jobs_cfg.get("model_types", []),
        gpu_distribution=jobs_cfg.get("gpu_distribution", {}),
        ci_distribution=jobs_cfg.get("ci_distribution", {}),
        job_count=count,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=overhead_factor,
        target_bw_bps=host_bw,
        workload_profile=workload_profile,
    )
    return loader.load()


def run_and_analyze(cfg: dict, policy_name: str, policy, seed: int, overhead_factor: float, overlap_factor: float = 1.0):
    """运行仿真并计算 SAS 指标。"""
    topo = build_topology(cfg)
    sim = Simulator(topo, policy, duration_ms=cfg["duration_ms"],
                    seed=seed, overhead_factor=overhead_factor, overlap_factor=overlap_factor)
    jobs = generate_jobs(cfg, seed, overhead_factor)
    for j in jobs:
        sim.submit(j)
    result = sim.run()
    stats = result.per_job_stats()

    # 计算每个 job 的 SAS
    job_sas = []
    for jid, s in stats.items():
        job = sim.jobs[jid]
        avg_iter_ms = s.get("avg_iter_ms", 0)
        meets_slo = s.get("meets_slo", False)
        target_iter_ms = s.get("target_iter_ms", 0)

        if avg_iter_ms > 0 and target_iter_ms > 0:
            # SAS = SLO允许时间 / 实际平均迭代时间
            # target_iter_ms 已经考虑了 overlap 模型
            sas = target_iter_ms / avg_iter_ms
        else:
            sas = 0.0

        job_sas.append({
            "jid": jid,
            "model": job.model,
            "dp": job.num_workers,
            "ci": job.slo_ci,
            "iter_solo_ms": job.iter_solo_ms,
            "avg_iter_ms": avg_iter_ms,
            "target_time_ms": target_iter_ms,
            "sas": sas,
            "meets_slo": meets_slo,
        })

    # 聚合指标
    sas_values = [j["sas"] for j in job_sas]
    mean_sas = np.mean(sas_values) if sas_values else 0.0
    median_sas = np.median(sas_values) if sas_values else 0.0
    min_sas = np.min(sas_values) if sas_values else 0.0
    max_sas = np.max(sas_values) if sas_values else 0.0

    # 按严重程度分类
    catastrophic = sum(1 for s in sas_values if s < 0.2)
    severe = sum(1 for s in sas_values if 0.2 <= s < 0.5)
    moderate = sum(1 for s in sas_values if 0.5 <= s < 0.8)
    minor = sum(1 for s in sas_values if 0.8 <= s < 1.0)
    satisfied = sum(1 for s in sas_values if s >= 1.0)

    # Binary attainment
    binary_attainment = sum(1 for j in job_sas if j["meets_slo"]) / len(job_sas) if job_sas else 0.0

    return {
        "policy": policy_name,
        "mean_sas": mean_sas,
        "median_sas": median_sas,
        "min_sas": min_sas,
        "max_sas": max_sas,
        "binary_attainment": binary_attainment,
        "distribution": {
            "catastrophic": catastrophic,
            "severe": severe,
            "moderate": moderate,
            "minor": minor,
            "satisfied": satisfied,
        },
        "job_details": job_sas,
    }


def main():
    cfg_path = Path(__file__).parent.parent / "configs" / "fatree_16host.yaml"
    cfg = load_config(str(cfg_path))
    overhead_factor = 2.0
    seed = 0

    policies = {
        "Fair": Fair(),
        "SRPT": SRPT(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu": LongLiu(K=2.0, use_dynamic_T_target=True),
    }

    print("=" * 80)
    print("SAS (SLO Achievement Score) 诊断报告")
    print("=" * 80)
    print(f"Config: {cfg_path}")
    print(f"Seed: {seed}")
    print(f"Overhead factor: {overhead_factor}")
    print()

    results = {}
    for policy_name, policy in policies.items():
        print(f"Running {policy_name} ...")
        results[policy_name] = run_and_analyze(cfg, policy_name, policy, seed, overhead_factor)

    # 输出聚合对比
    print()
    print("=" * 80)
    print("聚合指标对比")
    print("=" * 80)
    print(f"{'Policy':<12} {'Mean SAS':<10} {'Median SAS':<12} {'Min SAS':<10} {'Max SAS':<10} {'Binary%':<10}")
    print("-" * 80)
    for policy_name, r in results.items():
        print(f"{policy_name:<12} {r['mean_sas']:<10.3f} {r['median_sas']:<12.3f} "
              f"{r['min_sas']:<10.3f} {r['max_sas']:<10.3f} {r['binary_attainment']*100:<10.1f}")

    # 输出分布对比
    print()
    print("=" * 80)
    print("SAS 严重程度分布")
    print("=" * 80)
    print(f"{'Policy':<12} {'Catastrophic':<14} {'Severe':<8} {'Moderate':<10} {'Minor':<8} {'Satisfied':<10}")
    print("-" * 80)
    for policy_name, r in results.items():
        d = r["distribution"]
        print(f"{policy_name:<12} {d['catastrophic']:<14} {d['severe']:<8} "
              f"{d['moderate']:<10} {d['minor']:<8} {d['satisfied']:<10}")

    # 输出每个 job 的详细信息（只输出 CRUX 和 LongLiu）
    print()
    print("=" * 80)
    print("Job 级别 SAS 详情（CRUX vs LongLiu）")
    print("=" * 80)
    for policy_name in ["CRUX", "LongLiu"]:
        r = results[policy_name]
        print(f"\n{policy_name}:")
        print(f"{'JID':<6} {'Model':<16} {'DP':<4} {'CI':<4} {'iter_solo':<10} {'avg_iter':<10} {'target':<10} {'SAS':<8} {'Meets?'}")
        print("-" * 80)
        for j in r["job_details"]:
            meets = "✓" if j["meets_slo"] else "✗"
            print(f"{j['jid']:<6} {j['model']:<16} {j['dp']:<4} {j['ci']:<4.1f} "
                  f"{j['iter_solo_ms']:<10.1f} {j['avg_iter_ms']:<10.1f} {j['target_time_ms']:<10.1f} "
                  f"{j['sas']:<8.3f} {meets}")

    # 结论
    print()
    print("=" * 80)
    print("结论")
    print("=" * 80)
    longliu_mean_sas = results["LongLiu"]["mean_sas"]
    crux_mean_sas = results["CRUX"]["mean_sas"]
    if longliu_mean_sas > crux_mean_sas:
        print(f"✓ LongLiu Mean SAS ({longliu_mean_sas:.3f}) > CRUX Mean SAS ({crux_mean_sas:.3f})")
        print(f"  相对提升: {(longliu_mean_sas / crux_mean_sas - 1) * 100:.1f}%")
    else:
        print(f"✗ LongLiu Mean SAS ({longliu_mean_sas:.3f}) <= CRUX Mean SAS ({crux_mean_sas:.3f})")
        print(f"  需要调整 LongLiu 算法或 SAS 指标定义")


if __name__ == "__main__":
    main()
