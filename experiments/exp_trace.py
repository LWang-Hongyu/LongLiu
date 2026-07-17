"""
P3: Trace-driven 验证 - 使用 Alibaba Lingjun 数据集。

在真实 trace workload 上对比 LongLiu vs CRUX。

用法：
    python experiments/exp_trace.py --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from longliu_sim.network import FatTreeTopology
from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu
from longliu_sim.core import Simulator
from longliu_sim.trace import LingjunTraceLoader

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

CI_TIERS = [1.5, 2.0, 3.0]

TRACE_ZIP = "/home/why/LongLiu_rebuild/alibaba-lingjun-dataset-2023-main.zip"
TOPOLOGY_K = 4
HOST_BW = 100e9
SPINE_BW = 400e9
DURATION_MS = 300000
NUM_HOSTS = 16


def build_topology():
    return FatTreeTopology(
        k=TOPOLOGY_K,
        host_bw_bps=HOST_BW,
        spine_bw_bps=SPINE_BW,
    )


def generate_jobs(seed: int):
    loader = LingjunTraceLoader(
        zip_path=TRACE_ZIP,
        max_gpus=128,
        duration_ms=DURATION_MS,
        seed=seed,
        target_bw_bps=HOST_BW,
        overhead_factor=1.3,
    )
    return loader.load()


def compute_fairness_metrics(sas_values: list[float]) -> dict:
    n = len(sas_values)
    if n == 0:
        return {"jain_index": 1.0, "gini_coeff": 0.0, "catastrophic_rate": 0.0}
    sum_sas = sum(sas_values)
    sum_sas_sq = sum(v * v for v in sas_values)
    jain = (sum_sas ** 2) / (n * sum_sas_sq) if sum_sas_sq > 0 else 1.0
    sorted_sas = sorted(sas_values)
    total = sum_sas
    gini_sum = 0.0
    for i, v in enumerate(sorted_sas):
        gini_sum += (i + 1) * v
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n if total > 0 else 0.0
    catastrophic = sum(1 for v in sas_values if v < 0.2) / n
    return {"jain_index": jain, "gini_coeff": gini, "catastrophic_rate": catastrophic}


def run_experiment(policy, seed: int, overhead_factor: float = 1.3) -> dict:
    topo = build_topology()
    sim = Simulator(topo, policy, duration_ms=DURATION_MS, seed=seed,
                    overhead_factor=overhead_factor)
    jobs = generate_jobs(seed)
    if not jobs:
        return {"error": "no_jobs", "total_iters": 0}
    for j in jobs:
        sim.submit(j)
    result = sim.run()
    stats = result.per_job_stats()

    tier_meets = {c: [] for c in CI_TIERS}
    tier_sas = {c: [] for c in CI_TIERS}
    all_sas = []
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        tier_meets[ci].append(s["meets_slo"])
        tier_sas[ci].append(s["sas"])
        all_sas.append(s["sas"])

    out = {"total_iters": result.total_iterations(), "n_jobs": len(jobs)}
    for ci in CI_TIERS:
        if tier_meets[ci]:
            attainment = sum(tier_meets[ci]) / len(tier_meets[ci])
        else:
            attainment = 0.0
        out[f"slo_attainment_ci{ci}"] = attainment
        if tier_sas[ci]:
            mean_sas = sum(tier_sas[ci]) / len(tier_sas[ci])
            median_sas = sorted(tier_sas[ci])[len(tier_sas[ci]) // 2]
        else:
            mean_sas = 0.0
            median_sas = 0.0
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

    fairness = compute_fairness_metrics(all_sas)
    out.update(fairness)
    return out


def main():
    parser = argparse.ArgumentParser(description="P3: Trace-driven validation")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="outputs/trace")
    args = parser.parse_args()

    POLICIES = {
        "Fair": Fair(),
        "SRPT": SRPT(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu": LongLiu(K=2.0, use_dynamic_T_target=True),
    }

    results = {name: [] for name in POLICIES}
    per_seed_sas = {name: [] for name in POLICIES}

    for name, policy in POLICIES.items():
        print(f"  Running {name} ...")
        for seed in range(args.seeds):
            r = run_experiment(policy, seed)
            if "error" in r:
                print(f"    seed {seed}: {r['error']}")
                continue
            results[name].append(r)
            per_seed_sas[name].append(r["sas_mean_overall"])
            print(f"    seed {seed}: {r.get('n_jobs', 0)} jobs, SAS={r['sas_mean_overall']:.3f}")

    metric_names = ["total_iters", "slo_attainment_overall"] + \
                   [f"slo_attainment_ci{ci}" for ci in CI_TIERS] + \
                   ["sas_mean_overall", "sas_median_overall", "sas_min_overall", "sas_max_overall"] + \
                   [f"sas_mean_ci{ci}" for ci in CI_TIERS] + \
                   ["jain_index", "gini_coeff", "catastrophic_rate"]

    summary = {}
    for name in POLICIES:
        summary[name] = {}
        for m in metric_names:
            vals = [r[m] for r in results[name]]
            mean = sum(vals) / len(vals) if vals else 0.0
            std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            summary[name][m] = mean
            summary[name][f"{m}_std"] = std

    baseline_name = "CRUX"
    if baseline_name in per_seed_sas and _HAS_SCIPY:
        for name in POLICIES:
            if name != baseline_name:
                bv = per_seed_sas[baseline_name]
                tv = per_seed_sas[name]
                if len(bv) == len(tv) and len(bv) >= 2:
                    try:
                        _, pv = scipy_stats.ttest_rel(tv, bv)
                        summary[name]["p_vs_crux"] = pv
                    except Exception:
                        summary[name]["p_vs_crux"] = 1.0

    out_dir = os.path.join(os.path.dirname(__file__), "..", args.output)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "trace_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Policy"] + metric_names + ["p_vs_crux"]
        w.writerow(header)
        for name in POLICIES:
            row = [name]
            for m in metric_names:
                mean = summary[name].get(m, 0)
                std = summary[name].get(f"{m}_std", 0)
                row.append(f"{mean:.4f}±{std:.4f}")
            pv = summary[name].get("p_vs_crux", "")
            row.append(f"{pv:.4e}" if isinstance(pv, float) else "")
            w.writerow(row)
    print(f"\n  CSV → {csv_path}")

    print("\n  Summary (SAS - SLO Achievement Score):")
    print(f"  {'Policy':<12} {'Mean SAS':<10} {'Median SAS':<12} {'Min SAS':<10} {'Max SAS':<10} {'Catastrophic':<14} {'p_vs_CRUX':<12}")
    for name in POLICIES:
        s = summary[name]
        p_str = f"{s.get('p_vs_crux', 1.0):.4e}" if "p_vs_crux" in s else "N/A"
        print(f"  {name:<12} {s.get('sas_mean_overall', 0):<10.3f} {s.get('sas_median_overall', 0):<12.3f} "
              f"{s.get('sas_min_overall', 0):<10.3f} {s.get('sas_max_overall', 0):<10.3f} "
              f"{s.get('catastrophic_rate', 0):<14.4f} {p_str:<12}")


if __name__ == "__main__":
    main()
