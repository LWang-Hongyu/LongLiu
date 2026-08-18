"""
exp_e11b_overlap_waste: Overlap 带宽浪费率实验（回应"串行模型过度分配"关切）。

Waste Ratio = (Σ allocated − Σ used) / Σ allocated
  - allocated_i = v4 闭式解每窗口分配给 job i 的带宽（v4 trace 记录）
  - used_i      = job i 实际占用的物理带宽（仿真中 flow rate 的时间平均）

overlap_factor ∈ {0.0, 0.3, 0.5, 0.85, 1.0}，spine_bw ∈ {500, 630} Gbps。
LongLiu (v4) 主测，WFS 作 P-attn 参照。

用法：
    python experiments/exp_e11b_overlap_waste.py --seeds 5
    python experiments/exp_e11b_overlap_waste.py --seeds 2 --quick
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import yaml
import numpy as np

from longliu_sim.network import FatTreeTopology
from longliu_sim.policy.wfs import WFS
from longliu_sim.policy.dwrr import LongLiuAllocatorV4
from longliu_sim.trace.synthetic import (
    SyntheticTraceLoader,
    FEAS_BOUNDARY_V3_WORKLOAD,
)
from experiments._spine_probe import BwProbeSimulator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E11B_CONFIG = os.path.join(PROJECT_ROOT, "configs", "e11b_overlap_waste.yaml")

SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02
DURATION_MS = 600000.0


def load_config() -> dict:
    with open(E11B_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def get_policy(name: str, trace_file: str, overhead: float, overlap: float):
    if name == "WFS":
        return WFS()
    elif name == "v4":
        return LongLiuAllocatorV4(
            overhead_factor=overhead, overlap_factor=overlap,
            trace_file=trace_file,
        )
    raise ValueError(f"Unknown policy: {name}")


def read_allocated_trace(trace_path: str, duration_ms: float) -> dict:
    """从 v4 trace 读取 per-job 时间平均分配带宽 (bps)。

    同一 time_ms 可能有多个（跨 spine）allocate 调用行，按时间聚合相加。
    """
    rows = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}

    per_time: dict = {}
    for r in rows:
        t = r["time_ms"]
        d = per_time.setdefault(t, {})
        for k, v in r.items():
            if k.endswith("_bw_gbps"):
                jid = k[: -len("_bw_gbps")]
                d[jid] = d.get(jid, 0.0) + float(v)

    times = sorted(per_time)
    allocated = defaultdict(float)
    for i, t in enumerate(times):
        nxt = times[i + 1] if i + 1 < len(times) else duration_ms
        dt = max(nxt - t, 0.0)
        for jid, bw_gbps in per_time[t].items():
            allocated[jid] += bw_gbps * 1e9 * dt
    for jid in allocated:
        allocated[jid] /= duration_ms
    return dict(allocated)


def time_avg_used(samples: list, duration_ms: float) -> dict:
    """per-job 实际带宽的时间平均 (bps)。"""
    used = defaultdict(float)
    for i in range(1, len(samples)):
        t_prev, per_job_prev = samples[i - 1]
        t_cur, _ = samples[i]
        dt = t_cur - t_prev
        for jid, bw in per_job_prev.items():
            used[jid] += bw * dt
    if samples:
        t_last, per_job_last = samples[-1]
        dt = max(duration_ms - t_last, 0.0)
        for jid, bw in per_job_last.items():
            used[jid] += bw * dt
    return {jid: v / duration_ms for jid, v in used.items()}


def compute_attain_bw(job, overhead_factor: float, overlap_factor: float) -> float:
    """job 的 SLO 需求带宽 attain_bw (bps) —— 与 v4 的 _compute_attain_bw 同源。

    wire_bits = bits_per_iter × overhead_factor
    comm_budget = ci × comm_solo × overhead_factor
    target = max(comp, comm_budget) + (1−overlap) × min(comp, comm_budget)
    attain_bw = wire_bits / (target − comp)
    """
    wire_bits = job.bits_per_iter * overhead_factor
    comm_budget_ms = job.slo_ci * job.comm_solo_ms * overhead_factor
    if overlap_factor > 0:
        target_ms = max(job.comp_ms, comm_budget_ms) + \
                    (1.0 - overlap_factor) * min(job.comp_ms, comm_budget_ms)
    else:
        target_ms = job.comp_ms + comm_budget_ms
    effective_budget_ms = target_ms - job.comp_ms
    if effective_budget_ms <= 0:
        return float('inf')
    return wire_bits / (effective_budget_ms * 1e-3)


def allocation_precision(used_bw: dict, jobs: list,
                         overhead_factor: float, overlap_factor: float) -> float:
    """Allocation Precision = Σ min(a_i, b_i^att) / Σ a_i。

    a_i = job i 的时间平均分配带宽（flow-level 下分配即使用，用 bw_samples）
    b_i^att = job i 的 SLO 需求带宽（attain，与策略无关）

    v4 精确分配（a_i ≤ b_i^att）→ precision ≈ 1.0；
    WFS 权重分配可能给某些 job 超需求带宽（a_i > b_i^att）→ precision < 1.0。
    """
    total_alloc = 0.0
    total_capped = 0.0
    for job in jobs:
        a = used_bw.get(job.jid, 0.0)
        if a <= 0:
            continue
        att = compute_attain_bw(job, overhead_factor, overlap_factor)
        total_alloc += a
        total_capped += min(a, att) if att != float('inf') else a
    if total_alloc <= 0:
        return 1.0
    return total_capped / total_alloc


def run_single(scene: str, workload, spine_bw: float, overlap: float,
               policy_name: str, seed: int, frozen: dict) -> dict:
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"{scene}_{policy_name}_{int(spine_bw)}g_ov{overlap}_s{seed}"
    out_dir = f"outputs/e11b_overlap_waste/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)
    policy = get_policy(policy_name, trace_file, frozen["overhead_factor"], overlap)
    sim = BwProbeSimulator(topo, policy, duration_ms=DURATION_MS, seed=seed,
                           overhead_factor=frozen["overhead_factor"],
                           overlap_factor=overlap)

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=n_jobs, duration_ms=DURATION_MS, seed=seed,
        overhead_factor=frozen["overhead_factor"], target_bw_bps=100e9, num_hosts=16,
        workload_profile=list(workload),
    )
    jobs = loader.load()
    for i, j in enumerate(jobs):
        j.jid = f"J{i}"
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    if hasattr(policy, 'flush_trace'):
        policy.flush_trace()

    stats = result.per_job_stats(host_bw_gbps=HOST_BW_GBPS)

    n_premium = len(premium_jids)
    n_premium_attn = 0
    p_cap_total = 0.0
    s_sas_values = []
    for jid, s in stats.items():
        sas = s["sas"]
        if jid in premium_jids:
            if sas >= 1.0 - V4_TOLERANCE:
                n_premium_attn += 1
            p_cap_total += min(sas, 1.0)
        else:
            s_sas_values.append(min(sas, 1.0))

    p_attn = n_premium_attn / n_premium if n_premium > 0 else 1.0
    p_cap = p_cap_total / n_premium if n_premium > 0 else 1.0
    s_cont_cap = np.mean(s_sas_values) if s_sas_values else 0.0

    # waste ratio：物理层面 = 1 − per-spine 平均利用率（每条 spine 独立 cap 100%）
    # （flow-level 下"分配即使用"，物理浪费只体现为链路空闲时段；
    #   v4 在 twotier 下按总容量对每条 spine 分配，单链路利用率需 cap 到物理上限）
    total_util = sim.time_avg_spine_util(DURATION_MS)
    waste_ratio = 1.0 - total_util

    # per-job allocated / used（辅助分析）
    allocated_bw = None
    used_bw = time_avg_used(sim.bw_samples, DURATION_MS)
    if policy_name == "v4":
        allocated_bw = read_allocated_trace(trace_file, DURATION_MS)

    # Allocation Precision = Σ min(a_i, b_i^att) / Σ a_i
    # （回应"串行模型是否过度分配"：v4 精确分配 → ≈1.0；WFS 权重分配可能超需求 → <1.0）
    prec = allocation_precision(used_bw, jobs,
                                frozen["overhead_factor"], overlap)

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "scene": scene, "spine_bw": int(spine_bw),
        "overlap_factor": overlap,
        "policy": policy_name, "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "p_attn": round(p_attn, 4),
        "p_cap": round(p_cap, 4),
        "s_cont_cap": round(s_cont_cap, 4),
        "total_util": round(total_util, 4),
        "waste_ratio": round(waste_ratio, 4),
        "allocation_precision": round(prec, 4),
        "total_iters": result.total_iterations(),
    }
    if allocated_bw is not None:
        run_meta["sum_allocated_gbps"] = round(sum(allocated_bw.values()) / 1e9, 2)
        run_meta["sum_used_gbps"] = round(sum(used_bw.values()) / 1e9, 2)
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def main():
    parser = argparse.ArgumentParser(description="E11b Overlap waste-ratio experiment")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    n_seeds = 2 if args.quick else args.seeds
    seeds = list(range(n_seeds))

    print("=" * 80)
    print("E11b: Overlap Bandwidth Waste Ratio")
    print("=" * 80)
    print(f"Seeds: {seeds}  Overlap: {cfg['overlap_factors']}")
    print()

    all_runs = []
    for overlap in cfg["overlap_factors"]:
        for spine_bw in cfg["spine_bw_gbps"]:
            for pn in cfg["policies"]:
                seed_vals = []
                waste_vals = []
                for seed in seeds:
                    label = f"@{spine_bw}G ov={overlap} {pn} s{seed}"
                    print(f"[{label}] ", end="", flush=True)
                    try:
                        r = run_single("E1", FEAS_BOUNDARY_V3_WORKLOAD, spine_bw,
                                       overlap, pn, seed, frozen)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue
                    all_runs.append(r)
                    seed_vals.append(r["p_attn"])
                    waste_vals.append(r["waste_ratio"])
                    print(f"P-attn={r['p_attn']*100:5.1f}% "
                          f"waste={r['waste_ratio']*100:5.1f}%")
                if len(seed_vals) == len(seeds):
                    print(f"  → mean±std = {np.mean(seed_vals)*100:.1f}±{np.std(seed_vals)*100:.1f}%"
                          f"  waste={np.mean(waste_vals)*100:.1f}±{np.std(waste_vals)*100:.1f}%")
        print()

    # 聚合
    groups = {}
    for r in all_runs:
        k = (r["spine_bw"], r["overlap_factor"], r["policy"])
        groups.setdefault(k, []).append(r)

    os.makedirs("outputs/e11b_overlap_waste", exist_ok=True)
    with open("outputs/e11b_overlap_waste/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["spine_bw", "overlap_factor", "policy", "n_seeds",
                         "p_attn_mean", "p_attn_std",
                         "waste_ratio_mean", "waste_ratio_std",
                         "allocation_precision_mean", "allocation_precision_std",
                         "total_util_mean", "s_cont_cap_mean"])
        for k, runs in sorted(groups.items()):
            bw, ov, pol = k
            p_attns = [r["p_attn"] for r in runs]
            wastes = [r["waste_ratio"] for r in runs]
            precs = [r["allocation_precision"] for r in runs]
            utils = [r["total_util"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            writer.writerow([bw, ov, pol, len(runs),
                             round(np.mean(p_attns), 4),
                             round(np.std(p_attns), 4),
                             round(np.mean(wastes), 4),
                             round(np.std(wastes), 4),
                             round(np.mean(precs), 4),
                             round(np.std(precs), 4),
                             round(np.mean(utils), 4),
                             round(np.mean(s_caps), 4)])

    print("Summary saved to outputs/e11b_overlap_waste/summary.csv")
    print("*** E11b COMPLETE ***")


if __name__ == "__main__":
    main()
