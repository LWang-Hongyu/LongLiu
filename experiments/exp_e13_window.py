"""
exp_e13_window: 窗口大小 W 敏感性实验（论文 E13 章节）。

测试 LongLiu 控制环策略在不同 window_size 下的瞬态响应。
E3 swap 场景：t=300s 执行 tier swap，观察优先级调整速度。

预期目标：
- W 减小（如 5）：锚点估计受计算噪声影响变大，可能导致优先级抖动
- W 增大（如 50）：对 Tier Swap 的响应变慢
- W=20：较好的工程折中点

用法：
    python experiments/exp_e13_window.py --seeds 5
    python experiments/exp_e13_window.py --seeds 2 --quick   # 快速验证
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import yaml
import numpy as np

from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.policy.longliu import LongLiu
from longliu_sim.trace.synthetic import (
    SyntheticTraceLoader,
    FEAS_BOUNDARY_V3_PRIME_WORKLOAD,  # E2' Adversarial 场景
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E13_CONFIG = os.path.join(PROJECT_ROOT, "configs", "e13_window.yaml")

SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
V4_TOLERANCE = 0.02


def load_e13_config() -> dict:
    with open(E13_CONFIG) as f:
        return yaml.safe_load(f)


def load_frozen() -> dict:
    from longliu_sim.utils.config import load_config
    return load_config()["frozen"]


def run_single(window_size: int, workload, spine_bw: float,
               seed: int, frozen: dict) -> dict:
    """运行单次实验，返回 run_meta dict。

    使用 E2' Adversarial 场景（CRUX 杀场）：
    - 9 个 premium job (大模型低 intensity → LongLiu 低优先级)
    - 5 个 standard job (小模型高 intensity → LongLiu 高优先级)
    - 带宽 500G（过渡区）
    - 在 t=100s 时注入 3 个新 job 制造动态变化
    """
    import random
    rng = random.Random(seed)
    n_jobs = len(workload)

    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"w{window_size}_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/e13_window/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9,
                           spine_bw_bps=spine_bw * 1e9)

    # LongLiu 控制环策略，设置 window_size
    policy = LongLiu(
        window_size=window_size,
        use_dynamic_T_target=True,
    )

    sim = Simulator(topo, policy, duration_ms=300000, seed=seed,
                    overhead_factor=frozen["overhead_factor"],
                    overlap_factor=frozen["overlap_factor"])

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=n_jobs, duration_ms=300000, seed=seed,
        overhead_factor=frozen["overhead_factor"], target_bw_bps=100e9, num_hosts=16,
        workload_profile=list(workload),
    )
    jobs = loader.load()
    for i, j in enumerate(jobs):
        j.jid = f"J{i}"

    # 提交初始 jobs
    for j in jobs:
        sim.submit(j)

    # t=100s: 注入 3 个新 job（制造动态变化）
    new_jobs_profile = [
        ("LLaMA-2-13B", 8, 1.5),  # J14: premium
        ("BERT-Large-fp16", 4, 3.0),  # J15: standard
        ("ViT-Large", 2, 2.0),  # J16: standard
    ]
    new_loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=3, duration_ms=200000, seed=seed + 100,
        overhead_factor=frozen["overhead_factor"], target_bw_bps=100e9, num_hosts=16,
        workload_profile=new_jobs_profile,
    )
    new_jobs = new_loader.load()
    for i, j in enumerate(new_jobs):
        j.jid = f"J{n_jobs + i}"
        j.start_time_ms = 100000.0  # t=100s 开始

    # 提交新 jobs
    for j in new_jobs:
        sim.submit(j)

    # 更新 premium_jids 包含新注入的 premium job
    for i, (_, _, ci) in enumerate(new_jobs_profile):
        if ci <= 2.0:
            premium_jids.add(f"J{n_jobs + i}")

    result = sim.run()

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

    starv = sum(1 for jid in premium_jids
                if result.jobs[jid].completed_iters == 0)

    # 计算优先级抖动指标（标准差）
    # 通过 trace.jsonl 分析 DSCP 变化频率
    dscp_changes = 0
    if os.path.exists(trace_file):
        with open(trace_file) as f:
            prev_dscp = {}
            for line in f:
                try:
                    row = json.loads(line)
                    for jid in premium_jids:
                        dscp_key = f"{jid}_dscp"
                        if dscp_key in row:
                            curr_dscp = row[dscp_key]
                            if jid in prev_dscp and prev_dscp[jid] != curr_dscp:
                                dscp_changes += 1
                            prev_dscp[jid] = curr_dscp
                except:
                    pass

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "window_size": window_size, "spine_bw": int(spine_bw),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_premium": n_premium,
        "n_standard": len(stats) - n_premium,
        "p_attn": round(p_attn, 4),
        "p_cap": round(p_cap, 4),
        "s_cont_cap": round(s_cont_cap, 4),
        "starv": starv,
        "dscp_changes": dscp_changes,
        "total_iters": result.total_iterations(),
    }
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def main():
    parser = argparse.ArgumentParser(description="E13 Window size sensitivity experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick validation (2 seeds)")
    args = parser.parse_args()

    cfg = load_e13_config()
    frozen = load_frozen()

    global CONFIG_HASH
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    n_seeds = 2 if args.quick else args.seeds
    seeds = list(range(n_seeds))

    print("=" * 80)
    print("E13: Window Size W Sensitivity Analysis")
    print("=" * 80)
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
    print(f"Seeds: {seeds}")
    print(f"Window sizes: {cfg['window_sizes']}")
    print()

    all_runs = []
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9  # 630 Gbps

    for window_size in cfg["window_sizes"]:
        print(f"--- Window Size W = {window_size} ---")
        seed_vals = []
        for seed in seeds:
            label = f"W={window_size} s{seed}"
            print(f"[{label}] ", end="", flush=True)
            try:
                r = run_single(window_size, FEAS_BOUNDARY_V3_PRIME_WORKLOAD, spine_bw, seed, frozen)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            all_runs.append(r)
            seed_vals.append(r["p_attn"])
            print(f"P-attn={r['p_attn']*100:5.1f}%  DSCP changes={r['dscp_changes']}  starv={r['starv']}")

        if len(seed_vals) == len(seeds):
            mean_p = np.mean(seed_vals) * 100
            std_p = np.std(seed_vals) * 100
            print(f"  → mean±std = {mean_p:.1f}±{std_p:.1f}%")
        print()

    # 聚合
    groups = {}
    for r in all_runs:
        k = r["window_size"]
        groups.setdefault(k, []).append(r)

    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"\n{'W':>4s} | {'P-attn':>9s} | {'P-cap':>9s} | {'S-cap':>9s} | {'DSCP changes':>12s} | {'Starv':>5s}")
    print("-" * 70)
    for w in cfg["window_sizes"]:
        if w not in groups:
            continue
        runs = groups[w]
        p_attns = [r["p_attn"] for r in runs]
        p_caps = [r["p_cap"] for r in runs]
        s_caps = [r["s_cont_cap"] for r in runs]
        dscp_changes = [r["dscp_changes"] for r in runs]
        p_attn_mean = np.mean(p_attns) * 100
        p_attn_std = np.std(p_attns) * 100
        p_cap_mean = np.mean(p_caps)
        s_cap_mean = np.mean(s_caps)
        dscp_mean = np.mean(dscp_changes)
        starv = max(r["starv"] for r in runs)
        print(f"{w:>4d} | {p_attn_mean:5.1f}±{p_attn_std:3.1f}% | "
              f"{p_cap_mean:.3f} | {s_cap_mean:.3f} | {dscp_mean:>12.1f} | {starv:>5d}")

    # 保存 summary CSV
    os.makedirs("outputs/e13_window", exist_ok=True)
    with open("outputs/e13_window/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_size", "n_seeds", "seeds",
                         "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean",
                         "dscp_changes_mean"])
        for w in cfg["window_sizes"]:
            if w not in groups:
                continue
            runs = groups[w]
            p_attns = [r["p_attn"] for r in runs]
            p_caps = [r["p_cap"] for r in runs]
            s_caps = [r["s_cont_cap"] for r in runs]
            dscp_changes = [r["dscp_changes"] for r in runs]
            writer.writerow([w, len(runs),
                             f"[{', '.join(str(r['seed']) for r in runs)}]",
                             round(np.mean(p_attns), 4),
                             round(np.std(p_attns), 4),
                             round(np.mean(p_caps), 4),
                             round(np.mean(s_caps), 4),
                             round(np.mean(dscp_changes), 2)])

    print(f"\nSummary saved to outputs/e13_window/summary.csv")
    print("*** E13 COMPLETE ***")


if __name__ == "__main__":
    main()
