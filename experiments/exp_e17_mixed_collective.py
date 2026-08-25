"""E17：混合集合通信场景（DDP / ZeRO-3 / MoE 同场混合）。

背景：方案（SLO 保障 + 闭式分配器 + 优先级/加权）作用在带宽分配抽象层，
理论不依赖集合通信类型（任何集合通信都是带宽受限传输：时间 ∝ 数据量/带宽）。
E17 在实验上验证这一点：把主表 E1 的 14 个 job（8P/6S）替换为混合集合通信
负载——5× AllReduce（DDP）、4× AllGather（ZeRO-3 权重）、2× ReduceScatter
（ZeRO-3 梯度）、3× All-to-All（MoE 专家路由），同 model/dp/ci 结构，
仅按类型缩放每轮通信量（allgather/rs/alltoall = 0.5× allreduce）。

- 拓扑：FatTree k=4（同主表），spine 400/500/630/800 Gbps
- 策略：7（Fair/SRPT/CRUX/CASSINI/DF/LL-S/LongLiu）
- 种子：10（s0..s9）
- 输出：outputs/e17_mixed_collective/run_meta_*.json + summary.csv

运行：python3 experiments/exp_e17_mixed_collective.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy.fair import Fair
from longliu_sim.policy.crux import CRUX
from longliu_sim.policy.srpt import SRPT
from longliu_sim.policy.cassini import CASSINI
from longliu_sim.policy.longliu import LongLiu
from longliu_sim.policy.dwrr import LongLiuDWRR, LongLiuAllocatorV4
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.job import Job
from longliu_sim.utils.model_params import MODEL_PARAMS, get_comp_ms
from longliu_sim.trace.synthetic import place_workers_random
from longliu_sim.utils.config import load_config

_cfg = load_config()
SEMANTICS_VERSION = "e17-mixed-collective-v1"
HOST_BW_GBPS = 100.0
OVERLAP = _cfg["frozen"]["overlap_factor"]
OVERHEAD = _cfg["frozen"]["overhead_factor"]
K = _cfg["frozen"]["K"]

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")) as f:
    CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

V4_TOLERANCE = 0.02

POLICIES = ["Fair", "SRPT", "CRUX", "CASSINI", "DF", "LL-S", "LongLiu"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SPINES = [400, 500, 630, 800]

# (model, dp, ci, collective_type) —— 与主表 E1 相同的 model/dp/ci 结构，
# 仅按类型混合集合通信。通信量缩放：allgather/reduce_scatter/alltoall = 0.5×allreduce。
E17_WORKLOAD: List[Tuple[str, int, float, str]] = [
    # Premium tier: 8 jobs
    ("LLaMA-2-13B", 8, 1.5, "allreduce"),       # J0 DDP
    ("LLaMA-2-13B", 8, 1.5, "allgather"),       # J1 ZeRO-3
    ("LLaMA-2-13B", 8, 1.5, "allreduce"),       # J2 DDP
    ("LLaMA-2-13B", 8, 1.5, "alltoall"),        # J3 MoE
    ("LLaMA-2-7B",  8, 1.5, "allgather"),       # J4 ZeRO-3
    ("LLaMA-2-7B",  8, 1.5, "reduce_scatter"),  # J5 ZeRO-3
    ("BERT-Large-fp16", 4, 2.0, "allreduce"),   # J6 DDP
    ("T5-11B-fp16", 8, 2.0, "alltoall"),        # J7 MoE
    # Standard tier: 6 jobs
    ("LLaMA-2-13B", 8, 3.0, "allreduce"),       # J8 DDP
    ("LLaMA-2-13B", 8, 3.0, "allgather"),       # J9 ZeRO-3
    ("BERT-Large-fp16", 4, 3.0, "allgather"),   # J10 ZeRO-3
    ("BERT-Large-fp16", 4, 3.0, "alltoall"),    # J11 MoE
    ("BERT-Large-fp16", 2, 3.0, "allreduce"),   # J12 DDP
    ("BERT-Large-fp16", 2, 3.0, "reduce_scatter"),  # J13 ZeRO-3
]

COLLECTIVE_SCALE = {
    "allreduce": 1.0,        # 2×param（reduce-scatter + all-gather）
    "allgather": 0.5,        # 单阶段：权重 AllGather
    "reduce_scatter": 0.5,   # 单阶段：梯度 ReduceScatter
    "alltoall": 0.5,         # MoE 专家路由（每 worker 收发量级同 ZeRO）
}


def get_policy(name: str, trace_file: str):
    if name == "Fair":
        return Fair()
    elif name == "SRPT":
        return SRPT()
    elif name == "CRUX":
        return CRUX()
    elif name == "CASSINI":
        return CASSINI()
    elif name == "DF":
        return LongLiuDWRR(K=K, overlap_factor=OVERLAP,
                           overhead_factor=OVERHEAD, trace_file=trace_file)
    elif name == "LL-S":
        return LongLiu(use_dynamic_T_target=False)
    elif name == "LongLiu":
        return LongLiuAllocatorV4(overhead_factor=OVERHEAD,
                                  overlap_factor=OVERLAP, trace_file=trace_file)
    raise ValueError(name)


def _apply_cassini_offsets(jobs) -> None:
    """为所有 job 应用 CASSINI 静态通信相位偏移（time-shift）。"""
    offsets = CASSINI.compute_offsets([j.iter_interval_ms for j in jobs])
    for j, off in zip(jobs, offsets):
        j.comm_offset_ms = off


def create_jobs(workload, seed: int, duration_ms: float, num_hosts: int = 16) -> List[Job]:
    """构造 E17 混合集合通信 job 列表（Poisson 到达 + 随机 worker 放置）。"""
    rng = random.Random(seed)
    profile = list(workload)
    rng.shuffle(profile)
    mean_interval_ms = 2.0 * duration_ms / len(profile)

    jobs: List[Job] = []
    current_time = 0.0
    for i, (model, dp, slo_ci, ctype) in enumerate(profile):
        params = MODEL_PARAMS[model]
        base_mb = 2 * params["params"] * (2 if params.get("fp16", True) else 4) / dp / 1e6
        mb_per_iter = base_mb * COLLECTIVE_SCALE[ctype]
        raw_comm_ms = mb_per_iter * 8 * 1024 * 1024 / (HOST_BW_GBPS * 1e9) * 1000.0
        comp_ms = get_comp_ms(model, default=50.0)
        iter_interval_ms = raw_comm_ms + comp_ms

        interval = rng.expovariate(1.0 / mean_interval_ms)
        current_time += interval
        start_time_ms = min(current_time, duration_ms * 0.9)

        jobs.append(Job(
            jid=f"J{i}",
            model=model,
            mb_per_iter=mb_per_iter,
            iter_interval_ms=iter_interval_ms,
            target_iters=999999,
            slo_ci=slo_ci,
            num_workers=dp,
            collective_type=ctype,
            start_time_ms=start_time_ms,
            comm_solo_ms=raw_comm_ms,
            compute_ms=comp_ms,
            overhead_factor=OVERHEAD,
            worker_hosts=place_workers_random(dp, num_hosts, seed=rng.randint(0, 2**31)),
        ))
    return jobs


def run_single(spine_bw: float, policy_name: str, seed: int) -> dict:
    n_jobs = len(E17_WORKLOAD)
    premium_jids = {f"J{i}" for i, (_, _, ci, _) in enumerate(E17_WORKLOAD) if ci <= 2.0}

    tag = f"E17_{policy_name}_{int(spine_bw)}g_s{seed}"
    out_dir = f"outputs/e17_mixed_collective/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9, spine_bw_bps=spine_bw * 1e9)
    policy = get_policy(policy_name, trace_file)
    sim = Simulator(topo, policy, duration_ms=600000, seed=seed,
                    overhead_factor=OVERHEAD, overlap_factor=OVERLAP)

    jobs = create_jobs(E17_WORKLOAD, seed=seed, duration_ms=600000, num_hosts=16)
    for i, j in enumerate(jobs):
        j.jid = f"J{i}"
    if policy_name == "CASSINI":
        _apply_cassini_offsets(jobs)
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
    starv = sum(1 for jid in premium_jids
                if result.jobs[jid].completed_iters == 0)

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "scene": "E17", "spine_bw": int(spine_bw),
        "policy": policy_name, "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_premium": n_premium,
        "n_standard": n_jobs - n_premium,
        "p_attn": round(p_attn, 4),
        "p_cap": round(p_cap, 4),
        "s_cont_cap": round(s_cont_cap, 4),
        "starv": starv,
        "total_iters": result.total_iterations(),
    }
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    return run_meta


def main():
    start = time.time()
    rows = []
    for spine in SPINES:
        for policy in POLICIES:
            for seed in SEEDS:
                rows.append(run_single(spine, policy, seed))
                done = len(rows)
                total = len(SPINES) * len(POLICIES) * len(SEEDS)
                if done % 10 == 0 or done == total:
                    el = time.time() - start
                    print(f"[{done}/{total}] elapsed {el:.0f}s", flush=True)

    os.makedirs("outputs/e17_mixed_collective", exist_ok=True)
    out_csv = "outputs/e17_mixed_collective/e17_summary.csv"
    cols = ["scene", "spine_bw", "policy", "seed", "p_attn", "p_cap", "s_cont_cap", "starv", "total_iters"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"done in {time.time()-start:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main()
