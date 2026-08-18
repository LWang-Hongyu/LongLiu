"""
验证被动校准（passive_low）是否消除硬冻结现象（EMA 从不更新）。

对比 baseline（仅最高优先级更新）vs passive_low：
- hard_frozen: completed_iters>0 且 ema_update_count==0 的 job 数（从未校准）
- 更新覆盖率: 总 EMA 更新次数 / 总完成迭代数
- frozen: EMA > 2×expected_comm 的 job 数（原 E14 指标）

用法：
    python experiments/_verify_e14_hardfrozen.py
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import exp_e14_probe as E14
from exp_e14_probe import create_high_load_workload, load_e14_config, load_frozen, run_single

PASSIVE_LOW = [0.05, 0.1, 0.2, 0.4, 0.2, 0.4, 1.0]  # P4=0.2, P5=0.4


def measure(jobs, premium_jids, frozen):
    """统计硬冻结 / 更新覆盖率 / EMA 异常。"""
    n_hard_frozen = 0
    total_iters = 0
    total_updates = 0
    n_frozen_ema = 0
    n_completed = 0
    for jid in premium_jids:
        job = jobs[jid]
        total_iters += job.completed_iters
        total_updates += job.ema_update_count
        if job.completed_iters > 0:
            n_completed += 1
            if job.ema_update_count == 0:
                n_hard_frozen += 1
        if job.T_target_ema is not None:
            expected = job.comm_solo_ms * frozen["overhead_factor"]
            if job.T_target_ema > expected * 2.0:
                n_frozen_ema += 1
    coverage = total_updates / total_iters if total_iters > 0 else 0.0
    return n_completed, n_hard_frozen, coverage, n_frozen_ema


def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")) as f:
        E14.CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

    frozen = load_frozen()
    cfg = load_e14_config()
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9
    workload = create_high_load_workload(cfg["n_jobs"], seed=42)
    premium_jids = {f"J{i}" for i, (_, _, ci) in enumerate(workload) if ci <= 2.0}

    configs = [
        ("baseline",   False, None),
        ("passive_low", True, PASSIVE_LOW),
    ]
    seeds = [0, 1]

    print(f"{'config':>12s} | seed | {'completed':>9s} {'hard_frozen':>11s} "
          f"{'coverage':>8s} {'frozen_ema':>10s}")
    print("-" * 68)
    for name, passive, weights in configs:
        for seed in seeds:
            from longliu_sim.core import Simulator
            from longliu_sim.network import FatTreeTopology
            from longliu_sim.policy.longliu import LongLiu
            from longliu_sim.trace.synthetic import SyntheticTraceLoader

            topo = FatTreeTopology(k=4, host_bw_bps=100e9, spine_bw_bps=spine_bw * 1e9)
            policy = LongLiu(window_size=20, use_dynamic_T_target=True,
                             ema_passive=passive, ema_weights=weights)
            sim = Simulator(topo, policy, duration_ms=600000, seed=seed,
                            overhead_factor=frozen["overhead_factor"],
                            overlap_factor=frozen["overlap_factor"])
            loader = SyntheticTraceLoader(
                model_types=[], gpu_distribution={}, ci_distribution={},
                job_count=len(workload), duration_ms=600000, seed=seed,
                overhead_factor=frozen["overhead_factor"], target_bw_bps=100e9,
                num_hosts=16, workload_profile=list(workload))
            jobs = loader.load()
            for i, j in enumerate(jobs):
                j.jid = f"J{i}"
            for j in jobs:
                sim.submit(j)
            res = sim.run()

            n_comp, n_hf, cov, n_fe = measure(res.jobs, premium_jids, frozen)
            print(f"{name:>12s} | s{seed}  | {n_comp:>9d} {n_hf:>11d} {cov*100:>7.1f}% "
                  f"{n_fe:>10d}")
        print()


if __name__ == "__main__":
    main()
