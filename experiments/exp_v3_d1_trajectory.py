"""
exp_v3_d1_trajectory: D1@400G E1 轨迹分析
判定 D1 P-attn 37.5% < Fair 50% 是锁入（少数 premium 独占份额）还是噪声。
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy.dwrr import LongLiuDWRR
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V3_WORKLOAD
from longliu_sim.utils.config import load_config

_cfg = load_config()
OVERLAP = _cfg["frozen"]["overlap_factor"]
OVERHEAD = _cfg["frozen"]["overhead_factor"]

OUT_DIR = "outputs/v3_d1_trajectory_400g"
os.makedirs(OUT_DIR, exist_ok=True)


def run_d1_trace():
    """运行 D1@400G 并返回 trace 行列表。"""
    trace_file = f"{OUT_DIR}/trace.jsonl"
    workload = FEAS_BOUNDARY_V3_WORKLOAD
    n_jobs = len(workload)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9, spine_bw_bps=400e9)
    policy = LongLiuDWRR(
        K=2.0, overlap_factor=OVERLAP, overhead_factor=OVERHEAD,
        trace_file=trace_file,
    )
    sim = Simulator(
        topo, policy, duration_ms=600000, seed=0,
        overhead_factor=OVERHEAD, overlap_factor=OVERLAP,
    )

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=n_jobs, duration_ms=600000, seed=0,
        overhead_factor=OVERHEAD, target_bw_bps=100e9, num_hosts=16,
        workload_profile=list(workload),
    )
    jobs = loader.load()
    for i, j in enumerate(jobs):
        j.jid = f"J{i}"
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    policy.flush_trace()

    # Parse trace
    if not os.path.exists(trace_file):
        print("ERROR: trace file not generated")
        return [], result

    trace_rows = []
    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if line:
                trace_rows.append(json.loads(line))

    return trace_rows, result


def analyze_trajectory(trace_rows, result):
    """分析轨迹：提取 premium job (J0-J7) 的 π 和 share 时间序列。"""
    premium_jids = [f"J{i}" for i in range(8)]  # E1 premium: J0-J7

    # Extract per-job time series
    epochs = []
    pi_series = {jid: [] for jid in premium_jids}
    share_series = {jid: [] for jid in premium_jids}
    bw_series = {jid: [] for jid in premium_jids}

    for row in trace_rows:
        epoch = row.get("epoch", 0)
        epochs.append(epoch)
        for jid in premium_jids:
            pi_key = f"{jid}_pi"
            share_key = f"{jid}_share"
            bw_key = f"{jid}_bw_gbps"
            pi_series[jid].append(row.get(pi_key))
            share_series[jid].append(row.get(share_key))
            bw_series[jid].append(row.get(bw_key))

    n_epochs = len(epochs)
    print(f"Total epochs: {n_epochs}")
    print(f"Total iterations: {result.total_iterations()}")
    print()

    # --- 份额统计：逐 job 均值、标准差、中位数 ---
    print("=" * 80)
    print("Per-Premium Job Share Statistics (over all epochs)")
    print("=" * 80)
    print(f"{'JID':<6} {'Model':<18} {'Mean Share':>12} {'Std':>10} {'Median':>10} {'Max':>10} {'Min':>10}")
    print("-" * 80)

    job_stats = {}
    for jid in premium_jids:
        shares = [s for s in share_series[jid] if s is not None]
        if not shares:
            continue
        job = result.jobs[jid]
        mean_s = sum(shares) / len(shares)
        std_s = (sum((s - mean_s) ** 2 for s in shares) / len(shares)) ** 0.5
        sorted_s = sorted(shares)
        median_s = sorted_s[len(sorted_s) // 2]
        max_s = max(shares)
        min_s = min(shares)
        job_stats[jid] = {
            "model": job.model, "mean": mean_s, "std": std_s,
            "median": median_s, "max": max_s, "min": min_s,
        }
        print(f"{jid:<6} {job.model:<18} {mean_s:>12.4f} {std_s:>10.4f} "
              f"{median_s:>10.4f} {max_s:>10.4f} {min_s:>10.4f}")

    # --- Win/Loss 判据：份额 > 期望份额 的 epoch 计数 ---
    # 期望份额 ≈ 1/8 peers ≈ 0.125
    print()
    print("=" * 80)
    print("Lock-In / Starvation Analysis")
    print("=" * 80)
    print(f"Expected fair share per job: 1/8 = 0.1250")

    # For each epoch, rank jobs by share
    rank_counts = {jid: defaultdict(int) for jid in premium_jids}
    for ei in range(n_epochs):
        epoch_shares = []
        for jid in premium_jids:
            s = share_series[jid][ei]
            if s is not None:
                epoch_shares.append((jid, s))
        epoch_shares.sort(key=lambda x: x[1], reverse=True)
        for rank, (jid, _) in enumerate(epoch_shares):
            rank_counts[jid][rank] += 1

    print(f"\n{'JID':<6} {'Model':<18} {'#Rank1':>8} {'#Rank2':>8} "
          f"{'#Rank3':>8} {'#Top3%':>10} {'#Bot2':>8} {'#Bot2%':>10}")
    print("-" * 80)

    # Compute lock-in index: how concentrated are top ranks?
    top3_concentration = 0
    for jid in premium_jids:
        rc = rank_counts[jid]
        rank1 = rc.get(0, 0)
        rank2 = rc.get(1, 0)
        rank3 = rc.get(2, 0)
        top3 = rank1 + rank2 + rank3
        top3_pct = top3 / n_epochs * 100
        bot2 = rc.get(6, 0) + rc.get(7, 0)  # ranks 6-7 (bottom 2)
        bot2_pct = bot2 / n_epochs * 100
        job = result.jobs[jid]
        print(f"{jid:<6} {job.model:<18} {rank1:>8} {rank2:>8} "
              f"{rank3:>8} {top3_pct:>9.1f}% {bot2:>8} {bot2_pct:>9.1f}%")

        if top3_pct > 50:
            top3_concentration += 1

    # --- 锁入判定 ---
    print()
    print("=" * 80)
    print("Judgment")
    print("=" * 80)

    # Criteria for lock-in:
    # 1. Top-3 rank concentration: if any job has >60% of epochs in top-3 → lock-in
    # 2. Mean share std: if any job's std > 0.05 of the spine (0.05*400=20G variance) → suspicious
    # 3. Bottom-2 persistence: if any job spends >40% of epochs in bottom-2 → starved
    lock_in_jobs = []
    noise_jobs = []
    for jid in premium_jids:
        rc = rank_counts[jid]
        top3_pct = (rc.get(0, 0) + rc.get(1, 0) + rc.get(2, 0)) / n_epochs * 100
        bot2_pct = (rc.get(6, 0) + rc.get(7, 0)) / n_epochs * 100
        if top3_pct > 60:
            lock_in_jobs.append((jid, top3_pct))
        if bot2_pct > 40:
            noise_jobs.append((jid, bot2_pct))

    mean_shares = [job_stats[jid]["mean"] for jid in premium_jids]
    share_spread = max(mean_shares) - min(mean_shares) if mean_shares else 0

    print(f"Share spread (max-min mean): {share_spread:.4f}")
    print(f"Top-3 concentration ratio: {top3_concentration}/8 jobs with >50% top-3 epochs")

    if lock_in_jobs:
        for jid, pct in lock_in_jobs:
            print(f"  LOCK-IN DETECTED: {jid} in top-3 for {pct:.0f}% of epochs")
    if noise_jobs:
        for jid, pct in noise_jobs:
            print(f"  STARVATION: {jid} in bottom-2 for {pct:.0f}% of epochs")

    if lock_in_jobs and share_spread > 0.03:
        print()
        print(">>> VERDICT: LOCK-IN CONFIRMED <<<")
        print(f"  Mechanism: exp(pi*K) weighting at deep infeasibility (400G << 737.5G)")
        print(f"  creates a feedback loop where high-pi jobs receive more bandwidth,")
        print(f"  then lower their pi, while other premium jobs fail to ever catch up.")
        print(f"  Result: {len(lock_in_jobs)} jobs dominate, others starved → D1 P-attn < Fair.")
        verdict = "lock_in"
    elif share_spread < 0.02 and not lock_in_jobs:
        print()
        print(">>> VERDICT: NOISE (uniform distribution) <<<")
        print(f"  Share distribution is uniform; D1 P-attn 37.5% vs Fair 50% is")
        print(f"  likely a single-seed alignment artifact, expected to wash out in 3 seeds.")
        verdict = "noise"
    else:
        print()
        print(">>> VERDICT: MODERATE (ambiguous) <<<")
        print(f"  Some skew but not enough for definitive lock-in call.")
        verdict = "ambiguous"

    # --- Dump time series summary to file ---
    out = {
        "n_epochs": n_epochs,
        "n_iterations": result.total_iterations(),
        "verdict": verdict,
        "share_spread": share_spread,
        "top3_concentration_jobs": top3_concentration,
        "lock_in_jobs": [(jid, pct) for jid, pct in lock_in_jobs],
        "starved_jobs": [(jid, pct) for jid, pct in noise_jobs],
        "job_stats": {jid: {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in js.items()}
                      for jid, js in job_stats.items()},
        # Sample time series (every 10th epoch to keep size manageable)
        "sample_epochs": epochs[::10],
        "sample_shares": {jid: share_series[jid][::10] for jid in premium_jids},
        "sample_pi": {jid: pi_series[jid][::10] for jid in premium_jids},
    }

    with open(f"{OUT_DIR}/trajectory_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFull trajectory saved to {OUT_DIR}/trajectory_summary.json")
    return verdict


def main():
    print("D1 @400G E1 Trajectory Analysis")
    print("=" * 60)
    trace_rows, result = run_d1_trace()
    if not trace_rows:
        return
    verdict = analyze_trajectory(trace_rows, result)
    return verdict


if __name__ == "__main__":
    main()
