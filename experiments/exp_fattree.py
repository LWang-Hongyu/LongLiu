"""
论文 Table 5: TwoTier Spine-TOR 拓扑验证实验。

与 Table 3（单瓶颈链路）对比，验证 LongLiu 在分层拓扑下的泛化能力。

拓扑：8 racks × 16 hosts = 128 hosts
- 同 rack 内流量：rack link 640G（无竞争）
- 跨 rack 流量：spine link 1680G（竞争瓶颈）
- Worker 随机放置，不感知 rack

用法：
    python experiments/exp_fattree.py --seeds 2   # 快速验证
    python experiments/exp_fattree.py --seeds 10  # 完整实验
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from longliu_sim.network import TwoTierTopology
from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu
from longliu_sim.core import Simulator
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic_128 import TABLE4_TIERED_WORKLOAD_128

CI_TIERS = [1.5, 2.0, 3.0]
DEFAULT_OVERHEAD = 2.0

# TwoTier 拓扑默认参数
NUM_HOSTS = 128
HOSTS_PER_RACK = 16
HOST_BW_BPS = 40e9       # 40 Gbps
SPINE_BW_BPS = 1680e9     # 1680 Gbps（与 Table 4 单链路等压力）
DURATION_MS = 600_000     # 600s


def run_experiment(policy, seed: int, overhead_factor: float, duration_ms: float,
                   overlap_factor: float = 1.0) -> dict:
    """运行一次仿真。"""
    topo = TwoTierTopology(
        num_hosts=NUM_HOSTS,
        hosts_per_rack=HOSTS_PER_RACK,
        host_bw_bps=HOST_BW_BPS,
        spine_bw_bps=SPINE_BW_BPS,
    )

    loader = SyntheticTraceLoader(
        job_count=len(TABLE4_TIERED_WORKLOAD_128),
        duration_ms=duration_ms,
        seed=seed,
        overhead_factor=overhead_factor,
        target_bw_bps=HOST_BW_BPS,
        workload_profile=TABLE4_TIERED_WORKLOAD_128,
        num_hosts=NUM_HOSTS,  # 启用 worker placement
    )
    jobs = loader.load()

    sim = Simulator(topo, policy, duration_ms=duration_ms,
                    seed=seed, overhead_factor=overhead_factor,
                    overlap_factor=overlap_factor)
    for j in jobs:
        sim.submit(j)
    result = sim.run()
    stats = result.per_job_stats()

    tier_meets: dict[float, list[bool]] = {c: [] for c in CI_TIERS}
    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        tier_meets[ci].append(s["meets_slo"])

    out = {"total_iters": result.total_iterations()}
    for ci in CI_TIERS:
        if tier_meets[ci]:
            attainment = sum(tier_meets[ci]) / len(tier_meets[ci])
        else:
            attainment = 0.0
        out[f"slo_attainment_ci{ci}"] = attainment

    all_ok = sum(1 for s in stats.values() if s["meets_slo"])
    out["slo_attainment_overall"] = all_ok / len(stats) if stats else 0.0

    return out


def main():
    parser = argparse.ArgumentParser(description="Table 5: TwoTier Spine-TOR experiment")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="outputs/table5")
    parser.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD,
                        help=f"NCCL/PCIe overhead factor (default {DEFAULT_OVERHEAD})")
    parser.add_argument("--duration", type=float, default=DURATION_MS,
                        help=f"Simulation duration in ms (default {DURATION_MS})")
    parser.add_argument("--overlap", type=float, default=1.0,
                        help="Compute-comm overlap factor (0=serial, 1=full overlap, default 1.0)")
    args = parser.parse_args()

    POLICIES = {
        "Fair": Fair(),
        "SRPT": SRPT(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu": LongLiu(K=2.0, use_dynamic_T_target=True),
    }

    print(f"  Topology: TwoTier Spine-TOR ({NUM_HOSTS} hosts, {HOSTS_PER_RACK}/rack, {NUM_HOSTS // HOSTS_PER_RACK} racks)")
    print(f"  Spine BW: {SPINE_BW_BPS/1e9:.0f}G  |  Rack BW: {HOSTS_PER_RACK * HOST_BW_BPS/1e9:.0f}G")
    print(f"  Jobs: {len(TABLE4_TIERED_WORKLOAD_128)}  |  Seeds: {args.seeds}  |  Overhead: {args.overhead}  |  Overlap: {args.overlap}")

    results: dict[str, list[dict]] = {name: [] for name in POLICIES}
    for name, policy in POLICIES.items():
        print(f"  Running {name} ...")
        for seed in range(args.seeds):
            r = run_experiment(policy, seed, args.overhead, args.duration, args.overlap)
            results[name].append(r)
            print(f"    seed={seed}: total_iters={r['total_iters']}, "
                  f"tight={r['slo_attainment_ci1.5']*100:.1f}%, "
                  f"med={r['slo_attainment_ci2.0']*100:.1f}%, "
                  f"loose={r['slo_attainment_ci3.0']*100:.1f}%")

    metrics = ["total_iters"] + [f"slo_attainment_ci{ci}" for ci in CI_TIERS] + ["slo_attainment_overall"]

    summary: dict[str, dict] = {}
    for name in POLICIES:
        summary[name] = {}
        for m in metrics:
            vals = [r[m] for r in results[name]]
            summary[name][m] = sum(vals) / len(vals)

    out_dir = os.path.join(os.path.dirname(__file__), "..", args.output)
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "table5.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Policy"] + metrics
        w.writerow(header)
        for name in POLICIES:
            row = [name] + [f"{summary[name][m]:.4f}" for m in metrics]
            w.writerow(row)
    print(f"  CSV → {csv_path}")

    # LaTeX
    tex_path = os.path.join(out_dir, "table5.tex")
    with open(tex_path, "w") as f:
        f.write("% Table 5: TwoTier Spine-TOR SLO Attainment (128 hosts, 128 jobs)\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{SLO attainment on 2-tier Spine-TOR topology (128 hosts, 128 jobs)}\n")
        f.write("\\label{tab:twotier}\n")
        f.write("\\begin{tabular}{l|r|r|r|r|r}\n")
        f.write("\\hline\n")
        f.write("Policy & Total Iters ($\\times 10^4$) & Tight & Medium & Loose & Overall \\\\\n")
        f.write("\\hline\n")
        for name in POLICIES:
            s = summary[name]
            total_k = s["total_iters"] / 10000
            tight = s["slo_attainment_ci1.5"] * 100
            medium = s["slo_attainment_ci2.0"] * 100
            loose = s["slo_attainment_ci3.0"] * 100
            overall = s["slo_attainment_overall"] * 100
            f.write(f"{name} & {total_k:.2f} & {tight:.1f}\\% & {medium:.1f}\\% & {loose:.1f}\\% & {overall:.1f}\\% \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  LaTeX → {tex_path}")

    print("\n  Summary:")
    print(f"  {'Policy':<12} {'Total(K)':<10} {'Tight%':<8} {'Medium%':<8} {'Loose%':<8} {'Overall%':<8}")
    for name in POLICIES:
        s = summary[name]
        print(f"  {name:<12} {s['total_iters']/10000:<10.2f} "
              f"{s['slo_attainment_ci1.5']*100:<8.1f} "
              f"{s['slo_attainment_ci2.0']*100:<8.1f} "
              f"{s['slo_attainment_ci3.0']*100:<8.1f} "
              f"{s['slo_attainment_overall']*100:<8.1f}")


if __name__ == "__main__":
    main()
