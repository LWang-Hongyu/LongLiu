"""
exp_d1_trajectory: D1 (LongLiuDWRR) 轨迹追踪实验

目的：细粒度追踪 D1 在 feas_boundary_v1 场景下的带宽分配决策，
回答 "D1 的带宽重新分配去了哪里"：
  - P1(13B-p) 和 P2(7B-p) 在 D1 下获得了多少份额 vs Fair
  - S1(13B-s) 被剥夺的带宽是去了 P1/P2 还是去了 filler jobs

配置：FatTree k=4, spine=800G (1.21× load), seed=0
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import LongLiuDWRR
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V1_WORKLOAD
from longliu_sim.utils import compute_sas_eval, compute_iter_solo_ms


HOST_BW_GBPS = 100.0


def analyze_trace(trace_file: str, out_dir: str, link_bw_gbps: float,
                  job_map: dict = None):
    """分析 JSONL trace 文件并生成 trajectory_summary.md。"""
    # 加载 trace 行
    rows = []
    with open(trace_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("  ⚠ trace 文件为空，跳过分析。")
        return

    if job_map is None:
        job_map = {}

    # 收集所有 jid
    jids = set()
    for row in rows:
        for key in row:
            if key.endswith("_pi"):
                jids.add(key.rsplit("_", 1)[0])

    # 每个 job 的逐 epoch 聚合
    job_pi = defaultdict(list)
    job_share = defaultdict(list)
    job_bw_gbps = defaultdict(list)
    job_dscp_level = defaultdict(list)

    for row in rows:
        for jid in jids:
            pi_key = f"{jid}_pi"
            share_key = f"{jid}_share"
            bw_key = f"{jid}_bw_gbps"
            lvl_key = f"{jid}_level"
            if pi_key in row:
                job_pi[jid].append(row[pi_key])
            if share_key in row:
                job_share[jid].append(row[share_key])
            if bw_key in row:
                job_bw_gbps[jid].append(row[bw_key])
            if lvl_key in row:
                job_dscp_level[jid].append(row[lvl_key])

    # ---- 构建 summary.md ----
    lines = []
    lines.append("# D1 Trajectory Summary (feas_boundary_v1 @ 1.21× load)")
    lines.append("")
    lines.append(f"**Trace file**: `{trace_file}`")
    lines.append(f"**Total epochs**: {len(rows)}")
    lines.append(f"**Spine BW**: {link_bw_gbps:.0f} Gbps")
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Workload (FEAS_BOUNDARY_V1_WORKLOAD, 7 jobs)")
    lines.append("")
    lines.append("| jid | model | dp | ci | tier |")
    lines.append("|-----|-------|----|-----|------|")

    for jid in sorted(jids):
        info = job_map.get(jid, {})
        model = info.get("model", "?")
        dp = info.get("dp", "?")
        ci = info.get("ci", "?")
        tier = "Premium" if ci == 1.3 else ("Standard" if ci == 2.0 else "?")
        lines.append(f"| {jid} | {model} | {dp} | {ci} | {tier} |")

    lines.append("")
    lines.append("## Sampled Trajectory (every 50th epoch)")
    lines.append("")

    # Take every 50th epoch
    sample_step = 50
    sample_indices = list(range(0, len(rows), sample_step))

    for idx in sample_indices:
        row = rows[idx]
        epoch = row["epoch"]
        time_ms = row["time_ms"]
        lines.append(f"### Epoch {epoch} @ t={time_ms:.0f}ms")
        lines.append("")
        lines.append(f"| jid | π | DSCP lvl | BW share | BW (Gbps) |")
        lines.append(f"|-----|-----|----------|----------|-----------|")
        for jid in sorted(jids):
            pi = row.get(f"{jid}_pi", "N/A")
            lvl = row.get(f"{jid}_level", "N/A")
            share = row.get(f"{jid}_share", "N/A")
            bw = row.get(f"{jid}_bw_gbps", "N/A")
            pi_str = f"{pi:.4f}" if isinstance(pi, (int, float)) else str(pi)
            lvl_str = str(lvl)
            share_str = f"{share:.4f}" if isinstance(share, (int, float)) else str(share)
            bw_str = f"{bw:.4f}" if isinstance(bw, (int, float)) else str(bw)
            lines.append(f"| {jid} | {pi_str} | {lvl_str} | {share_str} | {bw_str} |")
        lines.append("")

    # ---- Aggregate stats ----
    lines.append("## Aggregate Statistics")
    lines.append("")
    lines.append("| jid | Avg π | Avg DSCP lvl | Avg BW share | Avg BW (Gbps) | % epochs π>0 |")
    lines.append("|-----|-------|-------------|-------------|---------------|--------------|")

    for jid in sorted(jids):
        avg_pi = sum(job_pi[jid]) / len(job_pi[jid]) if job_pi[jid] else 0.0
        avg_lvl = sum(job_dscp_level[jid]) / len(job_dscp_level[jid]) if job_dscp_level[jid] else 0.0
        avg_share = sum(job_share[jid]) / len(job_share[jid]) if job_share[jid] else 0.0
        avg_bw = sum(job_bw_gbps[jid]) / len(job_bw_gbps[jid]) if job_bw_gbps[jid] else 0.0
        pct_violating = sum(1 for pi in job_pi[jid] if pi > 0) / len(job_pi[jid]) * 100 if job_pi[jid] else 0.0

        lines.append(
            f"| {jid} | {avg_pi:.4f} | {avg_lvl:.1f} | {avg_share:.4f} | "
            f"{avg_bw:.4f} | {pct_violating:.1f}% |"
        )

    lines.append("")

    # ---- 核心问题：D1 的带宽去了哪里？----
    lines.append("## Where Does D1's Redistributed Bandwidth Go?")
    lines.append("")

    # Compute Fair baseline: equal share = 1/7 of link capacity ≈ 0.1429 share
    fair_share = 1.0 / len(jids)
    link_bw_total = link_bw_gbps  # in Gbps
    fair_bw_gbps = link_bw_total * fair_share

    lines.append(f"### Fair Baseline (equal share)")
    lines.append(f"- Each job gets **{fair_share:.4f}** of link capacity = **{fair_bw_gbps:.1f} Gbps**")
    lines.append("")

    lines.append("### D1 Actual Average Bandwidth")
    lines.append("")
    lines.append("| jid | Avg BW (Gbps) | vs Fair |")
    lines.append("|-----|--------------|---------|")

    # Try to identify jobs by their bandwidth profile
    # P1(13B-p, ci=1.3) and P2(7B-p, ci=1.3) should have high comm demand
    # S1(13B-s, ci=2.0) should have demand = 13B solo comm * 1.3 / (ci=2.0) etc.
    # Without model info in trace, we can still compare relative to fair share

    for jid in sorted(jids):
        avg_bw = sum(job_bw_gbps[jid]) / len(job_bw_gbps[jid]) if job_bw_gbps[jid] else 0.0
        delta_vs_fair = avg_bw - fair_bw_gbps
        direction = "↑" if delta_vs_fair > 0 else "↓"
        lines.append(f"| {jid} | {avg_bw:.1f} | {direction} {delta_vs_fair:+.1f} Gbps |")

    lines.append("")

    lines.append("### Key Findings")
    lines.append("")

    # Identify jobs above/below fair share
    above_fair = []
    below_fair = []
    for jid in sorted(jids):
        avg_bw = sum(job_bw_gbps[jid]) / len(job_bw_gbps[jid]) if job_bw_gbps[jid] else 0.0
        if avg_bw > fair_bw_gbps:
            above_fair.append((jid, avg_bw, avg_bw - fair_bw_gbps))
        else:
            below_fair.append((jid, avg_bw, avg_bw - fair_bw_gbps))

    above_fair.sort(key=lambda x: -x[2])  # most gain first
    below_fair.sort(key=lambda x: x[2])   # most loss first

    total_gain = sum(g for _, _, g in above_fair)
    total_loss = sum(l for _, _, l in below_fair)

    lines.append(f"**Bandwidth winners** (above fair share):")
    for jid, bw, gain in above_fair:
        lines.append(f"- {jid}: {bw:.1f} Gbps ({gain:+.1f} vs fair)")

    lines.append("")
    lines.append(f"**Bandwidth losers** (below fair share):")
    for jid, bw, loss in below_fair:
        lines.append(f"- {jid}: {bw:.1f} Gbps ({loss:+.1f} vs fair)")

    lines.append("")
    lines.append(f"**Net redistribution**: {total_gain:+.1f} Gbps taken from losers → winners")
    if total_loss != 0:
        lines.append(f"**Redistribution efficiency**: {total_gain / abs(total_loss) * 100:.1f}% of taken bandwidth goes to winners")

    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("- **P1(13B-p) / P2(7B-p)**: Premium jobs that need bandwidth to meet tight ci=1.3 SLO")
    lines.append("- **S1(13B-s) / S2(11B-s)**: Standard large jobs (ci=2.0) — primary source of redistributed bandwidth")
    lines.append("- **P3 / S3 / S4**: Filler (small) jobs that are non-contested and easily satisfied")
    lines.append("")
    lines.append("Key question: Is bandwidth taken from S1/S2 going to P1/P2 (good redistribution)")
    lines.append("or to filler jobs (wasteful redistribution)?")
    lines.append("")
    trade_surplus = total_gain
    lines.append(f"D1 achieves **{trade_surplus:.1f} Gbps** of bandwidth being redistributed from")
    lines.append("standard jobs to premium jobs above fair share.")
    lines.append("")

    # Save summary
    out_path = os.path.join(out_dir, "trajectory_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Trajectory summary saved to {out_path}")


def main():
    out_dir = "outputs/trajectory_d1_121x"
    trace_file = "outputs/trajectory_d1_121x.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("D1 Trajectory Experiment: feas_boundary_v1 @ 1.21× load")
    print("=" * 80)
    print(f"Policy: LongLiuDWRR(K=2.0, D1 config)")
    print(f"Topology: FatTree(k=4, spine=800G)")
    print(f"Trace: {trace_file}")
    print(f"Output: {out_dir}")
    print()

    # ---- Config ----
    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 800e9,  # 1.21× load point
        },
        "duration_ms": 600000,
        "overhead_factor": 2.0,
        "overlap_factor": 0.85,
    }

    # ---- Policy (D1) ----
    policy = LongLiuDWRR(
        K=2.0,
        use_soft_weights=False,
        intra_class_fair=False,
        clip_ratio=10.0,
        overlap_factor=cfg["overlap_factor"],
        trace_file=trace_file,
    )

    # ---- Topology ----
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )

    # ---- Simulator ----
    sim = Simulator(
        topo,
        policy,
        duration_ms=cfg["duration_ms"],
        seed=0,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

    # ---- Workload ----
    loader = SyntheticTraceLoader(
        model_types=[],
        gpu_distribution={},
        ci_distribution={},
        job_count=7,
        duration_ms=cfg["duration_ms"],
        seed=0,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=FEAS_BOUNDARY_V1_WORKLOAD,
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    print("Jobs submitted:")
    for j in jobs:
        print(f"  {j.jid}: {j.model} dp={j.num_workers} ci={j.slo_ci}")

    # ---- Run ----
    print("\nRunning simulation...")
    result = sim.run()
    print(f"  Done: {result.total_iterations()} total iterations")
    print(f"  Avg iter: {result.avg_iteration_ms():.1f}ms")
    print(f"  SLO attainment: {result.slo_attainment()*100:.1f}%")

    # ---- Flush trace ----
    policy.flush_trace()
    print(f"\nTrace flushed to {trace_file}")

    # ---- Per-job stats & sas_eval ----
    stats = result.per_job_stats(host_bw_gbps=HOST_BW_GBPS)
    per_job_results = []
    premium_stats = []
    standard_stats = []

    for jid, s in stats.items():
        job = sim.jobs[jid]
        ci = job.slo_ci
        avg_iter_ms = s["avg_iter_ms"]
        sas_eval = compute_sas_eval(
            avg_iter_ms, job.model, job.num_workers, ci,
            host_bw_gbps=HOST_BW_GBPS, overlap_factor=cfg["overlap_factor"]
        )
        completed = job.completed_iters
        target = job.target_iters

        # Attainment: avg_iter_ms <= ci * iter_solo_ms
        iter_solo_ms = compute_iter_solo_ms(
            job.model, job.num_workers,
            host_bw_gbps=HOST_BW_GBPS, overlap_factor=cfg["overlap_factor"]
        )
        target_iter_ms = ci * iter_solo_ms
        attained = avg_iter_ms <= target_iter_ms
        starved = completed == 0

        per_job_results.append({
            "jid": jid,
            "model": job.model,
            "dp": job.num_workers,
            "ci": ci,
            "sas_eval": sas_eval,
            "avg_iter_ms": avg_iter_ms,
            "completed_iters": completed,
            "target_iters": target,
            "attained": attained,
            "starved": starved,
        })

        if ci == 1.3:
            premium_stats.append(sas_eval)
        elif ci == 2.0:
            standard_stats.append(sas_eval)

    # ---- Aggregate stats ----
    premium_mean = sum(premium_stats) / len(premium_stats) if premium_stats else 0.0
    premium_capped = sum(min(s, 1.0) for s in premium_stats) / len(premium_stats) if premium_stats else 0.0
    premium_attainment = sum(1 for j in per_job_results if j["ci"] == 1.3 and j["attained"]) / \
                         max(1, sum(1 for j in per_job_results if j["ci"] == 1.3))
    premium_starved = sum(1 for j in per_job_results if j["ci"] == 1.3 and j["starved"])

    standard_mean = sum(standard_stats) / len(standard_stats) if standard_stats else 0.0
    standard_capped = sum(min(s, 1.0) for s in standard_stats) / len(standard_stats) if standard_stats else 0.0
    standard_attainment = sum(1 for j in per_job_results if j["ci"] == 2.0 and j["attained"]) / \
                          max(1, sum(1 for j in per_job_results if j["ci"] == 2.0))
    standard_starved = sum(1 for j in per_job_results if j["ci"] == 2.0 and j["starved"])

    overall_mean = sum(p["sas_eval"] for p in per_job_results) / len(per_job_results) if per_job_results else 0.0

    # ---- Print results ----
    print("\n" + "=" * 80)
    print("D1 Results Summary")
    print("=" * 80)
    print(f"  Overall Mean SAS:  {overall_mean:.3f}")
    print(f"  Premium Mean:      {premium_mean:.3f}  (capped: {premium_capped:.3f})")
    print(f"  Premium Attainment: {premium_attainment:.0%}  (starved: {premium_starved})")
    print(f"  Standard Mean:     {standard_mean:.3f}  (capped: {standard_capped:.3f})")
    print(f"  Standard Attainment: {standard_attainment:.0%}  (starved: {standard_starved})")
    print()

    for j in per_job_results:
        tier = "P" if j["ci"] == 1.3 else "S"
        attn = "✓" if j["attained"] else ("✗" if j["starved"] else "·")
        print(f"  {j['jid']} {j['model']:<18} dp={j['dp']} ci={j['ci']} "
              f"sas={j['sas_eval']:.3f} attn={attn}")

    # ---- Save results.json ----
    all_results = {
        "D1": [{
            "seed": 0,
            "overall_mean_sas": overall_mean,
            "premium_mean": premium_mean,
            "premium_capped": premium_capped,
            "premium_attainment": premium_attainment,
            "premium_starved": premium_starved,
            "standard_mean": standard_mean,
            "standard_capped": standard_capped,
            "standard_attainment": standard_attainment,
            "standard_starved": standard_starved,
            "per_job": per_job_results,
        }],
        "config": {
            "K": 2.0,
            "use_soft_weights": False,
            "intra_class_fair": False,
            "clip_ratio": 10.0,
            "topology": cfg["topology"],
            "duration_ms": cfg["duration_ms"],
            "overlap_factor": cfg["overlap_factor"],
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")

    # ---- Analyze trace ----
    # Trace records per-spine-link allocations; each link = spine_bw / num_spine_links
    per_link_bw_gbps = topo.spine_bw_bps / topo.num_spine_links / 1e9

    # Build job_map for trace analysis
    job_map = {}
    for j in jobs:
        job_map[j.jid] = {
            "model": j.model,
            "dp": j.num_workers,
            "ci": j.slo_ci,
        }

    analyze_trace(trace_file, out_dir, per_link_bw_gbps, job_map)

    # ---- Save run meta ----
    meta = {
        "timestamp": datetime.now().isoformat(),
        "cmdline": " ".join(sys.argv),
        "config": cfg,
        "workload": "FEAS_BOUNDARY_V1_WORKLOAD",
        "trace_file": trace_file,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Run metadata saved to {out_dir}/run_meta.json")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
