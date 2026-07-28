"""
exp_d1g_trajectory: D1G (LongLiuDWRRGap) 轨迹快检实验

目的：验证 D1G 的 gap 比例权重机制是否将 P1/P2 排在 S1/S2 前面。
预登记：
  - P1/P2 应持高 gap → 高权类 (P4-P6)
  - S1/S2 gap 应显著小于 P1/P2 → 低权类 (P0-P2)
  - gap→0 后权重回 floor_w → 不病态振荡

配置：FatTree k=4, spine=800G (1.21× load), seed=0
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy import LongLiuDWRRGap
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V1_WORKLOAD
from longliu_sim.utils import compute_sas_eval, compute_iter_solo_ms


HOST_BW_GBPS = 100.0


def analyze_trace(trace_file: str, out_dir: str):
    """分析 JSONL trace 文件并生成 trajectory_summary.md。"""
    rows = []
    with open(trace_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("  ⚠ trace 文件为空，跳过分析。")
        return

    link_bw_gbps = rows[0].get("link_bw_gbps", 800.0)
    G0 = rows[0].get("G0_gbps", 25.0)
    floor_w = rows[0].get("floor_w", 2.0)

    # 收集所有 jid
    jids = set()
    SUFFIX = "_demand_gbps"
    for row in rows:
        for key in row:
            if key.endswith(SUFFIX):
                jids.add(key[:-len(SUFFIX)])

    jids = sorted(jids)

    # ---- 逐 epoch 聚合 ----
    job_demand = defaultdict(list)
    job_gap = defaultdict(list)
    job_weight = defaultdict(list)
    job_level = defaultdict(list)
    job_bw = defaultdict(list)

    for row in rows:
        for jid in jids:
            d_key = f"{jid}_demand_gbps"
            g_key = f"{jid}_gap_gbps"
            w_key = f"{jid}_weight"
            l_key = f"{jid}_level"
            b_key = f"{jid}_bw_gbps"
            if d_key in row:
                job_demand[jid].append(row[d_key])
            if g_key in row:
                job_gap[jid].append(row[g_key])
            if w_key in row:
                job_weight[jid].append(row[w_key])
            if l_key in row:
                job_level[jid].append(row[l_key])
            if b_key in row:
                job_bw[jid].append(row[b_key])

    # ---- 构建 summary.md ----
    lines = []
    lines.append("# D1G Trajectory Quick Check (feas_boundary_v1 @ 1.21× load)")
    lines.append("")
    lines.append(f"**Trace file**: `{trace_file}`")
    lines.append(f"**Total epochs**: {len(rows)}")
    lines.append(f"**Spine BW**: {link_bw_gbps:.0f} Gbps")
    lines.append(f"**G0**: {G0} Gbps")
    lines.append(f"**floor_w**: {floor_w}")
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append("")

    # ---- Workload ----
    lines.append("## Workload (FEAS_BOUNDARY_V1_WORKLOAD, 7 jobs)")
    lines.append("")
    lines.append("| jid | model | dp | ci | tier |")
    lines.append("|-----|-------|----|-----|------|")
    job_map = {}
    for model, dp, ci in FEAS_BOUNDARY_V1_WORKLOAD:
        pass  # just for reference
    lines.append("")

    # ---- 预登记判定 ----
    lines.append("## 预登记判定")
    lines.append("")
    lines.append(f"1. **P1/P2 持高 gap 进高权类 (P4-P6)**: P1 avg level ≥4? P2 avg level ≥4?")
    lines.append(f"2. **S1/S2 gap 显著小于 P1/P2**: S1 avg gap < P1 avg gap? S2 avg gap < P1 avg gap?")
    lines.append(f"3. **gap→0 后权重回地板 (floor_w={floor_w})**: 是否有 epoch 中 gap=0 的 job 权重 = {floor_w}？")
    lines.append("")

    # ---- Aggregate Statistics ----
    lines.append("## Aggregate Statistics (Full Trace)")
    lines.append("")
    lines.append("| jid | Avg Demand (Gbps) | Avg Gap (Gbps) | Avg Weight | Avg Level | Avg BW (Gbps) | % epochs gap>0 |")
    lines.append("|-----|-------------------|---------------|------------|-----------|---------------|----------------|")

    for jid in jids:
        avg_demand = sum(job_demand[jid]) / len(job_demand[jid]) if job_demand[jid] else 0.0
        avg_gap = sum(job_gap[jid]) / len(job_gap[jid]) if job_gap[jid] else 0.0
        avg_weight = sum(job_weight[jid]) / len(job_weight[jid]) if job_weight[jid] else 0.0
        avg_level = sum(job_level[jid]) / len(job_level[jid]) if job_level[jid] else 0.0
        avg_bw = sum(job_bw[jid]) / len(job_bw[jid]) if job_bw[jid] else 0.0
        pct_gap_positive = sum(1 for g in job_gap[jid] if g > 0) / len(job_gap[jid]) * 100 if job_gap[jid] else 0.0

        lines.append(
            f"| {jid} | {avg_demand:.1f} | {avg_gap:.3f} | {avg_weight:.3f} | "
            f"{avg_level:.1f} | {avg_bw:.1f} | {pct_gap_positive:.1f}% |"
        )

    lines.append("")

    # ---- Sampled trajectory ----
    sample_step = 50
    sample_indices = list(range(0, len(rows), sample_step))

    lines.append("## Sampled Trajectory (every 50th epoch)")
    lines.append("")

    for idx in sample_indices:
        row = rows[idx]
        epoch = row["epoch"]
        time_ms = row["time_ms"]
        lines.append(f"### Epoch {epoch} @ t={time_ms:.0f}ms")
        lines.append("")
        lines.append("| jid | Demand (Gbps) | Gap (Gbps) | Weight | Level | BW (Gbps) | Share |")
        lines.append("|-----|--------------|------------|--------|-------|-----------|-------|")
        for jid in jids:
            demand = row.get(f"{jid}_demand_gbps", "N/A")
            gap = row.get(f"{jid}_gap_gbps", "N/A")
            weight = row.get(f"{jid}_weight", "N/A")
            level = row.get(f"{jid}_level", "N/A")
            bw = row.get(f"{jid}_bw_gbps", "N/A")
            share = row.get(f"{jid}_share", "N/A")

            d_str = f"{demand:.1f}" if isinstance(demand, (int, float)) else str(demand)
            g_str = f"{gap:.3f}" if isinstance(gap, (int, float)) else str(gap)
            w_str = f"{weight:.3f}" if isinstance(weight, (int, float)) else str(weight)
            l_str = str(level) if isinstance(level, (int, float)) else str(level)
            b_str = f"{bw:.1f}" if isinstance(bw, (int, float)) else str(bw)
            s_str = f"{share:.4f}" if isinstance(share, (int, float)) else str(share)

            lines.append(f"| {jid} | {d_str} | {g_str} | {w_str} | {l_str} | {b_str} | {s_str} |")
        lines.append("")

    # ---- Key Findings ----
    lines.append("## 关键诊断")
    lines.append("")

    # jid 映射: J3=13B-p (P1), J2=7B-p (P2)
    p1_gap_avg = sum(job_gap.get("J3", [])) / len(job_gap.get("J3", [1])) if job_gap.get("J3") else 0.0
    p2_gap_avg = sum(job_gap.get("J2", [])) / len(job_gap.get("J2", [1])) if job_gap.get("J2") else 0.0
    s1_gap_avg = sum(job_gap.get("J5", [])) / len(job_gap.get("J5", [1])) if job_gap.get("J5") else 0.0
    s2_gap_avg = sum(job_gap.get("J0", [])) / len(job_gap.get("J0", [1])) if job_gap.get("J0") else 0.0

    lines.append(f"**gap 排序验证**（正确映射: J3=P1 13B-p, J2=P2 7B-p, J5=S1 13B-s）:")
    lines.append(f"- P1 (J3, 13B-p): avg gap = {p1_gap_avg:.3f} Gbps")
    lines.append(f"- P2 (J2, 7B-p):  avg gap = {p2_gap_avg:.3f} Gbps")
    lines.append(f"- S1 (J5, 13B-s): avg gap = {s1_gap_avg:.3f} Gbps")
    lines.append(f"- S2 (J0, T5-s):  avg gap = {s2_gap_avg:.3f} Gbps")
    lines.append("")

    # Contested window: 仅看 gap>0 的 epoch（排除早期只有 J0 活跃的阶段）
    contested_p1_levels = [row.get("J3_level", 0) for row in rows if row.get("J3_gap_gbps", 0) > 0]
    contested_p2_levels = [row.get("J2_level", 0) for row in rows if row.get("J2_gap_gbps", 0) > 0]
    contested_s1_levels = [row.get("J5_level", 0) for row in rows if row.get("J5_gap_gbps", 0) > 0]
    contested_p1_avg = sum(contested_p1_levels) / len(contested_p1_levels) if contested_p1_levels else 0
    contested_p2_avg = sum(contested_p2_levels) / len(contested_p2_levels) if contested_p2_levels else 0
    contested_s1_avg = sum(contested_s1_levels) / len(contested_s1_levels) if contested_s1_levels else 0

    p1_lvl_avg = sum(job_level.get("J3", [])) / len(job_level.get("J3", [1])) if job_level.get("J3") else 0.0
    p2_lvl_avg = sum(job_level.get("J2", [])) / len(job_level.get("J2", [1])) if job_level.get("J2") else 0.0
    s1_lvl_avg = sum(job_level.get("J5", [])) / len(job_level.get("J5", [1])) if job_level.get("J5") else 0.0
    s2_lvl_avg = sum(job_level.get("J0", [])) / len(job_level.get("J0", [1])) if job_level.get("J0") else 0.0

    lines.append(f"**类级排序**（contested 窗口/全 trace 平均）:")
    lines.append(f"- P1 (J3): contested={contested_p1_avg:.1f} / full={p1_lvl_avg:.1f} ({len(contested_p1_levels)} contested epochs)")
    lines.append(f"- P2 (J2): contested={contested_p2_avg:.1f} / full={p2_lvl_avg:.1f} ({len(contested_p2_levels)} contested epochs)")
    lines.append(f"- S1 (J5): contested={contested_s1_avg:.1f} / full={s1_lvl_avg:.1f} ({len(contested_s1_levels)} contested epochs)")
    lines.append(f"- S2 (J0): full={s2_lvl_avg:.1f} (gap 始终为 0)")
    lines.append("")

    if contested_p1_avg >= 5 and contested_p2_avg >= 5:
        verdict = "✅ PASS (P1/P2 contested 窗口均在 P5-P6)"
    elif contested_p1_avg >= 4 and contested_p2_avg >= 4:
        verdict = "✅ PASS (P1/P2 contested 窗口均在 P4-P6)"
    else:
        verdict = "❌ FAIL (P1/P2 contested 窗口未进入高权类)"
    lines.append(f"**判定**: {verdict}")
    lines.append("")

    # ---- Floor weight check ----
    lines.append("**地板权重 (floor_w) 验证**:")
    weighed_samples = []
    for row in rows:
        for jid in jids:
            g_key = f"{jid}_gap_gbps"
            w_key = f"{jid}_weight"
            if g_key in row and w_key in row:
                if row[g_key] <= 0:
                    weighed_samples.append((jid, row[w_key]))
    min_weights = defaultdict(set)
    for jid, w in weighed_samples:
        min_weights[jid].add(w)
    for jid in jids:
        if jid in min_weights:
            vals = min_weights[jid]
            good = all(v == floor_w for v in vals)
            lines.append(f"- {jid}: gap=0 时权重 = {vals} → {'✅ floor_w={floor_w} 守住' if good else '❌ 偏离 floor_w'}")
    lines.append("")

    # 生成分析报告
    out_path = os.path.join(out_dir, "trajectory_summary.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Trajectory summary saved to {out_path}")

    # Print quick verdict to stdout
    print(f"\n  === Quick Check Verdict ===")
    print(f"  P1 avg level: {p1_lvl_avg:.1f}, P2 avg level: {p2_lvl_avg:.1f}")
    print(f"  S1 avg level: {s1_lvl_avg:.1f}, S2 avg level: {s2_lvl_avg:.1f}")
    print(f"  {verdict}")


def main():
    out_dir = "outputs/trajectory_d1g_121x"
    trace_file = "outputs/trajectory_d1g_121x.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("D1G Trajectory Quick Check: feas_boundary_v1 @ 1.21× load")
    print("=" * 80)
    print(f"Policy: LongLiuDWRRGap(floor_w=2.0, G0=25 Gbps)")
    print(f"Topology: FatTree(k=4, spine=800G)")
    print(f"Trace: {trace_file}")
    print(f"Output: {out_dir}")
    print()

    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 800e9,
        },
        "duration_ms": 600000,
        "overlap_factor": 0.85,
    }

    # ---- Policy (D1G) ----
    policy = LongLiuDWRRGap(
        floor_w=2.0,
        G0_gbps=25.0,
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
        overhead_factor=2.0,
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
        overhead_factor=2.0,
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

    # ---- Flush trace ----
    policy.flush_trace()
    print(f"\nTrace flushed to {trace_file}")

    # ---- Analyze ----
    analyze_trace(trace_file, out_dir)


if __name__ == "__main__":
    main()
