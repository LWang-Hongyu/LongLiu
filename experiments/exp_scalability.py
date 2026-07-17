"""
复现论文 Table 4: SLO attainment at 128 nodes (128 concurrent jobs).

与 Table 3 结构相同，但拓扑为 128 节点 Fat-Tree。
用法：
    python experiments/exp_scalability.py --seeds 10
    python experiments/exp_scalability.py --config configs/fatree_128host.yaml --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp_ablation import (
    load_config, build_topology, generate_jobs,
    run_experiment, CI_TIERS, DEFAULT_OVERHEAD,
)
from longliu_sim.policy import Fair, SRPT, CRUX, LongLiu


def main():
    parser = argparse.ArgumentParser(description="Table 4: SLO attainment scalability")
    parser.add_argument("--config", default="configs/fatree_128host.yaml")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", default="outputs/table4")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    cfg = load_config(cfg_path)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "outputs"), exist_ok=True)

    POLICIES = {
        "Fair": Fair(),
        "SRPT": SRPT(),
        "CRUX": CRUX(alpha=1.0),
        "LongLiu": LongLiu(K=2.0, use_dynamic_T_target=True),
    }

    # ── 运行所有策略 ──
    results: dict[str, list[dict]] = {name: [] for name in POLICIES}
    for name, policy in POLICIES.items():
        print(f"  Running {name} ...")
        for seed in range(args.seeds):
            r = run_experiment(cfg, policy, seed, DEFAULT_OVERHEAD)
            results[name].append(r)

    # ── 汇总 ──
    metrics = ["total_iters"] + [f"slo_attainment_ci{ci}" for ci in CI_TIERS] + ["slo_attainment_overall"]

    summary: dict[str, dict] = {}
    for name in POLICIES:
        summary[name] = {}
        for m in metrics:
            vals = [r[m] for r in results[name]]
            summary[name][m] = sum(vals) / len(vals)

    # ── 输出 CSV ──
    out_dir = os.path.join(os.path.dirname(__file__), "..", args.output)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "table4.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Policy"] + metrics
        w.writerow(header)
        for name in POLICIES:
            row = [name] + [f"{summary[name][m]:.4f}" for m in metrics]
            w.writerow(row)
    print(f"  CSV → {csv_path}")

    # ── 输出 LaTeX 表格 ──
    tex_path = os.path.join(out_dir, "table4.tex")
    with open(tex_path, "w") as f:
        f.write("% Table 4: SLO Attainment at 128 Nodes (128 concurrent jobs)\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{SLO attainment at 128 nodes (128 concurrent jobs)}\n")
        f.write("\\label{tab:slo_128host}\n")
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

    # ── 打印摘要 ──
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