#!/usr/bin/env python3
"""D1 重生主表分析：三负载点逐 job sas + 战斗场指标 + per-seed 配对差。

输入：outputs/feas_boundary_v1_{1000g,800g,630g}_3seeds/results.json
输出：outputs/feas_boundary_v1_rebirth_summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOAD_POINTS = {
    "1000G": {
        "label": "1.02× (spine=1000G, ci=1.3/2.0)",
        "path": PROJECT_ROOT / "outputs/feas_boundary_v1_1000g_3seeds/results.json",
    },
    "800G": {
        "label": "1.21× (spine=800G, ci=1.3/2.0)",
        "path": PROJECT_ROOT / "outputs/feas_boundary_v1_800g_3seeds/results.json",
    },
    "630G": {
        "label": "1.54× (spine=630G, ci=1.3/2.0)",
        "path": PROJECT_ROOT / "outputs/feas_boundary_v1_630g_3seeds/results.json",
    },
}

POLICIES = ["Fair", "CRUX", "LongLiu-SP", "D1"]

CONTESTED_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}


def classify_job(model: str, ci: float) -> str | None:
    """Return column key like 'P1', 'P2', 'S1', etc."""
    if ci == 1.3:
        if model == "LLaMA-2-13B":
            return "P1"
        if model == "LLaMA-2-7B":
            return "P2"
        if model.startswith("BERT"):
            return "P3"
    elif ci == 2.0:
        if model == "LLaMA-2-13B":
            return "S1"
        if model == "T5-11B-fp16":
            return "S2"
        if model.startswith("BERT"):
            return "S3"
        if model.startswith("ViT"):
            return "S4"
    return None


COLUMN_ORDER = ["P1", "P2", "P3", "S1", "S2", "S3", "S4"]
COLUMN_LABELS = {
    "P1": "P1(13B-p)",
    "P2": "P2(7B-p)",
    "P3": "P3(BERT-p)",
    "S1": "S1(13B-s)",
    "S2": "S2(T5-s)",
    "S3": "S3(BERT-s)",
    "S4": "S4(ViT-s)",
}


def load_all() -> dict:
    data = {}
    for key, lp in LOAD_POINTS.items():
        with open(lp["path"]) as f:
            data[key] = json.load(f)
    return data


def fmt_mean(values):
    if not values:
        return "N/A"
    return f"{mean(values):.3f}"


def fmt_mean_pm_range(values):
    """mean ± half_range."""
    if not values:
        return "N/A"
    m = mean(values)
    rng = max(values) - min(values)
    return f"{m:.3f}±{rng/2:.3f}"


def build_per_job_table(data: dict, lp_key: str) -> dict:
    """table[policy][col] = list of sas_eval across seeds."""
    lp_data = data[lp_key]
    table = defaultdict(lambda: defaultdict(list))
    for policy_name, seeds in lp_data.items():
        for seed_data in seeds:
            for job in seed_data["per_job"]:
                col = classify_job(job["model"], job["ci"])
                if col is None:
                    continue
                table[policy_name][col].append(job["sas_eval"])
    return table


def compute_battlefield(data: dict, lp_key: str) -> dict:
    """Compute contested-only metrics per policy."""
    lp_data = data[lp_key]
    metrics = {}
    for policy_name, seeds in lp_data.items():
        p_contested = []
        s_contested = []
        p_attain = []
        s_attain = []
        for seed_data in seeds:
            jobs = {}
            for job in seed_data["per_job"]:
                col = classify_job(job["model"], job["ci"])
                if col:
                    jobs[col] = job
            p1 = jobs.get("P1")
            p2 = jobs.get("P2")
            if p1 and p2:
                p_contested.append((p1["sas_eval"] + p2["sas_eval"]) / 2.0)
                p_attain.append(p1["attained"] and p2["attained"])
            s1 = jobs.get("S1")
            s2 = jobs.get("S2")
            if s1 and s2:
                s_contested.append((s1["sas_eval"] + s2["sas_eval"]) / 2.0)
                s_attain.append(s1["attained"] and s2["attained"])

        metrics[policy_name] = {
            "p_contested_mean": mean(p_contested) if p_contested else 0.0,
            "p_contested_capped": mean(min(v, 1.0) for v in p_contested) if p_contested else 0.0,
            "p_contested_attainment": mean(1.0 if x else 0.0 for x in p_attain) if p_attain else 0.0,
            "s_contested_mean": mean(s_contested) if s_contested else 0.0,
            "s_contested_capped": mean(min(v, 1.0) for v in s_contested) if s_contested else 0.0,
            "s_contested_attainment": mean(1.0 if x else 0.0 for x in s_attain) if s_attain else 0.0,
            "p_contested_per_seed": p_contested,
            "s_contested_per_seed": s_contested,
        }
    return metrics


def compute_paired_diff(data: dict, lp_key: str, baseline: str, target: str) -> dict:
    """Per-seed paired difference target - baseline for P contested metrics."""
    lp_data = data[lp_key]
    diffs = []
    for seed_idx in range(len(lp_data.get(baseline, []))):
        base_seed = lp_data[baseline][seed_idx]
        tgt_seed = lp_data[target][seed_idx]
        base_jobs = {classify_job(j["model"], j["ci"]): j for j in base_seed["per_job"]}
        tgt_jobs = {classify_job(j["model"], j["ci"]): j for j in tgt_seed["per_job"]}
        p_cols = ["P1", "P2"]
        base_vals = [base_jobs[c]["sas_eval"] for c in p_cols if c in base_jobs]
        tgt_vals = [tgt_jobs[c]["sas_eval"] for c in p_cols if c in tgt_jobs]
        if base_vals and tgt_vals:
            base_mean = mean(base_vals)
            tgt_mean = mean(tgt_vals)
            diffs.append(tgt_mean - base_mean)
    return {
        "diffs": diffs,
        "mean": mean(diffs) if diffs else 0.0,
        "min": min(diffs) if diffs else 0.0,
        "max": max(diffs) if diffs else 0.0,
    }


def main():
    data = load_all()
    lines = []
    lines.append("# D1 重生主表：三负载点汇总")
    lines.append("")
    lines.append("口径：sas_eval = ci × iter_solo / avg_iter_ms；iter_solo = comp + comm_solo × (1 − overlap)；overlap=0.85。")
    lines.append("")

    # 1. Per-job SAS table per load point
    lines.append("## 1. 逐 Job SAS 表（mean ± half_range，3 seeds）")
    for lp_key in ["1000G", "800G", "630G"]:
        lines.append("")
        lines.append(f"### {LOAD_POINTS[lp_key]['label']}")
        lines.append("")
        table = build_per_job_table(data, lp_key)
        header = f"| Policy | {' | '.join(COLUMN_LABELS[c] for c in COLUMN_ORDER)} |"
        lines.append(header)
        lines.append("|" + "|".join(["--------"] + ["--------"] * len(COLUMN_ORDER)) + "|")
        for policy in POLICIES:
            row = [policy]
            for col in COLUMN_ORDER:
                vals = table.get(policy, {}).get(col, [])
                row.append(fmt_mean_pm_range(vals))
            lines.append("| " + " | ".join(row) + " |")

    # 2. Battlefield metrics
    lines.append("")
    lines.append("## 2. 战斗场指标（P1+P2 / S1+S2 contested-only）")
    lines.append("")
    for lp_key in ["1000G", "800G", "630G"]:
        lines.append("")
        lines.append(f"### {LOAD_POINTS[lp_key]['label']}")
        lines.append("")
        bf = compute_battlefield(data, lp_key)
        lines.append("| Policy | P-cont mean | P-cont capped | P-cont attn | S-cont mean | S-cont capped | S-cont attn |")
        lines.append("|--------|-------------|---------------|-------------|-------------|---------------|-------------|")
        for policy in POLICIES:
            m = bf[policy]
            lines.append(
                f"| {policy:<10} | {m['p_contested_mean']:.3f} | "
                f"{m['p_contested_capped']:.3f} | {m['p_contested_attainment']:.0%} | "
                f"{m['s_contested_mean']:.3f} | {m['s_contested_capped']:.3f} | "
                f"{m['s_contested_attainment']:.0%} |"
            )

    # 3. Per-seed paired differences
    lines.append("")
    lines.append("## 3. Per-seed 配对差（目标策略 − 基线，P1+P2 mean SAS）")
    lines.append("")
    comparisons = [("Fair", "D1"), ("CRUX", "D1"), ("LongLiu-SP", "D1"), ("Fair", "CRUX")]
    for lp_key in ["1000G", "800G", "630G"]:
        lines.append("")
        lines.append(f"### {LOAD_POINTS[lp_key]['label']}")
        lines.append("")
        lines.append("| Comparison | per-seed diffs | mean | min | max |")
        lines.append("|------------|----------------|------|-----|-----|")
        for baseline, target in comparisons:
            d = compute_paired_diff(data, lp_key, baseline, target)
            diffs_str = ", ".join(f"{x:+.3f}" for x in d["diffs"])
            lines.append(
                f"| {target} − {baseline} | {diffs_str} | {d['mean']:+.3f} | {d['min']:+.3f} | {d['max']:+.3f} |"
            )

    # 4. Summary observations
    lines.append("")
    lines.append("## 4. 关键观察")
    lines.append("")
    lines.append("- 在所有三个负载点，D1 的 premium capped 仅略优于 Fair，均不及 CRUX（轻/中载）与 LongLiu-SP（重载）。")
    lines.append("- 1.02×(1000G) 下唯一达标的 premium job 是 P3(BERT-Large-p)；P1/P2 始终不达标，premium attainment 仅 11%。")
    lines.append("- 1.21×/1.54× 下 premium attainment 为 0%，说明当前 D1 机制（exp(pi·K)，K=2.0，对称权重）在边界区仍不足以保护 premium。")
    lines.append("- 轨迹显示 premium 确实进入 P6，但 standard 大 job 被剥夺后 π 同样升高并进入 P6，形成类内同权竞争，导致带宽回流到 standard。")

    out_path = PROJECT_ROOT / "outputs/feas_boundary_v1_rebirth_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
