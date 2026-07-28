#!/usr/bin/env python3
"""sas_eval 干净重算：从 workload config 读取 jid→model→基准映射。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS


def compute_iter_solo(model: str, dp: int, host_bw_gbps: float = 100.0, overlap_factor: float = 0.85) -> float:
    """计算 iter_solo（无竞争迭代时间）。

    iter_solo = comp + comm_solo × (1 - overlap)
    """
    params = MODEL_PARAMS[model]
    comp_ms = params.get("comp_ms", 50.0)

    # comm_solo = mb_per_iter / host_bw
    # mb_per_iter = 2 × params × bpp / dp
    bpp = 2 if params.get("fp16", True) else 4
    mb_per_iter = 2 * params["params"] * bpp / dp / 1e6

    # comm_solo (ms) = mb_per_iter (MB) × 8 / host_bw (Gbps)
    comm_solo_ms = mb_per_iter * 8 / host_bw_gbps

    # iter_solo = comp + comm_solo × (1 - overlap)
    iter_solo_ms = comp_ms + comm_solo_ms * (1 - overlap_factor)

    return iter_solo_ms


def main():
    print("=" * 160)
    print("sas_eval 干净重算（从 workload config 读取 jid→model→基准映射）")
    print("=" * 160)

    # 读取 v4 workload 配置
    workload = DEFAULT_TIERED_WORKLOAD

    print("v4 Workload 配置（jid → model → dp → ci）：")
    print("-" * 120)

    # 创建 jid → (model, dp, ci, iter_solo) 映射
    jid_to_config = {}

    for i, (model, dp, ci) in enumerate(workload):
        jid = f"J{i}"
        iter_solo = compute_iter_solo(model, dp)

        jid_to_config[jid] = {
            "model": model,
            "dp": dp,
            "ci": ci,
            "iter_solo": iter_solo,
            "tier": "premium" if ci <= 1.2 else ("standard" if ci <= 2.0 else "medium")
        }

        if i < 20:  # 只显示前 20 个
            print(f"{jid}: {model:<20} DP={dp}, ci={ci}, iter_solo={iter_solo:.1f}ms, tier={jid_to_config[jid]['tier']}")

    print()

    # 读取实验结果
    with open("outputs/dwrr_ablation_v4/dwrr_results.json") as f:
        data = json.load(f)

    # 计算 sas_eval
    print("=" * 160)
    print("sas_eval 七策略全表（干净重算）")
    print("=" * 160)

    policies = ["Fair", "SRPT", "CRUX", "LongLiu", "D1", "D2", "D3"]

    print(f"{'策略':<12} {'Premium Mean':>13} {'Premium Cap':>13} {'Premium 崩溃':>13} {'Premium Starv':>14} {'Premium 达成':>13} "
          f"{'Standard Mean':>14} {'Standard Cap':>14} {'Standard 崩溃':>14} {'Standard Starv':>15} {'Standard 达成':>14}")
    print("-" * 160)

    for policy in policies:
        results = data[policy]

        # 统计 premium 档
        all_sas_eval = []
        all_starved = 0
        total_premium = 0

        for r in results:
            for job in r["tier_stats"]["premium"]:
                jid = job["jid"]
                avg_iter_ms = job["avg_iter_ms"]

                # 从 config 读取基准
                if jid in jid_to_config:
                    config = jid_to_config[jid]
                    iter_solo = config["iter_solo"]
                    ci = config["ci"]
                else:
                    # 后备：使用默认值
                    iter_solo = 162.0
                    ci = 1.2

                total_premium += 1

                if avg_iter_ms == 0.0:  # 零迭代
                    all_starved += 1
                    all_sas_eval.append(0.0)
                else:
                    sas_eval = (ci * iter_solo) / avg_iter_ms
                    all_sas_eval.append(sas_eval)

        # Premium 统计
        mean_sas_eval = sum(all_sas_eval) / len(all_sas_eval) if all_sas_eval else 0.0
        capped_sas_eval = sum(min(s, 1.0) for s in all_sas_eval) / len(all_sas_eval) if all_sas_eval else 0.0
        collapsed_eval = sum(1 for s in all_sas_eval if s < 0.2)
        collapse_rate_eval = collapsed_eval / total_premium * 100 if total_premium > 0 else 0.0
        starve_rate = all_starved / total_premium * 100 if total_premium > 0 else 0.0
        achieved_eval = sum(1 for s in all_sas_eval if s >= 1.0)
        achieve_rate_eval = achieved_eval / total_premium * 100 if total_premium > 0 else 0.0

        # Standard 档统计
        all_sas_eval_std = []
        all_starved_std = 0
        total_standard = 0

        for r in results:
            for job in r["tier_stats"]["standard"]:
                jid = job["jid"]
                avg_iter_ms = job["avg_iter_ms"]

                # 从 config 读取基准
                if jid in jid_to_config:
                    config = jid_to_config[jid]
                    iter_solo = config["iter_solo"]
                    ci = config["ci"]
                else:
                    iter_solo = 143.0
                    ci = 2.0

                total_standard += 1

                if avg_iter_ms == 0.0:
                    all_starved_std += 1
                    all_sas_eval_std.append(0.0)
                else:
                    sas_eval = (ci * iter_solo) / avg_iter_ms
                    all_sas_eval_std.append(sas_eval)

        # Standard 统计
        mean_sas_eval_std = sum(all_sas_eval_std) / len(all_sas_eval_std) if all_sas_eval_std else 0.0
        capped_sas_eval_std = sum(min(s, 1.0) for s in all_sas_eval_std) / len(all_sas_eval_std) if all_sas_eval_std else 0.0
        collapsed_eval_std = sum(1 for s in all_sas_eval_std if s < 0.2)
        collapse_rate_eval_std = collapsed_eval_std / total_standard * 100 if total_standard > 0 else 0.0
        starve_rate_std = all_starved_std / total_standard * 100 if total_standard > 0 else 0.0
        achieve_rate_eval_std = sum(1 for s in all_sas_eval_std if s >= 1.0) / total_standard * 100 if total_standard > 0 else 0.0

        print(f"{policy:<12} {mean_sas_eval:>13.3f} {capped_sas_eval:>13.3f} {collapse_rate_eval:>13.1f}% {starve_rate:>14.1f}% {achieve_rate_eval:>13.1f}% "
              f"{mean_sas_eval_std:>14.3f} {capped_sas_eval_std:>14.3f} {collapse_rate_eval_std:>14.1f}% {starve_rate_std:>15.1f}% {achieve_rate_eval_std:>14.1f}%")

    print()


if __name__ == "__main__":
    main()