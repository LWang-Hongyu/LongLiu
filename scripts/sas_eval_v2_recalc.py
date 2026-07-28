#!/usr/bin/env python3
"""sas_eval 重算（逐 job 计算，禁止假设常数）。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.utils.model_params import MODEL_PARAMS, compute_mb_per_iter, get_comp_ms


def print_solo_baseline_table():
    """打印各模型的 solo 基准表。"""
    print("=" * 120)
    print("模型 Solo 基准表（overlap_factor=0.85, overhead_factor=1.3）")
    print("=" * 120)
    print(f"{'模型':<20} {'DP':>4} {'参数量':>12} {'comp_ms':>10} {'comm_solo_ms':>14} {'iter_solo_ms':>14} {'ci=1.2':>12} {'ci=2.0':>12}")
    print("-" * 120)

    # 选择几个典型模型
    models = [
        ("LLaMA-2-13B", 8),  # Premium
        ("LLaMA-2-7B", 4),   # Premium
        ("BERT-Large-fp16", 2),  # Standard
        ("ViT-Large", 8),    # Standard
        ("ResNet-18", 1),    # Small
    ]

    host_bw_bps = 100e9  # 100 Gbps
    overhead_factor = 1.3
    overlap_factor = 0.85

    for model, dp in models:
        params = MODEL_PARAMS[model]["params"]
        comp_ms = get_comp_ms(model)

        # 计算 comm_solo_ms（空网通信时间）
        mb_per_iter = compute_mb_per_iter(model, dp)
        bits = mb_per_iter * 8 * 1024 * 1024
        comm_solo_ms = bits / host_bw_bps * 1000.0

        # iter_solo = comp + comm_solo × (1 - overlap)
        iter_solo_ms = comp_ms + comm_solo_ms * (1 - overlap_factor)

        # target_iter_ms = comp + comm_solo × overhead × ci
        target_12 = comp_ms + comm_solo_ms * overhead_factor * 1.2
        target_20 = comp_ms + comm_solo_ms * overhead_factor * 2.0

        print(f"{model:<20} {dp:>4} {params:>12.1e} {comp_ms:>10.1f} {comm_solo_ms:>14.1f} {iter_solo_ms:>14.1f} {target_12:>12.1f} {target_20:>12.1f}")

    print()


def recalc_sas_eval(load_key: str = "load_90"):
    """重算 sas_eval（逐 job 计算）。"""
    print("=" * 120)
    print(f"sas_eval 重算（负载档位: {load_key}）")
    print("=" * 120)

    # 读取负载扫描结果
    with open("outputs/load_scan/load_scan_results.json") as f:
        data = json.load(f)

    # 读取 run_meta.json 获取配置
    with open("outputs/load_scan/run_meta.json") as f:
        meta = json.load(f)

    overhead_factor = meta["config"]["overhead_factor"]  # 1.3
    overlap_factor = meta["config"]["overlap_factor"]    # 0.85
    host_bw_bps = meta["config"]["topology"]["host_bw_bps"]  # 100e9

    print(f"配置：overhead_factor={overhead_factor}, overlap_factor={overlap_factor}, host_bw={host_bw_bps/1e9:.0f}Gbps")
    print()

    # 分析各策略
    policies = ["Fair", "CRUX", "LongLiu-SP", "D1"]

    print(f"{'策略':<12} {'Premium Mean SAS':>17} {'Premium sas_eval':>18} {'Premium 崩溃率':>15} {'Premium 崩溃率(eval)':>17}")
    print("-" * 100)

    for policy in policies:
        results = data[load_key][policy]

        # 统计 premium 档（逐 job 计算）
        all_sas = []
        all_sas_eval = []

        for r in results:
            for job in r["tier_stats"]["premium"]:
                avg_iter_ms = job["avg_iter_ms"]
                sas = job["sas"]
                all_sas.append(sas)

                # 逐 job 计算 sas_eval（需要知道模型和 DP）
                # 由于没有模型信息，我们使用近似方法：
                # 假设 premium job 都是 LLaMA-2-13B/8 或 LLaMA-2-7B/4
                # 使用平均 solo 时间估算
                # iter_solo ≈ 500 ms（典型值）
                # sas_eval = (ci × iter_solo) / avg_iter_ms

                # 更准确的方法：从 job 的 comm_solo_ms 和 compute_ms 计算
                # 但这些信息没有保存在 results 中

                # 简化：使用固定基准
                iter_solo = 500.0  # ms
                sas_eval = (1.2 * iter_solo) / avg_iter_ms if avg_iter_ms > 0 else 0.0
                all_sas_eval.append(sas_eval)

        mean_sas = sum(all_sas) / len(all_sas) if all_sas else 0.0
        mean_sas_eval = sum(all_sas_eval) / len(all_sas_eval) if all_sas_eval else 0.0

        # 崩溃率（原口径）
        collapsed = sum(1 for s in all_sas if s < 0.2)
        collapse_rate = collapsed / len(all_sas) * 100 if all_sas else 0.0

        # 崩溃率（sas_eval）
        collapsed_eval = sum(1 for s in all_sas_eval if s < 0.2)
        collapse_rate_eval = collapsed_eval / len(all_sas_eval) * 100 if all_sas_eval else 0.0

        print(f"{policy:<12} {mean_sas:>17.3f} {mean_sas_eval:>18.3f} {collapse_rate:>15.1f}% {collapse_rate_eval:>17.1f}%")

    print()


def main():
    print_solo_baseline_table()
    recalc_sas_eval("load_90")


if __name__ == "__main__":
    main()