#!/usr/bin/env python3
"""主场景 feas_boundary_v1 设计：800G spine，7 job 构型。

战斗场原则：区分度只存在于需求 > 公平份额的 job 身上。
"""

import sys
import os
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.trace.synthetic import MODEL_PARAMS


def compute_iter_solo(model: str, dp: int, overlap_factor: float = 0.85) -> float:
    """计算 iter_solo（无竞争迭代时间）。"""
    params = MODEL_PARAMS[model]
    comp_ms = params.get("comp_ms", 50.0)

    bpp = 2 if params.get("fp16", True) else 4
    mb_per_iter = 2 * params["params"] * bpp / dp / 1e6
    host_bw_gbps = 100.0
    comm_solo_ms = mb_per_iter * 8 / host_bw_gbps

    iter_solo_ms = comp_ms + comm_solo_ms * (1 - overlap_factor)
    return iter_solo_ms


def compute_target_rhythm_demand(mb_per_iter: float, ci: float, iter_solo_ms: float) -> float:
    """计算目标节奏需求（Gbps）。"""
    bits_per_iter = mb_per_iter * 8e6  # MB → Mb
    target_iter_ms = ci * iter_solo_ms
    if target_iter_ms <= 0:
        return 0.0
    demand_gbps = bits_per_iter / target_iter_ms * 1000 / 1e9
    return demand_gbps


def main():
    print("=" * 80)
    print("主场景 feas_boundary_v1 设计")
    print("=" * 80)

    print("\n战斗场原则：区分度只存在于需求 > 公平份额的 job 身上")
    print("公平份额 ≈ 800G / 7 ≈ 114 Gbps")
    print("小 job（<114 Gbps）在任何策略下天然达标，是纹理不是证据")

    # 配置：800G spine（2×400G）
    spine_bw_gbps = 800.0

    # 主场景构型（从 DEFAULT_TIERED_WORKLOAD config 提取）
    jobs_config = [
        # Premium tier (ci=1.2)
        {"jid": "P1", "model": "LLaMA-2-13B", "dp": 8, "ci": 1.2},
        {"jid": "P2", "model": "LLaMA-2-7B", "dp": 8, "ci": 1.2},
        {"jid": "P3", "model": "BERT-Large-fp16", "dp": 2, "ci": 1.2},
        # Standard tier (ci=2.0)
        {"jid": "S1", "model": "LLaMA-2-13B", "dp": 8, "ci": 2.0},
        {"jid": "S2", "model": "T5-11B-fp16", "dp": 8, "ci": 2.0},
        {"jid": "S3", "model": "BERT-Large-fp16", "dp": 4, "ci": 2.0},
        {"jid": "S4", "model": "ViT-Base", "dp": 2, "ci": 2.0},
    ]

    # 从 DEFAULT_TIERED_WORKLOAD 获取 mb_per_iter
    from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD

    print("\n" + "=" * 80)
    print("逐 Job 表（从 config 计算，禁止外推）")
    print("=" * 80)

    job_table = []
    total_demand_gbps = 0.0
    premium_demand_gbps = 0.0
    standard_demand_gbps = 0.0

    for job_cfg in jobs_config:
        jid = job_cfg["jid"]
        model = job_cfg["model"]
        dp = job_cfg["dp"]
        ci = job_cfg["ci"]

        # 从 DEFAULT_TIERED_WORKLOAD 查找对应的 mb_per_iter
        mb_per_iter = None
        for profile_model, profile_dp, profile_ci in DEFAULT_TIERED_WORKLOAD:
            if profile_model == model and profile_dp == dp:
                # 优先匹配相同的 ci，如果没有就取第一个匹配
                if mb_per_iter is None or profile_ci == ci:
                    # 从 MODEL_PARAMS 计算 mb_per_iter
                    params = MODEL_PARAMS[model]
                    bpp = 2 if params.get("fp16", True) else 4
                    mb_per_iter = 2 * params["params"] * bpp / dp / 1e6
                    if profile_ci == ci:
                        break

        if mb_per_iter is None:
            # Fallback：直接从 MODEL_PARAMS 计算
            params = MODEL_PARAMS[model]
            bpp = 2 if params.get("fp16", True) else 4
            mb_per_iter = 2 * params["params"] * bpp / dp / 1e6

        # 计算 iter_solo
        iter_solo_ms = compute_iter_solo(model, dp)

        # 计算目标节奏需求
        demand_gbps = compute_target_rhythm_demand(mb_per_iter, ci, iter_solo_ms)

        # 打印单位换算过程
        print(f"\n{jid} ({model}, dp={dp}, ci={ci}):")
        print(f"  params: {MODEL_PARAMS[model]['params']/1e9:.1f}B")
        print(f"  bpp: {bpp}")
        print(f"  mb_per_iter: 2 × {MODEL_PARAMS[model]['params']/1e9:.1f}B × {bpp} / {dp} / 1e6 = {mb_per_iter:.2f} MB")
        print(f"  bits_per_iter: {mb_per_iter:.2f} MB × 8 Mb/MB × 1e6 = {mb_per_iter * 8:.2f} Mb")
        print(f"  iter_solo: {iter_solo_ms:.2f} ms")
        print(f"  target_iter (ci×iter_solo): {ci} × {iter_solo_ms:.2f} = {ci * iter_solo_ms:.2f} ms")
        print(f"  demand: {mb_per_iter * 8:.2f} Mb / {ci * iter_solo_ms:.2f} ms × 1000 / 1e9 = {demand_gbps:.2f} Gbps")

        # 判断是否 contested
        fair_share_gbps = spine_bw_gbps / 7
        is_contested = demand_gbps > fair_share_gbps

        total_demand_gbps += demand_gbps
        if ci == 1.2:
            premium_demand_gbps += demand_gbps
        else:
            standard_demand_gbps += demand_gbps

        job_table.append({
            "jid": jid,
            "model": model,
            "dp": dp,
            "tier": "premium" if ci == 1.2 else "standard",
            "ci": ci,
            "mb_per_iter": mb_per_iter,
            "bits_per_iter_mb": mb_per_iter * 8,
            "iter_solo_ms": iter_solo_ms,
            "demand_gbps": demand_gbps,
            "is_contested": is_contested,
        })

    # 输出汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)

    oversub_ratio = total_demand_gbps / spine_bw_gbps
    print(f"\n总目标节奏需求: {total_demand_gbps:.2f} Gbps")
    print(f"Premium tier 需求: {premium_demand_gbps:.2f} Gbps")
    print(f"Standard tier 需求: {standard_demand_gbps:.2f} Gbps")
    print(f"\nSpine 容量: {spine_bw_gbps:.2f} Gbps")
    print(f"结构性超订倍数: {oversub_ratio:.2f}×")

    # 检查是否落在目标区间
    if 1.15 <= oversub_ratio <= 1.30:
        print(f"✓ 落在目标区间 [1.15, 1.30]×")
    else:
        print(f"✗ 不在目标区间 [1.15, 1.30]×")

    # 战斗场分析
    print("\n" + "=" * 80)
    print("战斗场分析")
    print("=" * 80)

    contested_jobs = [j for j in job_table if j["is_contested"]]
    print(f"\n公平份额: {spine_bw_gbps / 7:.2f} Gbps")
    print(f"Contested job（需求 > 公平份额）: {len(contested_jobs)} 个")
    for j in contested_jobs:
        print(f"  {j['jid']}: {j['demand_gbps']:.2f} Gbps ({j['tier']})")

    # 保存到文件
    os.makedirs("outputs/feas_boundary_v1_design", exist_ok=True)

    # CSV
    csv_path = "outputs/feas_boundary_v1_design/job_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "jid", "model", "dp", "tier", "ci", "mb_per_iter",
            "bits_per_iter_Mb", "iter_solo_ms", "demand_gbps", "is_contested"
        ])
        for row in job_table:
            writer.writerow([
                row["jid"],
                row["model"],
                row["dp"],
                row["tier"],
                f"{row['ci']:.1f}",
                f"{row['mb_per_iter']:.2f}",
                f"{row['bits_per_iter_mb']:.2f}",
                f"{row['iter_solo_ms']:.2f}",
                f"{row['demand_gbps']:.2f}",
                row["is_contested"],
            ])

    print(f"\n✓ 逐 job 表已保存: {csv_path}")

    # JSON
    summary = {
        "total_demand_gbps": total_demand_gbps,
        "premium_demand_gbps": premium_demand_gbps,
        "standard_demand_gbps": standard_demand_gbps,
        "spine_bw_gbps": spine_bw_gbps,
        "oversub_ratio": oversub_ratio,
        "num_contested_jobs": len(contested_jobs),
        "jobs": job_table,
    }

    with open("outputs/feas_boundary_v1_design/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ 摘要已保存: outputs/feas_boundary_v1_design/summary.json")

    # 验证 13B mb_per_iter
    print("\n" + "=" * 80)
    print("验证：13B mb_per_iter 来源")
    print("=" * 80)

    llama_13b_params = MODEL_PARAMS["LLaMA-2-13B"]["params"]
    bpp = 2  # fp16
    dp = 8
    mb_per_iter_calc = 2 * llama_13b_params * bpp / dp / 1e6

    print(f"\nLLaMA-2-13B params: {llama_13b_params/1e9:.1f}B")
    print(f"bpp: {bpp} (fp16)")
    print(f"dp: {dp}")
    print(f"mb_per_iter 计算: 2 × {llama_13b_params/1e9:.1f}B × {bpp} / {dp} / 1e6 = {mb_per_iter_calc:.2f} MB")
    print(f"DEFAULT_TIERED_WORKLOAD 中对应配置应为此值（而非 v2 的 6500 MB）")


if __name__ == "__main__":
    main()