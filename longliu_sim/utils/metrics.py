"""评估指标：唯一合法的 SAS / target_iter_ms 计算实现。

所有 SAS、SLO attainment、starvation、catastrophic/HANDOFF 计算必须调用此模块，
禁止在 simulator、实验脚本、分析脚本中复制粘贴公式。

SEMANTICS_VERSION: anchor-v2

锚点公式（paper-baseline-v2 tag，逐字还原）：
    comm_budget = slo_ci × comm_solo_ms × overhead_factor
    if overlap_factor > 0:
        target_iter_ms = max(comp_ms, comm_budget)
                        + (1 - overlap_factor) × min(comp_ms, comm_budget)
    else:
        target_iter_ms = comp_ms + comm_budget
    sas = target_iter_ms / avg_iter_ms

语义：通信 SLO + overlap 合成——SLO 管网络能控制的东西。
overhead_factor=1.3 是冻结的 wire 因子（非发明常数），代表 wire=1.3×逻辑bits。
48/48 逐值精确匹配 v2_test 锚点数据（2026-07-24 裁决实验）。

历史：现行公式（iter_solo = comp + comm_solo×(1-overlap), sas=ci×iter_solo/avg_iter）
于口径审计中被误判为正主，实际从未经过锚点验证。2026-07-24 复位。
"""

from __future__ import annotations

# 语义版本号：锚点公式
SEMANTICS_VERSION = "anchor-v2"

from .model_params import MODEL_PARAMS


def _get_comp_and_comm_solo(model: str, dp: int,
                            host_bw_gbps: float = 100.0) -> tuple[float, float]:
    """获取 comp_ms 和 comm_solo_ms。"""
    params = MODEL_PARAMS.get(model)
    if params is None:
        raise ValueError(f"Unknown model '{model}'")

    comp_ms = params.get("comp_ms", 50.0)
    bpp = 2 if params.get("fp16", True) else 4

    # mb_per_iter = 2 * params * bpp / dp (MB，十进制)
    mb_per_iter = 2 * params["params"] * bpp / max(dp, 1) / 1e6

    # comm_solo (ms) = mb_per_iter (MB) * 8 / host_bw (Gbps)
    comm_solo_ms = mb_per_iter * 8 / host_bw_gbps

    return comp_ms, comm_solo_ms


def compute_target_iter_ms(model: str, dp: int, slo_ci: float,
                           host_bw_gbps: float = 100.0,
                           overhead_factor: float = 1.3,
                           overlap_factor: float = 0.85) -> float:
    """计算目标迭代时间 target_iter_ms（锚点公式）。

    comm_budget = slo_ci × comm_solo_ms × overhead_factor
    if overlap_factor > 0:
        target = max(comp_ms, comm_budget) + (1-overlap_factor) × min(comp_ms, comm_budget)
    else:
        target = comp_ms + comm_budget

    参数：
        model: 模型名称
        dp: DDP worker 数量
        slo_ci: SLO 松弛系数
        host_bw_gbps: 无竞争链路带宽（Gbps）
        overhead_factor: wire 因子（默认 1.3，锚点冻结值）
        overlap_factor: compute-comm 重叠度（默认 0.85，锚点冻结值）
    """
    comp_ms, comm_solo_ms = _get_comp_and_comm_solo(model, dp, host_bw_gbps)

    comm_budget = slo_ci * comm_solo_ms * overhead_factor

    if overlap_factor > 0:
        target = max(comp_ms, comm_budget) + (1.0 - overlap_factor) * min(comp_ms, comm_budget)
    else:
        target = comp_ms + comm_budget

    return target


def compute_sas_eval(avg_iter_ms: float, model: str, dp: int, slo_ci: float,
                     host_bw_gbps: float = 100.0,
                     overhead_factor: float = 1.3,
                     overlap_factor: float = 0.85) -> float:
    """计算 sas_eval = target_iter_ms / avg_iter_ms（锚点公式）。

    返回 float；avg_iter_ms <= 0 时返回 0.0（调用方负责标记 starvation）。

    参数同 compute_target_iter_ms。
    """
    if avg_iter_ms <= 0.0:
        return 0.0
    target = compute_target_iter_ms(model, dp, slo_ci, host_bw_gbps,
                                    overhead_factor, overlap_factor)
    return target / avg_iter_ms


def compute_iter_solo_ms(model: str, dp: int, host_bw_gbps: float = 100.0,
                         overhead_factor: float = 1.3,
                         overlap_factor: float = 0.85) -> float:
    """计算无竞争迭代时间 iter_solo（锚点语义）。

    iter_solo = target_iter_ms / slo_ci（ci=1 时的 target）

    注意：此函数保留向后兼容，但语义已变。
    锚点语义下 iter_solo = max(comp, comm_solo×overhead)
                           + (1-overlap)×min(comp, comm_solo×overhead)
    不再是简单的 comp + comm_solo × (1-overlap)。
    """
    # ci=1.0 时的 target 就是 iter_solo
    return compute_target_iter_ms(model, dp, slo_ci=1.0,
                                  host_bw_gbps=host_bw_gbps,
                                  overhead_factor=overhead_factor,
                                  overlap_factor=overlap_factor)


def compute_attain_bw_bps(model: str, dp: int, slo_ci: float,
                          host_bw_gbps: float = 100.0,
                          overhead_factor: float = 1.3,
                          overlap_factor: float = 0.85) -> float:
    """计算 attain 带宽（锚点语义）。

    attain = wire_bits_per_iter / (target_iter_ms - comp_ms) × 1000

    其中 wire_bits_per_iter = bits_per_iter × overhead_factor，
    target_iter_ms 用锚点公式。

    若 target_iter_ms <= comp_ms（纯计算 SLO），返回 inf。
    """
    comp_ms, comm_solo_ms = _get_comp_and_comm_solo(model, dp, host_bw_gbps)

    # wire bits per iter
    bits_per_iter = comm_solo_ms * host_bw_gbps * 1e9 / 1000.0  # bits
    wire_bits = bits_per_iter * overhead_factor

    target = compute_target_iter_ms(model, dp, slo_ci, host_bw_gbps,
                                    overhead_factor, overlap_factor)
    effective_budget_ms = target - comp_ms

    if effective_budget_ms <= 0:
        return float('inf')

    return wire_bits / (effective_budget_ms * 1e-3)
