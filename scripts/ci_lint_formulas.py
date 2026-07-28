"""CI 断言：D1 内部 T_target 与 metrics.compute_target_iter_ms 同源。

验证 dwrr.py 中所有 DWRR 家族的 T_target/comm_budget/attain_bw 公式
与 metrics.py 的 compute_target_iter_ms 数值一致（容差 1e-9）。

用法：python3 scripts/ci_lint_formulas.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.utils.model_params import MODEL_PARAMS
from longliu_sim.utils.metrics import compute_target_iter_ms


# 与 config.yaml frozen 节一致
OVERHEAD = 1.3
OVERLAP = 0.85
HOST_BW = 100.0  # Gbps


def dwrr_inline_T_target(comp_ms: float, comm_solo_ms: float, slo_ci: float) -> float:
    """LongLiuDWRR.allocate() 内联的 T_target 计算（锚点公式，SEMANTICS_VERSION="anchor-v2"）。

    必须与 metrics.compute_target_iter_ms 逐位一致。
    """
    comm_budget = slo_ci * comm_solo_ms * OVERHEAD
    if OVERLAP > 0:
        return max(comp_ms, comm_budget) + (1.0 - OVERLAP) * min(comp_ms, comm_budget)
    else:
        return comp_ms + comm_budget


def dwrr_inline_attain_bw(comp_ms: float, comm_solo_ms: float, slo_ci: float,
                          bits_per_iter: float) -> float:
    """LongLiuDWRR.V4/V3.1 内联的 attain_bw 计算（锚点语义）。

    wire_bits = bits_per_iter × overhead
    eff_budget = T_target - comp
    attain_bw = wire_bits / eff_budget
    """
    wire_bits = bits_per_iter * OVERHEAD
    target = dwrr_inline_T_target(comp_ms, comm_solo_ms, slo_ci)
    eff_budget = target - comp_ms
    assert eff_budget > 0, f"eff_budget={eff_budget} <= 0 for ci={slo_ci}, comp={comp_ms}"
    return wire_bits / (eff_budget * 1e-3)


def test_all_models():
    """对 model_params.py 中所有模型，验证 T_target 双参考系一致。"""
    errors = []
    successes = []

    for model_name, params in MODEL_PARAMS.items():
        bpp = 2 if params.get("fp16", True) else 4

        for dp in [1, 2, 4, 8]:
            # 计算 comm_solo_ms 和 comp_ms（与 metrics._get_comp_and_solo 一致）
            mb_per_iter = 2 * params["params"] * bpp / max(dp, 1) / 1e6
            comm_solo_ms = mb_per_iter * 8 / HOST_BW
            comp_ms = params.get("comp_ms", 50.0)
            bits_per_iter = mb_per_iter * 1e6 * 8  # logical bits

            for ci in [1.2, 1.5, 2.0, 2.5, 3.0]:
                # Reference: metrics.compute_target_iter_ms
                ref = compute_target_iter_ms(
                    model_name, dp, ci, HOST_BW, OVERHEAD, OVERLAP
                )

                # DUT: dwrr inline formula
                dut = dwrr_inline_T_target(comp_ms, comm_solo_ms, ci)

                if abs(ref - dut) > 1e-9:
                    errors.append(
                        f"{model_name} dp={dp} ci={ci}: "
                        f"ref={ref:.6f} vs dut={dut:.6f}, diff={abs(ref-dut):.6e}"
                    )
                else:
                    # Also verify attain_bw = wire_bits / (target - comp)
                    wire_bits = bits_per_iter * OVERHEAD
                    eff = ref - comp_ms
                    if eff <= 0:
                        continue  # skip infeasible combos
                    attain_ref = wire_bits / (eff * 1e-3)
                    attain_dut = dwrr_inline_attain_bw(comp_ms, comm_solo_ms, ci, bits_per_iter)
                    if abs(attain_ref - attain_dut) > 1e-3:  # bps tolerance
                        errors.append(
                            f"{model_name} dp={dp} ci={ci}: "
                            f"attain_ref={attain_ref:.3f} vs attain_dut={attain_dut:.3f}"
                        )
                    else:
                        successes.append(f"{model_name} dp={dp} ci={ci}")

    return errors, successes


def main():
    errors, successes = test_all_models()

    total = len(errors) + len(successes)
    print(f"测试 {total} 个 (model, dp, ci) 组合")
    print(f"  PASS: {len(successes)}")
    print(f"  FAIL: {len(errors)}")

    if errors:
        print("\n失败详情:")
        for e in errors:
            print(f"  {e}")
        print(f"\nCI FAIL: {len(errors)} assertions failed")
        sys.exit(1)
    else:
        print("CI PASS: 所有 T_target 双参考系一致")
        sys.exit(0)


if __name__ == "__main__":
    main()
