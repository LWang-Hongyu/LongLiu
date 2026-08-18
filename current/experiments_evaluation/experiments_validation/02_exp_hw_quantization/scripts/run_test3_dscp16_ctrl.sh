#!/bin/bash
# ============================================================================
# run_test3_dscp16_ctrl.sh — 对照实验：并发两流同为 P3(DSCP16/tc:2)
# ============================================================================
# 目的：甄别 test3 中 P6:P3 = 6:4 带宽分配的成因（判别见 analyze_ctrl_sameprio.py）：
#   * 若优先级（SP）造成 → 同优先级下应观察到约 5:5（jobB 份额 ~50%）
#   * 若拥塞控制/其他机制固有 → 仍观察到 6:4（jobB 份额 ~59%）
#
# 与 test3 唯一差异：PRIO_A=PRIO_B=3（solo 校准也用 P3），其余参数与时序完全一致。
# 数据目录 exp2_ctrl_dscp16_r<round>_<ts>/，与主实验 exp2_test3_* 隔离（不混入分析）。
#
# Usage:
#   bash run_test3_dscp16_ctrl.sh [round]
# ============================================================================
set -euo pipefail

export RUN_PREFIX="exp2_ctrl_dscp16"
export PRIO_A=3
export PRIO_B=3
export SOLO_PRIO=3

exec bash "$(dirname "$0")/run_test3_sp_strict.sh" "$@"
