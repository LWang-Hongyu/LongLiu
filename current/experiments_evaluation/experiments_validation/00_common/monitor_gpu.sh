#!/bin/bash
# ============================================================================
# monitor_gpu.sh — GPU 状态周期采样（10.1 本机 + 226 远程）
# ============================================================================
# 使用 nvidia-smi --query-gpu 输出结构化时间序列，包含：
#   timestamp, power.draw, temperature.gpu, clocks.sm, clocks.max.sm,
#   clocks_throttle_reasons.active（降频原因，如 Thermal/Power/HW Slowdown）
#   clocks_throttle_reasons.hw_slowdown
# 用于检测 thermal throttling / power capping 等硬件层异常。
#
# Usage:
#   bash monitor_gpu.sh <run_id> <duration_sec> [interval_ms] [outdir]
#     若提供 outdir，则写入 $outdir/<run_id>/（否则写入 00_common/../data/<run_id>/）
#
# Output:
#   <outdir>/<run_id>/gpu_10.csv    （RTX 4000）
#   <outdir>/<run_id>/gpu_226.csv   （RTX 5000 ×2）
# ============================================================================
set -uo pipefail

RUN_ID=${1:?Usage: $0 <run_id> <duration_sec> [interval_ms] [outdir]}
DURATION=${2:?duration_sec required}
INTERVAL_MS=${3:-1000}
OVERRIDE_OUTDIR=${4:-}

NODE_226="192.10.10.226"

if [[ -n "$OVERRIDE_OUTDIR" ]]; then
    DATA_DIR="$OVERRIDE_OUTDIR"
else
    DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
fi
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"

QUERY="timestamp,index,name,power.draw,temperature.gpu,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,clocks_throttle_reasons.hw_slowdown,utilization.gpu"

echo "[monitor_gpu] run=$RUN_ID duration=${DURATION}s interval=${INTERVAL_MS}ms"
echo "[monitor_gpu] 10.1 -> $OUTDIR/gpu_10.csv"
echo "[monitor_gpu] 226  -> $OUTDIR/gpu_226.csv"

# 10.1（后台运行 DURATION 秒）
timeout "${DURATION}" nvidia-smi --query-gpu="$QUERY" --format=csv,nounits -lms "${INTERVAL_MS}" \
    > "$OUTDIR/gpu_10.csv" 2>&1 &
LOCAL_PID=$!

# 226（后台运行 DURATION 秒，经 ssh；写 226 本地 /tmp 后回传）
REMOTE_TMP="/tmp/gpu_226_${RUN_ID}.csv"
ssh "$NODE_226" "timeout ${DURATION} nvidia-smi --query-gpu=\"$QUERY\" --format=csv,nounits -lms ${INTERVAL_MS} > ${REMOTE_TMP} 2>&1" &
REMOTE_PID=$!

wait "$LOCAL_PID"
wait "$REMOTE_PID"

# 回传 226 数据
scp -q "$NODE_226:${REMOTE_TMP}" "$OUTDIR/gpu_226.csv" 2>/dev/null || echo "[monitor_gpu] WARN: failed to fetch gpu_226.csv"
ssh "$NODE_226" "rm -f ${REMOTE_TMP}" 2>/dev/null || true

echo "[monitor_gpu] done -> $OUTDIR/gpu_10.csv, gpu_226.csv"
