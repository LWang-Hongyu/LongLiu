#!/bin/bash
# ============================================================================
# run_exp1.sh — 实验1：主动重校准探针物理验证
# ============================================================================
# 流程：
#   Phase 0: env_check + 校准（solo，无背景流）→ T_target + solo 带宽基线
#   Phase 1: 启动背景流打满链路（单向 10.1->226，DSCP=P3）+ NIC/GPU 监控
#   Phase 2: 主阶段 — P3 冻结锚点作业 + 每 PROBE_EVERY epoch 触发 P6 单次探测
#   Phase 3: 停止监控与背景流，汇总
#
# Usage:
#   bash run_exp1.sh [round] [bg_total_gbps]
#     round = 1|2|3  重复轮次（默认 1）
#     bg_total_gbps 背景流总速率（默认 40）
# ============================================================================
set -uo pipefail

ROUND=${1:-1}
BG_TOTAL_GBPS=${2:-40}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_DIR="$(cd "$BASE_DIR/../00_common" && pwd)"
DATA_DIR="$BASE_DIR/data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ID="exp1_r${ROUND}_${TS}"
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
MASTER_PORT=$((29600 + ROUND))
REMOTE_EXP_DIR="/home/why/LongLiu_rebuild/experiments_validation/01_exp_probe_recalib/scripts"
REMOTE_DATA_BASE="/home/why/LongLiu_rebuild/experiments_validation/01_exp_probe_recalib/data"
REMOTE_OUTDIR="$REMOTE_DATA_BASE/$RUN_ID"

# 10.1 调度器源码路径；226 为扁平布局
SCHED_10="/home/why/LongLiu_rebuild/current/multi_comm_slo/src"
SCHED_226="/home/why/LongLiu_rebuild/multi_comm_slo/src"

PAYLOAD_MB=1024
SLEEP_US=30000
NUM_EPOCHS=15
ITERS_PER_EPOCH=20
PROBE_EVERY=3
TTARGET_FILE="$OUTDIR/ttarget.json"
MONITOR_DURATION=120   # 背景流与监控时长（秒），覆盖主阶段（~60s）
BG_PROTO=tcp           # tcp: 有拥塞控制，真实挤占链路拖慢 P3 作业（UDP 无PFC保护会被RoCE挤出）

echo "================================================================"
echo "Exp1 主动重校准探针 — Round $ROUND"
echo "  背景流: 单向 ${BG_TOTAL_GBPS}G P3（10.1->226，打满 50G 链路）"
echo "  作业:   ${PAYLOAD_MB}MB AllReduce, P3 冻结锚点, 每 ${PROBE_EVERY} epoch P6 探测"
echo "  输出:   $OUTDIR"
echo "  日期:   $(date -Iseconds)"
echo "================================================================"

# ------------------------------------------------------------------
# Phase 0: env_check + 校准
# ------------------------------------------------------------------
echo ""
echo "=== Phase 0: env_check + solo 校准 ==="
bash "$COMMON_DIR/env_check.sh" "exp1_r${ROUND}" >/dev/null 2>&1

# 校准（无背景流）
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_10 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_frozen_p3.py --job A --phase calib \
        --ttarget-file "$TTARGET_FILE" --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --calib-epochs 8 --outdir "$OUTDIR" \
        > "$OUTDIR/exp1_jobA_calib_rank0.log" 2>&1 &
JOB_CALIB_10=$!

ssh "$NODE_226" "mkdir -p '$REMOTE_OUTDIR' && cd $REMOTE_EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 RANK=1 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_226 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
    NCCL_SOCKET_IFNAME=enp59s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_frozen_p3.py --job A --phase calib \
        --ttarget-file $REMOTE_OUTDIR/ttarget.json --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --calib-epochs 8 --outdir $REMOTE_OUTDIR \
        > $REMOTE_OUTDIR/exp1_jobA_calib_rank1.log 2>&1" &
JOB_CALIB_226=$!

wait $JOB_CALIB_10
wait $JOB_CALIB_226

# 将 10.1 端 rank0 写入的 T_target 同步到 226（main 阶段 226 端要读）
if [ -f "$TTARGET_FILE" ]; then
    scp -q "$TTARGET_FILE" "$NODE_226:$REMOTE_OUTDIR/ttarget.json" && \
        echo "  T_target 已同步至 226"
fi
echo "  校准完成: $(cat $TTARGET_FILE 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("target_comm_time_ms"); s=d.get("solo_bw_gbps"); print("T_target=%.1fms solo_bw=%.2fGbps" % (t,s))' 2>/dev/null || echo 'FAILED')"

# ------------------------------------------------------------------
# Phase 1+2: 背景流 + 监控 + 主阶段
# ------------------------------------------------------------------
echo ""
echo "=== Phase 1: 启动背景流与监控 ==="
# 背景流：单向 BG_TOTAL_GBPS Gbps（bg_saturate.sh 按 12 路拆分；TCP 模式自动填满）
bash "$BASE_DIR/scripts/bg_saturate.sh" start "exp1_r${ROUND}" "$MONITOR_DURATION" "$BG_TOTAL_GBPS" "$BG_PROTO"

# 监控（写入实验 data 目录）
bash "$COMMON_DIR/monitor_nic.sh" "$RUN_ID" "$MONITOR_DURATION" 1 "$DATA_DIR" \
    > "$OUTDIR/monitor_nic.log" 2>&1 &
MON_NIC_PID=$!
bash "$COMMON_DIR/monitor_gpu.sh" "$RUN_ID" "$MONITOR_DURATION" 1000 "$DATA_DIR" \
    > "$OUTDIR/monitor_gpu.log" 2>&1 &
MON_GPU_PID=$!

# 等待背景流与监控稳定
sleep 5

echo ""
echo "=== Phase 2: 主阶段（冻结 P3 + P6 探测）==="
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_10 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_frozen_p3.py --job A --phase main \
        --ttarget-file "$TTARGET_FILE" --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --num-epochs $NUM_EPOCHS \
        --iters-per-epoch $ITERS_PER_EPOCH --probe-every $PROBE_EVERY \
        --outdir "$OUTDIR" \
        > "$OUTDIR/exp1_jobA_main_rank0.log" 2>&1 &
JOB_MAIN_10=$!

ssh "$NODE_226" "cd $REMOTE_EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 RANK=1 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_226 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
    NCCL_SOCKET_IFNAME=enp59s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_frozen_p3.py --job A --phase main \
        --ttarget-file $REMOTE_OUTDIR/ttarget.json --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --num-epochs $NUM_EPOCHS \
        --iters-per-epoch $ITERS_PER_EPOCH --probe-every $PROBE_EVERY \
        --outdir $REMOTE_OUTDIR \
        > $REMOTE_OUTDIR/exp1_jobA_main_rank1.log 2>&1" &
JOB_MAIN_226=$!

echo "  主阶段已启动（10.1 PID=$JOB_MAIN_10, 226 后台）..."
wait $JOB_MAIN_10
echo "  Job 10.1 done (exit=$?)"
wait $JOB_MAIN_226
echo "  Job 226 done (exit=$?)"

# 回传 226 端 rank1 产物（CSV/日志）
scp -q "$NODE_226:$REMOTE_OUTDIR/exp1_jobA_rank1_iter.csv" "$OUTDIR/" 2>/dev/null || true
scp -q "$NODE_226:$REMOTE_OUTDIR/exp1_jobA_main_rank1.log" "$OUTDIR/" 2>/dev/null || true
scp -q "$NODE_226:$REMOTE_OUTDIR/exp1_jobA_calib_rank1.log" "$OUTDIR/" 2>/dev/null || true
echo "  226 端数据已回传"

# ------------------------------------------------------------------
# Phase 3: 停止监控与背景流
# ------------------------------------------------------------------
echo ""
echo "=== Phase 3: 停止监控与背景流 ==="
wait $MON_NIC_PID 2>/dev/null || true
wait $MON_GPU_PID 2>/dev/null || true
bash "$BASE_DIR/scripts/bg_saturate.sh" stop "exp1_r${ROUND}"
sleep 2

echo ""
echo "================================================================"
echo "Exp1 Round $ROUND 完成"
echo "  数据: $OUTDIR"
echo "  文件: exp1_jobA_*_iter/epoch/probe.csv, nic_*.csv, gpu_*.csv, ttarget.json"
echo "================================================================"
