#!/bin/bash
# ============================================================================
# run_exp4.sh — 实验4：通信原语多样性验证（AllGather 替换 AllReduce）
# ============================================================================
# 保持 LongLiu 核心参数不变（π 调度 + EMA 锚点 + P3 背景流打满），
# 仅将通信原语从 AllReduce 换成 AllGather，多轮次验证：
#   1. DSCP 切换准确性（priority→DSCP 映射正确性，对照 NIC prio 计数器）
#   2. 锚点测量精度（calib solo 学到的 T_target/solo_bw 与 main 观测的偏差）
#
# 每轮流程：
#   Phase 0: env_check + solo 校准（AllGather）→ T_target + solo_bw
#   Phase 1: 启动双向背景流（P3）+ NIC/GPU 监控
#   Phase 2: main — LongLiu 调度 AllGather 作业（每 iter 记录 priority/dscp）
#   Phase 3: 停止监控与背景流，汇总
#
# Usage:
#   bash run_exp4.sh [round] [bg_total_gbps]
#     round = 1|2|3  重复轮次（默认 1）
#     bg_total_gbps 背景流每方向总速率（默认 48）
# ============================================================================
set -uo pipefail

ROUND=${1:-1}
BG_PER_DIR=${2:-48}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_DIR="$(cd "$BASE_DIR/../00_common" && pwd)"
DATA_DIR="$BASE_DIR/data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ID="exp4_r${ROUND}_${TS}"
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
MASTER_PORT=$((29800 + ROUND))
REMOTE_EXP_DIR="/home/why/LongLiu_rebuild/experiments_validation/04_exp_comm_primitives/scripts"

# 10.1 调度器源码路径；226 为扁平布局
SCHED_10="/home/why/LongLiu_rebuild/current/multi_comm_slo/src"
SCHED_226="/home/why/LongLiu_rebuild/multi_comm_slo/src"

PAYLOAD_MB=512
SLEEP_US=30000
NUM_ITERS=300
ITERS_PER_EPOCH=20
CI=1.7
INIT_PRIO=3
TTARGET_FILE="$OUTDIR/ttarget.json"
MONITOR_DURATION=300   # 背景流与监控时长（秒），覆盖主阶段

echo "================================================================"
echo "Exp4 通信原语多样性（AllGather）— Round $ROUND"
echo "  背景流: 双向 ${BG_PER_DIR}G P3（打满链路）"
echo "  原语:   AllGather（替换 AllReduce）"
echo "  LongLiu: ci=$CI initial=P$INIT_PRIO payload=${PAYLOAD_MB}MB"
echo "  输出:   $OUTDIR"
echo "  日期:   $(date -Iseconds)"
echo "================================================================"

# ------------------------------------------------------------------
# Phase 0: env_check + 校准
# ------------------------------------------------------------------
echo ""
echo "=== Phase 0: env_check + solo AllGather 校准 ==="
bash "$COMMON_DIR/env_check.sh" "exp4_r${ROUND}" >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_10 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_allgather.py --label A --phase calib \
        --ttarget-file "$TTARGET_FILE" --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --calib-epochs 5 --ci $CI \
        --outdir "$OUTDIR" \
        > "$OUTDIR/exp4_jobA_calib_rank0.log" 2>&1 &
JOB_CALIB_10=$!

ssh "$NODE_226" "cd $REMOTE_EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 RANK=1 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_226 \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
    NCCL_SOCKET_IFNAME=enp59s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_allgather.py --label A --phase calib \
        --ttarget-file $TTARGET_FILE --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --calib-epochs 5 --ci $CI \
        --outdir $OUTDIR \
        > $OUTDIR/exp4_jobA_calib_rank1.log 2>&1" &
JOB_CALIB_226=$!

wait $JOB_CALIB_10
wait $JOB_CALIB_226
echo "  校准完成: $(cat $TTARGET_FILE 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"T_target={d[\"target_comm_time_ms\"]}ms solo_bw={d[\"solo_bw_gbps\"]}Gbps")' 2>/dev/null || echo 'FAILED')"

# ------------------------------------------------------------------
# Phase 1+2: 背景流 + 监控 + 主阶段
# ------------------------------------------------------------------
echo ""
echo "=== Phase 1: 启动背景流与监控 ==="
bash "$BASE_DIR/scripts/bg_saturate.sh" start "exp4_r${ROUND}" "$MONITOR_DURATION" "$BG_PER_DIR"

bash "$COMMON_DIR/monitor_nic.sh" "$RUN_ID" "$MONITOR_DURATION" 1 "$DATA_DIR" \
    > "$OUTDIR/monitor_nic.log" 2>&1 &
MON_NIC_PID=$!
bash "$COMMON_DIR/monitor_gpu.sh" "$RUN_ID" "$MONITOR_DURATION" 1000 "$DATA_DIR" \
    > "$OUTDIR/monitor_gpu.log" 2>&1 &
MON_GPU_PID=$!

sleep 5

echo ""
echo "=== Phase 2: 主阶段（LongLiu 调度 + AllGather）==="
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_10 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_allgather.py --label A --phase main \
        --ttarget-file "$TTARGET_FILE" --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --num-iters $NUM_ITERS \
        --iters-per-epoch $ITERS_PER_EPOCH --ci $CI \
        --initial-priority $INIT_PRIO --outdir "$OUTDIR" \
        > "$OUTDIR/exp4_jobA_main_rank0.log" 2>&1 &
JOB_MAIN_10=$!

ssh "$NODE_226" "cd $REMOTE_EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 RANK=1 WORLD_SIZE=2 MASTER_ADDR=$RDMA_10 \
    MASTER_PORT=$MASTER_PORT MULTI_COMM_PORT=$MASTER_PORT \
    MULTI_COMM_SRC=$SCHED_226 \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
    NCCL_SOCKET_IFNAME=enp59s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u job_allgather.py --label A --phase main \
        --ttarget-file $TTARGET_FILE --payload-mb $PAYLOAD_MB \
        --sleep-us $SLEEP_US --num-iters $NUM_ITERS \
        --iters-per-epoch $ITERS_PER_EPOCH --ci $CI \
        --initial-priority $INIT_PRIO --outdir $OUTDIR \
        > $OUTDIR/exp4_jobA_main_rank1.log 2>&1" &
JOB_MAIN_226=$!

echo "  主阶段已启动（10.1 PID=$JOB_MAIN_10, 226 后台）..."
wait $JOB_MAIN_10
echo "  Job 10.1 done (exit=$?)"
wait $JOB_MAIN_226
echo "  Job 226 done (exit=$?)"

# ------------------------------------------------------------------
# Phase 3: 停止监控与背景流
# ------------------------------------------------------------------
echo ""
echo "=== Phase 3: 停止监控与背景流 ==="
wait $MON_NIC_PID 2>/dev/null || true
wait $MON_GPU_PID 2>/dev/null || true
bash "$BASE_DIR/scripts/bg_saturate.sh" stop "exp4_r${ROUND}"
sleep 2

echo ""
echo "================================================================"
echo "Exp4 Round $ROUND 完成"
echo "  数据: $OUTDIR"
echo "  文件: exp4_jobA_*_iter/epoch.csv, nic_*.csv, gpu_*.csv, ttarget.json"
echo "================================================================"
