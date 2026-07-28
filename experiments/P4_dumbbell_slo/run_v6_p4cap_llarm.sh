#!/bin/bash
# ============================================================================
# V6 P4 天花板钳制 — 仅运行 LL 臂一轮
# ============================================================================
# 目的：
#   将 LongLiu 的 max_priority 钳在 P4，防止动态升到 P5/P6。
#   若 tight job slowdown ≤ 1.2× → 确认问题在 P5/P6 未验证 TC 配置
#   若仍违约 > 1.2× → NIC 出口争抢方向
#
# 参数：
#   bash run_v6_p4cap_llarm.sh <bg_rate_gbps>
#     bg_rate_gbps: 背景流速率 (默认 6)
#
# 配置：
#   - 背景流: 12× iperf3 UDP, DSCP=P3 (TOS=96)
#   - Warmup: 5 min
#   - LL: initial_priority=3, max_priority=4
#   - c_i: tight=1.2, loose=3.0, epoch 7 翻转
#   - T_target: V5 校准
# ============================================================================
set -uo pipefail

BG_RATE_GBPS=${1:-6}
NUM_BG_FLOWS=12
BG_TOTAL_SEC=400  # 足够覆盖 warmup + job 全程

PAYLOAD_MB=1024
CI_TIGHT=1.2
CI_LOOSE=3.0
WARMUP_MINUTES=5

PORT_MAIN_A=29520
PORT_MAIN_B=29521

TTARGET_A="/tmp/ttarget_v5_jobA.json"
TTARGET_B="/tmp/ttarget_v5_jobB.json"

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
BG_PORT_START=6200
BG_PORT_END=6211
DSCP_P3_TOS=64  # P3 → DSCP=16 → TOS=64

EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"

echo "================================================================"
echo "V6 P4 天花板钳制 — 仅 LL 臂"
echo "  背景流: iperf3 UDP ${BG_RATE_GBPS} Gbps (12×$((BG_RATE_GBPS*1000/12))M), DSCP=P3"
echo "  LongLiu: initial_priority=3, max_priority=4 (钳在 P4)"
echo "  c_i tight/loose: ${CI_TIGHT} / ${CI_LOOSE} (epoch 7 翻转)"
echo "  T_target: V5 校准 (${PAYLOAD_MB}MB)"
echo "  Warmup: ${WARMUP_MINUTES} min"
echo "  日期: $(date -Iseconds)"
echo "================================================================"

cd "$EXP_DIR"

# Pre-flight check
if [[ ! -f "$TTARGET_A" || ! -f "$TTARGET_B" ]]; then
    echo "ERROR: T_target files not found."
    exit 1
fi

# ============================================================
# 清理函数
# ============================================================
cleanup_bg() {
    # 本地 iperf3 客户端
    for PID in $(pgrep -f "iperf3.*-p 62[0-9][0-9].*-u" 2>/dev/null); do
        kill $PID 2>/dev/null || true
    done
    # 远程 iperf3 服务器
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        local SRV_PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
        if [[ -n "$SRV_PID" ]]; then
            ssh $NODE_226 "kill $SRV_PID" 2>/dev/null || true
        fi
    done
}

cleanup_jobs() {
    for PID in $(pgrep -f "p4_job_reverse.py --job [AB]" 2>/dev/null); do
        kill $PID 2>/dev/null || true
    done
    ssh $NODE_226 "for PID in \$(pgrep -f 'p4_job_reverse.py --job [AB]' 2>/dev/null); do kill \$PID 2>/dev/null; done" 2>/dev/null || true
    sleep 2
}

cleanup_all() {
    echo "--- 清理所有进程 ---"
    cleanup_jobs
    cleanup_bg
    sleep 2
    echo "清理完成"
}

# ============================================================
# Step 1: 清理 + 启动背景流
# ============================================================
cleanup_all

echo ""
echo "=== Step 1: 启动 12 路 iperf3 服务器 (226) ==="
for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
    OLD_PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
    if [[ -n "$OLD_PID" ]]; then
        ssh $NODE_226 "kill $OLD_PID" 2>/dev/null || true
    fi
    ssh $NODE_226 "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null
done
sleep 2
SRV_COUNT=$(ssh $NODE_226 "pgrep -a iperf3 | grep '\-s' | wc -l")
echo "  服务器运行数: $SRV_COUNT"

echo ""
echo "=== Step 2: 启动 ${BG_RATE_GBPS}G P3 背景流 (${BG_TOTAL_SEC}s) ==="
PER_STREAM=$((BG_RATE_GBPS * 1000 / NUM_BG_FLOWS))
for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
    iperf3 -c $RDMA_226 -u -b ${PER_STREAM}M -t $BG_TOTAL_SEC \
        --tos $DSCP_P3_TOS -p $PORT -f g -l 8900 \
        > /tmp/v6_p4cap_bg_${PORT}.log 2>&1 &
done
echo "  ${NUM_BG_FLOWS} 路背景流已启动（每路 ${PER_STREAM}M）"
sleep 5

# ============================================================
# Step 3: Warmup
# ============================================================
echo ""
echo "=== Step 3: Warmup ${WARMUP_MINUTES} min (背景流运行中, 无 NCCL job) ==="
sleep ${WARMUP_MINUTES}m
echo "  Warmup 完成"

# ============================================================
# Step 4: 启动 LongLiu 实验 (max_priority=4)
# ============================================================
echo ""
echo "=== Step 4: 启动 LL 臂 (initial_priority=3, max_priority=4) ==="

cleanup_jobs

# Job A rank 0 (10.1)
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_MAIN_A \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u p4_job_reverse.py --job A --mode longliu --phase main \
        --ttarget-file $TTARGET_A \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --initial-priority 3 --max-priority 4 \
        > p4_jobA_v6_p4cap_longliu_node101.log 2>&1 &
JOB_A_101_PID=$!

# Job A rank 1 (226)
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_MAIN_A \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    python3 -u p4_job_reverse.py --job A --mode longliu --phase main \
        --ttarget-file $TTARGET_A \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --initial-priority 3 --max-priority 4 \
        > p4_jobA_v6_p4cap_longliu_node226.log 2>&1" &
JOB_A_226_PID=$!

echo "Job A 已启动, 等待 10s 让 Job A 初始化..."
sleep 10

# Job B rank 0 (10.1)
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_MAIN_B \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u p4_job_reverse.py --job B --mode longliu --phase main \
        --ttarget-file $TTARGET_B \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --initial-priority 3 --max-priority 4 \
        > p4_jobB_v6_p4cap_longliu_node101.log 2>&1 &
JOB_B_101_PID=$!

# Job B rank 1 (226)
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_MAIN_B \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    python3 -u p4_job_reverse.py --job B --mode longliu --phase main \
        --ttarget-file $TTARGET_B \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --initial-priority 3 --max-priority 4 \
        > p4_jobB_v6_p4cap_longliu_node226.log 2>&1" &
JOB_B_226_PID=$!

echo "Job B 已启动"
echo ""
echo "等待 Job A 和 Job B 完成..."
echo "(背景流持续 ${BG_TOTAL_SEC}s)"

# Wait for all jobs
wait $JOB_A_101_PID; echo "Job A 10.1 done (exit=$?)"
wait $JOB_A_226_PID; echo "Job A 226 done (exit=$?)"
wait $JOB_B_101_PID; echo "Job B 10.1 done (exit=$?)"
wait $JOB_B_226_PID; echo "Job B 226 done (exit=$?)"

# Rename CSV
for JOB in A B; do
    CSV_EPOCH="p4_job${JOB}_reverse_longliu_rank0_epoch.csv"
    CSV_ITER="p4_job${JOB}_reverse_longliu_rank0_iter.csv"
    [[ -f "$CSV_EPOCH" ]] && mv "$CSV_EPOCH" "p4_job${JOB}_v6_p4cap_longliu_rank0_epoch.csv"
    [[ -f "$CSV_ITER" ]] && mv "$CSV_ITER" "p4_job${JOB}_v6_p4cap_longliu_rank0_iter.csv"
done

# ============================================================
# Step 5: 停止背景流 + 汇总
# ============================================================
cleanup_bg

echo ""
echo "================================================================"
echo "P4 天花板钳制实验结果"
echo "================================================================"

for JOB in A B; do
    CSV="p4_job${JOB}_v6_p4cap_longliu_rank0_epoch.csv"
    if [[ -f "$CSV" ]]; then
        echo ""
        echo "=== Job ${JOB} ==="
        cat "$CSV"
    fi
done

echo ""
echo "=== 背景流汇总 ==="
TOTAL=0
for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
    GBPS=$(tail -3 /tmp/v6_p4cap_bg_${PORT}.log 2>/dev/null | grep -oP '[\d.]+(?= Gbits/sec)' | tail -1)
    if [[ -n "$GBPS" ]]; then
        TOTAL=$(echo "$TOTAL + $GBPS" | bc 2>/dev/null)
    fi
done
printf "  总吞吐: ~%.1f Gbps\n" "${TOTAL:-0}"

echo ""
echo "================================================================"
echo "完成。日志: p4_job[AB]_v6_p4cap_longliu_node*.log"
echo "CSV: p4_job[AB]_v6_p4cap_longliu_rank0_epoch.csv"
echo "================================================================"
