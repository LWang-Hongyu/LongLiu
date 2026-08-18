#!/bin/bash
# ============================================================================
# V6 Full Experiment: Background Flow + Warmup + Alternating Order
# ============================================================================
# Per the approved V6 plan:
#   1. Background flow: iperf3 UDP, DSCP=P3, ~30 Gbps
#   2. Warmup: 5-10 min before each round (data discarded)
#   3. LongLiu initial DSCP = P3 (same starting point as CRUX)
#   4. CRUX both jobs P3 (GPU intensity tie)
#   5. Alternating order: Round 1 = LL→CX, Round 2 = CX→LL
#   6. c_i: tight=1.2, loose=3.0 (swap at epoch 7)
#
# Usage:
#   bash run_v6_full.sh <round> <bg_rate_gbps>
#     round = 1 (LL→CX) or 2 (CX→LL)
#     bg_rate_gbps = background flow rate (default 30)
#
# Example:
#   bash run_v6_full.sh 1 30    # Round 1: LL→CX, 30 Gbps bg
#   bash run_v6_full.sh 2 30    # Round 2: CX→LL, 30 Gbps bg
# ============================================================================
set -euo pipefail

ROUND=${1:?Usage: $0 <round=1|2> <bg_rate_gbps>}
BG_RATE_GBPS=${2:-30}
RATE_MBPS=$((BG_RATE_GBPS * 1000))

# Validate round
if [[ "$ROUND" != "1" && "$ROUND" != "2" ]]; then
    echo "ERROR: round must be 1 (LL→CX) or 2 (CX→LL)"
    exit 1
fi

# Derive order from round
if [[ "$ROUND" == "1" ]]; then
    ORDER="LLthenCX"
    FIRST_MODE="longliu"
    SECOND_MODE="crux"
else
    ORDER="CXthenLL"
    FIRST_MODE="crux"
    SECOND_MODE="longliu"
fi

# ============================================================
# Configuration (matching V6 parameter table)
# ============================================================
PAYLOAD_MB=1024
CI_TIGHT=1.2
CI_LOOSE=3.0
WARMUP_MINUTES=5

# Ports for both modes
PORT_MAIN_A=29520
PORT_MAIN_B=29521

# 每个模式最多尝试次数（NCCL init 偶发挂死时自动重试）
MAX_MODE_RETRY=3

# T_target files (reuse V5 calibration — same 1024MB payload)
TTARGET_A="/tmp/ttarget_v5_jobA.json"
TTARGET_B="/tmp/ttarget_v5_jobB.json"

# Network
NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
BG_PORT_START=6200
BG_PORT_END=6211
NUM_BG_FLOWS=12
DSCP_P3_TOS=64  # P3 → DSCP=16 → TOS=64

EXP_DIR="/home/why/LongLiu_rebuild/current/experiments_evaluation/P4_dumbbell_slo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROUND_LABEL="round${ROUND}_${ORDER}"

echo "================================================================"
echo "V6 Full Experiment — Round $ROUND: $ORDER"
echo "  Background flow: iperf3 UDP ${BG_RATE_GBPS} Gbps, DSCP=P3"
echo "  Payload:        ${PAYLOAD_MB}MB × 2"
echo "  c_i tight/loose: ${CI_TIGHT} / ${CI_LOOSE} (swap at epoch 7)"
echo "  Warmup:         ${WARMUP_MINUTES} min before first mode"
echo "  CRUX priority:  both P3 (GPU intensity tie)"
echo "  LongLiu start:  P3 (same as CRUX starting point)"
echo "  Order:          ${FIRST_MODE} → ${SECOND_MODE}"
echo "  T_target:       V5 calibration (${PAYLOAD_MB}MB)"
echo "  Date:           $(date -Iseconds)"
echo "================================================================"

# Pre-flight: check T_target files
if [[ ! -f "$TTARGET_A" || ! -f "$TTARGET_B" ]]; then
    echo "ERROR: T_target files not found. Run V5 calibration first."
    echo "  bash run_p4_reverse.sh v5 both (with calibration)"
    exit 1
fi

# Sync T_target files to 226 — CRITICAL (2026-08-10 v7 死锁根因):
# 226 缺失该文件时，rank1 调度器回退到本地 EMA warmup 学习 T_target，
# 与 rank0 的 preset T_target 分叉 → π 不同 → 优先级切换不同步
# （rank0 切到 P4 而 rank1 仍留在 P2）→ 两侧使用不同 communicator
# 的 AllReduce 永久死锁（v7 实测）。
echo "--- Syncing T_target files to 226 ---"
scp -q "$TTARGET_A" "$NODE_226:/tmp/" && scp -q "$TTARGET_B" "$NODE_226:/tmp/"
if ssh "$NODE_226" "test -f $TTARGET_A && test -f $TTARGET_B"; then
    echo "  T_target files confirmed on 226"
else
    echo "ERROR: T_target files missing on 226 after sync"
    exit 1
fi

# ============================================================
# Helper: start/stop background flow
# ============================================================
start_bg_flow() {
    local duration_sec=$1
    local per_flow_mbps=$((BG_RATE_GBPS * 1000 / NUM_BG_FLOWS))
    echo "--- Starting iperf3 background flow: ${BG_RATE_GBPS} Gbps (${NUM_BG_FLOWS}×${per_flow_mbps}M), DSCP=P3, ${duration_sec}s ---"

    # Start 12 iperf3 servers on 226
    echo "  启动 ${NUM_BG_FLOWS} 路 iperf3 服务器 (226)..."
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        local OLD_PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
        if [[ -n "$OLD_PID" ]]; then
            ssh $NODE_226 "kill $OLD_PID" 2>/dev/null || true
        fi
        ssh $NODE_226 "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null
    done
    sleep 2
    local SRV_COUNT=$(ssh $NODE_226 "pgrep -a iperf3 | grep '\-s' | wc -l" 2>/dev/null)
    echo "  服务器运行数: $SRV_COUNT"

    # Start 12 iperf3 clients on 10.1
    echo "  启动 ${NUM_BG_FLOWS} 路客户端（每路 ${per_flow_mbps} Mbps）..."
    BG_PIDS=()
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        iperf3 -c $RDMA_226 -u -b ${per_flow_mbps}M -t $duration_sec \
            --tos $DSCP_P3_TOS -p $PORT -f g -l 8900 \
            > /tmp/v6_bgflow_${ROUND_LABEL}_${PORT}.log 2>&1 &
        BG_PIDS+=($!)
    done
    echo "  ${NUM_BG_FLOWS} 路客户端已启动（总 ${BG_RATE_GBPS} Gbps, duration ${duration_sec}s）"
    sleep 3
}

stop_bg_flow() {
    echo "--- Stopping background flow ---"
    # Kill all 12 client PIDs
    for PID in "${BG_PIDS[@]}"; do
        kill $PID 2>/dev/null || true
    done
    # 精确清理 12 路 iperf3 服务器
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        local SRV_PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
        if [[ -n "$SRV_PID" ]]; then
            ssh $NODE_226 "kill $SRV_PID" 2>/dev/null || true
        fi
    done
    echo "  ${NUM_BG_FLOWS} 路背景流已停止"
}

bg_flow_summary() {
    echo "=== 背景流总吞吐 ==="
    local TOTAL=0
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        local GBPS=$(tail -3 /tmp/v6_bgflow_${ROUND_LABEL}_${PORT}.log 2>/dev/null | grep -oP '[\d.]+(?= Gbits/sec)' | tail -1)
        if [[ -n "$GBPS" ]]; then
            TOTAL=$(echo "$TOTAL + $GBPS" | bc 2>/dev/null)
            printf "  Port %4d: %s Gbps\n" $PORT "$GBPS"
        fi
    done
    printf "  总吞吐: ~%.1f Gbps\n" "${TOTAL:-0}"
}

cleanup_jobs() {
    for PID in $(pgrep -f "p4_job_reverse.py --job [AB]" 2>/dev/null); do
        kill $PID 2>/dev/null || true
    done
    ssh $NODE_226 "for PID in \$(pgrep -f 'p4_job_reverse.py --job [AB]' 2>/dev/null); do kill \$PID 2>/dev/null; done" 2>/dev/null || true
    sleep 2
}

# 等待 4 个 job 全部结束（带超时与对端死亡检测，防止 101 侧挂死导致无限 wait）
# 返回 0 = 全部正常结束；1 = 超时/226 侧已死而 101 侧挂起（内部已清理残留）
wait_mode_jobs() {
    # 2026-08-17: 30Gbps 背景流下 1024MB allreduce 实测 ~2.7s/iter，
    # 300 iters + init ≈ 870s。原 420s 会在 Job 仍在正常推进时误杀，
    # 放宽到 1080s（18 分钟）覆盖单模式全时长。
    local MAX_WAIT=1080
    local t=0
    while [[ $t -lt $MAX_WAIT ]]; do
        local alive_101=0 alive_226=0
        kill -0 $JOB_A_101_PID 2>/dev/null && alive_101=1
        kill -0 $JOB_B_101_PID 2>/dev/null && alive_101=1
        kill -0 $JOB_A_226_PID 2>/dev/null && alive_226=1
        kill -0 $JOB_B_226_PID 2>/dev/null && alive_226=1
        if [[ $alive_101 -eq 0 && $alive_226 -eq 0 ]]; then
            echo "  4 个 job 均已结束"
            return 0
        fi
        if [[ $alive_101 -eq 1 && $alive_226 -eq 0 ]]; then
            echo "  226 侧已退出而 101 侧仍挂起（对端死亡），判定失败，清理残留"
            cleanup_jobs
            return 1
        fi
        sleep 15
        t=$((t + 15))
    done
    echo "  等待超时（${MAX_WAIT}s），判定失败，清理残留"
    cleanup_jobs
    return 1
}

# ============================================================
# Helper: run a single mode (longliu or crux)
# ============================================================
run_mode() {
    local MODE=$1
    local MODE_LABEL=$2  # human-readable label for file naming
    local PHASE_LABEL="${ROUND_LABEL}_${MODE}"

    local MODE_START_EPOCH=$(date +%s)
    echo ""
    echo "[$(date +%F_%T)] === MODE START: ${MODE_LABEL} (${MODE}) ==="
    echo "--- Running $MODE_LABEL (${MODE}) ---"

    cleanup_jobs

    # Determine CRUX priority flags
    local CRUX_FLAGS=""
    if [[ "$MODE" == "crux" ]]; then
        CRUX_FLAGS="--crux-priority-a 3 --crux-priority-b 3"
    fi

    # Determine LongLiu initial priority flags
    local LL_FLAGS=""
    if [[ "$MODE" == "longliu" ]]; then
        LL_FLAGS="--initial-priority 3"
    fi

    # Job A rank 0 on 10.1
    # 101 侧 Job rank 0 启动：必须显式用系统 NCCL（2026-08-17 实测
    # PyTorch bundled NCCL 2.18.6 比系统 2.30.7 慢 3 倍——与 226 侧不对称，
    # 是 V6 带宽仅 3.1 Gbps 的根因）
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
    PYTHONPATH=/home/why/LongLiu_rebuild/current/multi_comm_slo/src \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_MAIN_A \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
        NCCL_DEBUG_FILE=/tmp/nccl_jA_${MODE}_v6_${ROUND_LABEL}_101_%h_%p.log \
        python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS \
            > p4_jobA_v6_${PHASE_LABEL}_node101.log 2>&1 &
    JOB_A_101_PID=$!

    # Job A rank 1 on 226
    ssh $NODE_226 "cd $EXP_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_MAIN_A \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/current/multi_comm_slo/src:\$PYTHONPATH \
        NCCL_DEBUG_FILE=/tmp/nccl_jA_${MODE}_v6_${ROUND_LABEL}_226_%h_%p.log \
        python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS \
            > p4_jobA_v6_${PHASE_LABEL}_node226.log 2>&1" &
    JOB_A_226_PID=$!

    echo "Job A ($MODE) launched, waiting 10s for init..."
    sleep 10

    # Job B rank 0 on 10.1
    # 101 侧 Job rank 0 启动：必须显式用系统 NCCL（2026-08-17 实测
    # PyTorch bundled NCCL 2.18.6 比系统 2.30.7 慢 3 倍——与 226 侧不对称，
    # 是 V6 带宽仅 3.1 Gbps 的根因）
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
    PYTHONPATH=/home/why/LongLiu_rebuild/current/multi_comm_slo/src \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_MAIN_B \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
        NCCL_DEBUG_FILE=/tmp/nccl_jB_${MODE}_v6_${ROUND_LABEL}_101_%h_%p.log \
        python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS \
            > p4_jobB_v6_${PHASE_LABEL}_node101.log 2>&1 &
    JOB_B_101_PID=$!

    # Job B rank 1 on 226 (GPU 1 — 与 Job A 分离，避免同 GPU 双 NCCL 进程并发干扰)
    ssh $NODE_226 "cd $EXP_DIR && \
        CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_MAIN_B \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/current/multi_comm_slo/src:\$PYTHONPATH \
        NCCL_DEBUG_FILE=/tmp/nccl_jB_${MODE}_v6_${ROUND_LABEL}_226_%h_%p.log \
        python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS \
            > p4_jobB_v6_${PHASE_LABEL}_node226.log 2>&1" &
    JOB_B_226_PID=$!

    echo "Job B ($MODE) launched."
    echo "  (background flow active: ${BG_RATE_GBPS} Gbps DSCP=P3)"

    # Wait for all jobs in this mode (with per-mode retry for NCCL init flakiness)
    local EA10=1 EA26=1 EB10=1 EB26=1
    local MODE_OK=0
    for ATT in $(seq 1 $MAX_MODE_RETRY); do
        if [[ $ATT -gt 1 ]]; then
            echo "  [$MODE attempt $ATT] NCCL init 失败，清理后重试..."
            cleanup_jobs
            sleep 5
            # re-launch all 4 procs
            LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
            CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
                WORLD_SIZE=2 RANK=0 \
                MULTI_COMM_PORT=$PORT_MAIN_A \
                NCCL_SOCKET_IFNAME=enp130s0f0np0 \
                NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
                NCCL_DEBUG_FILE=/tmp/nccl_jA_${MODE}_v6_${ROUND_LABEL}_101_%h_%p.log \
                python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
                    --ttarget-file $TTARGET_A \
                    --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
                    $CRUX_FLAGS $LL_FLAGS \
                    > p4_jobA_v6_${PHASE_LABEL}_node101.log 2>&1 &
            JOB_A_101_PID=$!
            ssh $NODE_226 "cd $EXP_DIR && \
                CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
                WORLD_SIZE=2 RANK=1 \
                MULTI_COMM_PORT=$PORT_MAIN_A \
                NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
                NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
                LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH \
                PYTHONPATH=/home/why/LongLiu_rebuild/current/multi_comm_slo/src:\$PYTHONPATH \
                NCCL_DEBUG_FILE=/tmp/nccl_jA_${MODE}_v6_${ROUND_LABEL}_226_%h_%p.log \
                python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
                    --ttarget-file $TTARGET_A \
                    --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
                    $CRUX_FLAGS $LL_FLAGS \
                    > p4_jobA_v6_${PHASE_LABEL}_node226.log 2>&1" &
            JOB_A_226_PID=$!
            sleep 10
            LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
            CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
                WORLD_SIZE=2 RANK=0 \
                MULTI_COMM_PORT=$PORT_MAIN_B \
                NCCL_SOCKET_IFNAME=enp130s0f0np0 \
                NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
                NCCL_DEBUG_FILE=/tmp/nccl_jB_${MODE}_v6_${ROUND_LABEL}_101_%h_%p.log \
                python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
                    --ttarget-file $TTARGET_B \
                    --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
                    $CRUX_FLAGS $LL_FLAGS \
                    > p4_jobB_v6_${PHASE_LABEL}_node101.log 2>&1 &
            JOB_B_101_PID=$!
            ssh $NODE_226 "cd $EXP_DIR && \
                CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
                WORLD_SIZE=2 RANK=1 \
                MULTI_COMM_PORT=$PORT_MAIN_B \
                NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
                NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
                LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH \
                PYTHONPATH=/home/why/LongLiu_rebuild/current/multi_comm_slo/src:\$PYTHONPATH \
                NCCL_DEBUG_FILE=/tmp/nccl_jB_${MODE}_v6_${ROUND_LABEL}_226_%h_%p.log \
                python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
                    --ttarget-file $TTARGET_B \
                    --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
                    $CRUX_FLAGS $LL_FLAGS \
                    > p4_jobB_v6_${PHASE_LABEL}_node226.log 2>&1" &
            JOB_B_226_PID=$!
        fi
        if ! wait_mode_jobs; then
            EA10=1; EA26=1; EB10=1; EB26=1
        else
            # 进程已确认全部结束；不再 wait（历史上 wait 偶发挂住），
            # 短暂等待确保 CSV flush 后直接按文件检查判定
            echo "  PIDs: A10=$JOB_A_101_PID A226=$JOB_A_226_PID B10=$JOB_B_101_PID B226=$JOB_B_226_PID"
            sleep 2
            EA10=0; EA26=0; EB10=0; EB26=0
        fi
        echo "  Job A 10.1 done (exit=$EA10), Job A 226 done (exit=$EA26)"
        echo "  Job B 10.1 done (exit=$EB10), Job B 226 done (exit=$EB26)"
        if [[ $EA10 -eq 0 && $EA26 -eq 0 && $EB10 -eq 0 && $EB26 -eq 0 &&
              -f "p4_jobA_reverse_${MODE}_rank0_window.csv" &&
              -f "p4_jobB_reverse_${MODE}_rank0_window.csv" ]]; then
            MODE_OK=1
            break
        fi
        echo "  [$MODE attempt $ATT] 失败（exit=$EA10/$EA26/$EB10/$EB26）"
        # remove partial CSVs before retry（window+iter 都清，防止陈旧完整 CSV 干扰判定；
        # window 粒度改造后 p4_job_reverse.py 不再输出 epoch CSV，判定改用 window CSV）
        rm -f p4_jobA_reverse_${MODE}_rank0_window.csv p4_jobB_reverse_${MODE}_rank0_window.csv
        rm -f p4_jobA_v6_${PHASE_LABEL}_rank0_window.csv p4_jobB_v6_${PHASE_LABEL}_rank0_window.csv
        rm -f p4_jobA_reverse_${MODE}_rank0_iter.csv p4_jobB_reverse_${MODE}_rank0_iter.csv
        rm -f p4_jobA_reverse_${MODE}_rank0_window.csv p4_jobB_reverse_${MODE}_rank0_window.csv
        ssh $NODE_226 "rm -f /home/why/LongLiu_rebuild/current/experiments_evaluation/P4_dumbbell_slo/p4_job*_reverse_${MODE}_rank1_*.csv 2>/dev/null" 2>/dev/null || true
    done
    if [[ $MODE_OK -ne 1 ]]; then
        echo "ERROR: $MODE_LABEL 在 ${MAX_MODE_RETRY} 次尝试后仍失败，终止"
        exit 1
    fi

    # Rename CSV files to include round label (prevent overwrite)
    for JOB in A B; do
        local CSV_EPOCH="p4_job${JOB}_reverse_${MODE}_rank0_epoch.csv"
        local CSV_ITER="p4_job${JOB}_reverse_${MODE}_rank0_iter.csv"
        local CSV_LOG="p4_job${JOB}_reverse_${MODE}_rank0.log"
        if [[ -f "$CSV_EPOCH" ]]; then
            mv "$CSV_EPOCH" "p4_job${JOB}_v6_${PHASE_LABEL}_rank0_epoch.csv"
        fi
        if [[ -f "$CSV_ITER" ]]; then
            mv "$CSV_ITER" "p4_job${JOB}_v6_${PHASE_LABEL}_rank0_iter.csv"
        fi
        local CSV_WINDOW="p4_job${JOB}_reverse_${MODE}_rank0_window.csv"
        if [[ -f "$CSV_WINDOW" ]]; then
            mv "$CSV_WINDOW" "p4_job${JOB}_v6_${PHASE_LABEL}_rank0_window.csv"
        fi
    done

    echo "--- $MODE_LABEL completed ---"
    echo "[$(date +%F_%T)] === MODE END: ${MODE_LABEL} (${MODE}) | duration=$(( $(date +%s) - MODE_START_EPOCH ))s ==="
    echo ""
}

# ============================================================
# Main execution
# ============================================================

# Step 1: Start background flow (duration covers entire round + warmup)
# Total duration = warmup + first mode run + second mode run + inter-mode gap
#   NOTE (2026-08-10): 旧版 570s 会在第二个模式 phase2 后期耗尽背景流，
#   导致 CRUX 侧 slowdown 骤降到 <1（伪影）。840s 在实测模式耗时 ~5min
#   (NCCL 7 comm 初始化 + 300 iter × ~480ms) 下仍偏紧，现放宽到 ~18min：
#   warmup 300s + 模式 2×360s + 间隙 15s + 余量 60s，确保全程有背景流。
#   2026-08-17: 30Gbps 背景流实测 ~2.7s/iter → 单模式 ~900s，背景流
#   时长同步放宽（warmup 300 + 模式 2×900 + 间隙 15 + 余量 120）。
BG_TOTAL_SEC=$((WARMUP_MINUTES * 60 + 900 + 15 + 900 + 120))
start_bg_flow $BG_TOTAL_SEC

# Step 2: Warmup (background flow running, no NCCL jobs)
echo ""
echo "[$(date +%F_%T)] === WARMUP START (${WARMUP_MINUTES} min) ==="
echo "--- Warmup phase: ${WARMUP_MINUTES} min (data discarded) ---"
echo "  Background flow running at ${BG_RATE_GBPS} Gbps DSCP=P3"
echo "  Waiting for NIC/thermal/driver state stabilization..."
sleep ${WARMUP_MINUTES}m
echo "[$(date +%F_%T)] === WARMUP END === (planned ${WARMUP_MINUTES} min)"
echo "  Warmup complete."

# Step 3: Run first mode
run_mode $FIRST_MODE "${ORDER}_first(${FIRST_MODE})"

# Step 4: NCCL cleanup gap
echo "[$(date +%F_%T)] === GAP START (15s) ==="
echo "Waiting 15s between modes for NCCL cleanup..."
sleep 15
echo "[$(date +%F_%T)] === GAP END ==="

# Step 5: Run second mode
run_mode $SECOND_MODE "${ORDER}_second(${SECOND_MODE})"

# Step 6: Stop background flow
stop_bg_flow

# ============================================================
# Results summary
# ============================================================
echo ""
echo "================================================================"
echo "V6 Round $ROUND ($ORDER) — Results Summary"
echo "================================================================"

for MODE in longliu crux; do
    echo ""
    echo "=== $MODE ==="
    for JOB in A B; do
        PHASE_LABEL="${ROUND_LABEL}_${MODE}"
        CSV="p4_job${JOB}_v6_${PHASE_LABEL}_rank0_window.csv"
        if [[ -f "$CSV" ]]; then
            echo "--- Job $JOB ---"
            cat "$CSV"
        else
            echo "--- Job $JOB: CSV not found ---"
        fi
    done
done

# Background flow summary
echo ""
bg_flow_summary

echo ""
echo "================================================================"
echo "Round $ROUND complete."
echo "  Logs: p4_job[AB]_v6_${ROUND_LABEL}_*_node[101|226].log"
echo "  CSVs: p4_job[AB]_v6_${ROUND_LABEL}_*_rank0_window.csv"
echo "  Background: /tmp/v6_bgflow_${ROUND_LABEL}_*.log"
echo "================================================================"
