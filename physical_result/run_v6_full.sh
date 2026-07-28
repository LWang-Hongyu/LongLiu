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

EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
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

# ============================================================
# Helper: run a single mode (longliu or crux)
# ============================================================
run_mode() {
    local MODE=$1
    local MODE_LABEL=$2  # human-readable label for file naming
    local PHASE_LABEL="${ROUND_LABEL}_${MODE}"

    echo ""
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
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
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

    # Job B rank 1 on 226
    ssh $NODE_226 "cd $EXP_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_MAIN_B \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
        NCCL_DEBUG_FILE=/tmp/nccl_jB_${MODE}_v6_${ROUND_LABEL}_226_%h_%p.log \
        python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS \
            > p4_jobB_v6_${PHASE_LABEL}_node226.log 2>&1" &
    JOB_B_226_PID=$!

    echo "Job B ($MODE) launched."
    echo "  (background flow active: ${BG_RATE_GBPS} Gbps DSCP=P3)"

    # Wait for all jobs in this mode
    wait $JOB_A_101_PID; echo "  Job A 10.1 done (exit=$?)"
    wait $JOB_A_226_PID; echo "  Job A 226 done (exit=$?)"
    wait $JOB_B_101_PID; echo "  Job B 10.1 done (exit=$?)"
    wait $JOB_B_226_PID; echo "  Job B 226 done (exit=$?)"

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
    done

    echo "--- $MODE_LABEL completed ---"
    echo ""
}

# ============================================================
# Main execution
# ============================================================

# Step 1: Start background flow (duration covers entire round + warmup)
# Total duration = warmup + first mode run + second mode run + inter-mode gap
# Each mode runs ~300 iters × (30ms compute + ~300ms comm) = ~100s
# So total ~600s + 5min warmup + 15s gap ≈ 10 min
BG_TOTAL_SEC=$((WARMUP_MINUTES * 60 + 120 + 120 + 30))
start_bg_flow $BG_TOTAL_SEC

# Step 2: Warmup (background flow running, no NCCL jobs)
echo ""
echo "--- Warmup phase: ${WARMUP_MINUTES} min (data discarded) ---"
echo "  Background flow running at ${BG_RATE_GBPS} Gbps DSCP=P3"
echo "  Waiting for NIC/thermal/driver state stabilization..."
sleep ${WARMUP_MINUTES}m
echo "  Warmup complete."

# Step 3: Run first mode
run_mode $FIRST_MODE "${ORDER}_first(${FIRST_MODE})"

# Step 4: NCCL cleanup gap
echo "Waiting 15s between modes for NCCL cleanup..."
sleep 15

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
        CSV="p4_job${JOB}_v6_${PHASE_LABEL}_rank0_epoch.csv"
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
echo "  CSVs: p4_job[AB]_v6_${ROUND_LABEL}_*_rank0_epoch.csv"
echo "  Background: /tmp/v6_bgflow_${ROUND_LABEL}_*.log"
echo "================================================================"
