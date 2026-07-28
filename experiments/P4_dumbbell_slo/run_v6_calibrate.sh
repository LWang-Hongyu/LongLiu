#!/bin/bash
# ============================================================================
# V6 Step 2: iperf3 Background Flow Calibration
# ============================================================================
# Purpose: Verify that iperf3 UDP at DSCP=P3 creates ≥1.3× slowdown for
#          a CRUX P3-marked tight job.
#
# Method:
#   1. Start iperf3 server on 226 (192.10.10.226:6000)
#   2. Start iperf3 client on 10.1 (DSCP=P3, rate from $1)
#   3. Run CRUX experiment with background flow active
#   4. Measure tight job slowdown vs V5 baseline (no background flow)
#
# Usage:
#   bash run_v6_calibrate.sh <rate_gbps> [duration_sec]
#   e.g., bash run_v6_calibrate.sh 30  (30 Gbps background flow)
#
# Calibration target: CRUX tight job slowdown ≥1.3×
# ============================================================================
set -euo pipefail

# Args
BG_RATE_GBPS=${1:?Usage: $0 <rate_gbps> [duration_sec]}
BG_DURATION=${2:-120}  # 2 minutes default (enough for several epochs)
RATE_MBPS=$((BG_RATE_GBPS * 1000))

EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
BG_PORT=6000
DSCP_P3_TOS=64  # P3 → DSCP=16 → TOS=64

# V6 params
PAYLOAD_MB=1024
CI_TIGHT=1.2
CI_LOOSE=3.0
TTARGET_A="/tmp/ttarget_v5_jobA.json"
TTARGET_B="/tmp/ttarget_v5_jobB.json"
PORT_JA=29620
PORT_JB=29621

echo "=================================================================="
echo "V6 Step 2: Background Flow Calibration"
echo "  Background: iperf3 UDP ${BG_RATE_GBPS} Gbps, DSCP=P3 (TOS=${DSCP_P3_TOS})"
echo "  Direction:  10.1 (client) → 226 (server)"
echo "  Duration:   ${BG_DURATION}s"
echo "  CRUX c_i:   tight=${CI_TIGHT}, loose=${CI_LOOSE}"
echo "  Target:     tight job slowdown ≥1.3×"
echo "=================================================================="
echo ""

# Pre-flight checks
if [[ ! -f "$TTARGET_A" || ! -f "$TTARGET_B" ]]; then
    echo "ERROR: T_target files not found. Run V5 calibration first."
    exit 1
fi

# ============================================================
# Step 2a: Start iperf3 server on 226
# ============================================================
echo "--- Starting iperf3 server on 226 ---"
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $NODE_226 "pkill -f 'iperf3 -s -p $BG_PORT' 2>/dev/null; sleep 1; iperf3 -s -p $BG_PORT -D -B $RDMA_226 2>&1; sleep 1; ss -uln | grep $BG_PORT || echo 'iperf3 server NOT running'" 2>&1
echo ""

# ============================================================
# Step 2b: Verify server reachable (quick probe)
# ============================================================
echo "--- Quick connectivity probe ---"
sleep 2
iperf3 -c $RDMA_226 -u -b 100M -t 2 --tos $DSCP_P3_TOS -p $BG_PORT -f m 2>&1 | tail -5
echo ""

# ============================================================
# Step 2c: Start background flow (persistent, background)
# ============================================================
echo "--- Starting background flow: ${BG_RATE_GBPS} Gbps, DSCP=P3 ---"
iperf3 -c $RDMA_226 -u -${RATE_MBPS}M -t $BG_DURATION --tos $DSCP_P3_TOS -p $BG_PORT -f m -l 8900 > /tmp/v6_bgflow_101.log 2>&1 &
BG_PID=$!
echo "Background flow PID: $BG_PID"
sleep 3
echo ""

# ============================================================
# Step 2d: Run CRUX experiment with background flow active
# ============================================================
echo "--- Running CRUX experiment (background flow active) ---"
echo "  Using T_target: A=$(cat $TTARGET_A), B=$(cat $TTARGET_B)"
echo ""

# Cleanup any stale processes
pkill -9 -f "p4_job_reverse.py --job [AB] --mode crux --phase main" 2>/dev/null || true
ssh $NODE_226 "pkill -9 -f 'p4_job_reverse.py --job [AB] --mode crux --phase main'" 2>/dev/null || true
sleep 2
rm -f /tmp/nccl_j[AB]_crux_* p4_job[AB]_reverse_crux_rank*_*.csv
ssh $NODE_226 "rm -f /tmp/nccl_j[AB]_crux_* p4_job[AB]_reverse_crux_rank*_*.csv" 2>/dev/null || true

# Job A rank 0 on 10.1
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JA \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_JA \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    NCCL_DEBUG_FILE=/tmp/nccl_jA_crux_v6cal_101_%h_%p.log \
    python3 -u p4_job_reverse.py --job A --mode crux --phase main \
        --ttarget-file $TTARGET_A \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobA_reverse_crux_v6cal_node101.log 2>&1 &
JOB_A_101_PID=$!

# Job A rank 1 on 226
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JA \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_JA \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    NCCL_DEBUG_FILE=/tmp/nccl_jA_crux_v6cal_226_%h_%p.log \
    python3 -u p4_job_reverse.py --job A --mode crux --phase main \
        --ttarget-file $TTARGET_A \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobA_reverse_crux_v6cal_node226.log 2>&1" &
JOB_A_226_PID=$!

echo "Job A launched, waiting 10s for init..."
sleep 10

# Job B rank 0 on 10.1
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JB \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_JB \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    NCCL_DEBUG_FILE=/tmp/nccl_jB_crux_v6cal_101_%h_%p.log \
    python3 -u p4_job_reverse.py --job B --mode crux --phase main \
        --ttarget-file $TTARGET_B \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobB_reverse_crux_v6cal_node101.log 2>&1 &
JOB_B_101_PID=$!

# Job B rank 1 on 226
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_JB \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_JB \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    NCCL_DEBUG_FILE=/tmp/nccl_jB_crux_v6cal_226_%h_%p.log \
    python3 -u p4_job_reverse.py --job B --mode crux --phase main \
        --ttarget-file $TTARGET_B \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobB_reverse_crux_v6cal_node226.log 2>&1" &
JOB_B_226_PID=$!

echo "Job B launched. Waiting for experiment to complete..."
echo "  (Background flow will run for ${BG_DURATION}s)"
echo ""

# Wait for all jobs
wait $JOB_A_101_PID; JOB_A_101_EXIT=$?
wait $JOB_A_226_PID; JOB_A_226_EXIT=$?
wait $JOB_B_101_PID; JOB_B_101_EXIT=$?
wait $JOB_B_226_PID; JOB_B_226_EXIT=$?

# Kill background flow if still running
kill $BG_PID 2>/dev/null || true
ssh $NODE_226 "pkill -f 'iperf3 -s -p $BG_PORT'" 2>/dev/null || true

echo ""
echo "=================================================================="
echo "Calibration experiment completed"
echo "  Job A exits: $JOB_A_101_EXIT (10.1), $JOB_A_226_EXIT (226)"
echo "  Job B exits: $JOB_B_101_EXIT (10.1), $JOB_B_226_EXIT (226)"
echo "=================================================================="
echo ""

# ============================================================
# Step 2e: Quick analysis
# ============================================================
echo "=== Background flow stats ==="
tail -5 /tmp/v6_bgflow_101.log 2>/dev/null || echo "no bgflow log"
echo ""

echo "=== CRUX tight job (Phase 2, Job B) per-epoch ==="
if [[ -f "p4_jobB_reverse_crux_v6cal_rank0_epoch.csv" ]]; then
    cat p4_jobB_reverse_crux_v6cal_rank0_epoch.csv
else
    echo "CSV not found"
fi
echo ""

echo "=== CRUX tight job (Phase 1, Job A) per-epoch ==="
if [[ -f "p4_jobA_reverse_crux_v6cal_rank0_epoch.csv" ]]; then
    cat p4_jobA_reverse_crux_v6cal_rank0_epoch.csv
else
    echo "CSV not found"
fi
echo ""

echo "Calibration data saved. Analyze slowdown vs V5 baseline."
