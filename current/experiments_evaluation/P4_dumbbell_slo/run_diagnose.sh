#!/bin/bash
# ============================================================================
# Isolation Point Diagnosis: P0 (Job A) vs P6 (Job B)
# ============================================================================
# Goal: determine if contention is at switch (DSCP matters) or host NIC.
#   - Job A: heavy (2048MB), static P0 (DSCP=40)
#   - Job B: light (256MB), static P6 (DSCP=8)
#   - Both run contested for 10 epochs
#   - Measure B's comm time vs solo baseline (34.7ms)
#   - Expansion > 20% → host NIC contention
# ============================================================================

set -e

export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
export NCCL_PROTO=SIMPLE
export LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:$PYTHONPATH

NODE_226="192.10.10.226"
EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"

PORT_JA=29520
PORT_JB=29521

echo "=========================================="
echo "Isolation Point Diagnosis: P0 vs P6"
echo "  Job A: heavy(2048MB), P0 (DSCP=40)"
echo "  Job B: light(256MB),  P6 (DSCP=8)"
echo "  date: $(date -Iseconds)"
echo "=========================================="
echo ""

# Cleanup
pkill -9 -f "diagnose_isolation" 2>/dev/null || true
ssh $NODE_226 "pkill -9 -f 'diagnose_isolation'" 2>/dev/null || true
sleep 2

# Job A rank 0 on 10.1
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_JA \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_JA \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_DEBUG_FILE=/tmp/nccl_diagA_101_%h_%p.log \
    python3 -u diagnose_isolation.py --job A \
        > diag_jobA_node101.log 2>&1 &
JOB_A_101_PID=$!

# Job A rank 1 on 226
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_JA \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_JA \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    NCCL_DEBUG_FILE=/tmp/nccl_diagA_226_%h_%p.log \
    python3 -u diagnose_isolation.py --job A \
        > diag_jobA_node226.log 2>&1" &
JOB_A_226_PID=$!

echo "Job A launched (P0, heavy 2048MB)"
echo "Waiting 10s for Job A to initialize..."
sleep 10

# Job B rank 0 on 10.1
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_JB \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_JB \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_DEBUG_FILE=/tmp/nccl_diagB_101_%h_%p.log \
    python3 -u diagnose_isolation.py --job B \
        > diag_jobB_node101.log 2>&1 &
JOB_B_101_PID=$!

# Job B rank 1 on 226
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_JB \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_JB \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    NCCL_DEBUG_FILE=/tmp/nccl_diagB_226_%h_%p.log \
    python3 -u diagnose_isolation.py --job B \
        > diag_jobB_node226.log 2>&1" &
JOB_B_226_PID=$!

echo "Job B launched (P6, light 256MB)"
echo ""
echo "Running diagnosis (10 epochs each)..."
echo ""

wait $JOB_A_101_PID; JOB_A_101_EXIT=$?
wait $JOB_A_226_PID; JOB_A_226_EXIT=$?
wait $JOB_B_101_PID; JOB_B_101_EXIT=$?
wait $JOB_B_226_PID; JOB_B_226_EXIT=$?

echo ""
echo "=========================================="
echo "Diagnosis completed"
echo "  Job A exit codes: $JOB_A_101_EXIT (10.1), $JOB_A_226_EXIT (226)"
echo "  Job B exit codes: $JOB_B_101_EXIT (10.1), $JOB_B_226_EXIT (226)"
echo "=========================================="
echo ""

echo "=== Job B per-epoch comm time (P6, light 256MB) ==="
if [[ -f "diag_isolation_jobB_rank0.csv" ]]; then
    cat diag_isolation_jobB_rank0.csv
fi
echo ""

echo "=== Job A per-epoch comm time (P0, heavy 2048MB) ==="
if [[ -f "diag_isolation_jobA_rank0.csv" ]]; then
    cat diag_isolation_jobA_rank0.csv
fi
echo ""

echo "Logs: diag_job[AB]_node[101|226].log"
echo "CSVs: diag_isolation_job[AB]_rank0.csv"
