#!/bin/bash
# ============================================================================
# P4 V3: Asymmetric Workload — Two Independent Jobs
# ============================================================================
# Job1: 30ms compute, c_i=1.2 (strict SLO, communication-intensive)
# Job2: 80ms compute, c_i=2.0 (loose SLO, computation-intensive)
#
# 两个独立的分布式 Job：
# - Job1: rank 0 @ 10.1, rank 1 @ 226, PORT=29510
# - Job2: rank 0 @ 10.1, rank 1 @ 226, PORT=29511
#
# CRUX 模式：
# - Job1: P3 (DSCP=24) - 低 GPU Intensity
# - Job2: P4 (DSCP=32) - 高 GPU Intensity
#
# LongLiu 模式：
# - Job1: 动态调整优先级，保障严格 SLO
# - Job2: 动态调整优先级，宽松 SLO
# ============================================================================

set -e

MODE=${1:-longliu}

if [[ "$MODE" != "longliu" && "$MODE" != "crux" ]]; then
    echo "Usage: $0 <longliu|crux>"
    exit 1
fi

echo "=========================================="
echo "P4 V3 Asymmetric Workload (Two Jobs)"
echo "Mode: $MODE"
echo "=========================================="
echo ""
echo "Job1: 30ms compute, c_i=1.2 (strict SLO, comm-intensive)"
echo "Job2: 80ms compute, c_i=2.0 (loose SLO, compute-intensive)"
echo ""

# Cleanup
pkill -9 -f "p4_job[12]_asym" 2>/dev/null || true
ssh 192.10.10.226 "pkill -9 -f 'p4_job[12]_asym'" 2>/dev/null || true
sleep 2

# Clear NCCL logs
rm -f /tmp/nccl*
ssh 192.10.10.226 "rm -f /tmp/nccl*" 2>/dev/null || true

# Remove old CSV files
rm -f p4_job1_asym_${MODE}_rank*.csv p4_job2_asym_${MODE}_rank*.csv
ssh 192.10.10.226 "rm -f p4_job1_asym_${MODE}_rank*.csv p4_job2_asym_${MODE}_rank*.csv" 2>/dev/null || true

# Ports
PORT_J1=29510
PORT_J2=29511

# Environment variables
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
export NCCL_PROTO=SIMPLE

# Use NCCL 2.30.7 for both modes
export LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:$PYTHONPATH

echo "=========================================="
echo "Phase 1: Launching Job1 (30ms, c_i=1.2)"
echo "=========================================="

# Job1 rank 0 on 10.1
echo "Launching Job1 rank 0 on 10.1..."
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_J1 \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_J1 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_DEBUG_FILE=/tmp/nccl_j1_101_%h_%p.log \
    python3 -u p4_job1_asym.py --mode $MODE > p4_job1_asym_${MODE}_node101.log 2>&1 &
JOB1_101_PID=$!

# Job1 rank 1 on 226
echo "Launching Job1 rank 1 on 226..."
ssh 192.10.10.226 "cd /home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_J1 \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_J1 \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    NCCL_DEBUG_FILE=/tmp/nccl_j1_226_%h_%p.log \
    python3 -u p4_job1_asym.py --mode $MODE > p4_job1_asym_${MODE}_node226.log 2>&1" &
JOB1_226_PID=$!

echo "Job1 launched (PIDs: $JOB1_101_PID on 10.1, $JOB1_226_PID on 226)"
echo ""

# Wait for Job1 to initialize
echo "Waiting 10s for Job1 to initialize..."
sleep 10

echo "=========================================="
echo "Phase 2: Launching Job2 (80ms, c_i=2.0)"
echo "=========================================="

# Job2 rank 0 on 10.1
echo "Launching Job2 rank 0 on 10.1..."
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_J2 \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=$PORT_J2 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_DEBUG_FILE=/tmp/nccl_j2_101_%h_%p.log \
    python3 -u p4_job2_asym.py --mode $MODE > p4_job2_asym_${MODE}_node101.log 2>&1 &
JOB2_101_PID=$!

# Job2 rank 1 on 226
echo "Launching Job2 rank 1 on 226..."
ssh 192.10.10.226 "cd /home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_J2 \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=$PORT_J2 \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    NCCL_DEBUG_FILE=/tmp/nccl_j2_226_%h_%p.log \
    python3 -u p4_job2_asym.py --mode $MODE > p4_job2_asym_${MODE}_node226.log 2>&1" &
JOB2_226_PID=$!

echo "Job2 launched (PIDs: $JOB2_101_PID on 10.1, $JOB2_226_PID on 226)"
echo ""

echo "=========================================="
echo "All jobs running..."
echo "=========================================="
echo ""
echo "Job1 PIDs: $JOB1_101_PID (10.1), $JOB1_226_PID (226)"
echo "Job2 PIDs: $JOB2_101_PID (10.1), $JOB2_226_PID (226)"
echo ""
echo "Running experiment... (this will take ~10-15 minutes)"
echo ""

# Wait for all jobs to complete
wait $JOB1_101_PID
JOB1_101_EXIT=$?

wait $JOB1_226_PID
JOB1_226_EXIT=$?

wait $JOB2_101_PID
JOB2_101_EXIT=$?

wait $JOB2_226_PID
JOB2_226_EXIT=$?

echo ""
echo "=========================================="
echo "Experiment completed"
echo "Job1 exit codes: $JOB1_101_EXIT (10.1), $JOB1_226_EXIT (226)"
echo "Job2 exit codes: $JOB2_101_EXIT (10.1), $JOB2_226_EXIT (226)"
echo "=========================================="
echo ""

# Display results
echo "=== Job1 Results ==="
if [[ -f "p4_job1_asym_${MODE}_rank0.csv" ]]; then
    echo "Job1 rank 0 (last 10 iterations):"
    tail -n 11 p4_job1_asym_${MODE}_rank0.csv
    echo ""
fi

echo "=== Job2 Results ==="
if [[ -f "p4_job2_asym_${MODE}_rank0.csv" ]]; then
    echo "Job2 rank 0 (last 10 iterations):"
    tail -n 11 p4_job2_asym_${MODE}_rank0.csv
    echo ""
fi

echo "Results saved to:"
echo "  - p4_job1_asym_${MODE}_rank*.csv"
echo "  - p4_job2_asym_${MODE}_rank*.csv"
echo ""
echo "Logs saved to:"
echo "  - p4_job1_asym_${MODE}_node101.log"
echo "  - p4_job1_asym_${MODE}_node226.log"
echo "  - p4_job2_asym_${MODE}_node101.log"
echo "  - p4_job2_asym_${MODE}_node226.log"
