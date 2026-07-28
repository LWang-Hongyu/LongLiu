#!/bin/bash
# P4 V3: Asymmetric Workload Experiment
# Job1: 30ms compute, c_i=1.2 (strict SLO, communication-intensive)
# Job2: 80ms compute, c_i=2.0 (loose SLO, computation-intensive)
#
# This tests whether LongLiu can dynamically prioritize Job1's strict SLO
# despite Job2 having higher GPU Intensity (which CRUX would favor)

set -e

MODE=${1:-longliu}  # longliu or crux

if [[ "$MODE" != "longliu" && "$MODE" != "crux" ]]; then
    echo "Usage: $0 <longliu|crux>"
    exit 1
fi

echo "=========================================="
echo "P4 V3 Asymmetric Workload Experiment"
echo "Mode: $MODE"
echo "=========================================="
echo ""
echo "Job1: 30ms compute, c_i=1.2 (strict SLO, comm-intensive)"
echo "Job2: 80ms compute, c_i=2.0 (loose SLO, compute-intensive)"
echo ""

# Cleanup
pkill -9 -f "p4_job[12]_asym.py" 2>/dev/null || true
ssh 192.10.10.226 "pkill -9 -f 'p4_job[12]_asym.py'" 2>/dev/null || true
sleep 2

# Clear NCCL logs (only /tmp, not /dev/shm)
rm -f /tmp/nccl*
ssh 192.10.10.226 "rm -f /tmp/nccl*" 2>/dev/null || true

# Remove old CSV files
rm -f p4_job1_asym_${MODE}_rank0.csv p4_job2_asym_${MODE}_rank0.csv
ssh 192.10.10.226 "rm -f p4_job1_asym_${MODE}_rank0.csv p4_job2_asym_${MODE}_rank0.csv" 2>/dev/null || true

# Environment variables for MultiComm
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
export NCCL_PROTO=SIMPLE

# Use NCCL 2.30.7 for both modes (needed for trafficClass API)
export LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:$PYTHONPATH

# MultiComm ports
export MULTI_COMM_PORT_JOB1=29510
export MULTI_COMM_PORT_JOB2=29511

# Launch Job1 on 10.1 (rank 0)
echo "Launching Job1 on 10.1 (rank 0)..."
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$MULTI_COMM_PORT_JOB1 \
    WORLD_SIZE=2 RANK=0 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_DEBUG_FILE=/tmp/nccl_j1_101_%h_%p.log \
    python3 -u p4_job1_asym.py --mode $MODE > p4_job1_asym_${MODE}_node101.log 2>&1 &
JOB1_PID=$!

# Wait a bit for Job1 to initialize
sleep 3

# Launch Job2 on 226 (rank 1)
echo "Launching Job2 on 226 (rank 1)..."
ssh 192.10.10.226 "cd /home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$MULTI_COMM_PORT_JOB1 \
    WORLD_SIZE=2 RANK=1 \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    MULTI_COMM_PORT=$MULTI_COMM_PORT_JOB1 \
    NCCL_DEBUG_FILE=/tmp/nccl_j2_226_%h_%p.log \
    python3 -u p4_job2_asym.py --mode $MODE > p4_job2_asym_${MODE}_node226.log 2>&1" &
JOB2_PID=$!

echo ""
echo "Job1 PID: $JOB1_PID"
echo "Job2 PID: $JOB2_PID"
echo ""
echo "Running experiment... (this will take ~5-10 minutes)"
echo ""

# Wait for both jobs to complete
wait $JOB1_PID
JOB1_EXIT=$?

wait $JOB2_PID
JOB2_EXIT=$?

echo ""
echo "=========================================="
echo "Experiment completed"
echo "Job1 exit code: $JOB1_EXIT"
echo "Job2 exit code: $JOB2_EXIT"
echo "=========================================="
echo ""

# Display results
if [[ -f "p4_job1_asym_${MODE}_rank0.csv" ]]; then
    echo "Job1 results (last 10 iterations):"
    tail -n 11 p4_job1_asym_${MODE}_rank0.csv
    echo ""
fi

if [[ -f "p4_job2_asym_${MODE}_rank0.csv" ]]; then
    echo "Job2 results (last 10 iterations):"
    tail -n 11 p4_job2_asym_${MODE}_rank0.csv
    echo ""
fi

echo "Results saved to:"
echo "  - p4_job1_asym_${MODE}_rank0.csv"
echo "  - p4_job2_asym_${MODE}_rank0.csv"
echo ""
echo "Logs saved to:"
echo "  - p4_job1_asym_${MODE}_node101.log"
echo "  - p4_job2_asym_${MODE}_node226.log"
