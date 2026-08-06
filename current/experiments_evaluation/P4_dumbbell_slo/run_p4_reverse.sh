#!/bin/bash
# ============================================================================
# P4-1 Role-Reversal Experiment (SP queue, scheduler v1(π), c_i swap, same payload)
# ============================================================================
# Parameterized: supports V4 (512MB) and V5 (1GB) via VERSION arg
#   V4: payload=512MB, c_i 1.6/3.0 (transitional violation, proof of concept)
#   V5: payload=1024MB, c_i 1.7/3.0 (structural violation, paper-grade)
# ============================================================================
set -e

# ============================================================
# Args
# ============================================================
# Usage: bash run_p4_reverse.sh <version> <longliu|crux|both> [skip_calib]
#   version = v4 (512MB, c_i=1.6/3.0) or v5 (1024MB, c_i=1.7/3.0)
VERSION=${1:-v4}
MODE_ARG=${2:-both}
SKIP_CALIB=${3:-0}

if [[ "$VERSION" != "v4" && "$VERSION" != "v5" ]]; then
    echo "Usage: $0 <v4|v5> <longliu|crux|both> [skip_calib=0]"
    echo "  v4: payload=512MB, c_i=1.6/3.0"
    echo "  v5: payload=1024MB, c_i=1.7/3.0 (structural violation)"
    exit 1
fi

if [[ "$MODE_ARG" != "longliu" && "$MODE_ARG" != "crux" && "$MODE_ARG" != "both" ]]; then
    echo "Usage: $0 <v4|v5> <longliu|crux|both> [skip_calib=0]"
    exit 1
fi

# ============================================================
# Workload config from VERSION (parameterized — no hardcoded constants)
# ============================================================
case "$VERSION" in
    v4)
        PAYLOAD_MB=512
        CI_TIGHT=1.6
        CI_LOOSE=3.0
        ;;
    v5)
        PAYLOAD_MB=1024
        CI_TIGHT=1.7
        CI_LOOSE=3.0
        ;;
esac

# Port assignments (parameterized — no hardcoded constants)
PORT_CALIB_A=29510
PORT_CALIB_B=29511
PORT_JA=29520
PORT_JB=29521

# T_target files (version-specific)
TTARGET_A="/tmp/ttarget_${VERSION}_jobA.json"
TTARGET_B="/tmp/ttarget_${VERSION}_jobB.json"

NODE_226="192.10.10.226"
EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"

echo "=========================================="
echo "P4-1 Role-Reversal ${VERSION^^} (c_i swap, same payload)"
echo "  scheduler = v1(pi) / CRUX-static"
echo "  queue     = SP"
echo "  payload   = ${PAYLOAD_MB}MB x 2 (same for both jobs)"
echo "  c_i       = ${CI_TIGHT}/${CI_LOOSE} (swap at epoch 7)"
echo "  iters     = 300 (15 epochs x 20 iters/epoch)"
echo "  reversal  = c_i swap at epoch 7"
echo "  T_target  = solo pre-learning (Phase 0)"
echo "  date      = $(date -Iseconds)"
echo "=========================================="
echo ""

# ============================================================
# Phase 0: Solo calibration (sequential, shared across modes)
# ============================================================
if [[ "$SKIP_CALIB" != "1" ]]; then
    echo "=========================================="
    echo "Phase 0a: Calibrate Job A (solo, ${PAYLOAD_MB}MB, 5 epochs)"
    echo "=========================================="
    # Cleanup
    pkill -9 -f "p4_job_reverse.py --job A --phase calibrate" 2>/dev/null || true
    ssh $NODE_226 "pkill -9 -f 'p4_job_reverse.py --job A --phase calibrate'" 2>/dev/null || true
    sleep 2
    rm -f $TTARGET_A

    # Job A rank 0 on 10.1
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_CALIB_A \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_CALIB_A \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_DEBUG_FILE=/tmp/nccl_calibA_101_%h_%p.log \
        python3 -u p4_job_reverse.py --job A --mode longliu --phase calibrate \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobA_calib_node101.log 2>&1 &
    JOB_A_101_PID=$!

    # Job A rank 1 on 226
    ssh $NODE_226 "cd $EXP_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_CALIB_A \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_CALIB_A \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
        NCCL_DEBUG_FILE=/tmp/nccl_calibA_226_%h_%p.log \
        python3 -u p4_job_reverse.py --job A --mode longliu --phase calibrate \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobA_calib_node226.log 2>&1" &
    JOB_A_226_PID=$!

    echo "Job A calibration launched (PIDs: $JOB_A_101_PID on 10.1, $JOB_A_226_PID on 226)"
    wait $JOB_A_101_PID; JOB_A_101_EXIT=$?
    wait $JOB_A_226_PID; JOB_A_226_EXIT=$?
    echo "Job A calibration done (exits: $JOB_A_101_EXIT, $JOB_A_226_EXIT)"

    if [[ ! -f "$TTARGET_A" ]]; then
        echo "ERROR: T_target file $TTARGET_A not created. Aborting."
        exit 1
    fi
    # Copy T_target to 226 so rank 1 can read it in main phase
    scp -q $TTARGET_A $NODE_226:$TTARGET_A
    echo "T_target_A: $(cat $TTARGET_A)"
    echo "  (copied to $NODE_226:$TTARGET_A)"
    echo ""

    echo "=========================================="
    echo "Phase 0b: Calibrate Job B (solo, ${PAYLOAD_MB}MB, 5 epochs)"
    echo "=========================================="
    sleep 3  # let NCCL clean up
    pkill -9 -f "p4_job_reverse.py --job B --phase calibrate" 2>/dev/null || true
    ssh $NODE_226 "pkill -9 -f 'p4_job_reverse.py --job B --phase calibrate'" 2>/dev/null || true
    sleep 2
    rm -f $TTARGET_B

    # Job B rank 0 on 10.1
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_CALIB_B \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_CALIB_B \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_DEBUG_FILE=/tmp/nccl_calibB_101_%h_%p.log \
        python3 -u p4_job_reverse.py --job B --mode longliu --phase calibrate \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobB_calib_node101.log 2>&1 &
    JOB_B_101_PID=$!

    # Job B rank 1 on 226
    ssh $NODE_226 "cd $EXP_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_CALIB_B \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_CALIB_B \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
        NCCL_DEBUG_FILE=/tmp/nccl_calibB_226_%h_%p.log \
        python3 -u p4_job_reverse.py --job B --mode longliu --phase calibrate \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobB_calib_node226.log 2>&1" &
    JOB_B_226_PID=$!

    echo "Job B calibration launched (PIDs: $JOB_B_101_PID on 10.1, $JOB_B_226_PID on 226)"
    wait $JOB_B_101_PID; JOB_B_101_EXIT=$?
    wait $JOB_B_226_PID; JOB_B_226_EXIT=$?
    echo "Job B calibration done (exits: $JOB_B_101_EXIT, $JOB_B_226_EXIT)"

    if [[ ! -f "$TTARGET_B" ]]; then
        echo "ERROR: T_target file $TTARGET_B not created. Aborting."
        exit 1
    fi
    # Copy T_target to 226 so rank 1 can read it in main phase
    scp -q $TTARGET_B $NODE_226:$TTARGET_B
    echo "T_target_B: $(cat $TTARGET_B)"
    echo "  (copied to $NODE_226:$TTARGET_B)"
    echo ""
else
    echo "Skipping Phase 0 calibration (SKIP_CALIB=1)"
    if [[ ! -f "$TTARGET_A" || ! -f "$TTARGET_B" ]]; then
        echo "ERROR: T_target files missing on 10.1. Run without SKIP_CALIB first."
        exit 1
    fi
    # Ensure T_target files also exist on 226 (rank 1 needs them)
    ssh $NODE_226 "test -f $TTARGET_A" || scp -q $TTARGET_A $NODE_226:$TTARGET_A
    ssh $NODE_226 "test -f $TTARGET_B" || scp -q $TTARGET_B $NODE_226:$TTARGET_B
    echo "Reusing: T_target_A=$(cat $TTARGET_A)"
    echo "Reusing: T_target_B=$(cat $TTARGET_B)"
    echo "  (ensured both files exist on $NODE_226)"
    echo ""
fi

# ============================================================
# Phase 1+2: Main experiment runner
# ============================================================
run_main_experiment() {
    local MODE=$1
    local SCHEDULER_LABEL
    if [[ "$MODE" == "longliu" ]]; then
        SCHEDULER_LABEL="v1(pi)"
    else
        SCHEDULER_LABEL="CRUX-static"
    fi

    echo "=========================================="
    echo "Phase 1+2: Main experiment — Mode=$MODE (scheduler=$SCHEDULER_LABEL)"
    echo "=========================================="

    # Cleanup
    pkill -9 -f "p4_job_reverse.py --job [AB] --mode $MODE --phase main" 2>/dev/null || true
    ssh $NODE_226 "pkill -9 -f 'p4_job_reverse.py --job [AB] --mode $MODE --phase main'" 2>/dev/null || true
    sleep 2
    rm -f /tmp/nccl_j[AB]_${MODE}_*
    ssh $NODE_226 "rm -f /tmp/nccl_j[AB]_${MODE}_*" 2>/dev/null || true
    rm -f p4_job[AB]_reverse_${MODE}_rank*_*.csv
    ssh $NODE_226 "rm -f p4_job[AB]_reverse_${MODE}_rank*_*.csv" 2>/dev/null || true

    # Job A rank 0 on 10.1
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_JA \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_JA \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_DEBUG_FILE=/tmp/nccl_jA_${MODE}_101_%h_%p.log \
        python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobA_reverse_${MODE}_node101.log 2>&1 &
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
        NCCL_DEBUG_FILE=/tmp/nccl_jA_${MODE}_226_%h_%p.log \
        python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobA_reverse_${MODE}_node226.log 2>&1" &
    JOB_A_226_PID=$!

    echo "Job A launched (PIDs: $JOB_A_101_PID on 10.1, $JOB_A_226_PID on 226)"
    echo "Waiting 10s for Job A to initialize..."
    sleep 10

    # Job B rank 0 on 10.1
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=$PORT_JB \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_JB \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_DEBUG_FILE=/tmp/nccl_jB_${MODE}_101_%h_%p.log \
        python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobB_reverse_${MODE}_node101.log 2>&1 &
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
        NCCL_DEBUG_FILE=/tmp/nccl_jB_${MODE}_226_%h_%p.log \
        python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            > p4_jobB_reverse_${MODE}_node226.log 2>&1" &
    JOB_B_226_PID=$!

    echo "Job B launched (PIDs: $JOB_B_101_PID on 10.1, $JOB_B_226_PID on 226)"
    echo ""
    echo "Running $MODE experiment (scheduler=$SCHEDULER_LABEL, queue=SP)..."
    echo "Job A: ${PAYLOAD_MB}MB fixed, c_i ${CI_TIGHT}→${CI_LOOSE}; Job B: ${PAYLOAD_MB}MB fixed, c_i ${CI_LOOSE}→${CI_TIGHT}"
    echo "c_i swap at epoch 7 (same payload, CRUX blind to GPU intensity tie)"
    echo ""

    # Wait for all to complete (NO early exit — both run all 15 epochs)
    wait $JOB_A_101_PID; JOB_A_101_EXIT=$?
    wait $JOB_A_226_PID; JOB_A_226_EXIT=$?
    wait $JOB_B_101_PID; JOB_B_101_EXIT=$?
    wait $JOB_B_226_PID; JOB_B_226_EXIT=$?

    echo ""
    echo "=========================================="
    echo "$MODE experiment completed"
    echo "  scheduler = $SCHEDULER_LABEL"
    echo "  queue     = SP"
    echo "  payload   = ${PAYLOAD_MB}MB x 2 (same)"
    echo "  c_i       = ${CI_TIGHT}/${CI_LOOSE} (swapped at epoch 7)"
    echo "  T_target  = solo pre-learning"
    echo "  date      = $(date -Iseconds)"
    echo "Job A exit codes: $JOB_A_101_EXIT (10.1), $JOB_A_226_EXIT (226)"
    echo "Job B exit codes: $JOB_B_101_EXIT (10.1), $JOB_B_226_EXIT (226)"
    echo "=========================================="
    echo ""

    echo "=== Job A per-epoch summary ==="
    if [[ -f "p4_jobA_reverse_${MODE}_rank0_epoch.csv" ]]; then
        cat p4_jobA_reverse_${MODE}_rank0_epoch.csv
    fi
    echo ""
    echo "=== Job B per-epoch summary ==="
    if [[ -f "p4_jobB_reverse_${MODE}_rank0_epoch.csv" ]]; then
        cat p4_jobB_reverse_${MODE}_rank0_epoch.csv
    fi
    echo ""
    echo "Logs: p4_job[AB]_reverse_${MODE}_node[101|226].log"
    echo "CSVs: p4_job[AB]_reverse_${MODE}_rank*_*.csv"
    echo ""
}

# ============================================================
# Run main experiments
# ============================================================
if [[ "$MODE_ARG" == "longliu" || "$MODE_ARG" == "both" ]]; then
    run_main_experiment longliu
fi

if [[ "$MODE_ARG" == "crux" || "$MODE_ARG" == "both" ]]; then
    echo "Waiting 15s between modes for NCCL cleanup..."
    sleep 15
    run_main_experiment crux
fi

echo "=========================================="
echo "All experiments completed."
echo "T_target_A: $(cat $TTARGET_A 2>/dev/null)"
echo "T_target_B: $(cat $TTARGET_B 2>/dev/null)"
echo "=========================================="
