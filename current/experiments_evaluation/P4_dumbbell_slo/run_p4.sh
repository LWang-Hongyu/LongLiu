#!/bin/bash
# ============================================================================
# P4 V3: Synchronous Competition — Launch Script (Multi-Comm version)
# ============================================================================
# Two NCCL jobs with identical 2048MB payload and 50ms compute.
# AllReduce pulses naturally synchronize → collide at switch CoS queues.
# Job 1 = 15 epochs (tight SLO c_i=1.5), Job 2 = 10 epochs (loose c_i=2.5)
# Job 2 joins after Job 1 epoch 5 (~10s delay).
#
# Usage: bash run_p4.sh <mode> [schedule]
#   mode ∈ {solo, fair, longliu, crux, train_gpt}
#   schedule ∈ {solo, fair, longliu}  (only for train_gpt mode, default: longliu)
#
# Solo       — Only Job 1, standard NCCL  (baseline)
# Fair       — Both jobs, standard NCCL, no prioritization
# LongLiu    — Both jobs, MultiCommWrapper + NCCL 2.30.7 trafficClass (dynamic)
# CRUX       — Both jobs, MultiCommWrapper + static GPU Intensity priority (fixed)
# train_gpt  — Use real GPT-2 model instead of synthetic payload
# ============================================================================

set -e

MODE="${1:-longliu}"
SCHEDULE="${2:-}"

if [[ "$MODE" == "train_gpt" ]]; then
    if [[ -n "$SCHEDULE" ]]; then
        if [[ ! "$SCHEDULE" =~ ^(solo|fair|longliu)$ ]]; then
            echo "ERROR: Invalid schedule '$SCHEDULE' for train_gpt. Choose: solo fair longliu"
            exit 1
        fi
    fi
elif [[ ! "$MODE" =~ ^(solo|fair|longliu|crux)$ ]]; then
    echo "ERROR: Invalid mode '$MODE'. Choose: solo fair longliu crux train_gpt"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_ADDR="192.10.10.110"
NODE_226="192.10.10.226"
NODE_101_IF="enp130s0f0np0"
NODE_226_IF="enp59s0f0np0"

# NCCL with trafficClass API — used only in longliu mode.
# NOTE: nccl-master/build 目录在两节点均不存在。各节点 fallback 到系统 NCCL：
#   10.1 → /usr/local/lib/libnccl.so.2 (2.30.7)
#   226  → /usr/lib/x86_64-linux-gnu/libnccl.so.2 (2.29.7, 同样支持 trafficClass)
NCCL_2307_LIB="/usr/local/lib"
NCCL_2307_LIB_226="/usr/lib/x86_64-linux-gnu"
# Multi-Comm library path (两节点统一位于 current/ 下，与脚本目录相对位置一致)
MULTI_COMM_DIR="/home/why/LongLiu_rebuild/current/multi_comm_slo"

# PyTorch's bundled NCCL (for fair/solo mode)
PT_NCCL_101="/home/why/.local/lib/python3.8/site-packages/nvidia/nccl/lib"
PT_NCCL_226="/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib"

echo "========================================================"
echo "P4 V3: Synchronous Competition — MODE=$MODE"
if [[ "$MODE" == "train_gpt" ]]; then
    echo "Model: GPT-2 (124M params, 768D/12L), seq=64, batch=1"
    echo "Job1: 300 iters, c_i=1.5; Job2: 200 iters, c_i=2.5"
else
    echo "Job1: 2048MB, 50ms compute, 15 epochs (300 iters), c_i=1.5"
    [[ "$MODE" != "solo" ]] && echo "Job2: 2048MB, 50ms compute, 10 epochs (200 iters), c_i=2.5"
fi
echo "Topology: 10.1 GPU0 ──mlx5_0──[100G]── 226 GPU0 (Job1)"
[[ "$MODE" != "solo" && "$SCHED_MODE" != "solo" ]] && echo "          10.1 GPU0 ──mlx5_0──[100G]── 226 GPU1 (Job2)"
echo "========================================================"

# ============================================================
# Unique ports per run to avoid TIME_WAIT conflicts
# ============================================================
PORT_BASE=$(( 29510 + (RANDOM % 100) * 2 ))
PORT_J1=$PORT_BASE
PORT_J2=$(( PORT_BASE + 1 ))

# Determine script file per mode
if [[ "$MODE" == "train_gpt" ]]; then
    JOB1_SCRIPT="p4_train_gpt.py"
    JOB2_SCRIPT="p4_train_gpt.py"
    # 默认 GPT config（768D/12L ≈ 124M params, 495MB 梯度）+ 小 seq/batch
    # → 计算↓（~30ms/iter），通信占 ~80%+ → 两作业共享链路时真实拥塞 → P6 可触发
    JOB1_ARGS="--job 1 --seq-len 64 --batch-size 1"
    JOB2_ARGS="--job 2 --seq-len 64 --batch-size 1"
    # For train_gpt, the scheduler mode can be specified as 2nd arg (default: longliu)
    SCHED_MODE="${SCHEDULE:-longliu}"
elif [[ "$MODE" == "crux" ]]; then
    JOB1_SCRIPT="p4_job1_crux.py"
    JOB2_SCRIPT="p4_job2_crux.py"
    JOB1_ARGS="--mode crux"
    JOB2_ARGS="--mode crux"
    SCHED_MODE="crux"
else
    JOB1_SCRIPT="p4_job1.py"
    JOB2_SCRIPT="p4_job2.py"
    JOB1_ARGS=""
    JOB2_ARGS=""
    SCHED_MODE="$MODE"
fi

# Cleanup from previous runs
if [[ "$MODE" == "train_gpt" ]]; then
    rm -f /tmp/p4_train_JOB1_"$SCHED_MODE"_rank0.csv /tmp/p4_train_JOB2_"$SCHED_MODE"_rank0.csv
elif [[ "$MODE" == "crux" ]]; then
    rm -f "$SCRIPT_DIR"/p4_job1_crux_rank0.csv
    rm -f "$SCRIPT_DIR"/p4_job2_crux_rank0.csv
else
    rm -f "$SCRIPT_DIR"/p4_job1_"$MODE"_rank0.csv
    rm -f "$SCRIPT_DIR"/p4_job2_"$MODE"_rank0.csv
fi

pkill -9 -f "p4_job[12].py" 2>/dev/null || true
pkill -9 -f "p4_train_gpt.py" 2>/dev/null || true
ssh "$NODE_226" "pkill -9 -f 'p4_job[12].py'" 2>/dev/null || true
ssh "$NODE_226" "pkill -9 -f 'p4_train_gpt.py'" 2>/dev/null || true
rm -f /dev/shm/nccl* /tmp/nccl* 2>/dev/null || true
ssh "$NODE_226" "rm -f /dev/shm/nccl* /tmp/nccl*" 2>/dev/null || true
sleep 70  # wait for TIME_WAIT (default 60s) to expire

# ============================================================
# Configure env vars per mode
# ============================================================
case "$SCHED_MODE" in
    solo)
        J1_EXTRA=""
        J2_EXTRA=""
        ;;
    fair)
        J1_EXTRA=""
        J2_EXTRA=""
        ;;
    longliu|static_dscp|crux)
        # MultiComm mode: use NCCL 2.30.7 + libmulti_comm.so
        J1_EXTRA="MULTI_COMM_PORT=$PORT_J1"
        J2_EXTRA="MULTI_COMM_PORT=$PORT_J2"
        ;;
esac

if [[ "$SCHED_MODE" == "longliu" ]]; then
    MULTI_COMM_MSG="enabled (7 priorities, DSCP 0,8,16,24,32,40)"
elif [[ "$SCHED_MODE" == "crux" ]]; then
    MULTI_COMM_MSG="enabled (static GPU Intensity priority)"
else
    MULTI_COMM_MSG="disabled"
fi
echo "Multi-Comm: $MULTI_COMM_MSG"
echo "Job1 extra: ${J1_EXTRA:-none}"
echo "Job2 extra: ${J2_EXTRA:-none}"

# ============================================================
# Common NCCL env vars
# ============================================================
NCCL_COMMON="NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_NTHREADS=1"

# ============================================================
# Build env strings per job/node
# ============================================================
if [[ "$SCHED_MODE" == "longliu" || "$SCHED_MODE" == "static_dscp" || "$SCHED_MODE" == "crux" ]]; then
    # ---- LongLiu/CRUX: use system NCCL (trafficClass) + libmulti_comm.so ----
    LD_PATH_101="${NCCL_2307_LIB}"
    LD_PATH_226="${NCCL_2307_LIB_226}"
    PYTHONPATH_EXTRA="${MULTI_COMM_DIR}/src"
    PYTHONPATH_VAR="PYTHONPATH=${PYTHONPATH_EXTRA}:\${PYTHONPATH:-}"
else
    # ---- Fair/Solo: 统一用系统 NCCL 2.30.7（2026-08-17：bundled NCCL 慢 3 倍
    #      且竞争极不公平（实测 8.7 vs 1.1 Gbps），fair 基线与其他模式不可比）----
    LD_PATH_101="${NCCL_2307_LIB}"
    LD_PATH_226="${NCCL_2307_LIB_226}"
    PYTHONPATH_VAR=""
fi

# Job 1, rank 0 on 10.1 (GPU 0)
J1_101_ENV="LD_LIBRARY_PATH=${LD_PATH_101}:\$LD_LIBRARY_PATH $PYTHONPATH_VAR $NCCL_COMMON NCCL_SOCKET_IFNAME=$NODE_101_IF NCCL_DEBUG_FILE=/tmp/nccl_j1_101_%h_%p.log $J1_EXTRA"
# Job 1, rank 1 on 226 (GPU 0)
J1_226_ENV="LD_LIBRARY_PATH=${LD_PATH_226}:\$LD_LIBRARY_PATH $PYTHONPATH_VAR CUDA_VISIBLE_DEVICES=0 $NCCL_COMMON NCCL_SOCKET_IFNAME=$NODE_226_IF NCCL_DEBUG_FILE=/tmp/nccl_j1_226_%h_%p.log $J1_EXTRA"

# Job 2, rank 0 on 10.1 (GPU 0, shared with Job 1)
J2_101_ENV="LD_LIBRARY_PATH=${LD_PATH_101}:\$LD_LIBRARY_PATH $PYTHONPATH_VAR $NCCL_COMMON NCCL_SOCKET_IFNAME=$NODE_101_IF NCCL_DEBUG_FILE=/tmp/nccl_j2_101_%h_%p.log $J2_EXTRA"
# Job 2, rank 1 on 226 (GPU 0 shared — GPU 1 unavailable to PyTorch)
J2_226_ENV="LD_LIBRARY_PATH=${LD_PATH_226}:\$LD_LIBRARY_PATH $PYTHONPATH_VAR CUDA_VISIBLE_DEVICES=0 $NCCL_COMMON NCCL_SOCKET_IFNAME=$NODE_226_IF NCCL_DEBUG_FILE=/tmp/nccl_j2_226_%h_%p.log $J2_EXTRA"

# ============================================================
# Launch
# ============================================================
echo ""
echo "=== Phase 1: Launching Job 1 (15 epochs, 300 iters) ==="

# Job 1 rank 0 on 10.1
( cd "$SCRIPT_DIR" && env $J1_101_ENV \
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=$PORT_J1 WORLD_SIZE=2 RANK=0 \
    python3 $JOB1_SCRIPT --mode "$SCHED_MODE" $JOB1_ARGS \
    > /tmp/p4_job1_node101.log 2>&1 ) &
PID_J1_101=$!

# Job 1 rank 1 on 226
ssh "$NODE_226" "cd '$SCRIPT_DIR' && env $J1_226_ENV \
    MASTER_ADDR='$MASTER_ADDR' MASTER_PORT=$PORT_J1 WORLD_SIZE=2 RANK=1 \
    python3 $JOB1_SCRIPT --mode '$SCHED_MODE' $JOB1_ARGS \
    > /tmp/p4_job1_node226.log 2>&1" &
PID_J1_226=$!

echo "Job 1 launched (PIDs: $PID_J1_101 on 10.1, SSH to 226) [port=$PORT_J1]"

if [[ "$MODE" != "solo" && "$SCHED_MODE" != "solo" ]]; then
    # 2026-08-17: sleep 30→50。实测 30s 时 Job2 在 Job1 iter ~26 加入（窗口 1 中途），
    # 污染 warmup 期 comm baseline（窗口 1-6 全部带竞争，min 聚合也救不回 solo 基线），
    # comm_ratio≈1.0，P6 永不触发。50s 保证 Job2 在 Job1 窗口 2 及之后加入
    # （实测 startup≈23s + iter≈0.27s，50s→iter~80-100，慢机最坏也在窗口 2 内），
    # 窗口 1（iters 20-39）保持纯 solo → baseline 干净。
    echo "Waiting 50s (Job1 reaches window 2+, keeps comm-baseline window solo-clean)..."
    sleep 50

    echo ""
    echo "=== Phase 2: Launching Job 2 (10 epochs, 200 iters) ==="

    # Job 2 rank 0 on 10.1
    ( cd "$SCRIPT_DIR" && env $J2_101_ENV \
        MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=$PORT_J2 WORLD_SIZE=2 RANK=0 \
        python3 $JOB2_SCRIPT --mode "$SCHED_MODE" $JOB2_ARGS \
        > /tmp/p4_job2_node101.log 2>&1 ) &
    PID_J2_101=$!

    # Job 2 rank 1 on 226
    ssh "$NODE_226" "cd '$SCRIPT_DIR' && env $J2_226_ENV \
        MASTER_ADDR='$MASTER_ADDR' MASTER_PORT=$PORT_J2 WORLD_SIZE=2 RANK=1 \
        python3 $JOB2_SCRIPT --mode '$SCHED_MODE' $JOB2_ARGS \
        > /tmp/p4_job2_node226.log 2>&1" &
    PID_J2_226=$!

    echo "Job 2 launched (PIDs: $PID_J2_101 on 10.1, SSH to 226)"
fi

echo ""
echo "Waiting for all processes to complete..."
wait

echo ""
echo "=== Experiment ($MODE) completed ==="

# ============================================================
# Display results
# ============================================================
echo ""
echo "--- Job 1 results ---"
if [[ "$MODE" == "train_gpt" ]] && [ -f "/tmp/p4_train_JOB1_${SCHED_MODE}_rank0.csv" ]; then
    column -t -s, "/tmp/p4_train_JOB1_${SCHED_MODE}_rank0.csv" 2>/dev/null | tail -20
elif [ -f "$SCRIPT_DIR/p4_job1_${MODE}_rank0.csv" ]; then
    column -t -s, "$SCRIPT_DIR/p4_job1_${MODE}_rank0.csv" 2>/dev/null | tail -20 || cat "$SCRIPT_DIR/p4_job1_${MODE}_rank0.csv"
else
    echo "WARNING: No Job 1 CSV found. Check /tmp/p4_job1_node101.log"
fi

if [[ "$MODE" != "solo" && "$SCHED_MODE" != "solo" ]]; then
    echo ""
    echo "--- Job 2 results ---"
    if [[ "$MODE" == "train_gpt" ]] && [ -f "/tmp/p4_train_JOB2_${SCHED_MODE}_rank0.csv" ]; then
        column -t -s, "/tmp/p4_train_JOB2_${SCHED_MODE}_rank0.csv" 2>/dev/null | tail -20
    elif [ -f "$SCRIPT_DIR/p4_job2_${MODE}_rank0.csv" ]; then
        column -t -s, "$SCRIPT_DIR/p4_job2_${MODE}_rank0.csv" 2>/dev/null | tail -20 || cat "$SCRIPT_DIR/p4_job2_${MODE}_rank0.csv"
    else
        echo "WARNING: No Job 2 CSV found. Check /tmp/p4_job2_node101.log"
    fi
fi

echo ""
echo "P4 V3 experiment ($MODE) done."
