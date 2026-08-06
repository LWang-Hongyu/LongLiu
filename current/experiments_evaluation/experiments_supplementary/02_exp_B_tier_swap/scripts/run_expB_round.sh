#!/bin/bash
# ============================================================================
# Experiment B: Hardware Tier Swap — Single Round Runner
# ============================================================================
# Runs one complete round: both arms (LongLiu + CRUX) with background flow.
#
# Usage: bash run_expB_round.sh <round_num> [skip_bg=0]
#   round_num: 1-4 (determines job/arm order alternation)
#   skip_bg:   1 = skip background iperf3 flow (default: 0)
#
# Output: CSV + logs archived to data/round_<N>/
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/expB_config.sh"

ROUND_NUM=${1:?Usage: bash run_expB_round.sh <round_num> [skip_bg=0]}
SKIP_BG=${2:-0}

if [[ "$ROUND_NUM" -lt 1 || "$ROUND_NUM" -gt 4 ]]; then
    echo "ERROR: round_num must be 1-4, got $ROUND_NUM"
    exit 1
fi

JOB_ORDER="${ROUND_JOB_ORDER[$ROUND_NUM]}"
ARM_ORDER="${ROUND_ARM_ORDER[$ROUND_NUM]}"
ROUND_LABEL="round${ROUND_NUM}_${JOB_ORDER}_${ARM_ORDER}"

DATA_DIR="${EXP_B_DIR}/data/${ROUND_LABEL}"
LOGS_DIR="${EXP_B_DIR}/logs/${ROUND_LABEL}"

mkdir -p "$DATA_DIR" "$LOGS_DIR"

echo "================================================================"
echo "Experiment B — Tier Swap Round $ROUND_NUM"
echo "  Job order : $JOB_ORDER (which job starts first)"
echo "  Arm order : $ARM_ORDER (which scheduler runs first)"
echo "  Label     : $ROUND_LABEL"
echo "  Date      : $(date -Iseconds)"
echo "  Payload   : ${PAYLOAD_MB}MB, c_i ${CI_PREMIUM}↔${CI_STANDARD}"
echo "  Epochs    : $NUM_EPOCHS ($NUM_ITERS iters), swap at epoch $REVERSE_EPOCH"
echo "  Windows   : W1=[${W1_START}-${W1_END}] W2=[${W2_START}-${W2_END}] W3=[${W3_START}-${W3_END}]"
echo "  BG flow   : $([[ "$SKIP_BG" == "1" ]] && echo 'SKIP' || echo "${BG_NUM_FLOWS}×${BG_RATE_MBPS}Mbps DSCP=P3")"
echo "================================================================"

# ---- Verify prerequisites ----
verify_ttarget || exit 1
record_md5 "$DATA_DIR/md5_job_script.txt"

# ---- Cleanup any leftover jobs ----
cleanup_p4_jobs "pre-round"

# ============================================================
# Background flow setup
# ============================================================
BG_CLIENT_PIDS=""

start_bg_flow() {
    if [[ "$SKIP_BG" == "1" ]]; then
        echo "[BG] Skipping background flow (SKIP_BG=1)"
        return
    fi
    echo ""
    echo "================================================================"
    echo "Starting background flow: ${BG_NUM_FLOWS}×${BG_RATE_MBPS}Mbps UDP DSCP=P3"
    echo "================================================================"
    # Start iperf3 servers on 226
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        ssh -o ConnectTimeout=10 "$NODE_226" "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null || true
    done
    sleep 2
    # Start iperf3 clients on 10.1
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        iperf3 -c "$RDMA_226" -u -b "${BG_RATE_MBPS}M" -t "$BG_DURATION" \
            --tos "$BG_TOS" -p "$PORT" -f g -l 8900 \
            > "$LOGS_DIR/bgflow_${PORT}.log" 2>&1 &
        BG_CLIENT_PIDS="$BG_CLIENT_PIDS $!"
    done
    echo "[BG] Client PIDs: $BG_CLIENT_PIDS"
    sleep 5
    echo "[BG] Background flow active (~$((BG_NUM_FLOWS * BG_RATE_MBPS / 1000)) Gbps)"
}

stop_bg_flow() {
    if [[ "$SKIP_BG" == "1" || -z "$BG_CLIENT_PIDS" ]]; then
        return
    fi
    echo ""
    echo "[BG] Stopping background flow..."
    for PID in $BG_CLIENT_PIDS; do
        kill "$PID" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
        local SRV_PID
        SRV_PID=$(ssh -o ConnectTimeout=10 "$NODE_226" "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
        [[ -n "$SRV_PID" ]] && ssh -o ConnectTimeout=10 "$NODE_226" "kill $SRV_PID" 2>/dev/null || true
    done
    echo "[BG] Background flow stopped."
}

# ============================================================
# Run one arm (LongLiu or CRUX)
# ============================================================
run_arm() {
    local MODE=$1       # longliu or crux
    local MODE_LABEL=$2 # LL or CX

    echo ""
    echo "================================================================"
    echo "Running arm: $MODE_LABEL ($MODE) — Round $ROUND_NUM"
    echo "================================================================"
    cleanup_p4_jobs "pre-$MODE_LABEL"

    # Determine job start order
    local FIRST_JOB SECOND_JOB
    if [[ "$JOB_ORDER" == "AB" ]]; then
        FIRST_JOB="A"; SECOND_JOB="B"
    else
        FIRST_JOB="B"; SECOND_JOB="A"
    fi

    # Mode-specific flags
    local LL_FLAGS="" CRUX_FLAGS=""
    if [[ "$MODE" == "longliu" ]]; then
        LL_FLAGS="--initial-priority $LL_INITIAL_PRIORITY"
    else
        CRUX_FLAGS="--crux-priority-a $CRUX_PRIO_A --crux-priority-b $CRUX_PRIO_B"
    fi

    # Common args
    local COMMON_ARGS="--phase main --payload-mb $PAYLOAD_MB \
        --ci-phase1 $CI_PREMIUM --ci-phase2 $CI_STANDARD \
        --reverse-epoch $REVERSE_EPOCH \
        --num-iters $NUM_ITERS --iters-per-epoch $ITERS_PER_EPOCH \
        --sleep-us $SLEEP_US"

    # ---- Launch FIRST job ----
    local PORT_FIRST PORT_SECOND TTARGET_FIRST TTARGET_SECOND
    if [[ "$FIRST_JOB" == "A" ]]; then
        PORT_FIRST=$PORT_JA; PORT_SECOND=$PORT_JB
        TTARGET_FIRST=$TTARGET_A; TTARGET_SECOND=$TTARGET_B
    else
        PORT_FIRST=$PORT_JB; PORT_SECOND=$PORT_JA
        TTARGET_FIRST=$TTARGET_B; TTARGET_SECOND=$TTARGET_A
    fi

    echo "  Launching Job $FIRST_JOB first (port $PORT_FIRST)..."

    # First job rank 1 on 226
    ssh -o ConnectTimeout=30 "$NODE_226" "cd $P4_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_FIRST \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_FIRST \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
        timeout 900 python3 -u $JOB_SCRIPT --job $FIRST_JOB --mode $MODE \
            --ttarget-file $TTARGET_FIRST $COMMON_ARGS $LL_FLAGS $CRUX_FLAGS" \
        > "$LOGS_DIR/job${FIRST_JOB}_${MODE_LABEL}_226.log" 2>&1 &
    local FIRST_226_PID=$!
    sleep 5

    # First job rank 0 on 10.1
    cd "$P4_DIR"
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_FIRST \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_FIRST \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_DEBUG=WARN \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src \
        timeout 900 python3 -u "$JOB_SCRIPT" --job "$FIRST_JOB" --mode "$MODE" \
            --ttarget-file "$TTARGET_FIRST" $COMMON_ARGS $LL_FLAGS $CRUX_FLAGS \
        > "$LOGS_DIR/job${FIRST_JOB}_${MODE_LABEL}_101.log" 2>&1 &
    local FIRST_101_PID=$!

    echo "  Job $FIRST_JOB launched. Waiting 15s for init..."
    sleep 15

    # ---- Launch SECOND job ----
    echo "  Launching Job $SECOND_JOB (port $PORT_SECOND)..."

    ssh -o ConnectTimeout=30 "$NODE_226" "cd $P4_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_SECOND \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_SECOND \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
        timeout 900 python3 -u $JOB_SCRIPT --job $SECOND_JOB --mode $MODE \
            --ttarget-file $TTARGET_SECOND $COMMON_ARGS $LL_FLAGS $CRUX_FLAGS" \
        > "$LOGS_DIR/job${SECOND_JOB}_${MODE_LABEL}_226.log" 2>&1 &
    local SECOND_226_PID=$!
    sleep 5

    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_SECOND \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_SECOND \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_DEBUG=WARN \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src \
        timeout 900 python3 -u "$JOB_SCRIPT" --job "$SECOND_JOB" --mode "$MODE" \
            --ttarget-file "$TTARGET_SECOND" $COMMON_ARGS $LL_FLAGS $CRUX_FLAGS \
        > "$LOGS_DIR/job${SECOND_JOB}_${MODE_LABEL}_101.log" 2>&1 &
    local SECOND_101_PID=$!

    echo "  Both jobs running ($MODE_LABEL arm). BG flow: $([[ "$SKIP_BG" == "1" ]] && echo 'OFF' || echo 'ON')"

    # ---- Wait for completion ----
    wait "$FIRST_101_PID";  local FIRST_101_EXIT=$?
    wait "$FIRST_226_PID";  local FIRST_226_EXIT=$?
    wait "$SECOND_101_PID"; local SECOND_101_EXIT=$?
    wait "$SECOND_226_PID"; local SECOND_226_EXIT=$?

    echo "  Job $FIRST_JOB  exits: 10.1=$FIRST_101_EXIT, 226=$FIRST_226_EXIT"
    echo "  Job $SECOND_JOB exits: 10.1=$SECOND_101_EXIT, 226=$SECOND_226_EXIT"

    # ---- Archive CSV files ----
    cd "$P4_DIR"
    for JOB in A B; do
        local CSV_EPOCH="p4_job${JOB}_reverse_${MODE}_rank0_epoch.csv"
        local CSV_ITER="p4_job${JOB}_reverse_${MODE}_rank0_iter.csv"
        if [[ -f "$CSV_EPOCH" ]]; then
            mv "$CSV_EPOCH" "$DATA_DIR/job${JOB}_${MODE_LABEL}_epoch.csv"
            echo "  Archived: job${JOB}_${MODE_LABEL}_epoch.csv"
        else
            echo "  WARNING: $CSV_EPOCH not found!"
        fi
        if [[ -f "$CSV_ITER" ]]; then
            mv "$CSV_ITER" "$DATA_DIR/job${JOB}_${MODE_LABEL}_iter.csv"
        fi
    done

    # Print per-epoch summary
    echo ""
    echo "--- $MODE_LABEL arm per-epoch summary ---"
    for JOB in A B; do
        local CSV="$DATA_DIR/job${JOB}_${MODE_LABEL}_epoch.csv"
        if [[ -f "$CSV" ]]; then
            echo "=== Job $JOB ==="
            column -t -s, "$CSV" | head -30
        fi
    done
    echo ""
}

# ============================================================
# Main: run both arms with warmup between
# ============================================================
start_bg_flow

# Determine arm order
if [[ "$ARM_ORDER" == "ll_cx" ]]; then
    run_arm longliu "LL"
    echo "Waiting 120s between arms for NIC cleanup..."
    sleep 120
    run_arm crux "CX"
else
    run_arm crux "CX"
    echo "Waiting 120s between arms for NIC cleanup..."
    sleep 120
    run_arm longliu "LL"
fi

stop_bg_flow

# ---- Final summary ----
echo ""
echo "================================================================"
echo "Round $ROUND_NUM ($ROUND_LABEL) — Complete"
echo "================================================================"
echo "Data archived to: $DATA_DIR/"
ls -la "$DATA_DIR/"
echo ""
echo "Logs archived to: $LOGS_DIR/"
ls -la "$LOGS_DIR/"
echo "================================================================"
