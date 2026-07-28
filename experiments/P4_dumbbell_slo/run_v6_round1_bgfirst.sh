#!/bin/bash
# run_v6_round1_bgfirst.sh
# V6 Round 1: pre-start background flow, then run experiment
set -euo pipefail

EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
PORT_START=6200
PORT_END=6211
NUM_FLOWS=12
RATE_MBPS_PER_FLOW=500

PAYLOAD_MB=1024
CI_TIGHT=1.2
CI_LOOSE=3.0
TTARGET_A="/tmp/ttarget_v5_jobA.json"
TTARGET_B="/tmp/ttarget_v5_jobB.json"
PORT_MAIN_A=29520
PORT_MAIN_B=29521

ROUND_LABEL="round1_LLthenCX"

echo "================================================================"
echo "Step 1: Start iperf3 servers on 226 (ports $PORT_START-$PORT_END)"
echo "================================================================"
for PORT in $(seq $PORT_START $PORT_END); do
    ssh -o ConnectTimeout=30 $NODE_226 "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null || true
done
sleep 2
SRV_COUNT=$(ssh -o ConnectTimeout=30 $NODE_226 "pgrep -c iperf3" 2>/dev/null || echo "0")
echo "  Servers running: $SRV_COUNT"

echo ""
echo "================================================================"
echo "Step 2: Start iperf3 clients on 10.1"
echo "  ${NUM_FLOWS} x ${RATE_MBPS_PER_FLOW}Mbps UDP DSCP=P3, duration 600s"
echo "================================================================"
BG_CLIENT_PIDS=""
for PORT in $(seq $PORT_START $PORT_END); do
    iperf3 -c $RDMA_226 -u -b ${RATE_MBPS_PER_FLOW}M -t 600 \
        --tos 64 -p $PORT -f g -l 8900 \
        > /tmp/v6_bgflow_${ROUND_LABEL}_${PORT}.log 2>&1 &
    BG_CLIENT_PIDS="$BG_CLIENT_PIDS $!"
done
echo "  Client PIDs: $BG_CLIENT_PIDS"

# Quick verification (skip if port busy, use dedicated port)
sleep 5
iperf3 -c $RDMA_226 -u -b 500M -t 3 --tos 64 -p 6210 -f g 2>&1 | tail -3 || echo "  (verification skipped - port maybe busy)"
echo "  BG flow active."

echo ""
echo "================================================================"
echo "Step 3: Run experiment — Round 1 (LL→CX)"
echo "================================================================"

cleanup_jobs() {
    for PID in $(pgrep -f "p4_job_reverse.py --job [AB]" 2>/dev/null); do
        kill $PID 2>/dev/null || true
    done
    ssh -o ConnectTimeout=10 $NODE_226 \
        "for PID in \$(pgrep -f 'p4_job_reverse.py --job [AB]' 2>/dev/null); do kill \$PID 2>/dev/null; done" 2>/dev/null || true
    sleep 2
}

run_mode() {
    local MODE=$1
    local MODE_NAME=$2
    
    echo ""
    echo "--- Running $MODE_NAME ($MODE) ---"
    cleanup_jobs
    
    local LL_FLAGS=""
    if [[ "$MODE" == "longliu" ]]; then
        LL_FLAGS="--initial-priority 3"
    fi
    
    local CRUX_FLAGS=""
    if [[ "$MODE" == "crux" ]]; then
        CRUX_FLAGS="--crux-priority-a 3 --crux-priority-b 3"
    fi
    
    # Job A rank 1 on 226
    ssh -o ConnectTimeout=30 $NODE_226 "cd $EXP_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_MAIN_A \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
        timeout 300 python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS" \
        > p4_jobA_v6_${ROUND_LABEL}_${MODE}_226.log 2>&1 &
    JOB_A_226_PID=$!
    sleep 5
    
    # Job A rank 0 on 10.1
    cd $EXP_DIR
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_A \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_MAIN_A \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_DEBUG=WARN \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src \
        timeout 300 python3 -u p4_job_reverse.py --job A --mode $MODE --phase main \
            --ttarget-file $TTARGET_A \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS \
        > p4_jobA_v6_${ROUND_LABEL}_${MODE}_101.log 2>&1 &
    JOB_A_101_PID=$!
    
    echo "  Job A ($MODE) waiting for init..."
    sleep 10
    
    # Job B rank 1 on 226
    ssh -o ConnectTimeout=30 $NODE_226 "cd $EXP_DIR && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
        WORLD_SIZE=2 RANK=1 \
        MULTI_COMM_PORT=$PORT_MAIN_B \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
        timeout 300 python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS" \
        > p4_jobB_v6_${ROUND_LABEL}_${MODE}_226.log 2>&1 &
    JOB_B_226_PID=$!
    sleep 5
    
    # Job B rank 0 on 10.1
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$PORT_MAIN_B \
        WORLD_SIZE=2 RANK=0 \
        MULTI_COMM_PORT=$PORT_MAIN_B \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_DEBUG=WARN \
        PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src \
        timeout 300 python3 -u p4_job_reverse.py --job B --mode $MODE --phase main \
            --ttarget-file $TTARGET_B \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $CRUX_FLAGS $LL_FLAGS \
        > p4_jobB_v6_${ROUND_LABEL}_${MODE}_101.log 2>&1 &
    JOB_B_101_PID=$!
    
    echo "  Job B ($MODE) launched."
    echo "  (background flow active: 6 Gbps DSCP=P3)"
    
    # Wait for all job processes
    wait $JOB_A_101_PID; echo "  Job A 10.1 done (exit=$?)"
    wait $JOB_A_226_PID; echo "  Job A 226 done (exit=$?)"
    wait $JOB_B_101_PID; echo "  Job B 10.1 done (exit=$?)"
    wait $JOB_B_226_PID; echo "  Job B 226 done (exit=$?)"
    
    # Rename CSV files
    for JOB in A B; do
        CSV_EPOCH="p4_job${JOB}_reverse_${MODE}_rank0_epoch.csv"
        CSV_ITER="p4_job${JOB}_reverse_${MODE}_rank0_iter.csv"
        CSV_LOG="p4_job${JOB}_reverse_${MODE}_rank0.log"
        [[ -f "$CSV_EPOCH" ]] && mv "$CSV_EPOCH" "p4_job${JOB}_v6_${ROUND_LABEL}_${MODE}_rank0_epoch.csv"
        [[ -f "$CSV_ITER" ]] && mv "$CSV_ITER" "p4_job${JOB}_v6_${ROUND_LABEL}_${MODE}_rank0_iter.csv"
    done
    
    echo "--- $MODE_NAME done ---"
}

# Run LongLiu mode first
run_mode longliu "LongLiu"

# Gap
echo ""
echo "Waiting 15s between modes..."
sleep 15

# Run CRUX mode second
run_mode crux "CRUX"

# Stop background flow clients
echo ""
echo "================================================================"
echo "Cleanup: Stop background flow"
echo "================================================================"
for PID in $BG_CLIENT_PIDS; do
    kill $PID 2>/dev/null || true
done
wait 2>/dev/null || true

# Stop iperf3 servers on 226
for PORT in $(seq $PORT_START $PORT_END); do
    SRV_PID=$(ssh -o ConnectTimeout=10 $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
    [[ -n "$SRV_PID" ]] && ssh -o ConnectTimeout=10 $NODE_226 "kill $SRV_PID" 2>/dev/null || true
done

# Summary
echo ""
echo "================================================================"
echo "Round 1 (LL→CX) — Results Summary"
echo "================================================================"
for MODE in longliu crux; do
    echo ""
    echo "=== $MODE ==="
    for JOB in A B; do
        CSV="p4_job${JOB}_v6_${ROUND_LABEL}_${MODE}_rank0_epoch.csv"
        if [[ -f "$CSV" ]]; then
            echo "--- Job $JOB ---"
            cat "$CSV"
        fi
    done
done

# BG flow summary
echo ""
echo "=== Background flow summary ==="
TOTAL=0
for PORT in $(seq $PORT_START $PORT_END); do
    GBPS=$(tail -3 /tmp/v6_bgflow_${ROUND_LABEL}_${PORT}.log 2>/dev/null | grep -oP '[\d.]+(?= Gbits/sec)' | tail -1)
    if [[ -n "$GBPS" ]]; then
        TOTAL=$(echo "$TOTAL + $GBPS" | bc 2>/dev/null)
        printf "  Port %4d: %s Gbps\n" $PORT "$GBPS"
    fi
done
printf "  Total throughput: ~%.1f Gbps\n" "${TOTAL:-0}"

echo "================================================================"
echo "DONE"
echo "================================================================"
