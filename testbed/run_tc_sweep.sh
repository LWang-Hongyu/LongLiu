#!/bin/bash
# run_tc_sweep.sh - Systematically test NCCL_IB_TC values via tcpdump
# Run on 10.1 (guolab-10), orchestrates 226 as master
# Tests each TC value: sets NCCL_IB_TC, runs dscp_tc_sweep, captures pcap, verifies tos.

set -e

MASTER_HOST="10.157.197.107"
MASTER_USER="why"
MASTER_ADDR="192.10.10.226"
TEST_DIR="/home/why/LongLiu_rebuild/testbed"
RESULT_DIR="/tmp/dscp_tc_sweep_results"
PCAP_DIR="$RESULT_DIR/pcaps"
LOG_FILE="$RESULT_DIR/sweep_results.txt"

# TC values to test (spread across the range)
TC_VALUES=(0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60)

# NCCL common env
NCCL_ENV="NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=enp NCCL_DEBUG=WARN"
# LD_PRELOAD paths
LD_PRELOAD_226="/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2"
LD_PRELOAD_101="/home/why/.local/lib/python3.8/site-packages/nvidia/nccl/lib/libnccl.so.2"

mkdir -p "$PCAP_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "DSCP TC Sweep Test - $(date)" | tee -a "$LOG_FILE"
echo "Master: 226 (${MASTER_HOST}), Worker: 10.1 (local)" | tee -a "$LOG_FILE"
echo "Testing ${#TC_VALUES[@]} TC values: ${TC_VALUES[*]}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Kill any leftover processes on both nodes
cleanup() {
    echo "[cleanup] Killing leftover processes..."
    sudo pkill -9 -f tcpdump 2>/dev/null || true
    pkill -9 -f dscp_tc_sweep 2>/dev/null || true
    ssh ${MASTER_USER}@${MASTER_HOST} "pkill -9 -f dscp_tc_sweep 2>/dev/null || true" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT

for i in "${!TC_VALUES[@]}"; do
    TC="${TC_VALUES[$i]}"
    PORT=$((29600 + i))
    PCAP_FILE="$PCAP_DIR/tc_${TC}.pcap"
    
    echo "---" | tee -a "$LOG_FILE"
    echo "[$((i+1))/${#TC_VALUES[@]}] Testing NCCL_IB_TC=$TC (port=$PORT) ..." | tee -a "$LOG_FILE"
    
    # Clean up any leftover processes from previous test
    sudo pkill -9 -f tcpdump 2>/dev/null || true
    pkill -9 -f dscp_tc_sweep 2>/dev/null || true
    ssh ${MASTER_USER}@${MASTER_HOST} "pkill -9 -f dscp_tc_sweep 2>/dev/null || true" 2>/dev/null || true
    sleep 2
    
    # Remove old pcap
    rm -f "$PCAP_FILE"
    
    # Start tcpdump on mlx5_0 (local 10.1)
    echo "  Starting tcpdump on mlx5_0 (local)..." | tee -a "$LOG_FILE"
    sudo tcpdump -i mlx5_0 -s 200 -c 300 -w "$PCAP_FILE" udp port 4791 > /dev/null 2>&1 &
    TCPDUMP_PID=$!
    sleep 1
    
    # Start worker on 10.1 (local, non-blocking)
    echo "  Starting worker (local)..." | tee -a "$LOG_FILE"
    NCCL_IB_TC=${TC} ${NCCL_ENV} \
        MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${PORT} \
        WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
        LD_PRELOAD=${LD_PRELOAD_101} \
        python3 ${TEST_DIR}/dscp_tc_sweep.py > /tmp/dscp_tc_${TC}_worker.log 2>&1 &
    WORKER_PID=$!
    
    sleep 2
    
    # Start master on 226 via SSH (blocking)
    echo "  Starting master on 226..." | tee -a "$LOG_FILE"
    ssh ${MASTER_USER}@${MASTER_HOST} \
        "cd ${TEST_DIR} && \
         NCCL_IB_TC=${TC} ${NCCL_ENV} \
         MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${PORT} \
         WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
         LD_PRELOAD=${LD_PRELOAD_226} \
         python3 dscp_tc_sweep.py" > /tmp/dscp_tc_${TC}_master.log 2>&1
    
    MASTER_RC=$?
    
    # Wait for worker to finish
    wait $WORKER_PID 2>/dev/null || true
    
    # Wait a moment for any remaining packets
    sleep 2
    
    # Stop tcpdump
    sudo pkill -9 -f "tcpdump.*mlx5_0" 2>/dev/null || true
    wait $TCPDUMP_PID 2>/dev/null || true
    sleep 0.5
    
    # Analyze pcap
    if [ -f "$PCAP_FILE" ] && [ -s "$PCAP_FILE" ]; then
        PKT_COUNT=$(sudo tcpdump -r "$PCAP_FILE" 2>/dev/null | wc -l)
        
        # Extract unique ToS values
        TOS_VALUES=$(sudo tcpdump -r "$PCAP_FILE" -v -n 2>/dev/null | grep -oP 'tos 0x[0-9a-f]+' | sort -u)
        
        # Expected ToS = TC (direct mapping verified in previous test: TC=26 -> tos 0x1a)
        EXPECTED_TOS=$(printf "0x%02x" $TC)
        
        # Check if expected tos is present (all packets should match)
        if [ -n "$TOS_VALUES" ]; then
            UNEXPECTED=$(echo "$TOS_VALUES" | grep -v "tos $EXPECTED_TOS" || true)
        else
            UNEXPECTED="no_tos_found"
        fi
        
        if [ -z "$UNEXPECTED" ]; then
            RESULT="PASS"
        else
            RESULT="FAIL (expected=$EXPECTED_TOS, unexpected=$UNEXPECTED)"
        fi
        
        echo "  Result: $RESULT | $PKT_COUNT packets | ToS: $(echo $TOS_VALUES | tr '\n' ' ')" | tee -a "$LOG_FILE"
    else
        echo "  Result: FAIL (no pcap file or empty)" | tee -a "$LOG_FILE"
    fi
    
    # Show errors if any
    if [ "$MASTER_RC" -ne 0 ]; then
        echo "  WARNING: Master exited with code=$MASTER_RC" | tee -a "$LOG_FILE"
        echo "  Master log tail:" | tee -a "$LOG_FILE"
        tail -5 /tmp/dscp_tc_${TC}_master.log | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

echo "============================================================" | tee -a "$LOG_FILE"
echo "Summary Table" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

printf "%-6s %-8s %-8s %-10s %s\n" "TC" "ExpToS" "Result" "Packets" "Observed ToS" | tee -a "$LOG_FILE"
printf "%-6s %-8s %-8s %-10s %s\n" "------" "--------" "------" "--------" "------------" | tee -a "$LOG_FILE"

PASS_COUNT=0
FAIL_COUNT=0

for i in "${!TC_VALUES[@]}"; do
    TC="${TC_VALUES[$i]}"
    PCAP_FILE="$PCAP_DIR/tc_${TC}.pcap"
    EXPECTED_TOS=$(printf "0x%02x" $TC)
    
    if [ -f "$PCAP_FILE" ] && [ -s "$PCAP_FILE" ]; then
        TOS_VALUES=$(sudo tcpdump -r "$PCAP_FILE" -v -n 2>/dev/null | grep -oP 'tos 0x[0-9a-f]+' | sort -u | tr '\n' ' ')
        PKT_COUNT=$(sudo tcpdump -r "$PCAP_FILE" 2>/dev/null | wc -l)
        
        if echo "$TOS_VALUES" | grep -q "tos $EXPECTED_TOS"; then
            # Check all are expected
            UNEXPECTED=$(echo "$TOS_VALUES" | grep -oP 'tos 0x[0-9a-f]+' | grep -v "0x$(printf '%02x' $TC)" || true)
            if [ -z "$UNEXPECTED" ]; then
                STATUS="PASS"
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                STATUS="PARTIAL"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        else
            STATUS="FAIL"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        printf "%-6s %-8s %-8s %-10s %s\n" "$TC" "$EXPECTED_TOS" "$STATUS" "$PKT_COUNT" "$TOS_VALUES" | tee -a "$LOG_FILE"
    else
        printf "%-6s %-8s %-8s %-10s %s\n" "$TC" "$EXPECTED_TOS" "NODATA" "0" "no pcap" | tee -a "$LOG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "PASS: $PASS_COUNT / ${#TC_VALUES[@]}" | tee -a "$LOG_FILE"
echo "FAIL: $FAIL_COUNT / ${#TC_VALUES[@]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Pcap files: $PCAP_DIR/" | tee -a "$LOG_FILE"
echo "Done at $(date)" | tee -a "$LOG_FILE"
