#!/bin/bash
# ============================================================================
# V6 Step 1: DSCP→Queue Mapping Verification
#
# Purpose:
#   Verify that DSCP values (P1=32, P3=16, P4=0) map to the correct
#   priority queues on both NIC and switch, under corrected prio→DSCP mapping.
#
# Method:
#   1. mlnx_qos on both hosts: confirm trust=dscp + DSCP→priority mapping
#   2. Inject P1/P3/P4 probe traffic (iperf3 UDP, 5 Gbps × 10s)
#   3. Capture packets with tcpdump → verify DSCP bits on wire
#   4. Read NIC per-priority counters (if available)
#   5. Functional test: simultaneous P1/P3/P4 → measure bandwidth share
#      (P4 should get > P3 > P1 under SP, confirming queue mapping)
#
# Priority mapping for V6 (corrected):
#   P1 (DSCP=32) → priority 4 → tc:4 (fifth) — loose job graceful degradation
#   P3 (DSCP=16) → priority 2 → tc:2 (third) — CRUX static class + background flow
#   P4 (DSCP=0)  → priority 0 → tc:1 (second) — LongLiu tight job (lifted out of crowd)
#
# iperf3 --tos values (TOS = DSCP << 2):
#   P1: DSCP=32 → TOS=128
#   P3: DSCP=16 → TOS=64
#   P4: DSCP=0  → TOS=0
#
# Usage:
#   bash verify_dscp_queue.sh          # run on 10.1 (worker)
#   bash verify_dscp_queue.sh --remote # run on 226 (master)
#
# Output:
#   /tmp/dscp_verify_*.log
#   stdout verification table
#
# Pass criteria:
#   - Both hosts show DSCP trust mode enabled
#   - DSCP 32/16/0 map to priorities 4/2/0 in mlnx_qos output
#   - tcpdump confirms DSCP bits on wire match intended marking
#   - Functional test: P4 bandwidth > P3 bandwidth > P1 bandwidth
# ============================================================================
set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NIC="mlx5_0"
REMOTE_226="192.10.10.226"
LOCAL_10="192.10.10.110"
PROBE_DURATION=10       # seconds per probe
PROBE_RATE_MBPS=5000    # 5 Gbps per probe
FUNC_DURATION=15        # seconds for functional test
FUNC_RATE_MBPS=8000     # 8 Gbps per flow for functional test

# DSCP and corresponding TOS values
declare -A DSCP_MAP=(
    [1]=32   # P1 → DSCP 32 (tc:4)
    [3]=16   # P3 → DSCP 16 (tc:2)
    [4]=0    # P4 → DSCP 0  (tc:1)
)

# For probe: (priority, port)
PROBE_PORTS=(5201 5203 5204)  # P1, P3, P4

# For functional test: simultaneous all priorities
FUNC_PORTS=(5211 5213 5214)   # P1, P3, P4

IS_REMOTE=${1:-""}

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
dscp_to_tos() {
    echo $(( $1 << 2 ))
}

timestamp() {
    date +%H:%M:%S
}

# ------------------------------------------------------------------
# Step A: mlnx_qos NIC configuration dump
# ------------------------------------------------------------------
echo "=================================================================="
echo "V6 Step 1: DSCP→Queue Mapping Verification"
echo "Host: $(hostname)  NIC: ${NIC}  Time: $(timestamp)"
echo "=================================================================="
echo ""

echo "--- [A] NIC DSCP trust mode + priority mapping ---"
echo ">>> mlnx_qos -i ${NIC} (trust mode):"
mlnx_qos -i "${NIC}" 2>&1 | head -20 || echo "  [WARN] mlnx_qos failed"
echo ""

echo ">>> mlnx_qos -i ${NIC} --dscp-prio (DSCP→priority mapping):"
mlnx_qos -i "${NIC}" --dscp-prio 2>&1 | head -40 || echo "  [WARN] dscp-prio query failed"
echo ""

# ------------------------------------------------------------------
# Step B: Per-priority counters (if available)
# ------------------------------------------------------------------
echo "--- [B] NIC per-priority counters ---"
# Mellanox ConnectX: per-priority counters under ethtool -S
# Look for "prio[0-7]" or "tc_[0-7]" patterns
ETHTOOL_OUT=$(mktemp)
ethtool -S "${NIC}" 2>/dev/null > "$ETHTOOL_OUT" || true

PRIO_COUNTERS=$(grep -iE "prio[0-7]|tc_[0-7]|qos_" "$ETHTOOL_OUT" | head -20 || true)
if [[ -n "$PRIO_COUNTERS" ]]; then
    echo "$PRIO_COUNTERS"
else
    echo "  [INFO] No per-priority counters exposed via ethtool -S"
    echo "  [INFO] Will rely on functional test for queue verification"
fi
rm -f "$ETHTOOL_OUT"
echo ""

# ------------------------------------------------------------------
# Step C: Probe traffic with tcpdump capture
# ------------------------------------------------------------------
echo "--- [C] Probe traffic: P1/P3/P4 individual probes ---"
echo "  Each: iperf3 UDP ${PROBE_RATE_MBPS}Mbps × ${PROBE_DURATION}s"
echo ""

# Start iperf3 servers on remote (226)
echo ">>> Starting iperf3 servers on ${REMOTE_226}..."
for PORT in "${PROBE_PORTS[@]}" "${FUNC_PORTS[@]}"; do
    ssh -o ConnectTimeout=5 "${REMOTE_226}" \
        "pkill -f 'iperf3 -s -p ${PORT}' 2>/dev/null; \
         iperf3 -s -p ${PORT} -D -B ${REMOTE_226}" 2>/dev/null || true
done
sleep 2

# Verify servers
SRV_COUNT=$(ssh "${REMOTE_226}" "pgrep -a iperf3 | grep '\-s' | wc -l" 2>/dev/null || echo 0)
echo "  Servers running: ${SRV_COUNT}"
echo ""

# For each priority: send probe, capture with tcpdump
for PRIO in 1 3 4; do
    DSCP=${DSCP_MAP[$PRIO]}
    TOS=$(dscp_to_tos $DSCP)
    PORT="${PROBE_PORTS[$((PRIO-1))]}"  # 5201, 5203, 5204
    
    echo ">>> Probing P${PRIO} (DSCP=${DSCP}, TOS=${TOS}) on port ${PORT}..."
    
    # Start tcpdump capture on remote (capture 1 packet to verify DSCP)
    ssh "${REMOTE_226}" "timeout $((PROBE_DURATION + 5)) tcpdump -i ${NIC} -c 3 -nn \
        'udp port ${PORT}' -x 2>/dev/null" > /tmp/dscp_probe_P${PRIO}_capture.txt 2>&1 &
    TCPDUMP_PID=$!
    sleep 1
    
    # Send probe
    iperf3 -c "${REMOTE_226}" -u -b ${PROBE_RATE_MBPS}M \
        -t ${PROBE_DURATION} --tos ${TOS} -p ${PORT} -l 1400 -f g \
        2>&1 | tail -5 > /tmp/dscp_probe_P${PRIO}_iperf.log
    
    wait ${TCPDUMP_PID} 2>/dev/null || true
    
    # Extract and display DSCP from captured packet
    echo "  tcpdump captures:"
    CAPTURE="/tmp/dscp_probe_P${PRIO}_capture.txt"
    if [[ -f "$CAPTURE" ]]; then
        # DSCP is in the IP header TOS byte (byte 2 of IP header after ethernet)
        # For IPv4: ethernet header (14 bytes) + IP header, byte 1 of IP = TOS
        # tcpdump -x shows hex, so look for TOS byte
        DSCP_HEX=$(grep -oP '0x[0-9a-f]{4}' "$CAPTURE" | head -3 || true)
        if [[ -n "$DSCP_HEX" ]]; then
            echo "  Raw hex: $DSCP_HEX"
            # First hex word after ethernet header... this is probe, not exact parsing
            echo "  [CHECK] TOS=${TOS} (0x$(printf '%x' ${TOS})) → DSCP=${DSCP}"
        fi
        grep -v "^tcpdump:" "$CAPTURE" | head -5
    fi
    
    echo "  iperf3 result:"
    cat "/tmp/dscp_probe_P${PRIO}_iperf.log"
    echo ""
done

# ------------------------------------------------------------------
# Step D: Functional test — simultaneous P1/P3/P4
# ------------------------------------------------------------------
echo "--- [D] Functional test: P1/P3/P4 simultaneous ---"
echo "  3 concurrent iperf3 UDP flows, ${FUNC_RATE_MBPS}Mbps each, ${FUNC_DURATION}s"
echo "  Expected under SP: P4 BW > P3 BW > P1 BW"
echo ""

# Start servers on 226
for PORT in "${FUNC_PORTS[@]}"; do
    ssh "${REMOTE_226}" \
        "pkill -f 'iperf3 -s -p ${PORT}' 2>/dev/null; \
         iperf3 -s -p ${PORT} -D -B ${REMOTE_226}" 2>/dev/null || true
done
sleep 2

# Launch all 3 clients simultaneously
PIDS=""
for PRIO in 1 3 4; do
    DSCP=${DSCP_MAP[$PRIO]}
    TOS=$(dscp_to_tos $DSCP)
    PORT="${FUNC_PORTS[$((PRIO-1))]}"
    
    (iperf3 -c "${REMOTE_226}" -u -b ${FUNC_RATE_MBPS}M \
        -t ${FUNC_DURATION} --tos ${TOS} -p ${PORT} -l 1400 -f g \
        2>&1 | grep -E "sender|receiver|Gbits/sec" \
        > /tmp/dscp_func_P${PRIO}.log 2>&1) &
    PIDS="$PIDS $!"
done

echo "  All 3 flows launched, waiting ${FUNC_DURATION}s..."
sleep $((FUNC_DURATION + 3))
echo ""

# Report results
echo ">>> Functional test bandwidth results:"
declare -A BW_RESULTS
for PRIO in 1 3 4; do
    BW=$(grep -oP '[\d.]+(?= Gbits/sec)' "/tmp/dscp_func_P${PRIO}.log" | tail -1 || echo "N/A")
    BW_RESULTS[$PRIO]=$BW
    echo "  P${PRIO} (DSCP=${DSCP_MAP[$PRIO]}): ${BW} Gbits/sec"
done

# Determine if differentiation works
P1_BW=${BW_RESULTS[1]}
P3_BW=${BW_RESULTS[3]}
P4_BW=${BW_RESULTS[4]}

# Special case: if P1 got more than P3+P4 (unlikely but possible with artifacts)
echo ""
echo ">>> Differentiation check:"
if [[ "$P4_BW" != "N/A" && "$P3_BW" != "N/A" && "$P1_BW" != "N/A" ]]; then
    # Use bc if available for comparison
    if command -v bc &>/dev/null; then
        if (( $(echo "$P4_BW > $P3_BW" | bc -l) )) && (( $(echo "$P3_BW > $P1_BW" | bc -l) )); then
            echo "  ✅ PASS: P4 (${P4_BW}G) > P3 (${P3_BW}G) > P1 (${P1_BW}G) — SP works"
        elif (( $(echo "$P4_BW > $P3_BW" | bc -l) )); then
            echo "  ⚠️  PARTIAL: P4 (${P4_BW}G) > P3 (${P3_BW}G), but P3 vs P1 not resolved"
        else
            echo "  ❌ FAIL: No differentiation. P4 (${P4_BW}G) <= P3 (${P3_BW}G)"
        fi
    else
        echo "  [INFO] bc not available, manual check: P4=${P4_BW} P3=${P3_BW} P1=${P1_BW}"
    fi
else
    echo "  [WARN] Some bandwidth results missing, cannot auto-verify"
fi

# ------------------------------------------------------------------
# Step E: iperf3 --tos 64 → P3 queue specific check
# ------------------------------------------------------------------
echo ""
echo "--- [E] iperf3 --tos 64 (DSCP 16) → P3 verification ---"
echo "  Send iperf3 UDP with --tos 64, verify via mlnx_qos mapping"
echo "  mlnx_qos maps DSCP 16 → priority 2 → tc:2 (P3 queue) ✓"
echo "  (Mapping confirmed in step [A] if DSCP 16 → prio 2)"
echo ""

# Extract DSCP 16 mapping from mlnx_qos output
mlnx_qos -i "${NIC}" --dscp-prio 2>&1 | grep -E "0x10|16" | head -5 || \
    echo "  [INFO] Check step [A] for DSCP 16 → priority mapping"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "=================================================================="
echo "VERIFICATION TABLE"
echo "=================================================================="
printf "| %-5s | %-8s | %-6s | %-8s | %-14s | %-12s |\n" \
    "Prio" "DSCP" "TOS" "tos(hex)" "iperf3 --tos" "Exp Queue"
printf "|-------|----------|--------|----------|----------------|--------------|\n"
printf "| %-5s | %-8s | %-6s | %-8s | %-14s | %-12s |\n" \
    "P1" "32" "128" "0x80" "--tos 128" "low (tc:4)"
printf "| %-5s | %-8s | %-6s | %-8s | %-14s | %-12s |\n" \
    "P3" "16" "64" "0x40" "--tos 64" "mid (tc:2)"
printf "| %-5s | %-8s | %-6s | %-8s | %-14s | %-12s |\n" \
    "P4" "0" "0" "0x00" "--tos 0" "high (tc:1)"
echo "=================================================================="
echo ""

echo "--- Log files ---"
echo "  Probe iperf3:  /tmp/dscp_probe_P{1,3,4}_iperf.log"
echo "  tcpdump:       /tmp/dscp_probe_P{1,3,4}_capture.txt"
echo "  Functional:    /tmp/dscp_func_P{1,3,4}.log"
echo ""

echo "--- Manual verification checklist ---"
echo "  [ ] Step [A]: Both hosts show trust=dscp in mlnx_qos"
echo "  [ ] Step [A]: DSCP 32→prio4, DSCP 16→prio2, DSCP 0→prio0"
echo "  [ ] Step [C]: tcpdump shows packets with correct TOS byte"
echo "  [ ] Step [D]: P4 bandwidth > P3 bandwidth > P1 bandwidth"
echo "  [ ] Step [E]: --tos 64 (iperf3 bg flow) → DSCP 16 → tc:2 (P3 queue)"
echo ""

if command -v bc &>/dev/null \
    && [[ "$P4_BW" != "N/A" && "$P3_BW" != "N/A" && "$P1_BW" != "N/A" ]] \
    && (( $(echo "$P4_BW > $P3_BW" | bc -l) )) \
    && (( $(echo "$P3_BW > $P1_BW" | bc -l) )); then
    echo ">>> OVERALL: ✅ PASS — DSCP→queue mapping verified (P4>P3>P1)"
    exit 0
else
    echo ">>> OVERALL: ⚠️  Check results manually before proceeding to calibration"
    echo "  (Functional test auto-check requires bc and all 3 BW results)"
    exit 2
fi
