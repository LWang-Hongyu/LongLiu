#!/bin/bash
# ============================================================================
# Verify NCCL DSCP-on-wire for multi_comm (direct answer to: does the P6/P3
# priority actually change the DSCP that NCCL puts on the RoCEv2 wire?)
#
# Method (per priority):
#   1. Solo multi_comm allreduce at fixed priority (mc_solo_prio.py)
#   2. ethtool per-priority TX counters on 10.1 (sender) — which egress
#      priority queue (tx_prioN_bytes delta) the NIC actually used
#   3. tcpdump on 10.1 egress (sender) — parse the RoCEv2 outer IP TOS byte,
#      direct evidence of the DSCP NCCL puts on the wire
#      (NOTE: 226-side tcpdump needs root/CAP_NET_RAW and is unavailable, so
#       sender-side capture + ethtool are the evidence channels)
#
# Also verifies which libnccl the process actually resolved (via /proc maps),
# since torch bundles its own libnccl while libmulti_comm.so's dependency is
# resolved through LD_LIBRARY_PATH.
#
# Expected mapping (multi_comm.c prio_dscp table):
#   P6 -> ToS 0x20 (DSCP 8)  -> tc:0
#   P3 -> ToS 0x40 (DSCP 16) -> tc:2
#   P4 -> ToS 0x00 (DSCP 0)  -> tc:1
#
# Robustness: the testbed has a KNOWN flaky NCCL comm issue (transient
# "remote process exited" or first-collective hangs — same reason
# run_v6_full.sh sets MAX_MODE_RETRY=3). This script retries each priority
# up to 3 times and bounds rank0/rank1 with `timeout` so a hang cannot
# block forever.
#
# Usage (run on 10.1):
#   bash verify_nccl_dscp.sh 6       # test P6 only
#   bash verify_nccl_dscp.sh 6 3     # test P6 then P3
#
# Note: uses system NCCL (LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu) to match
# run_v6_full.sh. Requires GPU + RDMA — run manually in a real shell.
# ============================================================================
set -euo pipefail

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
NIC_226="enp59s0f0np0"      # 226 RDMA netdev (192.10.10.226)
NIC_10="enp130s0f0np0"      # 10.1 RDMA netdev (192.10.10.110)
PORT=29900
PAYLOAD_MB=1024
ITERS=60
WARMUP=10
RUN_TIMEOUT=120            # per rank, kills hung allreduce instead of blocking forever
MAX_ATTEMPTS=3             # per priority
NCCL_LD_PATH="/usr/lib/x86_64-linux-gnu"   # system NCCL 2.30.7
EXP_DIR="/home/why/LongLiu_rebuild/current/experiments_evaluation/P4_dumbbell_slo"
SO_PATH="/home/why/LongLiu_rebuild/current/multi_comm_slo/build/libmulti_comm.so"
DRIVER="$EXP_DIR/mc_solo_prio.py"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prio> [prio...]   (e.g. '$0 6 3')"
    exit 1
fi
PRIOS=("$@")

declare -A DSCP_OF=( [0]=40 [1]=32 [2]=24 [3]=16 [4]=0 [5]=16 [6]=8 )
declare -A TOS_OF=(  [0]=0xA0 [1]=0x80 [2]=0x60 [3]=0x40 [4]=0x00 [5]=0x40 [6]=0x20 )
declare -A TC_OF=(   [0]=5 [1]=4 [2]=3 [3]=2 [4]=1 [5]=2 [6]=0 )

for P in "${PRIOS[@]}"; do
    if [[ -z "${DSCP_OF[$P]+x}" ]]; then
        echo "ERROR: unsupported priority $P"; exit 1
    fi
done

cleanup() {
    pkill -f "mc_solo_prio.py" 2>/dev/null || true
    ssh "$NODE_226" "pkill -f 'mc_solo_prio.py'" 2>/dev/null || true
    sleep 2
}

echo "================================================================"
echo "NCCL DSCP-on-wire verification — priorities: ${PRIOS[*]}"
echo "  Payload: ${PAYLOAD_MB}MB  iters: ${ITERS}+${WARMUP} warmup"
echo "  NCCL:    system libnccl (${NCCL_LD_PATH})"
echo "  Retry:   ${MAX_ATTEMPTS} attempts/prio, timeout ${RUN_TIMEOUT}s"
echo "  Date:    $(date -Iseconds)"
echo "================================================================"

# ------------------------------------------------------------
# Pre-flight: .so + driver present on both nodes with identical md5
# ------------------------------------------------------------
echo "--- Pre-flight: libmulti_comm.so + driver consistency ---"
if [[ ! -f "$SO_PATH" ]]; then
    echo "ERROR: $SO_PATH not found on 10.1"; exit 1
fi
scp -q "$DRIVER" "$NODE_226:$EXP_DIR/" 2>/dev/null || {
    echo "ERROR: failed to sync driver to 226"; exit 1; }
ssh -o ConnectTimeout=5 "$NODE_226" "test -f $SO_PATH && test -f $DRIVER" || {
    echo "ERROR: .so or driver missing on 226"; exit 1; }
MD5_10=$(md5sum "$SO_PATH" | awk '{print $1}')
MD5_226=$(ssh "$NODE_226" "md5sum $SO_PATH" | awk '{print $1}')
if [[ "$MD5_10" == "$MD5_226" ]]; then
    echo "  OK: .so identical (${MD5_10:0:12})"
else
    echo "ERROR: .so mismatch 10.1=${MD5_10} 226=${MD5_226}"; exit 1
fi

# NIC DSCP->priority/TC trust mapping (sender side) — tells us which egress
# priority queue each DSCP is expected to hit, for interpreting ethtool delta
echo ""
echo "--- DSCP->priority/TC trust mapping (10.1 sender, $NIC_10) ---"
if command -v mlnx_qos >/dev/null 2>&1; then
    mlnx_qos -i "$NIC_10" 2>&1 | head -40 || echo "  (mlnx_qos query failed)"
else
    echo "  (mlnx_qos not installed — will infer mapping from ethtool tx_prio delta)"
fi

# ------------------------------------------------------------
# Loop over priorities
# ------------------------------------------------------------
for P in "${PRIOS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    OUT="$EXP_DIR/verify_nccl_dscp_${P}_${TS}"
    mkdir -p "$OUT"

    echo ""
    echo "================================================================"
    echo ">>> Testing priority P${P} (DSCP=${DSCP_OF[$P]} -> tc:${TC_OF[$P]})"
    echo "    Expected wire TOS: ${TOS_OF[$P]}"
    echo "================================================================"

    SUCCESS=0
    for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS"); do
        echo ""
        echo "  [attempt $ATTEMPT/$MAX_ATTEMPTS]"
        cleanup

        # ethtool baseline (10.1 sender) — full dump, diff later
        ethtool -S "$NIC_10" > "$OUT/eth_before.txt" 2>/dev/null || true

        # Start rank0 (server) on 10.1, bounded by timeout
        echo "  Starting rank0 (10.1)..."
        LD_LIBRARY_PATH="$NCCL_LD_PATH" MULTI_COMM_PRIOS="$P" MULTI_COMM_PORT="$PORT" \
            nohup timeout "$RUN_TIMEOUT" python3 "$DRIVER" --prio "$P" \
            --payload-mb "$PAYLOAD_MB" --iters "$ITERS" --warmup "$WARMUP" \
            > "$OUT/rank0.log" 2>&1 &
        RANK0_PID=$!

        # Wait for TCP listen (bounded)
        LISTENED=0
        for i in $(seq 1 60); do
            if ss -tln | grep -q ":$PORT "; then
                echo "  rank0 listening (attempt $i)"
                LISTENED=1
                break
            fi
            sleep 1
        done
        if [[ $LISTENED -ne 1 ]]; then
            echo "ERROR: rank0 did not start listening; aborting priority P$P"
            tail -20 "$OUT/rank0.log" || true
            exit 1
        fi

        # Version confirmation: which libnccl the running process actually resolved
        # (torch bundles libnccl 2.18.6, but libmulti_comm.so's dependency follows
        #  LD_LIBRARY_PATH — verify via /proc maps rather than torch's API)
        PY_PID=$(pgrep -f "mc_solo_prio.py --prio $P" | head -1 || true)
        echo "  [env] rank0 python3 pid: ${PY_PID:-n/a}"
        if [[ -n "${PY_PID:-}" ]]; then
            echo "  [env] libnccl loaded by rank0 (from /proc/$PY_PID/maps):"
            grep -E "/libnccl\.so" "/proc/$PY_PID/maps" 2>/dev/null | awk '{print $NF}' | sort -u || true
        fi

        # Start tcpdump on 10.1 (sender egress): capture RoCEv2 outbound TOS.
        # (226-side capture requires root/CAP_NET_RAW and is unavailable.)
        CAP_TIME=$((ITERS / 3 + 40))
        timeout "$CAP_TIME" tcpdump -i "$NIC_10" -nn -vv -c 40 \
            'udp dst port 4791' > "$OUT/tcpdump_10.txt" 2>&1 &
        TCPDUMP_PID=$!
        sleep 1

        # Launch rank1 (client) on 226, bounded by timeout
        echo "  Launching rank1 (226)..."
        timeout "$RUN_TIMEOUT" ssh "$NODE_226" "cd /tmp && LD_LIBRARY_PATH=$NCCL_LD_PATH \
            MULTI_COMM_PRIOS=$P MULTI_COMM_PORT=$PORT MASTER_ADDR=$RDMA_10 RANK=1 LOCAL_RANK=0 \
            python3 $DRIVER --prio $P --payload-mb $PAYLOAD_MB \
            --iters $ITERS --warmup $WARMUP" > "$OUT/rank1.log" 2>&1
        RANK1_EXIT=$?

        # Wait for rank0 (bounded by its own timeout)
        wait "$RANK0_PID" 2>/dev/null || true
        RANK0_OK=0
        if grep -q "\[mc_solo_prio\] done" "$OUT/rank0.log" 2>/dev/null; then
            RANK0_OK=1
        fi

        echo "  rank1 exit: $RANK1_EXIT, rank0 completed: $RANK0_OK"
        echo "--- rank0 log tail ---"
        tail -12 "$OUT/rank0.log" || true
        echo "--- rank1 log tail ---"
        tail -6 "$OUT/rank1.log" || true

        if [[ $RANK1_EXIT -eq 0 && $RANK0_OK -eq 1 ]]; then
            echo "  SUCCESS: attempt $ATTEMPT"
            SUCCESS=1
            break
        fi
        echo "  WARN: attempt $ATTEMPT failed (flaky NCCL comm on testbed); retrying..."
    done

    if [[ $SUCCESS -ne 1 ]]; then
        echo "ERROR: priority P$P failed after $MAX_ATTEMPTS attempts"
        exit 1
    fi

    # ethtool delta (10.1 sender egress per-priority TX counters)
    ethtool -S "$NIC_10" > "$OUT/eth_after.txt" 2>/dev/null || true
    echo ""
    echo "--- ethtool per-priority TX byte delta (10.1 egress, $NIC_10) ---"
    FOUND=0
    for i in 0 1 2 3 4 5 6 7; do
        B=$(grep -oE "tx_prio${i}_bytes: *[0-9]+" "$OUT/eth_before.txt" | grep -oE '[0-9]+$' | head -1 || true)
        A=$(grep -oE "tx_prio${i}_bytes: *[0-9]+" "$OUT/eth_after.txt" | grep -oE '[0-9]+$' | head -1 || true)
        [[ -n "$B" && -n "$A" ]] || continue
        FOUND=1
        D=$((A - B))
        printf "  tx_prio%d_bytes: %+d  (%.1f MB)\n" "$i" "$D" "$(awk -v d="$D" 'BEGIN{printf "%.1f", d/1e6}')"
    done
    if [[ $FOUND -eq 0 ]]; then
        echo "  (tx_prioN_bytes counters not found; showing raw delta below)"
        diff "$OUT/eth_before.txt" "$OUT/eth_after.txt" > "$OUT/eth_delta.txt" 2>&1 || true
        head -20 "$OUT/eth_delta.txt" || true
    fi

    # tcpdump TOS result (10.1 egress capture)
    echo ""
    echo "--- tcpdump TOS on wire (10.1 egress, dst port 4791) ---"
    wait "$TCPDUMP_PID" 2>/dev/null || true
    if grep -qi "permission" "$OUT/tcpdump_10.txt" 2>/dev/null; then
        echo "  (tcpdump: permission denied on 10.1 — relying on ethtool tx_prio delta)"
    elif [[ -s "$OUT/tcpdump_10.txt" ]]; then
        grep -oE 'tos 0x[0-9a-f]+' "$OUT/tcpdump_10.txt" | sort | uniq -c \
            > "$OUT/tos_summary.txt" 2>/dev/null || true
        cat "$OUT/tos_summary.txt" || echo "  (no TOS captured)"
        echo "  samples:"
        grep -E 'tos ' "$OUT/tcpdump_10.txt" | head -5 || true
    else
        echo "  (capture empty — check $OUT/tcpdump_10.txt)"
    fi

    # Verdict
    echo ""
    echo "--- Verdict for P${P} ---"
    echo "  Expected wire TOS: ${TOS_OF[$P]} (DSCP=${DSCP_OF[$P]}, tc:${TC_OF[$P]})"
    echo "  TOS summary (above) should contain tos ${TOS_OF[$P]}"
    echo "  Results: $OUT"
    echo ""

    rm -f "$OUT/eth_before.txt" "$OUT/eth_after.txt"
done

cleanup
echo "================================================================"
echo "DONE. Manual pass criteria:"
echo "  P6 -> wire TOS 0x20 (DSCP 8)   -> egress tx_prio of DSCP 8 (see mlnx_qos)"
echo "  P3 -> wire TOS 0x40 (DSCP 16)  -> egress tx_prio of DSCP 16"
echo "  P4 -> wire TOS 0x00 (DSCP 0)   -> egress tx_prio of DSCP 0"
echo "  Expected DSCP->tx_prio (P4 testbed): P6->0, P4->1, P3/P5->2, P2->3, P1->4, P0->5"
echo "  (cross-check with the mlnx_qos mapping printed above)"
echo "  If wire TOS == 0x00 for every priority, NCCL is IGNORING"
echo "  config.trafficClass -> DSCP never reaches the data plane."
echo "================================================================"
