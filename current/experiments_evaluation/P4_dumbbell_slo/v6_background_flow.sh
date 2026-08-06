#!/bin/bash
#=============================================================================
# V6 Background Flow: iperf3 UDP with DSCP=P3 (the "crowd")
#
# Purpose: Constant-rate background traffic at P3 priority to break the
#          phase-exclusion self-mitigation observed in V5.
#
# Usage:
#   On receiver (10.1):  bash v6_background_flow.sh server
#   On sender (226):     bash v6_background_flow.sh client <RATE_Gbps>
#
# DSCP P3 = 16 → TOS = 64 (16 << 2)
#=============================================================================

set -euo pipefail

ROLE="${1:-help}"
RATE="${2:-30}"  # default 30 Gbps
PORT=5202
DSCP_P3=16
TOS=$((DSCP_P3 << 2))  # = 64

case "$ROLE" in
    server)
        echo "[BG-FLOW] Starting iperf3 UDP server on port ${PORT}"
        echo "[BG-FLOW] DSCP=${DSCP_P3} (TOS=${TOS}) - P3 crowd traffic"
        iperf3 -s -p "$PORT" -D
        echo "[BG-FLOW] Server running. Stop with: pkill -f 'iperf3 -s -p ${PORT}'"
        ;;
    client)
        TARGET="${3:-10.157.197.26}"
        echo "[BG-FLOW] Starting iperf3 UDP client → ${TARGET}:${PORT}"
        echo "[BG-FLOW] Rate: ${RATE} Gbps, DSCP=${DSCP_P3} (TOS=${TOS})"
        echo "[BG-FLOW] Duration: infinite (Ctrl+C to stop)"
        iperf3 -c "$TARGET" -u -b "${RATE}G" -t 0 --tos "$TOS" -p "$PORT" -l 8960 -f m
        ;;
    stop)
        pkill -f "iperf3 -s -p ${PORT}" 2>/dev/null && echo "[BG-FLOW] Server stopped" || echo "[BG-FLOW] No server running"
        pkill -f "iperf3 -c.*-p ${PORT}" 2>/dev/null && echo "[BG-FLOW] Client stopped" || echo "[BG-FLOW] No client running"
        ;;
    *)
        echo "Usage: $0 {server|client <RATE_Gbps> [target_ip]|stop}"
        echo ""
        echo "Examples:"
        echo "  # On 10.1 (receiver):"
        echo "  bash $0 server"
        echo ""
        echo "  # On 226 (sender), 30 Gbps:"
        echo "  bash $0 client 30"
        echo ""
        echo "  # Stop all:"
        echo "  bash $0 stop"
        exit 1
        ;;
esac
