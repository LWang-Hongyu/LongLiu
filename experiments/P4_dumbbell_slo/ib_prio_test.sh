#!/bin/bash
# ib_prio_test.sh — RDMA dual-QP priority experiment
# Tests whether HCA/switch differentiates traffic at different SL/DSCP values
set -e

NODE_226="192.10.10.226"
DEV="mlx5_0"
DUR=10  # seconds per test

cleanup() {
    pkill -9 ib_write_bw 2>/dev/null || true
    ssh "$NODE_226" "pkill -9 ib_write_bw" 2>/dev/null || true
    sleep 2
}

run_single() {
    local label=$1 port=$2 sl=$3 server=$4
    if [ "$server" = "1" ]; then
        ssh "$NODE_226" "ib_write_bw --port=$port -d $DEV --report_gbits --sl=$sl -D $((DUR+5))" > /tmp/ib_srv_${port}.log 2>&1 &
    else
        ib_write_bw --port=$port "$NODE_226" -d "$DEV" --report_gbits --sl=$sl -D "$DUR" 2>&1
    fi
}

echo "=========================================="
echo "RDMA Priority Test — 10.1(50G) => 226(100G)"
echo "  Device: $DEV, Duration: ${DUR}s per test"
echo "=========================================="
echo ""

# === 实验1: 单 QP 基线 ===
cleanup
echo "--- 实验1: 单 QP 基线 (SL=0) ---"
run_single "srv" 21000 0 1
sleep 2
run_single "cli" 21000 0 0 | grep -E "^ 65536|BW average"
sleep 2
echo ""

# === 实验2: 双 QP 同 SL=0 (公平基线) ===
cleanup
echo "--- 实验2: 双 QP 同 SL=0 ---"
run_single "srvA" 21010 0 1
run_single "srvB" 21011 0 1
sleep 3
ib_write_bw --port=21010 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID1=$!
ib_write_bw --port=21011 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID2=$!
wait $PID1 $PID2 2>/dev/null
echo ""

# === 实验3: 双 QP 不同 SL (SL=0 vs SL=3) ===
cleanup
echo "--- 实验3: 双 QP SL=0(low) vs SL=3(high) ---"
run_single "srvA" 21020 0 1
run_single "srvB" 21021 3 1
sleep 3
ib_write_bw --port=21020 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID1=$!
ib_write_bw --port=21021 "$NODE_226" -d "$DEV" --report_gbits --sl=3 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID2=$!
wait $PID1 $PID2 2>/dev/null
echo ""

# === 实验4: 双 QP SL=0 vs SL=7 ===
cleanup
echo "--- 实验4: 双 QP SL=0(low) vs SL=7(high) ---"
run_single "srvA" 21030 0 1
run_single "srvB" 21031 7 1
sleep 3
ib_write_bw --port=21030 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID1=$!
ib_write_bw --port=21031 "$NODE_226" -d "$DEV" --report_gbits --sl=7 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID2=$!
wait $PID1 $PID2 2>/dev/null
echo ""

cleanup
echo "=== 实验完成 ==="
