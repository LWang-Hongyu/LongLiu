#!/bin/bash
# ib_prio_strict.sh — RDMA 严格优先级实验（3 QP 强制过载）
set -e

NODE_226="192.10.10.226"
DEV="mlx5_0"
DUR=10
OUTDIR="/tmp/ib_results"

mkdir -p "$OUTDIR"

cleanup() {
    ssh "$NODE_226" "pkill -9 ib_write_bw" 2>/dev/null || true
    pkill -9 ib_write_bw 2>/dev/null || true
    sleep 3
}

run_multi() {
    local label="$1"; shift
    # args: port sl port sl ...
    # 启动 server
    local i=0
    local pids=""
    while [ $# -gt 1 ]; do
        local port=$1; local sl=$2
        ssh "$NODE_226" "nohup ib_write_bw --port=$port -d $DEV --report_gbits --sl=$sl -D 30 > /tmp/ib_srv_${port}.log 2>&1 & echo PID=\$!" 2>/dev/null
        shift 2
    done
    sleep 4
    
    # 启动 client (全部并行)
    i=0; pids=""
    # 重新解析参数
    set -- $(cat /tmp/ib_args_${label}.txt 2>/dev/null || true)
    while [ $# -gt 1 ]; do
        local port=$1; local sl=$2
        local out="$OUTDIR/${label}_QP${i}_SL${sl}.txt"
        (ib_write_bw --port=$port "$NODE_226" -d "$DEV" --report_gbits --sl=$sl -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" > "$out") &
        pids="$pids $!"
        i=$((i+1))
        shift 2
    done
    
    for pid in $pids; do wait $pid 2>/dev/null; done
    echo "--- $label ---"
    cat "$OUTDIR/${label}"_*.txt
    echo ""
    sleep 2
}

# 实验1: 单 QP 基线
cleanup
echo "=== 实验1: 单 QP 基线 (SL=0) ==="
ssh "$NODE_226" "nohup ib_write_bw --port=21001 -d $DEV --report_gbits --sl=0 -D 30 > /tmp/ib_srv_21001.log 2>&1 & echo PID=\$!" 2>/dev/null
sleep 3
ib_write_bw --port=21001 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average"
sleep 2

# 实验2: 双 QP 同 SL=0 (并行)
cleanup
echo "=== 实验2: 双 QP 同 SL=0 (并行) ==="
ssh "$NODE_226" "nohup ib_write_bw --port=21002 -d $DEV --report_gbits --sl=0 -D 30 > /tmp/ib_srv_21002.log 2>&1 & echo PID=\$!" 2>/dev/null
ssh "$NODE_226" "nohup ib_write_bw --port=21003 -d $DEV --report_gbits --sl=0 -D 30 > /tmp/ib_srv_21003.log 2>&1 & echo PID=\$!" 2>/dev/null
sleep 4
ib_write_bw --port=21002 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID_A=$!
ib_write_bw --port=21003 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID_B=$!
wait $PID_A $PID_B 2>/dev/null
sleep 2

# 实验3: 三 QP (1×SL=0 + 2×SL=7) 强制链路过载
cleanup
echo "=== 实验3: 三 QP (1×SL=0 + 2×SL=7) ==="
ssh "$NODE_226" "nohup ib_write_bw --port=21010 -d $DEV --report_gbits --sl=0 -D 30 > /tmp/ib_srv_21010.log 2>&1 & echo PID=\$!" 2>/dev/null
ssh "$NODE_226" "nohup ib_write_bw --port=21011 -d $DEV --report_gbits --sl=7 -D 30 > /tmp/ib_srv_21011.log 2>&1 & echo PID=\$!" 2>/dev/null
ssh "$NODE_226" "nohup ib_write_bw --port=21012 -d $DEV --report_gbits --sl=7 -D 30 > /tmp/ib_srv_21012.log 2>&1 & echo PID=\$!" 2>/dev/null
sleep 4
echo "--- QP A (SL=0, low) ---"
ib_write_bw --port=21010 "$NODE_226" -d "$DEV" --report_gbits --sl=0 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID_A=$!
echo "--- QP B (SL=7, high) ---"
ib_write_bw --port=21011 "$NODE_226" -d "$DEV" --report_gbits --sl=7 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID_B=$!
echo "--- QP C (SL=7, high) ---"
ib_write_bw --port=21012 "$NODE_226" -d "$DEV" --report_gbits --sl=7 -D "$DUR" 2>&1 | grep -E "^ 65536|BW average" &
PID_C=$!
wait $PID_A $PID_B $PID_C 2>/dev/null
sleep 2

cleanup
echo "=== 全部完成 ==="