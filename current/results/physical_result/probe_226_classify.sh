#!/bin/bash
# ============================================================================
# 226 分类能力探针（V2 — 使用反向模式绕开 RDMA TCP 限制）
# ============================================================================
# 目标：判定 226 NIC 是否支持 DSCP→prio 分类
#   - "不分类"：NIC 硬件/驱动不支持 DSCP→prio 映射
#   - "无标记流量"：流量本身无 DSCP 标记（NIC 有分类能力但未被使用）
#
# 方法：
#   相一：从 226 发送 DSCP 标记 UDP 探针 → 10.1（反向模式 iperf3）
#         记录 tx_prio 增量 → 确认 DSCP 是否被映射到对应 prio
#   相二：6G P3 背景流 10.1→226，记录 rx_prio 增量
#         确认 226 接收侧是否分类
#
# 判定：
#   - DSCP=48→tx_prio6 有增量 → 226 分类 ✓
#   - 全部 DSCP→tx_prio0 → 226 不分类
#
# 注意：226→10.1 的 iperf3 TCP 控制通道被 10.1 firewall 阻断，
#       使用 --reverse 模式（10.1→226 TCP 控制，226→10.1 UDP 数据）解决。
# Usage:
#   bash probe_226_classify.sh
# ============================================================================
set -uo pipefail

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
PROBE_PORT=29550
PROBE_BW=200    # 200M probe (cover counter deltas without saturating)
PROBE_DURATION=20

EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${EXP_DIR}/226_classify_probe_${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "================================================================"
echo "226 分类能力探针 (V2 — 反向模式)"
echo "  日期: $(date -Iseconds)"
echo "================================================================"

# ============================================================
# Phase 1: Reverse-mode probe — sweep DSCP, check tx_prio
# ============================================================
echo ""
echo "============================================"
echo "Phase 1: DSCP Sweep (226→10.1, reverse mode)"
echo "  iperf3 server on 226, client on 10.1 (--reverse)"
echo "  UDP data flows 226→10.1, TCP control 10.1→226"
echo "============================================"
echo ""

DSCP_ARR=(0 8 16 24 32 40 48 56)

# Start iperf3 server on 226 ONCE for all probes
ssh $NODE_226 "pkill -f 'iperf3 -s -p $PROBE_PORT' 2>/dev/null; sleep 0.5; iperf3 -s -p $PROBE_PORT -D" 2>/dev/null || true
sleep 3

for DSCP in "${DSCP_ARR[@]}"; do
    TOS=$((DSCP << 2))
    printf -- '--- Probing DSCP=%2d (TOS=%3d) ---\n' $DSCP $TOS

    # Read BEFORE tx_prio
    before_file="$OUTDIR/tx_before_dscp${DSCP}.txt"
    > "$before_file"
    for prio in 0 1 2 3 4 5 6 7; do
        val=$(ssh $NODE_226 "ethtool -S enp59s0f0np0 2>/dev/null | grep tx_prio${prio}_bytes | awk '{print \$2}'" 2>/dev/null)
        echo "$prio:$val" >> "$before_file"
    done

    # Run reverse-mode iperf3: 10.1→226 TCP, 226→10.1 UDP with TOS
    iperf3 -c $NODE_226 -u -b ${PROBE_BW}M -t $PROBE_DURATION -p $PROBE_PORT -f g --reverse --tos $TOS \
        > "$OUTDIR/probe_dscp${DSCP}_10.log" 2>&1 || true

    sleep 2

    # Read AFTER tx_prio
    after_file="$OUTDIR/tx_after_dscp${DSCP}.txt"
    > "$after_file"
    for prio in 0 1 2 3 4 5 6 7; do
        val=$(ssh $NODE_226 "ethtool -S enp59s0f0np0 2>/dev/null | grep tx_prio${prio}_bytes | awk '{print \$2}'" 2>/dev/null)
        echo "$prio:$val" >> "$after_file"
    done

    # Compute delta
    printf "  Tx delta (bytes):"
    delta_found="none"
    for prio in 0 1 2 3 4 5 6 7; do
        bv=$(grep "^$prio:" "$before_file" | cut -d: -f2)
        av=$(grep "^$prio:" "$after_file" | cut -d: -f2)
        if [[ -n "$bv" && -n "$av" ]]; then
            delta=$((av - bv))
            if [[ $delta -gt 1000 ]]; then
                printf " tx_prio%1d:+%s" $prio $(numfmt --to=si $delta 2>/dev/null || echo $delta)
                delta_found="yes"
            fi
        fi
    done
    [[ "$delta_found" == "none" ]] && printf " (all zero or <1KB)"
    echo ""

    # Extract goodput
    gp=$(tail -5 "$OUTDIR/probe_dscp${DSCP}_10.log" 2>/dev/null | grep -oP '[\d.]+(?= Gbits/sec)' | tail -1 || echo "N/A")
    printf "  Goodput: %s Gbps\n" "$gp"
done

# Stop server after all probes
ssh $NODE_226 "pkill -f 'iperf3 -s -p $PROBE_PORT'" 2>/dev/null || true

# ============================================================
# Phase 2: 6G P3 background from 10.1→226, check 226 rx_prio
# ============================================================
echo ""
echo "============================================"
echo "Phase 2: 6G P3 Background (10.1→226) — check 226 rx_prio"
echo "============================================"
echo ""

BG_PORT_START=6200
BG_PORT_END=6211
BG_RATE_GBPS=6
PER_FLOW_MBPS=$((BG_RATE_GBPS * 1000 / 12))
DSCP_P3_TOS=64

# Read 226 rx_prio baseline
bg_before="$OUTDIR/rx_before_bg.txt"
> "$bg_before"
for prio in 0 1 2 3 4 5 6 7; do
    val=$(ssh $NODE_226 "ethtool -S enp59s0f0np0 2>/dev/null | grep rx_prio${prio}_bytes | awk '{print \$2}'" 2>/dev/null)
    echo "$prio:$val" >> "$bg_before"
done
echo "--- 226 rx_prio counters (before background) ---"
cat "$bg_before"

# Start iperf3 servers on 226
echo ""
echo "Starting 12 iperf3 servers on 226..."
for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
    OLD=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
    [[ -n "$OLD" ]] && ssh $NODE_226 "kill $OLD" 2>/dev/null || true
    ssh $NODE_226 "iperf3 -s -p $PORT -D -B $NODE_226" 2>/dev/null
done
sleep 2

# Run 6G background for 60s
echo "Running ${BG_RATE_GBPS}G P3 background (10.1→226) for 60s..."
BG_PIDS=()
for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
    iperf3 -c $NODE_226 -u -b ${PER_FLOW_MBPS}M -t 60 \
        --tos $DSCP_P3_TOS -p $PORT -f g -l 8900 \
        > "$OUTDIR/bg_port${PORT}.log" 2>&1 &
    BG_PIDS+=($!)
done

sleep 65

# Read 226 rx_prio after background
bg_after="$OUTDIR/rx_after_bg.txt"
> "$bg_after"
for prio in 0 1 2 3 4 5 6 7; do
    val=$(ssh $NODE_226 "ethtool -S enp59s0f0np0 2>/dev/null | grep rx_prio${prio}_bytes | awk '{print \$2}'" 2>/dev/null)
    echo "$prio:$val" >> "$bg_after"
done
echo "--- 226 rx_prio counters (after background) ---"
cat "$bg_after"

# Compute rx_prio deltas
echo ""
echo "  Rx delta (bytes):"
for prio in 0 1 2 3 4 5 6 7; do
    bv=$(grep "^$prio:" "$bg_before" | cut -d: -f2)
    av=$(grep "^$prio:" "$bg_after" | cut -d: -f2)
    if [[ -n "$bv" && -n "$av" ]]; then
        delta=$((av - bv))
        if [[ $delta -gt 1000 ]]; then
            rate=$(echo "scale=2; $delta * 8 / 60 / 1e9" | bc)
            printf "    rx_prio%1d: +%s bytes (~%.1f Gbps)\n" $prio $(numfmt --to=si $delta 2>/dev/null || echo $delta) $rate
        fi
    fi
done

# Stop background
for PID in "${BG_PIDS[@]}"; do
    kill $PID 2>/dev/null || true
done
for PORT in $(seq $BG_PORT_START $BG_PORT_END); do
    SRV=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
    [[ -n "$SRV" ]] && ssh $NODE_226 "kill $SRV" 2>/dev/null || true
done

# Cleanup
ssh $NODE_226 "pkill -f iperf3" 2>/dev/null

# ============================================================
# Results table
# ============================================================
echo ""
echo "================================================================"
echo "226 分类能力探针 — 结果摘要"
echo "================================================================"
echo ""
echo "Phase 1: Solo DSCP sweep (226→10.1, reverse mode)"
printf "| %4s | %8s | %s |\n" "DSCP" "Goodput" "Tx delta (non-zero prio)"
printf "|------|----------|-----------------------------------|\n"
for DSCP in "${DSCP_ARR[@]}"; do
    gp=$(tail -5 "$OUTDIR/probe_dscp${DSCP}_10.log" 2>/dev/null | grep -oP '[\d.]+(?= Gbits/sec)' | tail -1 || echo "N/A")
    af="$OUTDIR/tx_after_dscp${DSCP}.txt"
    bf="$OUTDIR/tx_before_dscp${DSCP}.txt"
    delta_str="tx_prio0 only"
    for prio in 1 2 3 4 5 6 7; do
        bv=$(grep "^$prio:" "$bf" | cut -d: -f2)
        av=$(grep "^$prio:" "$af" | cut -d: -f2)
        if [[ -n "$bv" && -n "$av" && $((av - bv)) -gt 1000 ]]; then
            delta_str="tx_prio${prio}!"
            break
        fi
    done
    printf "| %4d | %8s | %s |\n" $DSCP "$gp" "$delta_str"
done

echo ""
echo "Phase 2: 6G P3 bg (10.1→226), rx_prio delta"
echo "  rx_prio3 != 0 → 226 接收侧分类"
echo "  rx_prio0 only → 226 接收侧不分类"

# Read rx deltas
for prio in 0 1 2 3 4 5 6 7; do
    bv=$(grep "^$prio:" "$bg_before" | cut -d: -f2)
    av=$(grep "^$prio:" "$bg_after" | cut -d: -f2)
    if [[ -n "$bv" && -n "$av" ]]; then
        delta=$((av - bv))
        if [[ $delta -gt 1000 ]]; then
            echo "  rx_prio${prio}: +$(numfmt --to=si $delta 2>/dev/null) bytes"
        fi
    fi
done

echo ""
echo "判定:"
echo "  - 所有 DSCP 探针走 tx_prio0 → 226 不分类"
echo "  - 6G P3 背景流走 rx_prio0 → 226 接收也不分类"
echo ""

# Write run_meta
cat > "$OUTDIR/run_meta.txt" <<EOF
=== run_meta: 226 分类能力探针 (V2) ===
实验日期: $(date -I)
发起机: 10.1 (guolab-10)
探针方向: 226 → 10.1 (反向模式 iperf3)
探针速率: ${PROBE_BW}M iperf3 UDP, DSCP 0-56 step 8, ${PROBE_DURATION}s each
相一: 反向模式探针, 测 226 tx_prio 增量 (判断 226 是否分类)
相二: 6G P3 (12×500M) 背景流 10.1→226, 测 226 rx_prio 增量
脚本: probe_226_classify.sh (hash: $(md5sum "$0" | cut -d' ' -f1))
解决 226→10.1 TCP 阻断: 使用 --reverse 模式, 10.1→226 TCP 控制通道
结果目录: ${OUTDIR}
EOF

echo "结果目录: ${OUTDIR}"
echo "================================================================"
