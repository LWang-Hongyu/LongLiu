#!/bin/bash
# ============================================================================
# V6 全类探测 — P0-P7 vs 6G P3 背景流，两两争抢测相对带宽 + RoCE 重传
# ============================================================================
# 设计：
#   - 12路并行 iperf3 UDP, 每路 500M, 总 6G, DSCP=P3 (TOS=64) — 恒定背景
#   - 逐 DSCP 注入 1G 探针流（10s），测量 server 端 goodput
#   - 记录 NIC prio 计数器和 RoCE hw_counters 变化
#   - 输出真实优先级序
#
# 假设鉴别：
#   - P6 ≥ P4 → NIC/TC 方向（头号假设）
#   - P6 < P3 → 默认队列（二号假设）
#
# Usage:
#   bash probe_dscp_priority.sh
# ============================================================================
set -uo pipefail

NODE_226="192.10.10.226"
RDMA_226="192.10.10.226"
RDMA_10="192.10.10.110"

BG_TOTAL_GBPS=6
NUM_BG_FLOWS=12
BG_DURATION=300      # 总背景流时长
PROBE_RATE_MBPS=1000 # 1G 探针流
PROBE_DURATION=15    # 每探针持续 15s (5s 稳定 + 10s 测量)

# DSCP 值：P0-P7（DSCP = priority * 8）
DSCP_VALUES=(0 8 16 24 32 40 48 56)
PRIO_NAMES=("P0" "P1" "P2" "P3" "P4" "P5" "P6" "P7")

# 接口
IFACE_10="enp130s0f0np0"
IFACE_226="enp59s0f0np0"

EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="/tmp/dscp_probe_${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "================================================================"
echo "V6 全类探测 — P0-P7 vs 6G P3 背景流"
echo "  12路 iperf3 UDP 背景: 每路 500M, DSCP=P3 (TOS=64)"
echo "  探针: 1G UDP, DSCP 逐值 (0/8/16/24/32/40/48/56)"
echo "  每探针 ${PROBE_DURATION}s, server 侧 goodput 测量"
echo "  输出目录: ${OUTDIR}"
echo "================================================================"
echo ""

cd "$EXP_DIR"

# ============================================================
# 清理函数
# ============================================================
cleanup_all() {
    echo "--- 清理所有进程 ---"
    # 本地 iperf3 客户端
    for PID in $(pgrep -f "iperf3.*-p 62[0-9][0-9].*-u" 2>/dev/null); do
        kill $PID 2>/dev/null || true
    done
    # 远程 iperf3 服务器
    for PORT in $(seq 6200 6211); do
        local PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
        if [[ -n "$PID" ]]; then
            ssh $NODE_226 "kill $PID" 2>/dev/null || true
        fi
    done
    sleep 2
    echo "清理完成"
}

cleanup_all

# ============================================================
# 读取计数器的函数
# ============================================================
read_eth_prio() {
    local label="$1"
    local file="${OUTDIR}/eth_prio_${label}.txt"
    echo "=== eth_prio_${label} ===" > "$file"
    echo "--- 10.1 (${IFACE_10}) ---" >> "$file"
    ethtool -S ${IFACE_10} 2>/dev/null | grep -E 'prio[0-7]' >> "$file"
    echo "--- 226 (${IFACE_226}) ---" >> "$file"
    ssh $NODE_226 "ethtool -S ${IFACE_226} 2>/dev/null | grep -E 'prio[0-7]'" >> "$file"
}

read_roce_counters() {
    local label="$1"
    local file="${OUTDIR}/roce_${label}.txt"
    echo "=== roce_${label} ===" > "$file"
    echo "--- 10.1 IB hw_counters ---" >> "$file"
    for c in $(ls /sys/class/infiniband/mlx5_0/ports/1/hw_counters/); do
        echo "  $c: $(cat /sys/class/infiniband/mlx5_0/ports/1/hw_counters/$c)" >> "$file"
    done
    echo "--- 226 IB hw_counters ---" >> "$file"
    ssh $NODE_226 "for c in \$(ls /sys/class/infiniband/mlx5_0/ports/1/hw_counters/); do echo \"  \$c: \$(cat /sys/class/infiniband/mlx5_0/ports/1/hw_counters/\$c)\"; done" >> "$file"
}

# ============================================================
# Step 1: 启动 12 路 iperf3 服务器 (226)
# ============================================================
echo "=== Step 1: 启动 12 路 iperf3 服务器 (226) ==="
for PORT in $(seq 6200 6211); do
    OLD_PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT'" 2>/dev/null || true)
    if [[ -n "$OLD_PID" ]]; then
        ssh $NODE_226 "kill $OLD_PID" 2>/dev/null || true
    fi
    ssh $NODE_226 "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null
done
sleep 2
SRV_COUNT=$(ssh $NODE_226 "pgrep -a iperf3 | grep '\-s' | wc -l")
echo "  服务器运行数: $SRV_COUNT"

# ============================================================
# Step 2: 启动 12 路 P3 背景流
# ============================================================
echo "=== Step 2: 启动 12 路 P3 背景流 (${BG_TOTAL_GBPS}G) ==="
PER_STREAM=$((BG_TOTAL_GBPS * 1000 / NUM_BG_FLOWS))
for PORT in $(seq 6200 6211); do
    iperf3 -c $RDMA_226 -u -b ${PER_STREAM}M -t $BG_DURATION \
        --tos 64 -p $PORT -f g -l 8900 \
        > /tmp/bg_probe_${PORT}.log 2>&1 &
done
echo "  ${NUM_BG_FLOWS} 路 background 已启动"
sleep 5

# 读取基线
read_eth_prio "baseline"
read_roce_counters "baseline"

# ============================================================
# Step 3: 逐 DSCP 注入探针流
# ============================================================
echo ""
echo "=== Step 3: 逐 DSCP 探针 ==="

declare -A PROBE_BW

# 探针流使用端口 6212（独立于背景流）
PROBE_PORT=6212

for idx in "${!DSCP_VALUES[@]}"; do
    DSCP=${DSCP_VALUES[$idx]}
    PRIO_NAME=${PRIO_NAMES[$idx]}
    TOS=$((DSCP << 2))

    echo ""
    echo "--- ${PRIO_NAME}: DSCP=${DSCP}, TOS=${TOS} ---"

    # 清理旧探针服务器
    OLD_PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PROBE_PORT'" 2>/dev/null || true)
    if [[ -n "$OLD_PID" ]]; then
        ssh $NODE_226 "kill $OLD_PID" 2>/dev/null || true
    fi
    sleep 1

    # 启动探针服务器
    ssh $NODE_226 "iperf3 -s -p $PROBE_PORT -D -B $RDMA_226" 2>/dev/null
    sleep 1

    # 读取探针前计数器
    read_roce_counters "pre_${PRIO_NAME}"

    # 启动探针
    PROBE_LOG="${OUTDIR}/probe_${PRIO_NAME}.log"
    iperf3 -c $RDMA_226 -u -b ${PROBE_RATE_MBPS}M -t $PROBE_DURATION \
        --tos $TOS -p $PROBE_PORT -f g -l 8900 \
        > "$PROBE_LOG" 2>&1
    PROBE_EXIT=$?

    # 读取探针后计数器
    read_roce_counters "post_${PRIO_NAME}"

    # 解析探针吞吐
    if [[ $PROBE_EXIT -eq 0 ]]; then
        BW=$(grep -oP '[\d.]+(?= Gbits/sec)' "$PROBE_LOG" | tail -1)
        LOSS=$(grep -oP '[\d.]+(?=%)' "$PROBE_LOG" | tail -1)
        PROBE_BW[$PRIO_NAME]="${BW:-0}"
        echo "  ${PRIO_NAME} 探针: ${BW:-0} Gbps, 丢包: ${LOSS:-N/A}%"
    else
        PROBE_BW[$PRIO_NAME]="FAIL"
        echo "  ${PRIO_NAME} 探针: FAILED (exit=$PROBE_EXIT)"
    fi

    # 清理探针服务器
    SRV_PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PROBE_PORT'" 2>/dev/null || true)
    if [[ -n "$SRV_PID" ]]; then
        ssh $NODE_226 "kill $SRV_PID" 2>/dev/null || true
    fi
    sleep 2
done

# ============================================================
# Step 4: 停止背景流+服务器
# ============================================================
echo ""
echo "=== Step 4: 停止背景流 ==="
cleanup_all

# ============================================================
# Step 5: 输出结果汇总
# ============================================================
echo ""
echo "================================================================"
echo "全类探测结果汇总"
echo "================================================================"
echo ""
echo "背景流: ${BG_TOTAL_GBPS}G P3 (12×${PER_STREAM}M)"
echo "探针: ${PROBE_RATE_MBPS}M UDP, 15s 每值"
echo ""

# 排序输出（按带宽降序）
echo "优先级序（探针带宽降序）:"
echo "----------------------------"
for PRIO_NAME in "${PRIO_NAMES[@]}"; do
    BW=${PROBE_BW[$PRIO_NAME]:-"N/A"}
    DSCP_VAL=0
    # Find DSCP value for this prio name
    for i in "${!PRIO_NAMES[@]}"; do
        if [[ "${PRIO_NAMES[$i]}" == "$PRIO_NAME" ]]; then
            DSCP_VAL=${DSCP_VALUES[$i]}
            break
        fi
    done
    printf "  %s (DSCP=%3d): %s Gbps\n" "$PRIO_NAME" "$DSCP_VAL" "$BW"
done | sort -k3 -rn -t: || true

echo ""
echo "--- 原始探针日志: ${OUTDIR}/probe_*.log ---"
echo "--- NIC prio 计数器: ${OUTDIR}/eth_prio_*.txt ---"
echo "--- RoCE 计数器: ${OUTDIR}/roce_*.txt ---"

# P4 vs P6 RoCE 对比
echo ""
echo "=== P4 vs P6 RoCE 重传对比 ==="
for c in roce_adp_retrans out_of_buffer out_of_sequence rnr_nak_retry_err packet_seq_err; do
    PRE_P4=$(grep "  $c:" ${OUTDIR}/roce_pre_P4.txt 2>/dev/null | awk '{print $2}')
    POST_P4=$(grep "  $c:" ${OUTDIR}/roce_post_P4.txt 2>/dev/null | awk '{print $2}')
    PRE_P6=$(grep "  $c:" ${OUTDIR}/roce_pre_P6.txt 2>/dev/null | awk '{print $2}')
    POST_P6=$(grep "  $c:" ${OUTDIR}/roce_post_P6.txt 2>/dev/null | awk '{print $2}')
    echo "  $c:"
    echo "    P4: ${PRE_P4:-N/A} → ${POST_P4:-N/A} (delta: $(( ${POST_P4:-0} - ${PRE_P4:-0} )))"
    echo "    P6: ${PRE_P6:-N/A} → ${POST_P6:-N/A} (delta: $(( ${POST_P6:-0} - ${PRE_P6:-0} )))"
done

echo ""
echo "================================================================"
echo "探测完成"
echo "================================================================"
