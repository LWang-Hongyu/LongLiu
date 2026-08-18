#!/bin/bash
# ============================================================================
# bg_saturate.sh — 背景流打满链路（模拟持续拥塞，历史 V6 验证过的方案）
# ============================================================================
# 单方向 12 路 iperf3 UDP，DSCP=P3（TOS=64，与作业同队列 tc:2）。
#   方向: 10.1 → 226（server 在 226，client 在 10.1，与 run_v6_full.sh 一致）
# 10.1 侧 50G 链路成为瓶颈 → 背景流打满链路。
#
# 注：10.1 无法作为 iperf3 server 被 226 连接（防火墙/版本限制），
#     故不采用双向背景流；单向打满即可达到实验目的（V6 亦如此）。
#
# Usage:
#   bash bg_saturate.sh start [label] [duration_sec] [total_gbps]
#   bash bg_saturate.sh stop  [label]
#   bash bg_saturate.sh check [label]
# ============================================================================
set -uo pipefail

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
TOS_P3=64
FWD_START=6300
NUM_FLOWS=12

MODE="${1:-help}"
LABEL="${2:-exp1}"
DURATION="${3:-3600}"
TOTAL_GBPS="${4:-40}"
PROTO="${5:-udp}"   # udp=盲发（历史V6方案）；tcp=有拥塞控制，能真实挤占带宽

# 每路速率（Mbps，iperf3 用 -b XM 更稳定）
PER_FLOW_MBPS=$((TOTAL_GBPS * 1000 / NUM_FLOWS))

start_flow() {
    echo "[bg] forward 10.1->226: ${NUM_FLOWS}x${PER_FLOW_MBPS}M, DSCP=P3(TOS=${TOS_P3}), total=${TOTAL_GBPS}G, proto=${PROTO}"
    for p in $(seq $FWD_START $((FWD_START + NUM_FLOWS - 1))); do
        # 清理旧 server（若有）
        OLD_PID=$(ssh "$NODE_226" "pgrep -f 'iperf3 -s -p $p'" 2>/dev/null || true)
        if [[ -n "$OLD_PID" ]]; then
            ssh "$NODE_226" "kill $OLD_PID" 2>/dev/null || true
        fi
        ssh "$NODE_226" "iperf3 -s -p $p -D -B $RDMA_226" 2>/dev/null
        if [[ "$PROTO" == "tcp" ]]; then
            # TCP：无拥塞控制盲发问题，自动填满剩余带宽，与 P3 作业公平竞争
            iperf3 -c "$RDMA_226" -t "$DURATION" \
                --tos "$TOS_P3" -p "$p" -f g -P 1 > /tmp/bg_fwd_${LABEL}_${p}.log 2>&1 &
        else
            iperf3 -c "$RDMA_226" -u -b ${PER_FLOW_MBPS}M -t "$DURATION" \
                --tos "$TOS_P3" -p "$p" -f g -l 8900 > /tmp/bg_fwd_${LABEL}_${p}.log 2>&1 &
        fi
    done
    sleep 3
    echo "[bg] 背景流已启动（label=$LABEL, total=${TOTAL_GBPS}Gbps, proto=${PROTO}, duration=${DURATION}s）"
}

stop_flow() {
    echo "[bg] 停止所有背景流..."
    pkill -f "iperf3 -c $RDMA_226" 2>/dev/null || true
    for p in $(seq $FWD_START $((FWD_START + NUM_FLOWS - 1))); do
        ssh "$NODE_226" "pkill -f 'iperf3 -s -p $p'" 2>/dev/null || true
    done
    echo "[bg] 已停止"
}

check_flow() {
    local local_clients=$(pgrep -fc "iperf3 -c" 2>/dev/null || echo 0)
    echo "[bg] 本机 iperf3 客户端: $local_clients"
    # 汇总 forward 吞吐
    local total=0
    for p in $(seq $FWD_START $((FWD_START + NUM_FLOWS - 1))); do
        local g=$(grep -oP '[\d.]+(?= Gbits/sec)' /tmp/bg_fwd_${LABEL}_${p}.log 2>/dev/null | tail -1)
        total=$(echo "${total:-0} + ${g:-0}" | bc 2>/dev/null)
    done
    echo "[bg] forward 总吞吐: ~${total:-0} Gbps"
}

case "$MODE" in
    start) start_flow ;;
    stop)  stop_flow ;;
    check) check_flow ;;
    *) echo "Usage: $0 {start|stop|check} [label] [duration_sec] [total_gbps]"; exit 1 ;;
esac
