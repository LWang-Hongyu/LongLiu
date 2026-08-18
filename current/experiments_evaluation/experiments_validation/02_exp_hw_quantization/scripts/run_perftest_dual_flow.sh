#!/bin/bash
# ============================================================================
# run_perftest_dual_flow.sh — perftest 受控双流：持续饱和流下 SP 严格性判定
# ============================================================================
# 动机：NCCL 流量（test3）在包层面是突发（ON-OFF），无法判定 SP 队列是否
# 真正 per-packet 严格。perftest(ib_write_bw) 是持续饱和的 RDMA 流（占空比
# ≈100%），直接看"高优先级持续占满时，低优先级是否还能拿到带宽"：
#   * 流B(DSCP16) 被饿死(≈0)  → SP 严格成立，NCCL 的 58% 归因于突发间隙
#   * 流B 仍拿固定份额       → SP 非严格实锤，58% 是硬件物理上限
#
# 时序（Step 1）：
#   t=0   server A/B 就绪
#   t=2   流A(DSCP8/TOS32) 启动并占满（DUR_A=40s）
#   t=7   流B(DSCP16/TOS64) 插入（DUR_B=30s）→ 与流A 重叠 30s
# 并发期间对 10.1 出口 tx_prio 计数器逐秒采样（nic_prio_conc.csv）
#
# Usage:
#   bash run_perftest_dual_flow.sh [round]
# ============================================================================
set -uo pipefail

ROUND=${1:-1}

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$BASE_DIR/data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ID="exp2_perftest_r${ROUND}_${TS}"
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"

NODE_226="192.10.10.226"
IFACE_10="enp130s0f0np0"
PORT_A=29850
PORT_B=29851
MSG_SIZE=1048576          # 1MB 消息
DUR_A=40                  # 高优先级流 A 时长
DUR_B=30                  # 低优先级流 B 时长（t=7 插入，与 A 重叠 30s）
TOS_A=32                  # DSCP8  <<2（高优先级，tc:0）
TOS_B=${TOS_B:-64}             # 低优先级流：默认 DSCP16(TOS=64/tc:2)；覆盖：DSCP24→96、DSCP32→128
GID_IDX=3                 # IPv4 RoCEv2（与 NCCL_IB_GID_INDEX=3 一致）

echo "================================================================"
echo "Exp2 perftest 双流: 持续饱和流下 SP 严格性判定 — Round $ROUND"
echo "  流A: DSCP8(tc:0)  TOS=$TOS_A  先启动占满"
echo "  流B: TOS=$TOS_B  t=7s 后插入  (默认 DSCP16/tc:2; 覆盖 DSCP24→96、DSCP32→128)"
echo "  msg=${MSG_SIZE}B, 流A=${DUR_A}s, 流B=${DUR_B}s, gid_idx=$GID_IDX"
echo "  输出: $OUTDIR"
echo "================================================================"

cleanup() {
    pkill -f 'ib_write_bw' 2>/dev/null
    ssh "$NODE_226" "pkill -f 'ib_write_bw' 2>/dev/null; true"
    sleep 1
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Step 0: 单流分类验证（DSCP8，8s）——确认 TOS 映射到 tx_prio1
# ------------------------------------------------------------------
echo "=== Step 0: 分类验证（DSCP8 单流 8s）==="
cleanup
ssh "$NODE_226" "ib_write_bw -d mlx5_0 -R -x $GID_IDX -p $PORT_A -s $MSG_SIZE -D 8 \
    > /dev/null 2>&1" &
sleep 2
read -ra P0 <<< "$(ethtool -S "$IFACE_10" 2>/dev/null | awk '/tx_prio[12]_bytes:/{print $2}')"
ib_write_bw -d mlx5_0 -R -x $GID_IDX -p $PORT_A 192.10.10.226 -s $MSG_SIZE \
    -T $TOS_A -D 8 --report_gbits 2>&1 | tee "$OUTDIR/verifyA.log"
read -ra P1 <<< "$(ethtool -S "$IFACE_10" 2>/dev/null | awk '/tx_prio[12]_bytes:/{print $2}')"
sleep 1
read -ra P2 <<< "$(ethtool -S "$IFACE_10" 2>/dev/null | awk '/tx_prio[12]_bytes:/{print $2}')"
echo "[verify] 8s 内: tx_prio1 += $(( (${P1[0]:-0} - ${P0[0]:-0}) / 1024 / 1024 )) MB, tx_prio2 += $(( (${P1[1]:-0} - ${P0[1]:-0}) / 1024 / 1024 )) MB"
echo "[verify] 后 1s: tx_prio1 += $(( (${P2[0]:-0} - ${P1[0]:-0}) / 1024 / 1024 )) MB, tx_prio2 += $(( (${P2[1]:-0} - ${P1[1]:-0}) / 1024 / 1024 )) MB"

# ------------------------------------------------------------------
# Step 1: 双流并发（A 先占满 → B 后插入）
# ------------------------------------------------------------------
echo "=== Step 1: 双流并发（流A 先启动占满，流B t=7s 插入）==="
cleanup

ssh "$NODE_226" "ib_write_bw -d mlx5_0 -R -x $GID_IDX -p $PORT_A -s $MSG_SIZE -D $DUR_A \
    > /dev/null 2>&1" &
ssh "$NODE_226" "ib_write_bw -d mlx5_0 -R -x $GID_IDX -p $PORT_B -s $MSG_SIZE -D $DUR_B \
    > /dev/null 2>&1" &
sleep 2

# 流A（高优先级）先启动并占满
ib_write_bw -d mlx5_0 -R -x $GID_IDX -p $PORT_A 192.10.10.226 -s $MSG_SIZE \
    -T $TOS_A -D $DUR_A --report_gbits > "$OUTDIR/clientA.log" 2>&1 &
PID_A=$!
sleep 5

# 并发期间对 10.1 出口 tx_prio 逐秒采样
( for i in $(seq 1 45); do
    ts=$(date +%s.%3N)
    t=$(ethtool -S "$IFACE_10" 2>/dev/null | grep -E 'tx_prio[0-7]_bytes' | awk '{printf "%s,", $2}')
    echo "$ts,$t" >> "$OUTDIR/nic_prio_conc.csv"
    sleep 1
done ) &
MON_PID=$!

# 流B（低优先级）插入
ib_write_bw -d mlx5_0 -R -x $GID_IDX -p $PORT_B 192.10.10.226 -s $MSG_SIZE \
    -T $TOS_B -D $DUR_B --report_gbits > "$OUTDIR/clientB.log" 2>&1 &
PID_B=$!

wait $PID_A; echo "  流A(DSCP8) 结束 exit=$?"
wait $PID_B; echo "  流B(DSCP16) 结束 exit=$?"
kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

echo ""
echo "================================================================"
echo "perftest Round $ROUND 完成"
echo "  流A(DSCP8) 带宽:  $(grep -a -E '^[0-9]+ +[0-9]+ ' "$OUTDIR/clientA.log" | tail -1 | awk '{print $4}') Gb/s avg"
echo "  流B(TOS=${TOS_B}) 带宽: $(grep -a -E '^[0-9]+ +[0-9]+ ' "$OUTDIR/clientB.log" | tail -1 | awk '{print $4}') Gb/s avg"
echo "  数据: $OUTDIR"
echo "================================================================"
