#!/bin/bash
# ============================================================================
# monitor_nic.sh — NIC 硬件计数器周期采样（10.1 本机 + 226 远程）
# ============================================================================
# 采集内容（每 interval 一行，原始值，deltas 由分析脚本计算）：
#   [0]  ts                 Unix epoch 秒（3 位小数）
#   [1-4]  IB port counters  port_xmit_data / port_rcv_data / xmit_packets / rcv_packets
#   [5-12] RoCE hw_counters  roce_adp_retrans / rnr_nak_retry_err / out_of_buffer /
#                            out_of_sequence / packet_seq_err / retry_exceeded /
#                            duplicate_request / naks_recv
#   [13-19] ethtool 主计数   rx_bytes / tx_bytes / rx_packets / tx_packets /
#                            rx_dropped / tx_dropped / rx_pci_sig_err_errors
#   [20-27] prio 队列计数    tx_prio0_bytes … tx_prio7_bytes（验证探测流走哪个队列）
#   [28]    IRQ 计数          /proc/interrupts 中 mlx5_0 相关中断总数
#
# Usage:
#   bash monitor_nic.sh <run_id> <duration_sec> [interval_sec] [outdir]
#     若提供 outdir，则写入 $outdir/<run_id>/（否则写入 00_common/../data/<run_id>/）
#
# Output:
#   <outdir>/<run_id>/nic_10.csv    （10.1, enp130s0f0np0, ConnectX-6 Dx）
#   <outdir>/<run_id>/nic_226.csv   （226,  enp59s0f0np0,  BlueField-3）
# ============================================================================
set -uo pipefail

RUN_ID=${1:?Usage: $0 <run_id> <duration_sec> [interval_sec] [outdir]}
DURATION=${2:?duration_sec required}
INTERVAL=${3:-1}
OVERRIDE_OUTDIR=${4:-}

NODE_226="192.10.10.226"
IFACE_10="enp130s0f0np0"
IFACE_226="enp59s0f0np0"

if [[ -n "$OVERRIDE_OUTDIR" ]]; then
    DATA_DIR="$OVERRIDE_OUTDIR"
else
    DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
fi
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"
OUT_10="$OUTDIR/nic_10.csv"
OUT_226="$OUTDIR/nic_226.csv"

# 头部（与 sample 顺序一致）
HEADER="ts,port_xmit_data,port_rcv_data,port_xmit_packets,port_rcv_packets,roce_adp_retrans,rnr_nak_retry_err,out_of_buffer,out_of_sequence,packet_seq_err,retry_exceeded,duplicate_request,naks_recv,rx_bytes,tx_bytes,rx_packets,tx_packets,rx_dropped,tx_dropped,rx_pci_sig_err_errors,tx_prio0_bytes,tx_prio1_bytes,tx_prio2_bytes,tx_prio3_bytes,tx_prio4_bytes,tx_prio5_bytes,tx_prio6_bytes,tx_prio7_bytes,irq_count"

echo "$HEADER" > "$OUT_10"
echo "$HEADER" > "$OUT_226"

# ---------------------------------------------------------------------------
# 采样主体：在远程/本地 bash 中执行，需预置变量 iface
# ---------------------------------------------------------------------------
SAMPLE_BODY='
    pdir=/sys/class/infiniband/mlx5_0/ports/1/counters
    hdir=/sys/class/infiniband/mlx5_0/ports/1/hw_counters
    ts=$(date +%s.%3N)
    line=$ts
    for c in port_xmit_data port_rcv_data port_xmit_packets port_rcv_packets; do
        line=$line,$(cat $pdir/$c 2>/dev/null || echo NA)
    done
    for c in roce_adp_retrans rnr_nak_retry_err out_of_buffer out_of_sequence packet_seq_err retry_exceeded duplicate_request naks_recv; do
        line=$line,$(cat $hdir/$c 2>/dev/null || echo NA)
    done
    ethout=$(ethtool -S $iface 2>/dev/null)
    for k in rx_bytes tx_bytes rx_packets tx_packets rx_dropped tx_dropped rx_pci_sig_err_errors; do
        line=$line,$(echo "$ethout" | awk -v k="$k:" "\$1==k{print \$2; exit}" | head -1)
    done
    for p in 0 1 2 3 4 5 6 7; do
        line=$line,$(echo "$ethout" | awk -v k="tx_prio${p}_bytes:" "\$1==k{print \$2; exit}" | head -1)
    done
    irq=$(grep -c mlx5 /proc/interrupts 2>/dev/null || echo 0)
    echo "$line,$irq"
'

sample_one() {
    local iface=$1 is_local=$2 out=$3
    local line
    if [[ "$is_local" == "1" ]]; then
        line=$(IFACE_VAL="$iface" bash -c 'iface=$IFACE_VAL; '"$SAMPLE_BODY")
    else
        line=$(ssh "$NODE_226" "iface='$iface'; $SAMPLE_BODY")
    fi
    echo "$line" >> "$out"
}

echo "[monitor_nic] run=$RUN_ID duration=${DURATION}s interval=${INTERVAL}s"
echo "[monitor_nic] 10.1 -> $OUT_10"
echo "[monitor_nic] 226  -> $OUT_226"

END=$(( $(date +%s) + DURATION ))
N=0
while : ; do
    NOW=$(date +%s)
    if (( NOW >= END )); then break; fi
    sample_one "$IFACE_10" 1 "$OUT_10" &
    sample_one "$IFACE_226" 0 "$OUT_226" &
    wait
    N=$((N+1))
    sleep "$INTERVAL"
done

echo "[monitor_nic] done: $N samples per node -> $OUTDIR"
