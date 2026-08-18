#!/bin/bash
# ============================================================================
# rerun_rep2_r1.sh — 实验3：图11异常点深度分析（重跑 rep2 r1 + BF-3 DPU 监控）
# ============================================================================
# 复刻 run_meta_round1.txt 的 rep2 r1 条件（V6 Round 1: LL→CX）：
#   bg 12×500M=6G iperf3 UDP, DSCP=P3 (TOS=64), warmup 5min
#   payload 1024MB, sleep 30ms, 300 iters (20/epoch, 15 epochs)
#   c_i tight=1.2 / loose=3.0, swap at epoch 7
#   T_target: A=4201.087ms (ttarget_v5_jobA), B=3905.163ms (ttarget_v5_jobB)
#   LongLiu: --initial-priority 3（无 max cap，允许 π>0.3→P6）
#   CRUX:    --crux-priority-a 3 --crux-priority-b 3
#   顺序:    longliu → crux（Round 1 = LL→CX）
# 新增监控（Forensic）：
#   monitor_nic.sh  — 10.1+226 NIC 计数器（RoCE 重传/out_of_buffer/prio 队列/IRQ）
#   monitor_gpu.sh  — 双端 GPU（温度/功耗/时钟/降频原因）
#
# Usage:
#   bash rerun_rep2_r1.sh
# ============================================================================
set -uo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_DIR="$(cd "$BASE_DIR/../00_common" && pwd)"
DATA_DIR="$BASE_DIR/data"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ID="exp3_rerun_${TS}"
OUTDIR="$DATA_DIR/$RUN_ID"
mkdir -p "$OUTDIR"
date +%s > "$OUTDIR/run_start.epoch"    # 时间对齐锚点（monitor ts 为 Unix 秒）

NODE_226="192.10.10.226"
RDMA_10="192.10.10.110"
RDMA_226="192.10.10.226"
REMOTE_EXP_DIR="/home/why/LongLiu_rebuild/experiments_validation/03_exp_anomaly_forensic/scripts"
REMOTE_TMP="/tmp/exp3_rerun_226"
SCHED_10="/home/why/LongLiu_rebuild/current/multi_comm_slo/src"
SCHED_226="/home/why/LongLiu_rebuild/multi_comm_slo/src"

# ---- V6 rep2 r1 参数（与 run_meta_round1.txt 一致）----
PAYLOAD_MB=1024
CI_TIGHT=1.2
CI_LOOSE=3.0
WARMUP_MINUTES=5
PORT_MAIN_A=29530
PORT_MAIN_B=29531
TTARGET_A="/tmp/ttarget_v5_jobA.json"
TTARGET_B="/tmp/ttarget_v5_jobB.json"

# ---- 背景流 6G P3（与原实验一致）----
BG_TOTAL_GBPS=6
NUM_BG_FLOWS=12
BG_PORT_START=6200
DSCP_P3_TOS=64
MON_DURATION=600   # 10 分钟，覆盖 warmup + 两个 mode

echo "================================================================"
echo "Exp3 异常点 Forensic — 重跑 rep2 r1 (LL→CX)"
echo "  背景流: ${BG_TOTAL_GBPS}G P3, warmup ${WARMUP_MINUTES}min"
echo "  payload=${PAYLOAD_MB}MB, c_i=${CI_TIGHT}/${CI_LOOSE} swap@7"
echo "  LongLiu initial=P3 (无 cap) | CRUX 静态 P3"
echo "  监控: NIC + GPU（Forensic）"
echo "  输出: $OUTDIR"
echo "  日期: $(date -Iseconds)"
echo "================================================================"

if [[ ! -f "$TTARGET_A" || ! -f "$TTARGET_B" ]]; then
    echo "ERROR: 缺少 T_target 文件。请先运行校准："
    echo "  python3 p4_job_reverse_ts.py --job A --mode longliu --phase calibrate --ci-phase1 1.7 --ci-phase2 1.2 --payload-mb 1024 --ttarget-file $TTARGET_A"
    echo "  python3 p4_job_reverse_ts.py --job B --mode longliu --phase calibrate --ci-phase1 3.0 --ci-phase2 1.2 --payload-mb 1024 --ttarget-file $TTARGET_B"
    exit 1
fi

# ------------------------------------------------------------------
# 背景流（12×500M P3，与原实验相同端口/速率）
# ------------------------------------------------------------------
start_bg() {
    echo "[bg] 启动 ${NUM_BG_FLOWS} 路 500M iperf3 UDP, DSCP=P3(TOS=${DSCP_P3_TOS}) = ${BG_TOTAL_GBPS}G"
    for PORT in $(seq $BG_PORT_START $((BG_PORT_START + NUM_BG_FLOWS - 1))); do
        ssh "$NODE_226" "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null
        iperf3 -c "$RDMA_226" -u -b $((BG_TOTAL_GBPS * 1000 / NUM_BG_FLOWS))M -t $MON_DURATION \
            --tos "$DSCP_P3_TOS" -p "$PORT" -f g -l 8900 \
            > "$OUTDIR/bg_${PORT}.log" 2>&1 &
    done
    sleep 3
}

stop_bg() {
    echo "[bg] 停止背景流"
    pkill -f "iperf3 -c $RDMA_226 -u -b" 2>/dev/null || true
    for PORT in $(seq $BG_PORT_START $((BG_PORT_START + NUM_BG_FLOWS - 1))); do
        ssh "$NODE_226" "pkill -f 'iperf3 -s -p $PORT'" 2>/dev/null || true
    done
}

# ------------------------------------------------------------------
# 作业启动（10.1 rank0 / 226 rank1）
# ------------------------------------------------------------------
launch_job() {  # <job A|B> <mode longliu|crux> <port>
    local job=$1 mode=$2 port=$3
    local crux_flags="" ll_flags=""
    [[ "$mode" == "crux" ]] && crux_flags="--crux-priority-a 3 --crux-priority-b 3"
    [[ "$mode" == "longliu" ]] && ll_flags="--initial-priority 3"

    # 10.1 rank0（CWD=OUTDIR，rank0 CSV 落盘于数据目录）
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$port \
        WORLD_SIZE=2 RANK=0 MULTI_COMM_PORT=$port MULTI_COMM_SRC=$SCHED_10 \
        NCCL_SOCKET_IFNAME=enp130s0f0np0 NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
        python3 -u "$BASE_DIR/scripts/p4_job_reverse_ts.py" --job "$job" --mode "$mode" --phase main \
            --ttarget-file $([ "$job" == "A" ] && echo "$TTARGET_A" || echo "$TTARGET_B") \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $crux_flags $ll_flags \
            > "$OUTDIR/exp3_job${job}_${mode}_node101.log" 2>&1 &
    local pid10=$!

    ssh "$NODE_226" "mkdir -p $REMOTE_TMP && cd $REMOTE_TMP && \
        CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$RDMA_10 MASTER_PORT=$port \
        WORLD_SIZE=2 RANK=1 MULTI_COMM_PORT=$port MULTI_COMM_SRC=$SCHED_226 \
        LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
        NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
        NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
        NCCL_DEBUG_FILE=/tmp/nccl_exp3_${job}_${mode}_%h_%p.log \
        python3 -u p4_job_reverse_ts.py --job $job --mode $mode --phase main \
            --ttarget-file $([ "$job" == "A" ] && echo "$TTARGET_A" || echo "$TTARGET_B") \
            --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
            $crux_flags $ll_flags \
            > $REMOTE_TMP/exp3_job${job}_${mode}_node226.log 2>&1" &
    local pid26=$!

    wait "$pid10"
    echo "  Job $job ($mode) 10.1 done (exit=$?)"
    wait "$pid26"
    echo "  Job $job ($mode) 226 done (exit=$?)"

    # 重命名 rank0 CSV 防覆盖
    for J in A B; do
        if [[ -f "$OUTDIR/p4_job${J}_reverse_${mode}_rank0_epoch.csv" ]]; then
            mv "$OUTDIR/p4_job${J}_reverse_${mode}_rank0_epoch.csv" \
               "$OUTDIR/exp3_job${J}_${mode}_rank0_epoch.csv"
        fi
        if [[ -f "$OUTDIR/p4_job${J}_reverse_${mode}_rank0_iter.csv" ]]; then
            mv "$OUTDIR/p4_job${J}_reverse_${mode}_rank0_iter.csv" \
               "$OUTDIR/exp3_job${J}_${mode}_rank0_iter.csv"
        fi
    done
}

# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
# 1) 启动监控（先于背景流与作业）
echo "=== 启动 Forensic 监控（NIC + GPU）==="
bash "$COMMON_DIR/monitor_nic.sh" "$RUN_ID" "$MON_DURATION" 1 "$DATA_DIR" \
    > "$OUTDIR/monitor_nic.log" 2>&1 &
MON_NIC=$!
bash "$COMMON_DIR/monitor_gpu.sh" "$RUN_ID" "$MON_DURATION" 1000 "$DATA_DIR" \
    > "$OUTDIR/monitor_gpu.log" 2>&1 &
MON_GPU=$!
sleep 2

# 2) 启动背景流
echo "=== 启动 6G P3 背景流 ==="
start_bg

# 3) Warmup 5 分钟（数据丢弃）
echo "=== Warmup ${WARMUP_MINUTES} min（背景流运行中）==="
sleep $((WARMUP_MINUTES * 60))

# 4) LongLiu 模式（先跑）
echo "=== Mode 1: LongLiu（LL→CX）==="
launch_job A longliu $PORT_MAIN_A
launch_job B longliu $PORT_MAIN_B
sleep 15

# 5) CRUX 模式（后跑）
echo "=== Mode 2: CRUX（LL→CX）==="
launch_job A crux $PORT_MAIN_A
launch_job B crux $PORT_MAIN_B

# 6) 收尾
wait $MON_NIC 2>/dev/null || true
wait $MON_GPU 2>/dev/null || true
stop_bg

echo ""
echo "================================================================"
echo "Exp3 重跑完成"
echo "  数据: $OUTDIR"
echo "  epoch CSV: exp3_job[AB]_{longliu,crux}_rank0_epoch.csv"
echo "  NIC: $DATA_DIR/$RUN_ID/nic_10.csv, nic_226.csv"
echo "  GPU: $DATA_DIR/$RUN_ID/gpu_10.csv, gpu_226.csv"
echo "================================================================"
