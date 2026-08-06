#!/bin/bash
# V6 背景流校准 — 原子化执行，避免终端间干扰
# 注意：不使用 set -e — bc 在空值时可能返回非零，
# 但脚本应继续执行。关键步骤（job 启动、等待）有手动 exit check。
set -uo pipefail

BG_RATE_GBPS=${1:-30}
BG_DURATION=${2:-400}

PAYLOAD_MB=1024
CI_TIGHT=1.2
CI_LOOSE=3.0
EXP_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"
NODE_226="192.10.10.226"
RDMA_226="192.10.10.226"
DSCP_P3_TOS=64  # P3 → DSCP=16 → TOS=64

echo "================================================================"
echo "V6 背景流校准（原子化执行）"
echo "  12路并行 UDP, 每路 $((BG_RATE_GBPS * 1000 / 12)) Mbps, DSCP=P3"
echo "  目标总吞吐: ${BG_RATE_GBPS} Gbps"
echo "  持续: ${BG_DURATION}s"
echo "  CRUX c_i: tight=${CI_TIGHT}, loose=${CI_LOOSE}"
echo "================================================================"
echo ""

cd "$EXP_DIR"

# 清理残留（精确 PID，禁用 pkill -9 -f 宽匹配）
echo "--- 清理残留进程 ---"
for PID in $(pgrep -f "p4_job_reverse.py" 2>/dev/null); do
    kill $PID 2>/dev/null || true
done
ssh $NODE_226 "for PID in \$(pgrep -f 'p4_job_reverse.py' 2>/dev/null); do kill \$PID 2>/dev/null; done" 2>/dev/null || true
# iperf3 服务器进程用精确模式清理
for PORT in $(seq 6200 6211); do
    PID=$(ssh $NODE_226 "pgrep -f 'iperf3 -s -p $PORT' 2>/dev/null" 2>/dev/null || true)
    if [[ -n "$PID" ]]; then
        ssh $NODE_226 "kill $PID" 2>/dev/null || true
    fi
done
# 清理本地残留 iperf3 客户端（仅匹配背景流端口模式）
for PID in $(pgrep -f "iperf3.*-p 62[0-9][0-9].*-u" 2>/dev/null); do
    kill $PID 2>/dev/null || true
done
sleep 3
echo "清理完成"

# 启动 iperf3 服务器
echo "--- 启动 12 路 iperf3 服务器 (226) ---"
for PORT in $(seq 6200 6211); do
    ssh -o ConnectTimeout=5 $NODE_226 "iperf3 -s -p $PORT -D -B $RDMA_226" 2>/dev/null
done
sleep 2
SRV_COUNT=$(ssh $NODE_226 "pgrep -a iperf3 | grep '\-s' | wc -l")
echo "  服务器运行数: $SRV_COUNT"

# 启动 12 路并行 UDP 背景流
echo "--- 启动背景流: 12路 UDP, DSCP=P3 ---"
PER_STREAM_RATE=$((BG_RATE_GBPS * 1000 / 12))
for PORT in $(seq 6200 6211); do
    iperf3 -c $RDMA_226 -u -b ${PER_STREAM_RATE}M -t $BG_DURATION \
        --tos $DSCP_P3_TOS -p $PORT -f g -l 8900 \
        > /tmp/v6_bgflow_${PORT}.log 2>&1 &
done
echo "  12 路客户端已启动（每路 ${PER_STREAM_RATE} Mbps）"
sleep 15

# 检查背景流吞吐（15s 数据足够出第一行报告）
echo "--- 背景流吞吐确认（15s 采样） ---"
TOTAL=0
for PORT in $(seq 6200 6211); do
    GBPS=$(grep -oP '[\d.]+(?= Gbits/sec)' /tmp/v6_bgflow_${PORT}.log 2>/dev/null | head -1)
    if [[ -n "$GBPS" ]] && [[ "$GBPS" != "0.00" ]]; then
        TOTAL=$(echo "$TOTAL + $GBPS" | bc 2>/dev/null)
        printf "  Port %4d: %s Gbps\n" $PORT "$GBPS"
    else
        printf "  Port %4d: starting...\n" $PORT
    fi
done
printf "  总吞吐: ~%.1f Gbps\n" "${TOTAL:-0}"
echo ""

# 启动 CRUX 实验
echo "--- 启动 CRUX 校准实验 ---"
rm -f p4_job[AB]_reverse_crux_v6cal_*.log p4_job[AB]_reverse_crux_rank*_*.csv

# Job A rank 0 (10.1)
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=29620 \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=29620 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u p4_job_reverse.py --job A --mode crux --phase main \
        --ttarget-file /tmp/ttarget_v5_jobA.json \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobA_reverse_crux_v6cal_node101.log 2>&1 &
JOB_A_101_PID=$!

# Job A rank 1 (226)
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=29620 \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=29620 \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    python3 -u p4_job_reverse.py --job A --mode crux --phase main \
        --ttarget-file /tmp/ttarget_v5_jobA.json \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobA_reverse_crux_v6cal_node226.log 2>&1" &
JOB_A_226_PID=$!

echo "Job A launched (PIDs: $JOB_A_101_PID on 10.1, $JOB_A_226_PID on 226)"
echo "等待 12s 让 Job A 初始化..."
sleep 12

# Job B rank 0 (10.1)
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=29621 \
    WORLD_SIZE=2 RANK=0 \
    MULTI_COMM_PORT=29621 \
    NCCL_SOCKET_IFNAME=enp130s0f0np0 \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE \
    python3 -u p4_job_reverse.py --job B --mode crux --phase main \
        --ttarget-file /tmp/ttarget_v5_jobB.json \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobB_reverse_crux_v6cal_node101.log 2>&1 &
JOB_B_101_PID=$!

# Job B rank 1 (226)
ssh $NODE_226 "cd $EXP_DIR && \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=192.10.10.110 MASTER_PORT=29621 \
    WORLD_SIZE=2 RANK=1 \
    MULTI_COMM_PORT=29621 \
    NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO \
    NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_SOCKET_IFNAME=enp59s0f0np0 \
    LD_LIBRARY_PATH=/home/why/LongLiu_rebuild/nccl-master/build/lib:\$LD_LIBRARY_PATH \
    PYTHONPATH=/home/why/LongLiu_rebuild/multi_comm_slo/src:\$PYTHONPATH \
    python3 -u p4_job_reverse.py --job B --mode crux --phase main \
        --ttarget-file /tmp/ttarget_v5_jobB.json \
        --payload-mb $PAYLOAD_MB --ci-phase1 $CI_TIGHT --ci-phase2 $CI_LOOSE \
        --crux-priority-a 3 --crux-priority-b 3 \
        > p4_jobB_reverse_crux_v6cal_node226.log 2>&1" &
JOB_B_226_PID=$!

echo "Job B launched"
echo "等待 Job A 和 Job B 完成（~90s）..."
echo "背景流持续 ${BG_DURATION}s（已在后台运行）"
echo ""

# 等待两个 job 完成
wait $JOB_A_101_PID; echo "Job A 10.1 done (exit=$?)"
wait $JOB_A_226_PID; echo "Job A 226 done (exit=$?)"
wait $JOB_B_101_PID; echo "Job B 10.1 done (exit=$?)"
wait $JOB_B_226_PID; echo "Job B 226 done (exit=$?)"

echo ""
echo "================================================================"
echo "校准实验结果"
echo "================================================================"

# 输出 epoch 汇总
echo ""
echo "=== Job A per-epoch ==="
cat p4_jobA_reverse_crux_rank0_epoch.csv 2>/dev/null || echo "CSV not found"

echo ""
echo "=== Job B per-epoch ==="
cat p4_jobB_reverse_crux_rank0_epoch.csv 2>/dev/null || echo "CSV not found"

echo ""
echo "=== 背景流总吞吐 ==="
TOTAL=0
for PORT in $(seq 6200 6211); do
    GBPS=$(tail -3 /tmp/v6_bgflow_${PORT}.log 2>/dev/null | grep -oP '[\d.]+(?= Gbits/sec)' | tail -1)
    if [[ -n "$GBPS" ]]; then
        TOTAL=$(echo "$TOTAL + $GBPS" | bc)
    fi
done
echo "  总吞吐: ~${TOTAL} Gbps"

echo ""
echo "================================================================"
echo "校准完成"
echo "  tight job Phase 2 avg_comm / (c_i × T_target_per_iter) 应 ≥1.3"
echo "================================================================"
