#!/bin/bash
# ============================================================================
# sync_to_226.sh — 同步实验脚本与调度器到 226 节点
# ============================================================================
# 226 使用旧式扁平布局（无 current/ 前缀），同步后：
#   /home/why/LongLiu_rebuild/experiments_validation/   （本目录脚本）
#   /home/why/LongLiu_rebuild/multi_comm_slo/{src,build}（调度器 + .so）
#
# Usage:
#   bash sync_to_226.sh
# ============================================================================
set -e

NODE_226="192.10.10.226"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/home/why/LongLiu_rebuild/experiments_validation"

echo "=== Syncing experiments_validation to $NODE_226 ==="
ssh "$NODE_226" "mkdir -p '$REMOTE_DIR'"

rsync -avz --delete \
    --exclude 'data/' \
    --exclude '*/data/' \
    --exclude '__pycache__/' \
    "$LOCAL_DIR/" "$NODE_226:$REMOTE_DIR/"

echo ""
echo "=== Syncing scheduler + libmulti_comm.so to $NODE_226 ==="
SCHED_SRC="/home/why/LongLiu_rebuild/current/multi_comm_slo/src"
REMOTE_SCHED="/home/why/LongLiu_rebuild/multi_comm_slo/src"
REMOTE_BUILD="/home/why/LongLiu_rebuild/multi_comm_slo/build"

rsync -avz "$SCHED_SRC/slo_scheduler.py" "$NODE_226:$REMOTE_SCHED/"
rsync -avz "$SCHED_SRC/multi_comm.c" "$NODE_226:$REMOTE_SCHED/"
rsync -avz "$SCHED_SRC/multi_comm.h" "$NODE_226:$REMOTE_SCHED/"
rsync -avz "$SCHED_SRC/Makefile" "$NODE_226:$REMOTE_SCHED/"
# vendored 头文件（规避 226 端 /usr/local/include/nccl.h v21700 遮蔽）
rsync -avz -r "$SCHED_SRC/include" "$NODE_226:$REMOTE_SCHED/"
rsync -avz "$SCHED_SRC/../build/libmulti_comm.so" "$NODE_226:$REMOTE_BUILD/"

# 226 端也重建一次 .so，确保链接到 226 本机系统 NCCL（libnccl.so.2.30.7）
ssh "$NODE_226" "cd $REMOTE_SCHED && make clean >/dev/null 2>&1; make 2>&1 | tail -3 && nm -D ../build/libmulti_comm.so | grep multi_comm_allgather"

echo ""
echo "=== Sync complete ==="
ssh "$NODE_226" "ls -la '$REMOTE_DIR/' && ls -la '$REMOTE_SCHED/' '$REMOTE_BUILD/'"
