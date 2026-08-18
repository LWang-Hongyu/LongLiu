#!/bin/bash
# ============================================================================
# Sync experiment scripts + multi_comm_slo to remote node 226
# ============================================================================
# 目标目录必须与 run_p4.sh 中的 SCRIPT_DIR 完全一致（226 上 cd '$SCRIPT_DIR'
# 才能成功）。两节点统一使用 current/ 路径。
# Usage: bash sync_to_226.sh
# ============================================================================

set -e

NODE_226="192.10.10.226"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/home/why/LongLiu_rebuild/current/experiments_evaluation/P4_dumbbell_slo"

echo "=== Syncing experiment scripts to $NODE_226 ==="
ssh "$NODE_226" "mkdir -p '$REMOTE_DIR'"

# 全量同步脚本目录（排除缓存/结果/log）
rsync -avz --delete \
    --exclude '*.csv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '/tmp' \
    "$SCRIPT_DIR/" \
    "$NODE_226:$REMOTE_DIR/"

# ============================================================================
# 同步 multi_comm_slo（scheduler + libmulti_comm.so + C 源码）
# 两节点统一位于 current/ 下，与脚本目录的相对位置保持一致
# ============================================================================
echo "=== Syncing multi_comm_slo to $NODE_226 ==="
LOCAL_MC="/home/why/LongLiu_rebuild/current/multi_comm_slo"
REMOTE_MC="/home/why/LongLiu_rebuild/current/multi_comm_slo"
ssh "$NODE_226" "mkdir -p '$REMOTE_MC'"
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$LOCAL_MC/" \
    "$NODE_226:$REMOTE_MC/"

echo "=== Sync complete ==="
echo ""
echo "Verify with:"
echo "  ssh $NODE_226 'ls -la $REMOTE_DIR/ | head'"
echo "  ssh $NODE_226 'ls -la $REMOTE_MC/build/ $REMOTE_MC/src/'"
