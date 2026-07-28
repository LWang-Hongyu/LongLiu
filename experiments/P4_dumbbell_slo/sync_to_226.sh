#!/bin/bash
# ============================================================================
# Sync experiment scripts to remote node 226
# ============================================================================
# Run this from the experiment directory to push the latest scripts to 226.
# Usage: bash sync_to_226.sh
# ============================================================================

set -e

NODE_226="192.10.10.226"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo"

echo "=== Syncing experiment scripts to $NODE_226 ==="

# Create remote directory
ssh "$NODE_226" "mkdir -p '$REMOTE_DIR'"

# Sync all experiment scripts and configs
rsync -avz \
    "$SCRIPT_DIR/p4_job1.py" \
    "$SCRIPT_DIR/p4_job2.py" \
    "$SCRIPT_DIR/p4_job1_asym.py" \
    "$SCRIPT_DIR/p4_job2_asym.py" \
    "$SCRIPT_DIR/p4_job_reverse.py" \
    "$SCRIPT_DIR/run_p4.sh" \
    "$SCRIPT_DIR/run_p4_asym.sh" \
    "$SCRIPT_DIR/run_p4_asym_v2.sh" \
    "$SCRIPT_DIR/run_p4_reverse.sh" \
    "$SCRIPT_DIR/plot_p4.py" \
    "$SCRIPT_DIR/analyze_results.py" \
    "$SCRIPT_DIR/sync_to_226.sh" \
    "$NODE_226:$REMOTE_DIR/"

# Also sync the updated scheduler (lives outside this dir)
SCHED_DIR="/home/why/LongLiu_rebuild/multi_comm_slo/src"
REMOTE_SCHED_DIR="/home/why/LongLiu_rebuild/multi_comm_slo/src"
echo "=== Syncing scheduler and libmulti_comm to $NODE_226 ==="
rsync -avz \
    "$SCHED_DIR/slo_scheduler.py" \
    "$NODE_226:$REMOTE_SCHED_DIR/"
# Sync compiled libmulti_comm.so
rsync -avz \
    "$SCHED_DIR/../build/libmulti_comm.so" \
    "$NODE_226:$REMOTE_SCHED_DIR/../build/"

# Sync updated shell scripts
echo "=== Syncing updated shell scripts ==="
rsync -avz \
    "$SCRIPT_DIR/run_v6_calibrate.sh" \
    "$SCRIPT_DIR/run_v6_calib_atomic.sh" \
    "$SCRIPT_DIR/run_v6_full.sh" \
    "$SCRIPT_DIR/run_v6_p4cap_llarm.sh" \
    "$SCRIPT_DIR/v6_background_flow.sh" \
    "$SCRIPT_DIR/probe_dscp_priority.sh" \
    "$SCRIPT_DIR/verify_dscp_queue.sh" \
    "$SCRIPT_DIR/run_diagnose.sh" \
    "$NODE_226:$REMOTE_DIR/"

echo "=== Sync complete ==="
echo ""
echo "Verify with: ssh $NODE_226 'ls -la $REMOTE_DIR/ && ls -la $REMOTE_SCHED_DIR/'"
