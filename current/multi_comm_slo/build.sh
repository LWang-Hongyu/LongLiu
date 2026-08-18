#!/bin/bash
# Build script for multi_comm_slo
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/src"

echo "Building libmulti_comm.so..."
make clean
make

echo ""
echo "Build complete: $SCRIPT_DIR/build/libmulti_comm.so"
echo ""
echo "Usage in Python:"
echo "  from slo_scheduler import SLOScheduler, MultiCommWrapper"
echo "  sched = SLOScheduler(slo_threshold=1.5)"
echo "  mc = MultiCommWrapper(sched, rank=0, world_size=2, device_list='0',"
echo "                        master_addr='192.10.10.110', port=29500)"
echo "  for window in range(num_windows):"
echo "      mc.window_start(window)"
echo "      # ... do allreduce ...  (内部自动累计窗口纯通信时间)"
echo "      mc.window_end(window, data_size=2048*1024*1024)"
