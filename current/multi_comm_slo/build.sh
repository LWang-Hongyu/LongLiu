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
echo "  from src.slo_scheduler import init_slo, epoch_start, epoch_end"
echo "  mc = init_slo(rank=0, world_size=2, device_list='0', slo_threshold=1.5)"
echo "  mc.epoch_start(epoch)"
echo "  # ... do allreduce ..."
echo "  mc.epoch_end(epoch, data_size=2048*1024*1024)"
