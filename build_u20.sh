#!/bin/bash
set -e

cd /workspace/nccl

echo "=== container env ==="
ldd --version | head -n 1
nvcc --version

echo "=== start build ==="
rm -rf build && mkdir -p build
make -j8 src.build BUILDDIR=/workspace/nccl/build 2>&1 | tee /tmp/nccl_build_u20.log

echo "=== verify artifacts ==="
ls -la /workspace/nccl/build/lib/libnccl.so*

echo "=== verify DSCP symbols ==="
nm -C /workspace/nccl/build/lib/libnccl.so.2.18.3 | grep ncclDscpAdapter | head -n 5

echo "=== verify glibc deps ==="
objdump -T /workspace/nccl/build/lib/libnccl.so.2.18.3 | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -n 5

echo "=== DONE ==="
