#!/usr/bin/env python3
"""Direct bandwidth test using libmulti_comm.so (bypass PyTorch NCCL).
Launches on both nodes via subprocess, uses the multi_comm C library directly.
"""
import os, sys, time, json, ctypes, numpy as np

PAYLOAD_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
RANK = int(os.environ.get('RANK', '0'))
MASTER_ADDR = os.environ.get('MASTER_ADDR', '192.10.10.110')
PORT = int(os.environ.get('MULTI_COMM_PORT', '29500'))
CUDA_DEV = int(os.environ.get('LOCAL_RANK', '0'))

import torch
torch.cuda.set_device(CUDA_DEV)

# Load multi_comm library
lib = ctypes.CDLL('/home/why/LongLiu_rebuild/multi_comm_slo/build/libmulti_comm.so')

# Init
lib.multi_comm_init.restype = ctypes.c_int
lib.multi_comm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                 ctypes.c_char_p, ctypes.c_int]

ret = lib.multi_comm_init(RANK, 2, b"0", 
                           MASTER_ADDR.encode(), PORT)
assert ret == 0, f"multi_comm_init failed: {ret}"

# Set priority to P4 (DSCP=0, middle-of-the-road for solo testing)
lib.multi_comm_set_priority(4)

# Allocate tensor
num_elements = PAYLOAD_MB * 1024 * 1024 // 4
data = torch.ones(num_elements, dtype=torch.float32, device='cuda')

# Warmup
for _ in range(5):
    lib.multi_comm_allreduce(
        ctypes.c_void_p(data.data_ptr()),
        ctypes.c_void_p(data.data_ptr()),
        ctypes.c_size_t(data.numel()),
        0, 0, 0)
torch.cuda.synchronize()

# Benchmark
WARMUP = 10
ITERS = 100
times = []
bytes_per_iter = PAYLOAD_MB * 1024 * 1024

for i in range(WARMUP + ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lib.multi_comm_allreduce(
        ctypes.c_void_p(data.data_ptr()),
        ctypes.c_void_p(data.data_ptr()),
        ctypes.c_size_t(data.numel()),
        0, 0, 0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_s = t1 - t0
    if i >= WARMUP:
        times.append(elapsed_s)
    if RANK == 0 and (i - WARMUP) % 20 == 0:
        bw = (bytes_per_iter * 8 / 1e9) * 0.5 / elapsed_s  # world_size=2
        print(f"[{PAYLOAD_MB}MB] iter={i-WARMUP}  comm={elapsed_s*1000:.1f}ms  bw={bw:.1f}Gbps")

if RANK == 0:
    avg_s = sum(times) / len(times)
    avg_bw = (bytes_per_iter * 8 / 1e9) * 0.5 / avg_s
    print(f"\n=== RESULT: {PAYLOAD_MB}MB ===")
    print(f"  Avg comm: {avg_s*1000:.1f}ms")
    print(f"  Avg bw:   {avg_bw:.1f} Gbps")
    print(f"  Line util: {avg_bw/50*100:.0f}% (50Gbps link)")

lib.multi_comm_destroy()
