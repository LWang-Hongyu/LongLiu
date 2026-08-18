#!/usr/bin/env python3
"""Solo multi_comm allreduce at a FIXED priority — for DSCP-on-wire verification.

Launches on both nodes (rank0 = 10.1 server, rank1 = 226 client) like
bench_bw_direct.py. While it runs, the wrapper script (verify_nccl_dscp.sh)
samples ethtool per-priority TX counters and tcpdump-captures the RoCEv2
outer IP TOS byte on 226 to confirm the DSCP NCCL actually puts on the wire.

Usage:
  RANK=0 MULTI_COMM_PORT=29900 MULTI_COMM_PRIOS=6 \
      python3 mc_solo_prio.py --prio 6
  RANK=1 MASTER_ADDR=192.10.10.110 MULTI_COMM_PORT=29900 MULTI_COMM_PRIOS=6 \
      python3 mc_solo_prio.py --prio 6

Env: RANK, LOCAL_RANK (GPU idx), MASTER_ADDR (rank0 TCP bind for id exchange),
     MULTI_COMM_PORT. MULTI_COMM_PRIOS limits which priority communicators are
     created (set to just the tested priority for fast init).
"""
import os
import sys
import time
import ctypes
import argparse

parser = argparse.ArgumentParser(description='Solo multi_comm allreduce at fixed priority')
parser.add_argument('--prio', type=int, required=True, help='Priority 0-6 (6=DSCP8/tc:0, 3=DSCP16/tc:2)')
parser.add_argument('--payload-mb', type=int, default=1024)
parser.add_argument('--iters', type=int, default=60)
parser.add_argument('--warmup', type=int, default=10)
args = parser.parse_args()

RANK = int(os.environ.get('RANK', '0'))
MASTER_ADDR = os.environ.get('MASTER_ADDR', '192.10.10.110')
PORT = int(os.environ.get('MULTI_COMM_PORT', '29500'))
CUDA_DEV = int(os.environ.get('LOCAL_RANK', '0'))

import torch
torch.cuda.set_device(CUDA_DEV)

lib = ctypes.CDLL('/home/why/LongLiu_rebuild/current/multi_comm_slo/build/libmulti_comm.so')

lib.multi_comm_init.restype = ctypes.c_int
lib.multi_comm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                ctypes.c_char_p, ctypes.c_int]
lib.multi_comm_set_priority.argtypes = [ctypes.c_int]
lib.multi_comm_set_priority.restype = ctypes.c_int
lib.multi_comm_allreduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_size_t, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int]
lib.multi_comm_allreduce.restype = ctypes.c_int

ret = lib.multi_comm_init(RANK, 2, b"0", MASTER_ADDR.encode(), PORT)
assert ret == 0, f"multi_comm_init failed: {ret}"

ret = lib.multi_comm_set_priority(args.prio)
assert ret == 0, f"multi_comm_set_priority({args.prio}) failed: {ret}"
print(f"[mc_solo_prio] rank={RANK} prio=P{args.prio} payload={args.payload_mb}MB "
      f"iters={args.iters}", flush=True)

num_elements = args.payload_mb * 1024 * 1024 // 4
data = torch.ones(num_elements, dtype=torch.float32, device='cuda')
bytes_per_iter = args.payload_mb * 1024 * 1024

for i in range(args.warmup + args.iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ret = lib.multi_comm_allreduce(
        ctypes.c_void_p(data.data_ptr()),
        ctypes.c_void_p(data.data_ptr()),
        ctypes.c_size_t(data.numel()),
        7, 0, 0)  # ncclFloat32, ncclSum, device 0
    assert ret == 0, f"allreduce failed: {ret}"
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if i >= args.warmup and RANK == 0 and (i - args.warmup) % 20 == 0:
        bw = (bytes_per_iter * 8 / 1e9) * 0.5 / (t1 - t0)
        print(f"  iter={i - args.warmup}  comm={(t1 - t0) * 1000:.1f}ms  bw={bw:.1f}Gbps",
              flush=True)

if RANK == 0:
    print("[mc_solo_prio] done", flush=True)
lib.multi_comm_destroy()
