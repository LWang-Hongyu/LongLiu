#!/usr/bin/env python3
import os, sys, time, torch, torch.distributed as dist

print(f"[debug] rank env start", flush=True)
print(f"[debug] RANK={os.environ.get('RANK')}", flush=True)
print(f"[debug] LOCAL_RANK={os.environ.get('LOCAL_RANK')}", flush=True)
print(f"[debug] WORLD_SIZE={os.environ.get('WORLD_SIZE')}", flush=True)
print(f"[debug] MASTER_ADDR={os.environ.get('MASTER_ADDR')}", flush=True)
print(f"[debug] MASTER_PORT={os.environ.get('MASTER_PORT')}", flush=True)
print(f"[debug] LD_PRELOAD={os.environ.get('LD_PRELOAD')}", flush=True)
print(f"[debug] NCCL_DEBUG={os.environ.get('NCCL_DEBUG')}", flush=True)
print(f"[debug] NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA')}", flush=True)
print(f"[debug] torch.version.cuda={torch.version.cuda}", flush=True)
print(f"[debug] torch.__version__={torch.__version__}", flush=True)

print(f"[debug] checking NCCL version via ctypes", flush=True)
import ctypes
try:
    n = ctypes.CDLL('libnccl.so.2')
    v = ctypes.c_int()
    n.ncclGetVersion(ctypes.byref(v))
    print(f"[debug] ncclGetVersion={v.value}", flush=True)
except Exception as e:
    print(f"[debug] ncclGetVersion error: {e}", flush=True)

print(f"[debug] calling torch.cuda.set_device", flush=True)
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
print(f"[debug] cuda device set ok", flush=True)

print(f"[debug] calling init_process_group", flush=True)
dist.init_process_group("nccl")
print(f"[debug] init_process_group ok", flush=True)

rank = dist.get_rank()
print(f"[debug] rank={rank}", flush=True)

data = torch.randn(1024, device="cuda")
torch.cuda.synchronize()
print(f"[debug] starting all_reduce", flush=True)
dist.all_reduce(data)
torch.cuda.synchronize()
print(f"[debug] all_reduce ok", flush=True)

dist.destroy_process_group()
print(f"[debug] rank={rank} exit", flush=True)
