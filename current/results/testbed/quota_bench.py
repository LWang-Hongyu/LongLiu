#!/usr/bin/env python3
"""LongLiu Quota Bandwidth Benchmark - pure AllReduce with hardcoded quota"""
import os, time, torch, torch.distributed as dist

QUOTA = int(os.environ.get("LONGLIU_HARDCODE_QUOTA", "0"))
WARMUP = 10
ITERS = 200

if QUOTA > 0:
    os.environ["LONGLIU_ENABLED"] = "1"
    os.environ["LONGLIU_C_I"] = "1.5"
    os.environ["LONGLIU_HARDCODE_QUOTA"] = str(QUOTA)
    mode = f"hardcoded_quota={QUOTA}"
else:
    ll_enabled = os.environ.get("LONGLIU_ENABLED", "0")
    mode = "LongLiu_dynamic" if ll_enabled == "1" else "baseline_no_LL"

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
print(f"[rank={rank}] mode={mode}", flush=True)

data = torch.randn(50 * 1024 * 1024 // 4, device="cuda")
torch.cuda.synchronize()

times = []
for i in range(WARMUP + ITERS):
    torch.cuda.synchronize()
    t0 = time.monotonic()
    dist.all_reduce(data)
    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - t0) * 1000
    if i >= WARMUP:
        times.append(elapsed_ms)
    if rank == 0 and i % 40 == 0:
        print(f"[{mode}] iter={i-WARMUP} {elapsed_ms:.1f}ms", flush=True)

dist.barrier()
avg = sum(times) / len(times)
p95 = sorted(times)[int(len(times) * 0.95)]
print(f"\n[RESULT] {mode} avg={avg:.2f}ms p95={p95:.2f}ms n={len(times)}", flush=True)
dist.destroy_process_group()
print(f"[rank={rank}] exit", flush=True)
