import os, time, torch, torch.distributed as dist, ctypes

# Try loading LongLiu signals from libnccl.so
try:
    libc = ctypes.CDLL("libnccl.so.2")
    longliu_start = libc.ncclLongLiuIterStart
    longliu_end = libc.ncclLongLiuIterEnd
except Exception:
    longliu_start = None
    longliu_end = None

TARGET_MS = int(os.environ.get("LONGLIU_TARGET_MS", "100"))
WARMUP = 10
ITERS = 200

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
print(f"[rank={rank}] target={TARGET_MS}ms longliu={longliu_start is not None}", flush=True)

# 50MB AllReduce
data = torch.randn(50 * 1024 * 1024 // 4, device="cuda")
torch.cuda.synchronize()

times = []
for i in range(WARMUP + ITERS):
    if longliu_start:
        longliu_start()
    torch.cuda.synchronize()
    t0 = time.monotonic()
    dist.all_reduce(data)
    torch.cuda.synchronize()
    if longliu_end:
        longliu_end()
    elapsed_ms = (time.monotonic() - t0) * 1000
    if i >= WARMUP:
        times.append(elapsed_ms)
        if i % 20 == 0:
            avg = sum(times[-20:]) / len(times[-20:])
            print(f"[rank={rank}] i={i-WARMUP} {elapsed_ms:.1f}ms avg20={avg:.1f}ms", flush=True)

dist.barrier()
avg = sum(times) / len(times)
p95 = sorted(times)[int(len(times) * 0.95)]
print(f"\n[RESULT] target={TARGET_MS}ms avg={avg:.1f}ms p95={p95:.1f}ms n={len(times)}", flush=True)
dist.destroy_process_group()
print(f"[rank={rank}] exit", flush=True)
