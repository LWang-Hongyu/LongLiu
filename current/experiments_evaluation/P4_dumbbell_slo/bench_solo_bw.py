#!/usr/bin/env python3
"""Solo bandwidth benchmark: test payload sizes to see how close to line rate we get."""
import os, sys, time, torch, torch.distributed as dist

PAYLOAD_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 1024

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

bytes_per_iter = PAYLOAD_MB * 1024 * 1024
data = torch.ones(bytes_per_iter // 4, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()

WARMUP = 10
ITERS = 100

times = []
for i in range(WARMUP + ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(data)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_s = t1 - t0
    if i >= WARMUP:
        times.append(elapsed_s)
    if rank == 0 and (i - WARMUP) % 20 == 0:
        bw_gbps = (bytes_per_iter * 8 / 1e9) * (dist.get_world_size() - 1) / dist.get_world_size() / elapsed_s
        print(f"[{PAYLOAD_MB}MB] iter={i-WARMUP}  comm={elapsed_s*1000:.1f}ms  bw={bw_gbps:.1f}Gbps")

if rank == 0:
    avg_s = sum(times) / len(times)
    avg_bw = (bytes_per_iter * 8 / 1e9) * (dist.get_world_size() - 1) / dist.get_world_size() / avg_s
    print(f"\n=== RESULT: {PAYLOAD_MB}MB ===")
    print(f"  Avg comm: {avg_s*1000:.1f}ms")
    print(f"  Avg bw:   {avg_bw:.1f} Gbps")
    print(f"  Line util: {avg_bw/50*100:.0f}% (50Gbps link)")

dist.destroy_process_group()
