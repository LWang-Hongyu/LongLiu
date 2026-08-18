#!/usr/bin/env python3
"""
Benchmark payload sizes using the SAME MultiCommWrapper path as the experiment.
Measures solo allreduce time for various payloads to find the performance cliff.
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', 'multi_comm_slo', 'src'))
from slo_scheduler import MultiCommWrapper, SLOScheduler

rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
device_idx = 0
torch.cuda.set_device(device_idx)
device = torch.cuda.current_device()

# Dummy scheduler (we only care about comm time, not priority)
scheduler = SLOScheduler(slo_threshold=1.0, target_comm_time_ms=1000, preset_target=True)
mc = MultiCommWrapper(scheduler, rank, world_size, str(0),
                       os.environ.get('MASTER_ADDR', '192.10.10.110'),
                       int(os.environ.get('MULTI_COMM_PORT',
                           os.environ.get('MASTER_PORT', '29500'))))

if rank == 0:
    print(f"Benchmarking via MultiCommWrapper on {torch.cuda.get_device_name(device)}")
    print(f"{'payload_mb':>10} | {'avg_ms':>10} | {'bw_gbps':>10} | {'duty_30ms':>10}")
    print("-" * 55)

sizes = [64, 128, 256, 512, 768, 1024]
compute_s = 0.030

for payload_mb in sizes:
    num_elements = payload_mb * 1024 * 1024 // 4
    tensor = torch.ones(num_elements, dtype=torch.float32, device=device)

    # Warmup 3 iters
    for _ in range(3):
        mc.allreduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), 7, 0, 0)  # ncclFloat32
    torch.cuda.synchronize()

    # Measure 5 iters
    times = []
    for _ in range(5):
        if SLEEP_US := int(os.environ.get('SLEEP_US', '30000')):
            torch.cuda._sleep(SLEEP_US)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mc.allreduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), 7, 0, 0)  # ncclFloat32
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_s = sum(times) / len(times)
    avg_ms = avg_s * 1000
    bytes_total = payload_mb * 1024 * 1024
    bw_gbps = (bytes_total * 8 / 1e9) * 1 / world_size / avg_s if avg_s > 0 else 0
    duty = avg_s / (avg_s + compute_s)

    if rank == 0:
        flag = " <-- V4" if payload_mb == 512 else (" <-- V5 target" if payload_mb == 1024 else "")
        print(f"{payload_mb:>10} | {avg_ms:>10.1f} | {bw_gbps:>10.2f} | {duty:>10.3f}{flag}")

    del tensor

mc.destroy()
if rank == 0:
    print("\nDone.")
