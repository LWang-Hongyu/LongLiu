#!/usr/bin/env python3
"""
P4 V3: Asymmetric Workload — Job 2 (Computation-intensive, Loose SLO)

Workload: 80ms compute, 2048MB payload
SLO: c_i = 2.0 (loose)
GPU Intensity: I_2 = 80ms / 85ms ≈ 0.94 (computation-intensive)

This job is computation-intensive with a loose SLO requirement.
CRUX would assign higher priority (high GPU Intensity).
LongLiu should keep priority low since SLO is loose and Job1 is more urgent.
"""

import os
import sys
import time
import csv
import ctypes
import argparse
import torch

# ============================================================
# Parse args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='longliu',
                    choices=['longliu', 'crux'])
args = parser.parse_args()
MODE = args.mode

# ============================================================
# Job 2 constants (asymmetric: computation-intensive)
# ============================================================
PAYLOAD_MB = 2048           # 2 GB fp32
SLEEP_US   = 80000          # 80 ms simulated compute (computation-intensive)
SLO_C_I    = 2.0            # Loose SLO
NUM_ITERS  = 300
ITERS_PER_EPOCH = 20
NUM_EPOCHS = NUM_ITERS // ITERS_PER_EPOCH  # 15

BYTES_PER_ITER = PAYLOAD_MB * 1024 * 1024
NUM_ELEMENTS = BYTES_PER_ITER // 4  # float32

# CRUX: Static GPU Intensity-based priority
# Job1: I_1 = 30ms / 85ms ≈ 0.35 → assign P3 (lower priority)
# Job2: I_2 = 80ms / 85ms ≈ 0.94 → assign P4 (higher priority)
CRUX_STATIC_PRIORITY = 4  # P4 (DSCP=32)

def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    """Compute per-direction wire bandwidth for NCCL Ring AllReduce."""
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur

# ============================================================
# Scheduler setup
# ============================================================
import torch.distributed as dist
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', 'multi_comm_slo', 'src'))
from slo_scheduler import MultiCommWrapper, SLOScheduler

if MODE == 'longliu':
    _scheduler = SLOScheduler(slo_threshold=SLO_C_I)
else:  # crux
    class StaticPriorityScheduler:
        def __init__(self, static_priority):
            self.current_priority = static_priority
            self.priority_history = [self.current_priority]
        def update(self, actual_comm_time: float, data_size: float) -> int:
            return self.current_priority
        def get_dscp(self) -> int:
            return self.current_priority * 8
    _scheduler = StaticPriorityScheduler(CRUX_STATIC_PRIORITY)

_mc_wrapper = None

def epoch_start(epoch):
    if _mc_wrapper is not None:
        _mc_wrapper.epoch_start(epoch)

def epoch_end(epoch):
    if _mc_wrapper is not None:
        _mc_wrapper.epoch_end(epoch, data_size=BYTES_PER_ITER)

def allreduce(tensor):
    if _mc_wrapper is not None:
        _mc_wrapper.allreduce(tensor.data_ptr(), tensor.data_ptr(),
                               tensor.numel(), 0, 0, 0)
    else:
        dist.all_reduce(tensor)

def main():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    master_addr = os.environ.get('MASTER_ADDR', '192.10.10.110')
    port = int(os.environ.get('MULTI_COMM_PORT',
                os.environ.get('MASTER_PORT', '29500')))

    device_idx = 0
    torch.cuda.set_device(device_idx)
    device = torch.cuda.current_device()

    global _mc_wrapper
    _mc_wrapper = MultiCommWrapper(
        _scheduler, rank, world_size, str(device_idx),
        master_addr, port)

    if rank == 0:
        print(f"[Job2-{MODE.upper()}] MODE={MODE}, Payload={PAYLOAD_MB}MB, "
              f"sleep={SLEEP_US}us, SLO c_i={SLO_C_I}")
        if MODE == 'crux':
            print(f"[Job2-CRUX] Static Priority: P{CRUX_STATIC_PRIORITY} "
                  f"(DSCP={CRUX_STATIC_PRIORITY*8})")
            print(f"[Job2-CRUX] GPU Intensity I_2 = {SLEEP_US/1000:.0f}ms / ~85ms ≈ 0.94")
        print(f"[Job2-{MODE.upper()}] {NUM_ITERS} iters, {ITERS_PER_EPOCH}/epoch, "
              f"{NUM_EPOCHS} epochs")

    if rank == 0 and device is not None:
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"[Job2-{MODE.upper()}] GPU {device}: {torch.cuda.get_device_name(device)}, "
              f"free={free_mem:.1f} GB")

    # ============================================================
    # Allocate tensor
    # ============================================================
    tensor = torch.ones(NUM_ELEMENTS, dtype=torch.float32, device='cuda')

    # Warmup
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device='cuda')
    for _ in range(2):
        allreduce(warmup)
    torch.cuda.synchronize()
    if rank == 0:
        print("[Job2] Warmup done.")

    # ============================================================
    # Main training loop
    # ============================================================
    results = []
    t_total_start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        epoch_start(epoch)

        for i in range(ITERS_PER_EPOCH):
            global_iter = epoch * ITERS_PER_EPOCH + i

            # Simulated forward/backward compute
            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)

            # AllReduce (communication)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            allreduce(tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            comm_dur = t1 - t0
            bw_gbps = bus_bw_gbps(BYTES_PER_ITER, comm_dur, world_size) if comm_dur > 0 else 0.0

            # All iters are contested (Job2 joins after Job1 has started)
            phase = 'contested'

            if rank == 0:
                print(f"[Job2-{MODE.upper()}] iter {global_iter:2d} (epoch {epoch}, "
                      f"iter {i}): comm={comm_dur*1000:.1f}ms, "
                      f"bw={bw_gbps:.2f} Gbps [{phase}]")

            results.append({
                'iter': global_iter,
                'epoch': epoch,
                'comm_dur_s': round(comm_dur, 6),
                'bw_gbps': round(bw_gbps, 4),
                'phase': phase,
            })

        epoch_end(epoch)

    t_total_end = time.perf_counter()

    # ============================================================
    # Save results
    # ============================================================
    # Job2 is rank 1, but we still save results for analysis
    csv_path = f'p4_job2_asym_{MODE}_rank{rank}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'epoch', 'comm_dur_s',
                         'bw_gbps', 'phase'])
        for r in results:
            writer.writerow([r['iter'], r['epoch'],
                             r['comm_dur_s'], r['bw_gbps'],
                             r['phase']])
    print(f"[Job2-{MODE.upper()}] Results saved to {csv_path}")

    # Summary
    contested_iters = [r for r in results if r['phase'] == 'contested']

    if contested_iters:
        avg_cont = sum(r['comm_dur_s'] for r in contested_iters) / len(contested_iters)
        avg_bw_cont = sum(r['bw_gbps'] for r in contested_iters) / len(contested_iters)
        print(f"[Job2-{MODE.upper()}] Contested ({len(contested_iters)} iters): "
              f"avg_comm={avg_cont*1000:.1f}ms, avg_bw={avg_bw_cont:.2f} Gbps")

    print(f"[Job2-{MODE.upper()}] Total wall time: {t_total_end - t_total_start:.1f}s")

    # Cleanup
    if _mc_wrapper is not None:
        _mc_wrapper.destroy()

if __name__ == '__main__':
    main()
