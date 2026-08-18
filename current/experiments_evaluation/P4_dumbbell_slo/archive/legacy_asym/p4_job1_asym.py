#!/usr/bin/env python3
"""
P4 V3: Asymmetric Workload — Job 1 (Communication-intensive, Strict SLO)

Workload: 30ms compute, 2048MB payload
SLO: c_i = 1.2 (very strict)
GPU Intensity: I_1 = 30ms / 85ms ≈ 0.35 (communication-intensive)

This job is communication-intensive but has a strict SLO requirement.
CRUX would assign lower priority (low GPU Intensity).
LongLiu should dynamically boost priority to meet strict SLO.
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
# Job 1 constants (asymmetric: communication-intensive)
# ============================================================
PAYLOAD_MB = 2048           # 2 GB fp32
SLEEP_US   = 30000          # 30 ms simulated compute (communication-intensive)
SLO_C_I    = 1.2            # VERY strict SLO
NUM_ITERS  = 300
ITERS_PER_WINDOW = 20
NUM_WINDOWS = NUM_ITERS // ITERS_PER_WINDOW  # 15

BYTES_PER_ITER = PAYLOAD_MB * 1024 * 1024
NUM_ELEMENTS = BYTES_PER_ITER // 4  # float32

# CRUX: Static GPU Intensity-based priority
# Job1: I_1 = 30ms / 85ms ≈ 0.35 → assign P3 (lower priority)
# Job2: I_2 = 80ms / 85ms ≈ 0.94 → assign P4 (higher priority)
CRUX_STATIC_PRIORITY = 3  # P3 (DSCP=24)

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
        def update(self, actual_comm_time: float, data_size: float,
                   window_comm_time: float = 0.0) -> int:
            return self.current_priority
        def get_dscp(self) -> int:
            return self.current_priority * 8
    _scheduler = StaticPriorityScheduler(CRUX_STATIC_PRIORITY)

_mc_wrapper = None

def window_start(window):
    if _mc_wrapper is not None:
        _mc_wrapper.window_start(window)

def window_end(window):
    if _mc_wrapper is not None:
        _mc_wrapper.window_end(window, data_size=BYTES_PER_ITER)

def allreduce(tensor):
    if _mc_wrapper is not None:
        _mc_wrapper.allreduce(tensor.data_ptr(), tensor.data_ptr(),
                               tensor.numel(), 7, 0, 0)  # ncclFloat32
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
        print(f"[Job1-{MODE.upper()}] MODE={MODE}, Payload={PAYLOAD_MB}MB, "
              f"sleep={SLEEP_US}us, SLO c_i={SLO_C_I}")
        if MODE == 'crux':
            print(f"[Job1-CRUX] Static Priority: P{CRUX_STATIC_PRIORITY} "
                  f"(DSCP={CRUX_STATIC_PRIORITY*8})")
            print(f"[Job1-CRUX] GPU Intensity I_1 = {SLEEP_US/1000:.0f}ms / ~85ms ≈ 0.35")
            print(f"[Job1-CRUX] NOTE: Both jobs use P4 (same priority) for MultiComm compatibility")
            print(f"[Job1-CRUX] CRUX cannot dynamically prioritize Job1's strict SLO!")
        print(f"[Job1-{MODE.upper()}] {NUM_ITERS} iters, {ITERS_PER_WINDOW}/window, "
              f"{NUM_WINDOWS} windows")

    if rank == 0 and device is not None:
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"[Job1-{MODE.upper()}] GPU {device}: {torch.cuda.get_device_name(device)}, "
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
        print("[Job1] Warmup done.")

    # ============================================================
    # Main training loop
    # ============================================================
    results = []
    t_total_start = time.perf_counter()

    for window in range(NUM_WINDOWS):
        window_start(window)

        for i in range(ITERS_PER_WINDOW):
            global_iter = window * ITERS_PER_WINDOW + i

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

            # Phase: first 5 windows (100 iters) are solo baseline
            if global_iter < 100:
                phase = 'solo_rampup'
            else:
                phase = 'contested'

            if rank == 0:
                print(f"[Job1-{MODE.upper()}] iter {global_iter:2d} (window {window}, "
                      f"iter {i}): comm={comm_dur*1000:.1f}ms, "
                      f"bw={bw_gbps:.2f} Gbps [{phase}]")

            results.append({
                'iter': global_iter,
                'window': window,
                'comm_dur_s': round(comm_dur, 6),
                'bw_gbps': round(bw_gbps, 4),
                'phase': phase,
            })

        window_end(window)

    t_total_end = time.perf_counter()

    # ============================================================
    # Save results
    # ============================================================
    if rank == 0:
        csv_path = f'p4_job1_asym_{MODE}_rank0.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'window', 'comm_dur_s',
                             'bw_gbps', 'phase'])
            for r in results:
                writer.writerow([r['iter'], r['window'],
                                 r['comm_dur_s'], r['bw_gbps'],
                                 r['phase']])
        print(f"[Job1-{MODE.upper()}] Results saved to {csv_path}")

        # Summary
        solo_iters = [r for r in results if r['phase'] != 'contested']
        contested_iters = [r for r in results if r['phase'] == 'contested']

        if solo_iters:
            avg_solo = sum(r['comm_dur_s'] for r in solo_iters) / len(solo_iters)
            avg_bw_solo = sum(r['bw_gbps'] for r in solo_iters) / len(solo_iters)
            print(f"[Job1-{MODE.upper()}] Solo ({len(solo_iters)} iters): "
                  f"avg_comm={avg_solo*1000:.1f}ms, avg_bw={avg_bw_solo:.2f} Gbps")
        if contested_iters:
            avg_cont = sum(r['comm_dur_s'] for r in contested_iters) / len(contested_iters)
            avg_bw_cont = sum(r['bw_gbps'] for r in contested_iters) / len(contested_iters)
            slowdown = avg_cont / avg_solo if solo_iters else float('nan')
            print(f"[Job1-{MODE.upper()}] Contested ({len(contested_iters)} iters): "
                  f"avg_comm={avg_cont*1000:.1f}ms, avg_bw={avg_bw_cont:.2f} Gbps, "
                  f"slowdown={slowdown:.2f}x")

        print(f"[Job1-{MODE.upper()}] Total wall time: {t_total_end - t_total_start:.1f}s")

    # Cleanup
    if _mc_wrapper is not None:
        _mc_wrapper.destroy()

if __name__ == '__main__':
    main()
