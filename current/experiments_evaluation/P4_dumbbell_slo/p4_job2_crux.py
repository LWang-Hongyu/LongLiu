#!/usr/bin/env python3
"""
P4 V3: CRUX Baseline — Job 2 (Static GPU Intensity Priority)

CRUX assigns priority based on GPU Intensity I_j = compute_time / comm_time.
Higher I_j → higher priority (more compute-intensive, needs bandwidth to keep GPU busy).

Job1: compute=50ms, comm=~85ms → I_1 ≈ 0.59 → P4 (DSCP=32)
Job2: compute=50ms, comm=~85ms → I_2 ≈ 0.59 → P3 (DSCP=24)

In this experiment, both jobs have similar GPU Intensity, so we assign
Job2 a slightly lower static priority (P3) vs Job1 (P4) to demonstrate
CRUX's static allocation behavior.

Launch:   python3 p4_job2_crux.py --mode crux
Env vars: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, CUDA_VISIBLE_DEVICES,
          MULTI_COMM_PORT (for MultiComm DSCP control)
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
parser.add_argument('--mode', type=str, default='crux',
                    choices=['crux'])
args = parser.parse_args()
MODE = args.mode

# ============================================================
# Job 2 constants (CRUX baseline)
# ============================================================
PAYLOAD_MB = 2048           # 2 GB fp32
SLEEP_US   = 50000          # 50 ms simulated compute per iteration
NUM_ITERS  = 200
ITERS_PER_WINDOW = 20
NUM_WINDOWS = NUM_ITERS // ITERS_PER_WINDOW  # 10

BYTES_PER_ITER = PAYLOAD_MB * 1024 * 1024
NUM_ELEMENTS = BYTES_PER_ITER // 4  # float32

# CRUX: Static GPU Intensity-based priority
# Job1: I_1 = 50ms / 85ms ≈ 0.59 → assign P4 (DSCP=32)
# Job2: I_2 = 50ms / 85ms ≈ 0.59 → assign P3 (DSCP=24)
# Job2 gets slightly lower priority than Job1
CRUX_STATIC_PRIORITY = 3  # P3 (DSCP=24)

def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    """Compute per-direction wire bandwidth for NCCL Ring AllReduce."""
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur

# ============================================================
# CRUX: Use MultiCommWrapper with STATIC priority (no dynamic adjustment)
# ============================================================
import torch.distributed as dist
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', 'multi_comm_slo', 'src'))
from slo_scheduler import MultiCommWrapper

# Create a dummy scheduler that always returns the static CRUX priority
class StaticPriorityScheduler:
    """CRUX baseline: static priority based on GPU Intensity, never changes."""
    def __init__(self, static_priority):
        self.current_priority = static_priority
        self.priority_history = [self.current_priority]
    
    def update(self, actual_comm_time: float, data_size: float,
               window_comm_time: float = 0.0) -> int:
        # CRUX: priority is STATIC, never updated
        return self.current_priority
    
    def get_dscp(self) -> int:
        return self.current_priority * 8

_static_scheduler = StaticPriorityScheduler(CRUX_STATIC_PRIORITY)
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
        _static_scheduler, rank, world_size, str(device_idx),
        master_addr, port)

    if rank == 0:
        print(f"[Job2-CRUX] MODE={MODE}, Payload={PAYLOAD_MB}MB, "
              f"sleep={SLEEP_US}us")
        print(f"[Job2-CRUX] Static Priority: P{CRUX_STATIC_PRIORITY} "
              f"(DSCP={CRUX_STATIC_PRIORITY*8})")
        print(f"[Job2-CRUX] GPU Intensity I_2 = {SLEEP_US/1000:.0f}ms / ~85ms ≈ 0.59")
        print(f"[Job2-CRUX] {NUM_ITERS} iters, {ITERS_PER_WINDOW}/window, "
              f"{NUM_WINDOWS} windows, MultiComm mode (STATIC priority)")

    if rank == 0 and device is not None:
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"[Job2-CRUX] GPU {device}: {torch.cuda.get_device_name(device)}, "
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
        print("[Job2-CRUX] Warmup done.")

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

            # All iters are contested (Job2 joins after Job1 has started)
            phase = 'contested'

            if rank == 0:
                print(f"[Job2-CRUX] iter {global_iter:2d} (window {window}, "
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
        csv_path = f'p4_job2_{MODE}_rank0.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'window', 'comm_dur_s',
                             'bw_gbps', 'phase'])
            for r in results:
                writer.writerow([r['iter'], r['window'],
                                 r['comm_dur_s'], r['bw_gbps'],
                                 r['phase']])
        print(f"[Job2-CRUX] Results saved to {csv_path}")

        # Summary
        contested_iters = [r for r in results if r['phase'] == 'contested']

        if contested_iters:
            avg_cont = sum(r['comm_dur_s'] for r in contested_iters) / len(contested_iters)
            avg_bw_cont = sum(r['bw_gbps'] for r in contested_iters) / len(contested_iters)
            print(f"[Job2-CRUX] Contested ({len(contested_iters)} iters): "
                  f"avg_comm={avg_cont*1000:.1f}ms, avg_bw={avg_bw_cont:.2f} Gbps")

        print(f"[Job2-CRUX] Total wall time: {t_total_end - t_total_start:.1f}s")

    # Cleanup
    if _mc_wrapper is not None:
        _mc_wrapper.destroy()


if __name__ == '__main__':
    main()
