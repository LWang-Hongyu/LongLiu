#!/usr/bin/env python3
"""
P4 V3: Synchronous Competition — Job 1 (Tight SLO)

Payload:  2048 MB fp32 — same as Job2 for natural AllReduce sync
Compute:  50 ms simulated forward/backward per iteration
SLO:      c_i = 1.5 (tight deadline)
Schedule: 300 iterations, 20 per epoch, 15 total epochs
          (Job2 joins after epoch 5)

Launch:   python3 p4_job1.py
Env vars: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, CUDA_VISIBLE_DEVICES,
          P4_MODE ∈ {solo, fair, longliu}
          MULTI_COMM_PORT (optional, for longliu mode TCP ID exchange)
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
                    choices=['solo', 'fair', 'longliu', 'static_dscp'])
parser.add_argument('--window', type=int, default=-1,
                    help='Single window to run (default -1 = all windows)')
parser.add_argument('--no-warmup', action='store_true',
                    help='Skip warmup (for per-window manual runs)')
args = parser.parse_args()
MODE = args.mode
SINGLE_WINDOW = args.window if args.window >= 0 else None
SKIP_WARMUP = args.no_warmup

# ============================================================
# Job 1 constants
# ============================================================
PAYLOAD_MB = 2048           # 2 GB fp32 — same as Job2 for sync competition
SLEEP_US   = 50000          # 50 ms simulated compute per iteration
SLO_C_I    = 1.5            # tight SLO threshold
NUM_ITERS  = 300
ITERS_PER_WINDOW = 20
NUM_WINDOWS = NUM_ITERS // ITERS_PER_WINDOW  # 15

BYTES_PER_ITER = PAYLOAD_MB * 1024 * 1024
NUM_ELEMENTS = BYTES_PER_ITER // 4  # float32

# Bandwidth correction: for Ring AllReduce with 2 ranks, the actual
# per-direction wire bandwidth is:
#   wire_bw_gbps = size_Gb * (n-1)/n / time
# For n=2: wire_bw = size_Gb / 2 / time
# This reflects the physical link utilization (vs algorithm bandwidth
# which counts the tensor size only once, ignoring wire overhead).
def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    """Compute per-direction wire bandwidth for NCCL Ring AllReduce."""
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur

# ============================================================
# LongLiu adapter — two modes:
#   longliu : use MultiCommWrapper (7 priority communicators via trafficClass)
#   fair/solo : use standard PyTorch distributed
# ============================================================
_adapter_enabled = (MODE == 'longliu')

if _adapter_enabled:
    import torch.distributed as dist  # only for env parsing helpers
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), '..', '..', 'multi_comm_slo', 'src'))
    from slo_scheduler import MultiCommWrapper, SLOScheduler
    _slo_scheduler = SLOScheduler(slo_threshold=SLO_C_I)
    _mc_wrapper = None  # initialized in main()
else:
    import torch.distributed as dist


def window_start(window):
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.window_start(window)


def window_end(window):
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.window_end(window, data_size=BYTES_PER_ITER)


def allreduce(tensor):
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.allreduce(tensor.data_ptr(), tensor.data_ptr(),
                               tensor.numel(), 7, 0, 0)  # ncclFloat32
    else:
        dist.all_reduce(tensor)


def main():
    if _adapter_enabled:
        # ---- LongLiu mode: use MultiCommWrapper ----
        rank = int(os.environ.get('RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        master_addr = os.environ.get('MASTER_ADDR', '192.10.10.110')
        port = int(os.environ.get('MULTI_COMM_PORT',
                    os.environ.get('MASTER_PORT', '29500')))

        # CUDA_VISIBLE_DEVICES renumbers GPUs starting from 0
        device_idx = 0
        torch.cuda.set_device(device_idx)
        device = torch.cuda.current_device()

        global _mc_wrapper
        _mc_wrapper = MultiCommWrapper(
            _slo_scheduler, rank, world_size, str(device_idx),
            master_addr, port)

        if rank == 0:
            print(f"[Job1] MODE={MODE}, Payload={PAYLOAD_MB}MB, "
                  f"sleep={SLEEP_US}us, SLO c_i={SLO_C_I}")
            print(f"[Job1] {NUM_ITERS} iters, {ITERS_PER_WINDOW}/window, "
                  f"{NUM_WINDOWS} windows, MultiComm mode")
    else:
        # ---- Fair/Solo mode: standard PyTorch distributed ----
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.cuda.current_device()

        if rank == 0:
            print(f"[Job1] MODE={MODE}, Payload={PAYLOAD_MB}MB, "
                  f"sleep={SLEEP_US}us, SLO c_i={SLO_C_I}")
            print(f"[Job1] {NUM_ITERS} iters, {ITERS_PER_WINDOW}/window, "
                  f"{NUM_WINDOWS} windows")

    if rank == 0 and device is not None:
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"[Job1] GPU {device}: {torch.cuda.get_device_name(device)}, "
              f"free={free_mem:.1f} GB")

    # ============================================================
    # Allocate tensor
    # ============================================================
    tensor = torch.ones(NUM_ELEMENTS, dtype=torch.float32, device='cuda')

    # Warmup
    if not SKIP_WARMUP:
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

    if SINGLE_WINDOW is not None:
        window_range = [SINGLE_WINDOW]
    else:
        window_range = range(NUM_WINDOWS)

    for window in window_range:
        window_start(window)  # sets priority based on scheduler state

        for i in range(ITERS_PER_WINDOW):
            if SINGLE_WINDOW is not None:
                global_iter = i
            else:
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
            # Corrected bus bandwidth: accounts for Ring AllReduce wire overhead
            # For n=2, wire data = 2× tensor, so per-direction bw = size/2/time
            bw_gbps = bus_bw_gbps(BYTES_PER_ITER, comm_dur, world_size) if comm_dur > 0 else 0.0

            # Phase: first 5 windows (100 iters) are solo baseline;
            # Job2 joins at window 5, subsequent iters are contested.
            if MODE == 'solo':
                phase = 'solo'
            elif SINGLE_WINDOW is not None:
                phase = 'contested'
            elif global_iter < 100:
                phase = 'solo_rampup'
            else:
                phase = 'contested'

            if rank == 0:
                print(f"[Job1] iter {global_iter:2d} (window {window}, "
                      f"iter {i}): comm={comm_dur*1000:.1f}ms, "
                      f"bw={bw_gbps:.2f} Gbps [{phase}]")

            results.append({
                'iter': global_iter,
                'window': window,
                'comm_dur_s': round(comm_dur, 6),
                'bw_gbps': round(bw_gbps, 4),
                'phase': phase,
            })

        window_end(window)  # measures time, updates priority

    t_total_end = time.perf_counter()

    # ============================================================
    # Save results
    # ============================================================
    if rank == 0:
        if SINGLE_WINDOW is not None:
            csv_path = f'p4_job1_manual_window{SINGLE_WINDOW}.csv'
        else:
            csv_path = f'p4_job1_{MODE}_rank0.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'window', 'comm_dur_s',
                             'bw_gbps', 'phase'])
            for r in results:
                writer.writerow([r['iter'], r['window'],
                                 r['comm_dur_s'], r['bw_gbps'],
                                 r['phase']])
        print(f"[Job1] Results saved to {csv_path}")

        # Summary
        solo_iters = [r for r in results if r['phase'] != 'contested']
        contested_iters = [r for r in results if r['phase'] == 'contested']

        if solo_iters:
            avg_solo = sum(r['comm_dur_s'] for r in solo_iters) / len(solo_iters)
            avg_bw_solo = sum(r['bw_gbps'] for r in solo_iters) / len(solo_iters)
            print(f"[Job1] Solo ({len(solo_iters)} iters): "
                  f"avg_comm={avg_solo*1000:.1f}ms, avg_bw={avg_bw_solo:.2f} Gbps")
        if contested_iters:
            avg_cont = sum(r['comm_dur_s'] for r in contested_iters) / len(contested_iters)
            avg_bw_cont = sum(r['bw_gbps'] for r in contested_iters) / len(contested_iters)
            slowdown = avg_cont / avg_solo if solo_iters else float('nan')
            print(f"[Job1] Contested ({len(contested_iters)} iters): "
                  f"avg_comm={avg_cont*1000:.1f}ms, avg_bw={avg_bw_cont:.2f} Gbps, "
                  f"slowdown={slowdown:.2f}x")

        print(f"[Job1] Total wall time: {t_total_end - t_total_start:.1f}s")

    # Cleanup
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.destroy()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
