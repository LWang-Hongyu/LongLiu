#!/usr/bin/env python3
"""
P4 V3: Synchronous Competition — Job 2 (Loose SLO)

Payload:  2048 MB fp32 — same as Job1 for natural AllReduce sync
Compute:  50 ms simulated forward/backward per iteration
SLO:      c_i = 2.5 (loose deadline)
Schedule: 200 iterations, 20 per epoch, 10 total epochs
          (joins after Job1 epoch 5)

Launch:   python3 p4_job2.py
Env vars: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, CUDA_VISIBLE_DEVICES,
          P4_MODE ∈ {fair, longliu}
          MULTI_COMM_PORT (optional, for longliu mode TCP ID exchange)
Note:     Job 2 has no "solo" mode — it is not launched in Solo experiments.
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
                    choices=['fair', 'longliu', 'static_dscp'])
parser.add_argument('--epoch', type=int, default=-1,
                    help='Single epoch to run (default -1 = all epochs)')
parser.add_argument('--no-warmup', action='store_true',
                    help='Skip warmup (for per-epoch manual runs)')
args = parser.parse_args()
MODE = args.mode
SINGLE_EPOCH = args.epoch if args.epoch >= 0 else None
SKIP_WARMUP = args.no_warmup

# ============================================================
# Job 2 constants
# ============================================================
PAYLOAD_MB = 2048           # 2 GB fp32 — same as Job1 for sync competition
SLEEP_US   = 50000          # 50 ms simulated compute per iteration
SLO_C_I    = 2.5            # loose SLO threshold
NUM_ITERS  = 200
ITERS_PER_EPOCH = 20
NUM_EPOCHS = NUM_ITERS // ITERS_PER_EPOCH  # 10

BYTES_PER_ITER = PAYLOAD_MB * 1024 * 1024
NUM_ELEMENTS = BYTES_PER_ITER // 4  # float32

# Bandwidth correction: for Ring AllReduce with 2 ranks, the actual
# per-direction wire bandwidth is:
#   wire_bw_gbps = size_Gb * (n-1)/n / time
# For n=2: wire_bw = size_Gb / 2 / time
def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    """Compute per-direction wire bandwidth for NCCL Ring AllReduce."""
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur

# ============================================================
# LongLiu adapter — two modes:
#   longliu : use MultiCommWrapper (7 priority communicators via trafficClass)
#   fair    : use standard PyTorch distributed
# ============================================================
_adapter_enabled = (MODE == 'longliu')

if _adapter_enabled:
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), '..', '..', 'multi_comm_slo', 'src'))
    from slo_scheduler import MultiCommWrapper, SLOScheduler
    _slo_scheduler = SLOScheduler(slo_threshold=SLO_C_I)
    _mc_wrapper = None  # initialized in main()
    import torch.distributed as dist  # only for env parsing
else:
    import torch.distributed as dist


def epoch_start(epoch):
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.epoch_start(epoch)


def epoch_end(epoch):
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.epoch_end(epoch, data_size=BYTES_PER_ITER)


def allreduce(tensor):
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.allreduce(tensor.data_ptr(), tensor.data_ptr(),
                               tensor.numel(), 0, 0, 0)
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
            print(f"[Job2] MODE={MODE}, Payload={PAYLOAD_MB}MB, "
                  f"sleep={SLEEP_US}us, SLO c_i={SLO_C_I}, "
                  f"MultiComm mode")
            print(f"[Job2] {NUM_ITERS} iters, {ITERS_PER_EPOCH}/epoch, "
                  f"{NUM_EPOCHS} epochs")
    else:
        # ---- Fair mode: standard PyTorch distributed ----
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.cuda.current_device()

        if rank == 0:
            print(f"[Job2] MODE={MODE}, Payload={PAYLOAD_MB}MB, "
                  f"sleep={SLEEP_US}us, SLO c_i={SLO_C_I}")
            print(f"[Job2] {NUM_ITERS} iters, {ITERS_PER_EPOCH}/epoch, "
                  f"{NUM_EPOCHS} epochs")

    if rank == 0 and device is not None:
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"[Job2] GPU {device}: {torch.cuda.get_device_name(device)}, "
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
            print("[Job2] Warmup done.")

    # ============================================================
    # Main training loop
    # ============================================================
    results = []
    t_total_start = time.perf_counter()

    if SINGLE_EPOCH is not None:
        epoch_range = [SINGLE_EPOCH]
    else:
        epoch_range = range(NUM_EPOCHS)

    for epoch in epoch_range:
        epoch_start(epoch)

        for i in range(ITERS_PER_EPOCH):
            if SINGLE_EPOCH is not None:
                global_iter = i  # 0-4 within this epoch
            else:
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
            # Corrected bus bandwidth: accounts for Ring AllReduce wire overhead
            bw_gbps = bus_bw_gbps(BYTES_PER_ITER, comm_dur, world_size) if comm_dur > 0 else 0.0

            # Job 2 is always in contested mode (joined after Job 1 ramp-up)
            phase = 'contested'

            if rank == 0:
                print(f"[Job2] iter {global_iter:2d} (epoch {epoch}, "
                      f"iter {i}): comm={comm_dur*1000:.1f}ms, "
                      f"bw={bw_gbps:.2f} Gbps")

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
    if rank == 0:
        if SINGLE_EPOCH is not None:
            csv_path = f'p4_job2_manual_epoch{SINGLE_EPOCH}.csv'
        else:
            csv_path = f'p4_job2_{MODE}_rank0.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'epoch', 'comm_dur_s',
                             'bw_gbps', 'phase'])
            for r in results:
                writer.writerow([r['iter'], r['epoch'],
                                 r['comm_dur_s'], r['bw_gbps'],
                                 r['phase']])
        print(f"[Job2] Results saved to {csv_path}")

        avg_comm = sum(r['comm_dur_s'] for r in results) / len(results)
        avg_bw = sum(r['bw_gbps'] for r in results) / len(results)
        print(f"[Job2] Avg ({len(results)} iters): "
              f"comm={avg_comm*1000:.1f}ms, bw={avg_bw:.2f} Gbps")
        print(f"[Job2] Total wall time: {t_total_end - t_total_start:.1f}s")

    # Cleanup
    if _adapter_enabled and _mc_wrapper is not None:
        _mc_wrapper.destroy()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
