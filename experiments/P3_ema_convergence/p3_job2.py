#!/usr/bin/env python3
"""
P3 Job 2: Waits for Job 1's sync signal, then runs epochs 6-10.
Creates network competition during these epochs.

Launched with CUDA_VISIBLE_DEVICES=1 on 226 (uses GPU 1 to avoid contention with Job 1).
On 10.1 (1 GPU), shares GPU 0 with Job 1 — competition is at RDMA NIC level.
"""

import os
import sys
import time
import csv
import ctypes
import torch
import torch.distributed as dist

# Load DSCP-modified NCCL library for epoch trigger functions
_nccl_lib = ctypes.CDLL('libnccl.so.2')
_nccl_lib.ncclDscpEpochStart.argtypes = [ctypes.c_int]
_nccl_lib.ncclDscpEpochStart.restype = ctypes.c_int
_nccl_lib.ncclDscpEpochEnd.argtypes = [ctypes.c_int]
_nccl_lib.ncclDscpEpochEnd.restype = ctypes.c_int


def run_epoch(num_elements, iters_per_epoch):
    """Run one epoch of NCCL AllReduce, return (comm_dur_s, total_bytes, bw_gbps)."""
    tensor = torch.ones(num_elements, dtype=torch.float32, device='cuda')
    total_bytes = num_elements * 4 * iters_per_epoch

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for _ in range(iters_per_epoch):
        dist.all_reduce(tensor)

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    comm_dur_s = t_end - t_start
    bw_gbps = (total_bytes * 8 / 1e9) / comm_dur_s
    return comm_dur_s, total_bytes, bw_gbps


def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES already restricted

    payload_mb = int(os.environ.get('P3_PAYLOAD_MB', '256'))
    iters_per_epoch = int(os.environ.get('P3_ITERS', '20'))
    num_elements = payload_mb * 1024 * 1024 // 4

    sync_file = '/home/why/LongLiu_rebuild/experiments/P3_ema_convergence/.sync_ready'
    csv_path = f'/home/why/LongLiu_rebuild/experiments/P3_ema_convergence/p3_job2_rank{rank}.csv'

    # Warmup
    warmup = torch.ones(4096, dtype=torch.float32, device='cuda')
    for _ in range(3):
        dist.all_reduce(warmup)
    torch.cuda.synchronize()

    # Wait for Job 1 epoch 5 to complete
    if rank == 0:
        print(f"[Job2] Waiting for Job 1 sync signal ({sync_file})...", flush=True)
        while not os.path.exists(sync_file):
            time.sleep(0.5)
        print(f"[Job2] Sync received, starting contested epochs...", flush=True)
    dist.barrier()

    results = []

    for epoch in range(5, 10):
        # Set NCCL_STATS_ITERATION so each epoch's ops go into separate iteration bucket
        os.environ['NCCL_STATS_ITERATION'] = str(epoch)

        # Signal DSCP adapter: epoch start (processed by next NCCL op)
        _nccl_lib.ncclDscpEpochStart(epoch)

        comm_dur_s, total_bytes, bw_gbps = run_epoch(num_elements, iters_per_epoch)

        # Signal DSCP adapter: epoch end (processed by next NCCL op, triggers EMA+DSCP)
        _nccl_lib.ncclDscpEpochEnd(epoch)

        results.append({
            'epoch': epoch, 'total_bytes': total_bytes,
            'comm_dur_s': round(comm_dur_s, 6),
            'bw_obs_gbps': round(bw_gbps, 3), 'phase': 'contested'
        })

        print(f"[Job2 Rank{rank}] Epoch {epoch:2d} (contested): "
              f"{total_bytes/1e9:.2f}GB, {comm_dur_s*1000:.1f}ms, "
              f"Bw_obs={bw_gbps:.2f} Gbps", flush=True)

        dist.barrier()

    # Save
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'total_bytes',
                                                'comm_dur_s', 'bw_obs_gbps', 'phase'])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Job2 Rank{rank}] Saved {len(results)} epochs → {csv_path}", flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
