#!/usr/bin/env python3
"""
P3 Job 1: Runs epochs 1-10.
- Epochs 1-5: Solo (no competition)
- Epochs 6-10: Contested (Job 2 active)
After epoch 5, creates sync flag file to signal Job 2.
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
    torch.cuda.set_device(0)

    payload_mb = int(os.environ.get('P3_PAYLOAD_MB', '256'))
    iters_per_epoch = int(os.environ.get('P3_ITERS', '20'))
    num_elements = payload_mb * 1024 * 1024 // 4

    sync_file = '/home/why/LongLiu_rebuild/experiments/P3_ema_convergence/.sync_ready'
    csv_path = f'/home/why/LongLiu_rebuild/experiments/P3_ema_convergence/p3_job1_rank{rank}.csv'

    # Remove stale sync file
    if rank == 0 and os.path.exists(sync_file):
        os.remove(sync_file)

    # Warmup
    warmup = torch.ones(4096, dtype=torch.float32, device='cuda')
    for _ in range(3):
        dist.all_reduce(warmup)
    torch.cuda.synchronize()
    if rank == 0:
        print("[Job1] Warmup done, starting epochs...", flush=True)
    dist.barrier()

    results = []

    for epoch in range(0, 10):
        # Set NCCL_STATS_ITERATION so each epoch's ops go into separate iteration bucket
        os.environ['NCCL_STATS_ITERATION'] = str(epoch)

        # Signal DSCP adapter: epoch start (processed by next NCCL op)
        _nccl_lib.ncclDscpEpochStart(epoch)

        comm_dur_s, total_bytes, bw_gbps = run_epoch(num_elements, iters_per_epoch)

        # Signal DSCP adapter: epoch end (processed by next NCCL op, triggers EMA+DSCP)
        _nccl_lib.ncclDscpEpochEnd(epoch)

        phase = 'solo' if epoch <= 4 else 'contested'
        results.append({
            'epoch': epoch, 'total_bytes': total_bytes,
            'comm_dur_s': round(comm_dur_s, 6),
            'bw_obs_gbps': round(bw_gbps, 3), 'phase': phase
        })

        print(f"[Job1 Rank{rank}] Epoch {epoch:2d} ({phase:>9s}): "
              f"{total_bytes/1e9:.2f}GB, {comm_dur_s*1000:.1f}ms, "
              f"Bw_obs={bw_gbps:.2f} Gbps", flush=True)

        # Signal Job 2 after epoch 4 (5th epoch, end of solo phase)
        if epoch == 4 and rank == 0:
            with open(sync_file, 'w') as f:
                f.write(str(time.time()))
            print(f"[Job1] Sync flag written → Job 2 can start", flush=True)

        dist.barrier()

    # Save
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'total_bytes',
                                                'comm_dur_s', 'bw_obs_gbps', 'phase'])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Job1 Rank{rank}] Saved {len(results)} epochs → {csv_path}", flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
