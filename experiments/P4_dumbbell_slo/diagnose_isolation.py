#!/usr/bin/env python3
"""
Isolation Point Diagnosis: extreme priority gap test (P0 vs P6).

Goal: determine whether the contention point is at the switch (DSCP matters)
or at the host NIC (DSCP doesn't help).

Setup:
  - Job A: heavy (2048MB), STATIC P0 (DSCP=0, lowest)
  - Job B: light (256MB), STATIC P6 (DSCP=48, highest)
  - Both run contested for 10 epochs

Metric:
  - B's comm time under extreme priority gap vs B's solo baseline (34.7ms)
  - Expansion ratio = contested_comm / solo_comm
  - If expansion > 1.2 (20%) → contention at host NIC, DSCP ineffective
  - If expansion ≈ 1.0 → switch isolation works, DSCP effective

This is a one-shot diagnostic, not part of the main experiment series.
"""

import os
import sys
import time
import csv
import argparse
import torch

# ============================================================
# Parse args
# ============================================================
parser = argparse.ArgumentParser(description='Isolation point diagnosis (P0 vs P6)')
parser.add_argument('--job', type=str, required=True, choices=['A', 'B'],
                    help='Job identity (A=heavy/P0, B=light/P6)')
parser.add_argument('--num-iters', type=int, default=200)
parser.add_argument('--iters-per-epoch', type=int, default=20)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

JOB = args.job
NUM_ITERS = args.num_iters
ITERS_PER_EPOCH = args.iters_per_epoch
NUM_EPOCHS = args.epochs

# ============================================================
# Fixed configuration for diagnosis
# ============================================================
SLEEP_US = 30000               # 30ms compute, fixed

# Job A: heavy payload, P0 (lowest priority)
# Job B: light payload, P6 (highest priority)
JOB_CONFIG = {
    'A': {'payload_mb': 2048, 'priority': 0, 'label': 'heavy/P0'},
    'B': {'payload_mb': 256,  'priority': 6, 'label': 'light/P6'},
}

PAYLOAD_MB = JOB_CONFIG[JOB]['payload_mb']
STATIC_PRIORITY = JOB_CONFIG[JOB]['priority']
BYTES_PER_ITER = PAYLOAD_MB * 1024 * 1024
NUM_ELEMENTS = BYTES_PER_ITER // 4  # float32


def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    """Per-direction wire bandwidth for NCCL Ring AllReduce."""
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur


# ============================================================
# Use MultiCommWrapper with STATIC priority (no dynamic adjustment)
# ============================================================
import torch.distributed as dist
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', 'multi_comm_slo', 'src'))
from slo_scheduler import MultiCommWrapper


class StaticPriorityScheduler:
    """Static priority — never changes (for diagnosis)."""
    def __init__(self, static_priority):
        self.current_priority = static_priority
        self.priority_history = [self.current_priority]
        self.last_pi = float('nan')
        self.target_comm_time_s = None
        self.slo_threshold = None
        self.cumulative_actual_s = 0.0
        self.completed_iters = 0
        self.preset_target = False

    def update(self, actual_comm_time, data_size):
        self.completed_iters += 1
        self.cumulative_actual_s += actual_comm_time
        return self.current_priority

    def get_dscp(self):
        return self.current_priority * 8

    def set_slo_threshold(self, new_threshold):
        self.slo_threshold = new_threshold


_mc_wrapper = None


def epoch_start(epoch):
    if _mc_wrapper is not None:
        _mc_wrapper.epoch_start(epoch)


def epoch_end(epoch, data_size):
    if _mc_wrapper is not None:
        _mc_wrapper.epoch_end(epoch, data_size=data_size)


def allreduce(tensor):
    if _mc_wrapper is not None:
        _mc_wrapper.allreduce(tensor.data_ptr(), tensor.data_ptr(),
                               tensor.numel(), 0, 0, 0)
    else:
        dist.all_reduce(tensor)


def main():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    device_idx = 0
    torch.cuda.set_device(device_idx)
    device = torch.cuda.current_device()

    _scheduler = StaticPriorityScheduler(STATIC_PRIORITY)
    global _mc_wrapper
    _mc_wrapper = MultiCommWrapper(
        _scheduler, rank, world_size, str(device_idx),
        os.environ.get('MASTER_ADDR', '192.10.10.110'),
        int(os.environ.get('MULTI_COMM_PORT',
            os.environ.get('MASTER_PORT', '29500'))))

    if rank == 0:
        print(f"[Diag-{JOB}] payload={PAYLOAD_MB}MB, priority=P{STATIC_PRIORITY} "
              f"(DSCP={STATIC_PRIORITY*8}), {JOB_CONFIG[JOB]['label']}")
        print(f"[Diag-{JOB}] {NUM_ITERS} iters, {ITERS_PER_EPOCH}/epoch, "
              f"{NUM_EPOCHS} epochs")

    tensor = torch.ones(NUM_ELEMENTS, dtype=torch.float32, device=device)

    # Warmup
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        allreduce(warmup)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[Diag-{JOB}] Warmup done.")

    results = []
    t_total_start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        epoch_start(epoch)
        epoch_comm_times = []
        epoch_bws = []

        for i in range(ITERS_PER_EPOCH):
            global_iter = epoch * ITERS_PER_EPOCH + i

            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            allreduce(tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            comm_dur = t1 - t0
            bw_gbps = bus_bw_gbps(BYTES_PER_ITER, comm_dur, world_size) if comm_dur > 0 else 0.0
            epoch_comm_times.append(comm_dur)
            epoch_bws.append(bw_gbps)

            if rank == 0 and i == 0:
                print(f"[Diag-{JOB}] epoch {epoch}: comm={comm_dur*1000:.1f}ms, "
                      f"bw={bw_gbps:.2f} Gbps")

            results.append({
                'iter': global_iter,
                'epoch': epoch,
                'comm_dur_s': round(comm_dur, 6),
                'bw_gbps': round(bw_gbps, 4),
            })

        epoch_end(epoch, data_size=BYTES_PER_ITER)

    t_total_end = time.perf_counter()

    # Save results
    if rank == 0:
        csv_path = f'diag_isolation_job{JOB}_rank0.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'epoch', 'comm_dur_s', 'bw_gbps'])
            for r in results:
                writer.writerow([r['iter'], r['epoch'],
                                 r['comm_dur_s'], r['bw_gbps']])
        print(f"[Diag-{JOB}] Results saved to {csv_path}")

        # Summary
        avg_comm = sum(r['comm_dur_s'] for r in results) / len(results)
        avg_bw = sum(r['bw_gbps'] for r in results) / len(results)
        print(f"[Diag-{JOB}] Summary: avg_comm={avg_comm*1000:.1f}ms, "
              f"avg_bw={avg_bw:.2f} Gbps")
        print(f"[Diag-{JOB}] Total wall time: {t_total_end - t_total_start:.1f}s")

    _mc_wrapper.destroy()


if __name__ == '__main__':
    main()
