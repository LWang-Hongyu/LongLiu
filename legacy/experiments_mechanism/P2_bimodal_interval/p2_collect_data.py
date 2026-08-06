#!/usr/bin/env python3
"""
P2: Bimodal Interval Distribution - Data Collection Script
Runs 100 AllReduce iterations on 2 nodes and records timestamps.
"""

import os
import sys
import time
import csv
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_addr', default='10.157.197.107')
    parser.add_argument('--master_port', type=int, default=29510)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', default='p2_timestamps.csv')
    args = parser.parse_args()

    # Initialize NCCL process group
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    os.environ['NCCL_IB_HCA'] = 'mlx5_0'
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp'

    dist.init_process_group(backend='nccl', rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')

    # Tensor: 4096 float32 = 16KB
    tensor = torch.ones(4096, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    # Collect timestamps
    timestamps = []
    clock = time.perf_counter_ns  # nanosecond precision

    for i in range(args.iterations):
        # Issue multiple collectives per iteration to reveal intra-iteration gaps
        for j in range(5):
            torch.cuda.synchronize()
            t_start = clock()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            t_end = clock()
            timestamps.append((i * 5 + j, t_start, t_end, t_end - t_start))

        # Simulate per-iteration computation delay (~50ms host-side)
        time.sleep(0.05)

    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'start_ns', 'end_ns', 'duration_ns'])
        for row in timestamps:
            writer.writerow(row)

    print(f"[Rank {args.rank}] Saved {len(timestamps)} timestamps to {output_path}")

    # Print summary
    durations = [r[3] for r in timestamps]
    print(f"[Rank {args.rank}] Duration stats: min={min(durations)/1e6:.3f}ms, "
          f"max={max(durations)/1e6:.3f}ms, avg={sum(durations)/len(durations)/1e6:.3f}ms")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
