#!/usr/bin/env python3
"""
DSCP TC Sweep Test - Tests a single NCCL_IB_TC value.
Run with: NCCL_IB_TC=<value> python3 dscp_tc_sweep.py

Designed for repeated invocations with different TC values.
Each invocation uses a unique port to avoid NCCL port conflicts.
"""
import os, sys, time, torch, torch.distributed as dist

rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '2'))
master_addr = os.environ.get('MASTER_ADDR', '192.10.10.226')
master_port = os.environ.get('MASTER_PORT', '29550')
tc = os.environ.get('NCCL_IB_TC', 'unknown')

os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['RANK'] = str(rank)

torch.cuda.set_device(0)

dist.init_process_group("nccl")
my_rank = dist.get_rank()

# Run 20 all_reduce iterations to generate enough RoCEv2 packets
data = torch.randn(4096, device="cuda")
for i in range(20):
    dist.all_reduce(data)
torch.cuda.synchronize()

# Barrier to ensure both ranks finish
dist.barrier()

# Report
if my_rank == 0:
    print(f"[TC_SWEEP_RESULT] NCCL_IB_TC={tc} RANK={my_rank} OK", flush=True)

dist.destroy_process_group()
