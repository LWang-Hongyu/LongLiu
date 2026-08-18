#!/usr/bin/env python3
"""
Exp2 — 固定优先级 NCCL 作业（不调度）
用于测试1（P6 vs P3 抢占）与测试2（3×P3 共享）的 worker。

行为：
  - 以 --priority 指定优先级固定运行（不做 LongLiu 调度）
  - 每 iter：模拟计算 sleep + AllReduce，记录通信时间与带宽
  - 输出：data/<run_id>/exp2_<label>_rank<r>_iter.csv（rank0 另存汇总）

Usage（由 run_test*.sh 调用）：
  RANK=0/1 WORLD_SIZE=2 MASTER_ADDR=... MULTI_COMM_PORT=... \
  MULTI_COMM_SRC=... python3 fixed_prio_job.py \
      --label <job名> --priority <3|6> --payload-mb <MB> \
      --num-iters <N> --sleep-us <us> --outdir <dir> [--solo-bw <G>]
"""
import os
import sys
import time
import csv
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, required=True, help='作业标识（用于文件名）')
parser.add_argument('--priority', type=int, required=True, help='固定优先级 0-6')
parser.add_argument('--payload-mb', type=int, default=512)
parser.add_argument('--num-iters', type=int, default=60)
parser.add_argument('--sleep-us', type=int, default=10000)
parser.add_argument('--outdir', type=str, default='.')
parser.add_argument('--solo-bw', type=float, default=0.0, help='solo 参考带宽（可选，用于即时记录）')
args = parser.parse_args()

SCHED_DIR = os.environ.get('MULTI_COMM_SRC',
                           '/home/why/LongLiu_rebuild/current/multi_comm_slo/src')
sys.path.insert(0, SCHED_DIR)
from slo_scheduler import MultiCommWrapper  # noqa: E402

# 修正后映射：P6→DSCP8(tc:0 最高), P4→DSCP0(tc:1), P3→DSCP16(tc:2),
# P2→DSCP24(tc:3), P1→DSCP32(tc:4), P0→DSCP40(tc:5)
# （对齐硬件实测，见 HANDOFF_physical_evidence.md §e）
PRIORITY_TO_DSCP = {0: 40, 1: 32, 2: 24, 3: 16, 4: 0, 5: 16, 6: 8}


class FixedPrioScheduler:
    """固定优先级 stub（MultiCommWrapper 需要 scheduler 接口）。"""
    def __init__(self, priority):
        self.current_priority = priority
        self.priority_history = [priority]
        self.last_pi = float('nan')
        self.target_comm_time_s = None
        self.slo_threshold = None
        self.cumulative_actual_s = 0.0
        self.completed_iters = 0

    def update(self, actual_comm_time, data_size):
        self.completed_iters += 1
        self.cumulative_actual_s += actual_comm_time
        return self.current_priority

    def get_dscp(self):
        return PRIORITY_TO_DSCP.get(self.current_priority, self.current_priority * 8)


def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur


def main():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(0)
    device = torch.cuda.current_device()

    sched = FixedPrioScheduler(args.priority)
    # 只创建本 job 使用的优先级 communicator（P0-P6 全建时 NCCL 偶发死锁）
    os.environ['MULTI_COMM_PRIOS'] = str(args.priority)
    mc = MultiCommWrapper(
        sched, rank, world_size, str(0),
        os.environ.get('MASTER_ADDR', '192.10.10.110'),
        int(os.environ.get('MULTI_COMM_PORT',
            os.environ.get('MASTER_PORT', '29500'))))

    # 固定优先级，只设置一次
    mc._lib.multi_comm_set_priority(args.priority)

    payload_mb = args.payload_mb
    n_elem = payload_mb * 1024 * 1024 // 4
    tensor = torch.ones(n_elem, dtype=torch.float32, device=device)
    bytes_per_iter = payload_mb * 1024 * 1024

    # warmup
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        mc.allreduce(warmup, warmup, warmup.numel(), 0, 0, 0)
    torch.cuda.synchronize()

    if rank == 0:
        print(f"[Exp2-{args.label}] priority=P{args.priority} (DSCP={sched.get_dscp()}) "
              f"payload={payload_mb}MB iters={args.num_iters} sleep={args.sleep_us}us")

    records = []
    for i in range(args.num_iters):
        if args.sleep_us > 0:
            torch.cuda._sleep(args.sleep_us)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mc.allreduce(tensor, tensor, tensor.numel(), 0, 0, 0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dur = t1 - t0
        bw = bus_bw_gbps(bytes_per_iter, dur, world_size)
        records.append({'iter': i, 'ts': round(time.time(), 3),
                        'comm_dur_s': round(dur, 6), 'bw_gbps': round(bw, 4)})
        if rank == 0 and (i % 10 == 0 or i == args.num_iters - 1):
            print(f"[Exp2-{args.label}] iter {i}: comm={dur*1000:.1f}ms bw={bw:.2f}Gbps")

    # 落盘
    os.makedirs(args.outdir, exist_ok=True)
    out = f'{args.outdir}/exp2_{args.label}_rank{rank}_iter.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iter', 'ts', 'comm_dur_s', 'bw_gbps'])
        for r in records:
            w.writerow([r['iter'], r['ts'], r['comm_dur_s'], r['bw_gbps']])

    if rank == 0:
        avg_bw = sum(r['bw_gbps'] for r in records) / len(records)
        print(f"[Exp2-{args.label}] P{args.priority} 平均带宽 = {avg_bw:.2f} Gbps "
              f"(solo={args.solo_bw:.2f}) -> {out}")

    mc.destroy()


if __name__ == '__main__':
    main()
