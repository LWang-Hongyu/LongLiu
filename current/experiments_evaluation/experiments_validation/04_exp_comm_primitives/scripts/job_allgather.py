#!/usr/bin/env python3
"""
Exp4 — 通信原语多样性验证：AllGather 替换 AllReduce

保持 LongLiu 核心参数不变（π 调度 + EMA 锚点），仅将通信原语从
AllReduce 换成 AllGather，多轮次验证：
  1. DSCP 切换准确性 —— 调度器随 π 更新切换 priority → DSCP，
     每 iter 记录 priority/dscp，供分析阶段对照 NIC prio 计数器。
  2. 锚点测量精度 —— calib 阶段 solo 学到的 T_target / solo_bw 与
     main 阶段（有背景流）观测值的偏差。

输出（写入 --outdir）：
  exp4_<label>_rank<r>_iter.csv    per-iter: priority/dscp/pi/comm_dur/bw
  exp4_<label>_rank0_epoch.csv     per-epoch 汇总
  exp4_<label>_<phase>_rank<r>.log 日志

Usage（由 run_exp4.sh 调用）：
  RANK=0/1 WORLD_SIZE=2 MASTER_ADDR=... MULTI_COMM_PORT=... \
  MULTI_COMM_SRC=... python3 job_allgather.py \
      --label A --phase calib|main --ttarget-file <json> \
      --payload-mb 512 --sleep-us 30000 --num-iters 300 \
      --iters-per-epoch 20 --ci 1.7 --outdir <dir>
"""
import os
import sys
import time
import csv
import json
import argparse
import torch

parser = argparse.ArgumentParser(description='Exp4: AllGather 原语验证')
parser.add_argument('--label', type=str, required=True, choices=['A', 'B'])
parser.add_argument('--phase', type=str, required=True, choices=['calib', 'main'])
parser.add_argument('--num-iters', type=int, default=300)
parser.add_argument('--iters-per-epoch', type=int, default=20)
parser.add_argument('--calib-epochs', type=int, default=5)
parser.add_argument('--ttarget-file', type=str, default=None)
parser.add_argument('--payload-mb', type=int, default=512)
parser.add_argument('--sleep-us', type=int, default=30000)
parser.add_argument('--ci', type=float, default=1.7,
                    help='SLO 松弛系数 c_i（与 V6 实验保持一致）')
parser.add_argument('--initial-priority', type=int, default=3,
                    help='LongLiu 初始优先级（V6 用 P3）')
parser.add_argument('--max-priority', type=int, default=6,
                    help='LongLiu 最大优先级（默认不封顶，允许 P6）')
parser.add_argument('--outdir', type=str, default='.')
args = parser.parse_args()

JOB = args.label
PHASE = args.phase
NUM_ITERS = args.num_iters
ITERS_PER_EPOCH = args.iters_per_epoch
NUM_EPOCHS = NUM_ITERS // ITERS_PER_EPOCH
CALIB_EPOCHS = args.calib_epochs
PAYLOAD_MB = args.payload_mb
SLEEP_US = args.sleep_us
C_I = args.ci

PRIORITY_TO_DSCP = {6: 8, 4: 0, 3: 16, 2: 24, 1: 32, 0: 40}

SCHED_DIR = os.environ.get('MULTI_COMM_SRC',
                           '/home/why/LongLiu_rebuild/current/multi_comm_slo/src')
sys.path.insert(0, SCHED_DIR)
from slo_scheduler import MultiCommWrapper, SLOScheduler  # noqa: E402


def bus_bw_gbps(sendbytes, comm_dur, world_size):
    """AllGather 总线带宽：
    每 rank 链路上传输 (sendbytes × world_size) × (W-1)/W = sendbytes×(W-1)。
    """
    if comm_dur <= 0:
        return 0.0
    return (sendbytes * (world_size - 1)) * 8.0 / 1e9 / comm_dur


def allocate_send_tensor(payload_mb, device):
    n_elem = payload_mb * 1024 * 1024 // 4
    return torch.ones(n_elem, dtype=torch.float32, device=device)


def make_wrapper(rank, world_size, target_ms=None, preset=False):
    sched = SLOScheduler(slo_threshold=C_I,
                         target_comm_time_ms=target_ms,
                         preset_target=preset,
                         initial_priority=args.initial_priority,
                         max_priority=args.max_priority)
    mc = MultiCommWrapper(
        sched, rank, world_size, str(0),
        os.environ.get('MASTER_ADDR', '192.10.10.110'),
        int(os.environ.get('MULTI_COMM_PORT',
            os.environ.get('MASTER_PORT', '29500'))))
    return sched, mc


def allgather(mc, send_tensor, device, world_size):
    """AllGather: sendbuff=每 rank 数据，recvbuff=W×sendcount 拼接。"""
    recv = torch.empty(send_tensor.numel() * world_size,
                       dtype=torch.float32, device=device)
    mc.allgather(send_tensor.data_ptr(), recv.data_ptr(),
                 send_tensor.numel(), 0, 0)
    return recv


def run_calib(rank, world_size, device):
    sendbytes = PAYLOAD_MB * 1024 * 1024
    sched, mc = make_wrapper(rank, world_size, target_ms=None, preset=False)
    tensor = allocate_send_tensor(PAYLOAD_MB, device)

    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        allgather(mc, warmup, device, world_size)
    torch.cuda.synchronize()

    bws = []
    for epoch in range(CALIB_EPOCHS):
        mc.epoch_start(epoch)
        for i in range(ITERS_PER_EPOCH):
            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            allgather(mc, tensor, device, world_size)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dur = t1 - t0
            bw = bus_bw_gbps(sendbytes, dur, world_size)
            bws.append(bw)
            if rank == 0:
                print(f"[Exp4-CALIB-J{JOB}] epoch {epoch} iter {i}: "
                      f"comm={dur*1000:.1f}ms bw={bw:.2f}Gbps")
        mc.epoch_end(epoch, data_size=sendbytes)

    if rank == 0 and args.ttarget_file:
        ttarget_ms = sched.target_comm_time_s * 1000.0
        solo_bw = sum(bws) / len(bws) if bws else 0.0
        data = {
            'job': JOB,
            'primitive': 'allgather',
            'payload_mb': PAYLOAD_MB,
            'sleep_us': SLEEP_US,
            'ci': C_I,
            'calib_epochs': CALIB_EPOCHS,
            'iters_per_epoch': ITERS_PER_EPOCH,
            'target_comm_time_ms': round(ttarget_ms, 3),
            'solo_bw_gbps': round(solo_bw, 4),
            'unit': 'per_epoch_ms',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        os.makedirs(os.path.dirname(args.ttarget_file) or '.', exist_ok=True)
        with open(args.ttarget_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Exp4-CALIB-J{JOB}] T_target={ttarget_ms:.2f}ms "
              f"solo_bw={solo_bw:.2f}Gbps -> {args.ttarget_file}")

    mc.destroy()


def run_main(rank, world_size, device):
    sendbytes = PAYLOAD_MB * 1024 * 1024
    solo_bw = 0.0
    target_ms = None
    if args.ttarget_file and os.path.exists(args.ttarget_file):
        with open(args.ttarget_file) as f:
            tdata = json.load(f)
        target_ms = tdata.get('target_comm_time_ms')
        solo_bw = tdata.get('solo_bw_gbps', 0.0)
        if rank == 0:
            print(f"[Exp4-J{JOB}-MAIN] 锚点: T_target={target_ms:.2f}ms "
                  f"solo_bw={solo_bw:.2f}Gbps (preset_target=True)")

    sched, mc = make_wrapper(rank, world_size, target_ms=target_ms, preset=True)
    if rank == 0:
        print(f"[Exp4-J{JOB}-MAIN] AllGather payload={PAYLOAD_MB}MB "
              f"sleep={SLEEP_US}us iters={NUM_ITERS} ci={C_I} "
              f"init_prio=P{args.initial_priority}")

    tensor = allocate_send_tensor(PAYLOAD_MB, device)
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        allgather(mc, warmup, device, world_size)
    torch.cuda.synchronize()

    iter_records = []
    epoch_records = []
    for epoch in range(NUM_EPOCHS):
        mc.epoch_start(epoch)
        ep_dur = []
        ep_bw = []
        for i in range(ITERS_PER_EPOCH):
            giter = epoch * ITERS_PER_EPOCH + i
            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            allgather(mc, tensor, device, world_size)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dur = t1 - t0
            bw = bus_bw_gbps(sendbytes, dur, world_size)
            ep_dur.append(dur)
            ep_bw.append(bw)
            iter_records.append({
                'iter': giter, 'epoch': epoch,
                'priority': sched.current_priority,
                'dscp': PRIORITY_TO_DSCP.get(sched.current_priority, 0),
                'pi': round(sched.last_pi, 6) if sched.last_pi == sched.last_pi else 'nan',
                'comm_dur_s': round(dur, 6), 'bw_gbps': round(bw, 4),
            })
        mc.epoch_end(epoch, data_size=sendbytes)

        if rank == 0:
            avg_comm = sum(ep_dur) / len(ep_dur)
            avg_bw = sum(ep_bw) / len(ep_bw)
            iter_target = (sched.target_comm_time_s / ITERS_PER_EPOCH
                           if sched.target_comm_time_s else None)
            slowdown = (avg_comm / (C_I * iter_target)
                        if iter_target else float('nan'))
            epoch_records.append({
                'epoch': epoch,
                'avg_comm_s': round(avg_comm, 6),
                'avg_bw_gbps': round(avg_bw, 4),
                'pi': round(sched.last_pi, 6) if sched.last_pi == sched.last_pi else 'nan',
                'priority': sched.current_priority,
                'dscp': PRIORITY_TO_DSCP.get(sched.current_priority, 0),
                'slowdown': round(slowdown, 4),
            })

    os.makedirs(args.outdir, exist_ok=True)
    if rank == 0:
        ip = f'{args.outdir}/exp4_job{JOB}_rank0_iter.csv'
        with open(ip, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iter', 'epoch', 'priority', 'dscp', 'pi',
                        'comm_dur_s', 'bw_gbps'])
            for r in iter_records:
                w.writerow([r['iter'], r['epoch'], r['priority'], r['dscp'],
                            r['pi'], r['comm_dur_s'], r['bw_gbps']])

        ep = f'{args.outdir}/exp4_job{JOB}_rank0_epoch.csv'
        with open(ep, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'avg_comm_s', 'avg_bw_gbps', 'pi',
                        'priority', 'dscp', 'slowdown'])
            for r in epoch_records:
                w.writerow([r['epoch'], r['avg_comm_s'], r['avg_bw_gbps'],
                            r['pi'], r['priority'], r['dscp'], r['slowdown']])

        print(f"[Exp4-J{JOB}-MAIN] 完成 {NUM_ITERS} iters，落盘 -> {args.outdir}")
        print(f"  锚点精度: solo_bw={solo_bw:.2f}Gbps, main 均值="
              f"{sum(r['avg_bw_gbps'] for r in epoch_records)/len(epoch_records):.2f}Gbps")

    mc.destroy()


def main():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(0)
    device = torch.cuda.current_device()

    if PHASE == 'calib':
        run_calib(rank, world_size, device)
    else:
        run_main(rank, world_size, device)


if __name__ == '__main__':
    main()
