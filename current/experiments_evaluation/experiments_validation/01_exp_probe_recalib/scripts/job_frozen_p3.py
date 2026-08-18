#!/usr/bin/env python3
"""
Experiment 1 — 主动重校准探针物理验证

作业行为：
  - 以 P3 优先级启动，锚点参数冻结（preset_target=True，T_target 不更新；
    同时 max_priority=3 强制优先级停留在 P3，模拟"锚点已失效、无法自救"的作业）
  - 每 PROBE_EVERY 个 window，在 window 边界触发一次 P6 单次 AllReduce 探测
    （显式切换到 P6 communicator，单次 AllReduce，测带宽，切回 P3）
  - 背景流（P3 打满链路）由 run_exp1.sh 负责，本脚本只负责作业 + 探测

输出（写入 --outdir）：
  exp1_job<JOB>_rank<r>_iter.csv    per-iter 通信时间/带宽
  exp1_job<JOB>_rank0_window.csv     per-window 统计（π、priority、slowdown）
  exp1_job<JOB>_rank0_probe.csv     探测样本（时间戳、带宽；EMA 在分析阶段计算）
  exp1_job<JOB>_<phase>_rank<r>.log 日志

校准阶段（--phase calib）：solo 运行，学习 T_target 与 solo 带宽基线，
写入 --ttarget-file（含 solo_bw_gbps）。
"""
import os
import sys
import time
import csv
import json
import argparse
import torch

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Exp1: P3 冻结作业 + P6 探测')
parser.add_argument('--job', type=str, required=True, choices=['A', 'B'])
parser.add_argument('--phase', type=str, required=True, choices=['calib', 'main'])
parser.add_argument('--num-windows', type=int, default=15)
parser.add_argument('--iters-per-window', type=int, default=20)
parser.add_argument('--calib-windows', type=int, default=8)
parser.add_argument('--ttarget-file', type=str, default=None,
                    help='calib: 写 T_target+solo_bw；main: 读入冻结锚点')
parser.add_argument('--payload-mb', type=int, default=1024)
parser.add_argument('--sleep-us', type=int, default=30000)
parser.add_argument('--probe-every', type=int, default=3,
                    help='每 N 个 window 触发一次 P6 探测')
parser.add_argument('--outdir', type=str, default='.')
args = parser.parse_args()

JOB = args.job
PHASE = args.phase
NUM_WINDOWS = args.num_windows
ITERS_PER_WINDOW = args.iters_per_window
CALIB_WINDOWS = args.calib_windows
PAYLOAD_MB = args.payload_mb
SLEEP_US = args.sleep_us
PROBE_EVERY = args.probe_every

# 冻结配置：初始 P3 + 锚点冻结 + 优先级封顶 P3（模拟失效锚点作业）
INITIAL_PRIORITY = 3
MAX_PRIORITY = 3
PROBE_PRIORITY = 6

PRIORITY_TO_DSCP = {6: 8, 4: 0, 3: 16, 2: 24, 1: 32, 0: 40}

# 调度器源码路径（双端不同，由运行脚本设置）
SCHED_DIR = os.environ.get('MULTI_COMM_SRC',
                           '/home/why/LongLiu_rebuild/current/multi_comm_slo/src')
sys.path.insert(0, SCHED_DIR)
from slo_scheduler import MultiCommWrapper, SLOScheduler  # noqa: E402


def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    """Per-direction wire bandwidth for NCCL Ring AllReduce."""
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur


def allocate_tensor(payload_mb, device):
    num_elements = payload_mb * 1024 * 1024 // 4
    return torch.ones(num_elements, dtype=torch.float32, device=device)


def make_wrapper(rank, world_size, target_ms=None, preset=False):
    """创建冻结锚点的 SLOScheduler + MultiCommWrapper。"""
    sched = SLOScheduler(slo_threshold=1.5,
                         target_comm_time_ms=target_ms,
                         preset_target=preset,
                         initial_priority=INITIAL_PRIORITY,
                         max_priority=MAX_PRIORITY)
    mc = MultiCommWrapper(
        sched, rank, world_size, str(0),
        os.environ.get('MASTER_ADDR', '192.10.10.110'),
        int(os.environ.get('MULTI_COMM_PORT',
            os.environ.get('MASTER_PORT', '29500'))))
    return sched, mc


def run_probe(mc, tensor, bytes_per_iter, world_size):
    """P6 单次 AllReduce 探测。返回 (comm_dur_s, bw_gbps)。"""
    mc._lib.multi_comm_set_priority(PROBE_PRIORITY)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mc.allreduce(tensor, tensor, tensor.numel(), 0, 0, 0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    mc._lib.multi_comm_set_priority(INITIAL_PRIORITY)
    dur = t1 - t0
    bw = bus_bw_gbps(bytes_per_iter, dur, world_size) if dur > 0 else 0.0
    return dur, bw


# ---------------------------------------------------------------------------
# 校准阶段（solo，无背景流）
# ---------------------------------------------------------------------------
def run_calib(rank, world_size, device):
    bytes_per_iter = PAYLOAD_MB * 1024 * 1024
    sched, mc = make_wrapper(rank, world_size, target_ms=None, preset=False)

    tensor = allocate_tensor(PAYLOAD_MB, device)
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        mc.allreduce(warmup, warmup, warmup.numel(), 0, 0, 0)
    torch.cuda.synchronize()

    bws = []
    for window in range(CALIB_WINDOWS):
        mc.window_start(window)
        window_bws = []
        for i in range(ITERS_PER_WINDOW):
            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mc.allreduce(tensor, tensor, tensor.numel(), 0, 0, 0)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dur = t1 - t0
            bw = bus_bw_gbps(bytes_per_iter, dur, world_size)
            window_bws.append(bw)
            if rank == 0:
                print(f"[Exp1-CALIB-J{JOB}] window {window} iter {i}: comm={(dur)*1000:.1f}ms bw={bw:.2f}G")
        bws.extend(window_bws)
        mc.window_end(window, data_size=bytes_per_iter)

    if rank == 0 and args.ttarget_file:
        ttarget_ms = sched.target_comm_time_s * 1000.0
        solo_bw = sum(bws) / len(bws) if bws else 0.0
        data = {
            'job': JOB,
            'payload_mb': PAYLOAD_MB,
            'sleep_us': SLEEP_US,
            'calib_windows': CALIB_WINDOWS,
            'iters_per_window': ITERS_PER_WINDOW,
            'target_comm_time_ms': round(ttarget_ms, 3),
            'solo_bw_gbps': round(solo_bw, 4),
            'unit': 'per_window_ms',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        os.makedirs(os.path.dirname(args.ttarget_file) or '.', exist_ok=True)
        with open(args.ttarget_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Exp1-CALIB-J{JOB}] T_target={ttarget_ms:.2f}ms solo_bw={solo_bw:.2f}Gbps -> {args.ttarget_file}")

    mc.destroy()


# ---------------------------------------------------------------------------
# 主阶段（背景流 + 冻结 P3 + 周期 P6 探测）
# ---------------------------------------------------------------------------
def run_main(rank, world_size, device):
    bytes_per_iter = PAYLOAD_MB * 1024 * 1024

    # 读取冻结锚点
    target_ms = None
    solo_bw = 0.0
    if args.ttarget_file and os.path.exists(args.ttarget_file):
        with open(args.ttarget_file) as f:
            tdata = json.load(f)
        target_ms = tdata.get('target_comm_time_ms')
        solo_bw = tdata.get('solo_bw_gbps', 0.0)
        if rank == 0:
            print(f"[Exp1-J{JOB}-MAIN] 冻结锚点: T_target={target_ms:.2f}ms solo_bw={solo_bw:.2f}Gbps "
                  f"(preset_target=True, max_priority={MAX_PRIORITY})")

    sched, mc = make_wrapper(rank, world_size, target_ms=target_ms, preset=True)
    if rank == 0:
        print(f"[Exp1-J{JOB}-MAIN] payload={PAYLOAD_MB}MB sleep={SLEEP_US}us "
              f"windows={NUM_WINDOWS} probe_every={PROBE_EVERY}")

    tensor = allocate_tensor(PAYLOAD_MB, device)
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        mc.allreduce(warmup, warmup, warmup.numel(), 0, 0, 0)
    torch.cuda.synchronize()

    iter_records = []
    window_records = []
    probe_records = []

    for window in range(NUM_WINDOWS):
        mc.window_start(window)
        window_comm = []
        window_bws = []
        for i in range(ITERS_PER_WINDOW):
            giter = window * ITERS_PER_WINDOW + i
            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mc.allreduce(tensor, tensor, tensor.numel(), 0, 0, 0)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dur = t1 - t0
            bw = bus_bw_gbps(bytes_per_iter, dur, world_size)
            window_comm.append(dur)
            window_bws.append(bw)
            iter_records.append({
                'iter': giter, 'window': window,
                'comm_dur_s': round(dur, 6), 'bw_gbps': round(bw, 4),
            })
        mc.window_end(window, data_size=bytes_per_iter)

        if rank == 0:
            avg_comm = sum(window_comm) / len(window_comm)
            avg_bw = sum(window_bws) / len(window_bws)
            ttarget_per_iter = (sched.target_comm_time_s / ITERS_PER_WINDOW
                                if sched.target_comm_time_s else None)
            slowdown = (avg_comm / (1.5 * ttarget_per_iter)
                        if ttarget_per_iter else float('nan'))
            window_records.append({
                'window': window,
                'avg_comm_s': round(avg_comm, 6),
                'avg_bw_gbps': round(avg_bw, 4),
                'pi': round(sched.last_pi, 6),
                'priority': sched.current_priority,
                'dscp': PRIORITY_TO_DSCP.get(sched.current_priority, 0),
                'slowdown': round(slowdown, 4),
            })

        # 周期性 P6 探测（window 边界）
        if (window + 1) % PROBE_EVERY == 0:
            pdur, pbw = run_probe(mc, tensor, bytes_per_iter, world_size)
            probe_records.append({
                'probe_idx': len(probe_records),
                'window': window,
                'wall_ts': round(time.time(), 3),
                'comm_dur_s': round(pdur, 6),
                'bw_gbps': round(pbw, 4),
                'solo_bw_gbps': round(solo_bw, 4),
                'clean': int(pbw >= 0.9 * solo_bw) if solo_bw > 0 else -1,
            })
            if rank == 0:
                print(f"[Exp1-J{JOB}-PROBE #{len(probe_records)}] window {window}: "
                      f"P6 探测 comm={pdur*1000:.1f}ms bw={pbw:.2f}Gbps "
                      f"(solo={solo_bw:.2f}G, clean={'Y' if pbw>=0.9*solo_bw and solo_bw>0 else 'N'})")

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    if rank == 0:
        ip = f'{args.outdir}/exp1_job{JOB}_rank0_iter.csv'
        with open(ip, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iter', 'window', 'comm_dur_s', 'bw_gbps'])
            for r in iter_records:
                w.writerow([r['iter'], r['window'], r['comm_dur_s'], r['bw_gbps']])

        ep = f'{args.outdir}/exp1_job{JOB}_rank0_window.csv'
        with open(ep, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['window', 'avg_comm_s', 'avg_bw_gbps', 'pi', 'priority', 'dscp', 'slowdown'])
            for r in window_records:
                w.writerow([r['window'], r['avg_comm_s'], r['avg_bw_gbps'], r['pi'],
                            r['priority'], r['dscp'], r['slowdown']])

        pp = f'{args.outdir}/exp1_job{JOB}_rank0_probe.csv'
        with open(pp, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['probe_idx', 'window', 'wall_ts', 'comm_dur_s', 'bw_gbps',
                        'solo_bw_gbps', 'clean'])
            for r in probe_records:
                w.writerow([r['probe_idx'], r['window'], r['wall_ts'], r['comm_dur_s'],
                            r['bw_gbps'], r['solo_bw_gbps'], r['clean']])

        n_clean = sum(r['clean'] == 1 for r in probe_records)
        print(f"[Exp1-J{JOB}-MAIN] 探测样本 {len(probe_records)} 个，无拥塞样本 "
              f"{n_clean} 个 ({n_clean/max(len(probe_records),1)*100:.1f}%)")
        print(f"[Exp1-J{JOB}-MAIN] 落盘 -> {args.outdir}")

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
