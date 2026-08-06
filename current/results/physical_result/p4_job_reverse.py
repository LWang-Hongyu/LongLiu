#!/usr/bin/env python3
"""
P4-1 Role-Reversal Experiment V4 (SP queue, scheduler v1(π), c_i swap, same payload)

Scenario redesign (2026-07-20, V4 — same payload, asymmetric c_i):
  - Phase 0 (calibrate): each job runs SOLO for 5 epochs to establish an
    uncontaminated T_target via EMA. T_target is written to a JSON file
    and reused by the main experiment (preset_target=True).
  - Both jobs use the SAME payload (512 MB) throughout — only c_i differs.
    This is the "money experiment": when jobs are identical except for SLO
    tightness, CRUX's GPU-intensity-based static priority has NOTHING to
    discriminate on (both jobs have identical GPU intensity → CRUX is blind).
  - c_i SWAP at REVERSE_EPOCH:
      Phase 1 (epochs 0-6): A c_i=1.6 (tight), B c_i=3.0 (loose)
      Phase 2 (epochs 7-14): A c_i=3.0 (loose), B c_i=1.6 (tight)
  - Both jobs run all 15 epochs — no early exit.

Why same payload:
  V3 (different payloads) showed host-NIC contention makes the light job's
  SLO physically unreachable regardless of policy. V4 uses same payload so
  both jobs have equal bandwidth claim — SLO feasibility depends only on
  c_i assignment, not on asymmetric bandwidth competition.

Expected hypothesis (falsifiable):
  CRUX Phase 1: GPU intensity tie → bandwidth split ~50/50 → tight A violates
  CRUX Phase 2: B becomes tight → also violates → CRUX fails tight job both phases
  LongLiu both phases: π tracks actual SLO status → both jobs ~0 violations,
  priority trajectories cross at epoch 7.

CSV output (main phase): per-iter and per-epoch with π, priority, dscp, slowdown, c_i.
"""

import os
import sys
import time
import csv
import json
import argparse
import torch

# ============================================================
# Parse args
# ============================================================
parser = argparse.ArgumentParser(description='P4-1 Role-Reversal V3 (c_i swap)')
parser.add_argument('--job', type=str, required=True, choices=['A', 'B'],
                    help='Job identity (A=heavy/tight→loose, B=light/loose→tight)')
parser.add_argument('--mode', type=str, required=True, choices=['longliu', 'crux'])
parser.add_argument('--phase', type=str, required=True, choices=['calibrate', 'main'],
                    help='calibrate=solo T_target learning; main=contested experiment')
parser.add_argument('--reverse-epoch', type=int, default=7)
parser.add_argument('--num-iters', type=int, default=300)
parser.add_argument('--iters-per-epoch', type=int, default=20)
parser.add_argument('--calib-epochs', type=int, default=5,
                    help='Number of solo epochs for T_target calibration')
parser.add_argument('--ttarget-file', type=str, default=None,
                    help='JSON file path for T_target (calibrate: write; main: read)')
parser.add_argument('--payload-mb', type=int, default=512,
                    help='Payload size in MB (default: 512)')
parser.add_argument('--ci-phase1', type=float, required=True,
                    help='c_i for Phase 1 (tight job)')
parser.add_argument('--ci-phase2', type=float, required=True,
                    help='c_i for Phase 2 (loose job)')
parser.add_argument('--sleep-us', type=int, default=30000,
                    help='Compute time in us (default: 30000 = 30ms)')
parser.add_argument('--crux-priority-a', type=int, default=4,
                    help='CRUX static priority for Job A (default: 4, V6: 3)')
parser.add_argument('--crux-priority-b', type=int, default=3,
                    help='CRUX static priority for Job B (default: 3, V6: 3)')
parser.add_argument('--initial-priority', type=int, default=None,
                    help='Initial priority for LongLiu scheduler (default: 4, V6: 3)')
parser.add_argument('--max-priority', type=int, default=None,
                    help='Max priority cap for LongLiu scheduler (e.g., 4 caps at P4). '
                         'Default: None (no cap, allow P6)')
args = parser.parse_args()

JOB = args.job
MODE = args.mode
PHASE = args.phase
REVERSE_EPOCH = args.reverse_epoch
NUM_ITERS = args.num_iters
ITERS_PER_EPOCH = args.iters_per_epoch
CALIB_EPOCHS = args.calib_epochs
NUM_EPOCHS = NUM_ITERS // ITERS_PER_EPOCH  # 15

# ============================================================
# Workload configuration (parameterized via args)
# ============================================================
SLEEP_US = args.sleep_us
PAYLOAD_MB = args.payload_mb

# Payload is the SAME for both jobs (money experiment)
JOB_PAYLOAD_MB = {'A': PAYLOAD_MB, 'B': PAYLOAD_MB}

# c_i SWAPS at REVERSE_EPOCH — values from command line
JOB_C_I_PHASE1 = {'A': args.ci_phase1, 'B': args.ci_phase2}
JOB_C_I_PHASE2 = {'A': args.ci_phase2, 'B': args.ci_phase1}

# CRUX static priority: GPU Intensity-based assignment (identical to
# the original p4_job1_crux.py / p4_job2_crux.py pattern).
# Job A (tight) gets P4, Job B (loose) gets P3 — this CRUX-style
# static differentiation has been validated in earlier experiments.
# Both jobs have identical GPU intensity (same payload, same compute)
# but CRUX in practice assigns discrete priorities anyway.
JOB_CRUX_PRIORITY = {'A': args.crux_priority_a, 'B': args.crux_priority_b}

CRUX_STATIC_PRIORITY = JOB_CRUX_PRIORITY[JOB]
PAYLOAD_MB = JOB_PAYLOAD_MB[JOB]
C_I_PHASE1 = JOB_C_I_PHASE1[JOB]
C_I_PHASE2 = JOB_C_I_PHASE2[JOB]

# Hardware DSCP mapping (corrected for 10.1 NIC TC order):
# tc:0(prio1,dscp8-15) > tc:1(prio0,dscp0-7) > tc:2(prio2,dscp16-23) > ...
# Higher software priority → higher hardware TC
# P6(highest)→DSCP=8(tc:0), P4→DSCP=0(tc:1), P3→DSCP=16(tc:2),
# P2→DSCP=24(tc:3), P1(lowest)→DSCP=32(tc:4)
PRIORITY_TO_DSCP = {6: 8, 4: 0, 3: 16, 2: 24, 1: 32, 0: 40}


def bus_bw_gbps(bytes_per_iter, comm_dur, world_size):
    """Per-direction wire bandwidth for NCCL Ring AllReduce."""
    return (bytes_per_iter * 8.0 / 1e9) * (world_size - 1) / world_size / comm_dur


def get_c_i(epoch):
    """c_i value based on current epoch (swap at REVERSE_EPOCH)."""
    return C_I_PHASE1 if epoch < REVERSE_EPOCH else C_I_PHASE2


def get_phase_label(epoch):
    return 'phase1' if epoch < REVERSE_EPOCH else 'phase2'


def allocate_tensor(payload_mb, device):
    """Allocate a float32 tensor of given payload size (in MB)."""
    num_elements = payload_mb * 1024 * 1024 // 4
    return torch.ones(num_elements, dtype=torch.float32, device=device)


# ============================================================
# Scheduler setup
# ============================================================
import torch.distributed as dist
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', 'multi_comm_slo', 'src'))
from slo_scheduler import MultiCommWrapper, SLOScheduler


def make_scheduler(c_i=None, ttarget_ms=None):
    """Create scheduler based on MODE and PHASE."""
    # Determine initial priority for LongLiu (default 4, V6 uses 3)
    ll_initial_prio = args.initial_priority if args.initial_priority is not None else 4
    ll_max_prio = args.max_priority  # None = no cap (allow P6)
    if MODE == 'longliu':
        if PHASE == 'main' and ttarget_ms is not None:
            # Main phase with preset T_target from calibration
            return SLOScheduler(slo_threshold=c_i,
                                target_comm_time_ms=ttarget_ms,
                                preset_target=True,
                                initial_priority=ll_initial_prio,
                                max_priority=ll_max_prio)
        else:
            # Calibration phase: learn T_target via EMA warmup
            return SLOScheduler(slo_threshold=c_i,
                                initial_priority=ll_initial_prio,
                                max_priority=ll_max_prio)
    else:  # crux — static priority, never updates (but still need T_target for slowdown)
        class StaticPriorityScheduler:
            """CRUX baseline: static priority based on GPU Intensity, never changes.
            Paper definition (CRUX SIGCOMM 2024):
              I_j = compute_time / comm_time (high → high priority)
              Job A low I → P3; Job B high I → P4.
            """
            def __init__(self, static_priority, c_i=None, ttarget_ms=None):
                self.current_priority = static_priority
                self.priority_history = [self.current_priority]
                self.last_pi = float('nan')
                self.slo_threshold = c_i
                self.target_comm_time_s = (ttarget_ms / 1000.0
                                           if ttarget_ms is not None else None)
                self.cumulative_actual_s = 0.0
                self.completed_iters = 0
                self.preset_target = (ttarget_ms is not None)
            def update(self, actual_comm_time, data_size):
                self.completed_iters += 1
                self.cumulative_actual_s += actual_comm_time
                # Compute π for logging/slowdown (even though CRUX ignores it)
                if (self.target_comm_time_s is not None
                        and self.target_comm_time_s > 0
                        and self.slo_threshold is not None):
                    expected = (self.slo_threshold * self.target_comm_time_s
                                * self.completed_iters)
                    self.last_pi = (self.cumulative_actual_s / expected - 1.0
                                    if expected > 0 else float('nan'))
                return self.current_priority
            def get_dscp(self):
                return PRIORITY_TO_DSCP.get(self.current_priority, self.current_priority * 8)
            def set_slo_threshold(self, new_threshold):
                self.slo_threshold = new_threshold
        return StaticPriorityScheduler(CRUX_STATIC_PRIORITY, c_i=c_i,
                                       ttarget_ms=ttarget_ms)


_mc_wrapper = None
_scheduler = None


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


# ============================================================
# Phase 0: Calibration (solo, learn T_target)
# ============================================================
def run_calibration(rank, world_size, device):
    """Run CALIB_EPOCHS solo epochs with fixed payload, EMA-learn T_target."""
    global _scheduler, _mc_wrapper

    bytes_per_iter = PAYLOAD_MB * 1024 * 1024

    _scheduler = make_scheduler(c_i=C_I_PHASE1, ttarget_ms=None)  # EMA warmup mode
    _mc_wrapper = MultiCommWrapper(
        _scheduler, rank, world_size, str(0),
        os.environ.get('MASTER_ADDR', '192.10.10.110'),
        int(os.environ.get('MULTI_COMM_PORT',
            os.environ.get('MASTER_PORT', '29500'))))

    if rank == 0:
        print(f"[Job{JOB}-{MODE.upper()}-CALIB] solo calibration: "
              f"{CALIB_EPOCHS} epochs, payload={PAYLOAD_MB}MB, c_i={C_I_PHASE1}")

    tensor = allocate_tensor(PAYLOAD_MB, device)

    # Brief warmup
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        allreduce(warmup)
    torch.cuda.synchronize()

    for epoch in range(CALIB_EPOCHS):
        epoch_start(epoch)
        for i in range(ITERS_PER_EPOCH):
            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            allreduce(tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if rank == 0:
                print(f"[Job{JOB}-CALIB] epoch {epoch} iter {i}: "
                      f"comm={(t1-t0)*1000:.1f}ms")
        epoch_end(epoch, data_size=bytes_per_iter)

    # Rank 0 writes T_target to JSON file
    if rank == 0 and args.ttarget_file:
        ttarget_ms = _scheduler.target_comm_time_s * 1000.0
        ttarget_data = {
            'job': JOB,
            'mode': MODE,
            'payload_mb': PAYLOAD_MB,
            'c_i_calib': C_I_PHASE1,
            'sleep_us': SLEEP_US,
            'calib_epochs': CALIB_EPOCHS,
            'iters_per_epoch': ITERS_PER_EPOCH,
            'target_comm_time_ms': round(ttarget_ms, 3),
            'unit': 'per_epoch_ms',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        os.makedirs(os.path.dirname(args.ttarget_file) or '.', exist_ok=True)
        with open(args.ttarget_file, 'w') as f:
            json.dump(ttarget_data, f, indent=2)
        print(f"[Job{JOB}-CALIB] T_target={ttarget_ms:.2f}ms written to {args.ttarget_file}")

    _mc_wrapper.destroy()


# ============================================================
# Phase 1+2: Main experiment (contested, with c_i swap)
# ============================================================
def run_main(rank, world_size, device):
    """Run NUM_EPOCHS contested epochs with c_i swap at REVERSE_EPOCH."""
    global _scheduler, _mc_wrapper

    # Read T_target from calibration file
    ttarget_ms = None
    if args.ttarget_file and os.path.exists(args.ttarget_file):
        with open(args.ttarget_file) as f:
            tdata = json.load(f)
        # Unit assertion: prevent bare-number T_target misuse
        unit = tdata.get('unit', None)
        if unit is not None and unit != 'per_epoch_ms':
            raise ValueError(
                f"T_target unit '{unit}' != expected 'per_epoch_ms' "
                f"in {args.ttarget_file}")
        ttarget_ms = tdata['target_comm_time_ms']
        if rank == 0:
            print(f"[Job{JOB}-{MODE.upper()}-MAIN] Loaded T_target={ttarget_ms:.2f}ms "
                  f"from {args.ttarget_file} (preset_target=True)")
    else:
        if rank == 0:
            print(f"[Job{JOB}-{MODE.upper()}-MAIN] WARNING: no T_target file, "
                  f"falling back to EMA warmup (preset_target=False)")

    # Start with phase-1 c_i
    _scheduler = make_scheduler(c_i=C_I_PHASE1, ttarget_ms=ttarget_ms)
    _mc_wrapper = MultiCommWrapper(
        _scheduler, rank, world_size, str(0),
        os.environ.get('MASTER_ADDR', '192.10.10.110'),
        int(os.environ.get('MULTI_COMM_PORT',
            os.environ.get('MASTER_PORT', '29500'))))

    if rank == 0:
        print(f"[Job{JOB}-{MODE.upper()}-MAIN] scheduler="
              f"{'v1(pi)' if MODE == 'longliu' else 'CRUX-static'}, "
              f"queue=SP, payload={PAYLOAD_MB}MB (fixed), "
              f"reverse_epoch={REVERSE_EPOCH} (c_i swap), "
              f"num_iters={NUM_ITERS}")
        print(f"[Job{JOB}-{MODE.upper()}-MAIN] Phase 1 (epoch 0-{REVERSE_EPOCH-1}): "
              f"c_i={C_I_PHASE1}")
        print(f"[Job{JOB}-{MODE.upper()}-MAIN] Phase 2 (epoch {REVERSE_EPOCH}-{NUM_EPOCHS-1}): "
              f"c_i={C_I_PHASE2}")
        if MODE == 'crux':
            print(f"[Job{JOB}-CRUX] Static priority: P{CRUX_STATIC_PRIORITY} "
                  f"(DSCP={CRUX_STATIC_PRIORITY*8}) — based on GPU intensity, held fixed")
        free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
        print(f"[Job{JOB}-{MODE.upper()}-MAIN] GPU {device}: "
              f"{torch.cuda.get_device_name(device)}, free={free_mem:.1f} GB")

    # Pre-allocate tensor (fixed payload, no realloc needed)
    tensor = allocate_tensor(PAYLOAD_MB, device)
    bytes_per_iter = PAYLOAD_MB * 1024 * 1024

    # Warmup
    warmup = torch.ones(1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(2):
        allreduce(warmup)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[Job{JOB}-{MODE.upper()}-MAIN] Warmup done.")

    results = []        # per-iter records
    epoch_stats = []    # per-epoch aggregated records
    t_total_start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        c_i = get_c_i(epoch)
        phase_label = get_phase_label(epoch)

        # At REVERSE_EPOCH, swap c_i in scheduler
        if epoch == REVERSE_EPOCH:
            _scheduler.set_slo_threshold(c_i)
            if rank == 0:
                print(f"[Job{JOB}-{MODE.upper()}-MAIN] *** c_i SWAP at epoch {epoch}: "
                      f"now c_i={c_i} ***")

        epoch_start(epoch)
        epoch_comm_times = []
        epoch_bws = []

        for i in range(ITERS_PER_EPOCH):
            global_iter = epoch * ITERS_PER_EPOCH + i

            # Fixed compute (same for both jobs, both phases)
            if SLEEP_US > 0:
                torch.cuda._sleep(SLEEP_US)

            # AllReduce with fixed payload
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            allreduce(tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            comm_dur = t1 - t0
            bw_gbps = bus_bw_gbps(bytes_per_iter, comm_dur, world_size) if comm_dur > 0 else 0.0
            epoch_comm_times.append(comm_dur)
            epoch_bws.append(bw_gbps)

            if rank == 0:
                print(f"[Job{JOB}-{MODE.upper()}-MAIN] iter {global_iter:3d} (epoch {epoch}): "
                      f"payload={PAYLOAD_MB}MB, c_i={c_i}, sleep={SLEEP_US}us, "
                      f"comm={comm_dur*1000:.1f}ms, bw={bw_gbps:.2f} Gbps [{phase_label}]")

            results.append({
                'iter': global_iter,
                'epoch': epoch,
                'payload_mb': PAYLOAD_MB,
                'c_i': c_i,
                'sleep_us': SLEEP_US,
                'comm_dur_s': round(comm_dur, 6),
                'bw_gbps': round(bw_gbps, 4),
                'phase': phase_label,
            })

        # End of epoch — scheduler updates priority
        epoch_end(epoch, data_size=bytes_per_iter)

        # Per-epoch aggregated stats (with π, priority, slowdown)
        if rank == 0:
            avg_comm = sum(epoch_comm_times) / len(epoch_comm_times)
            avg_bw = sum(epoch_bws) / len(epoch_bws)
            # Slowdown = actual_comm / (c_i × T_target_per_iter)
            # T_target_per_iter = T_target_epoch / ITERS_PER_EPOCH
            if _scheduler.target_comm_time_s is not None and _scheduler.target_comm_time_s > 0:
                ttarget_per_iter = _scheduler.target_comm_time_s / ITERS_PER_EPOCH
                slowdown = avg_comm / (c_i * ttarget_per_iter)
            else:
                slowdown = float('nan')
            pi_val = (_scheduler.last_pi if _scheduler.last_pi == _scheduler.last_pi
                      else 'nan')
            epoch_stats.append({
                'epoch': epoch,
                'phase': phase_label,
                'payload_mb': PAYLOAD_MB,
                'c_i': c_i,
                'sleep_us': SLEEP_US,
                'avg_comm_s': round(avg_comm, 6),
                'avg_bw_gbps': round(avg_bw, 4),
                'pi': round(pi_val, 6) if pi_val == pi_val else 'nan',
                'priority': _scheduler.current_priority,
                'dscp': PRIORITY_TO_DSCP.get(_scheduler.current_priority, 0),
                'slowdown': round(slowdown, 4) if slowdown == slowdown else 'nan',
                't_target_ms': round(_scheduler.target_comm_time_s * 1000, 3)
                               if _scheduler.target_comm_time_s else 'nan',
            })

    t_total_end = time.perf_counter()

    # ============================================================
    # Save results
    # ============================================================
    csv_iter_path = f'p4_job{JOB}_reverse_{MODE}_rank{rank}_iter.csv'
    with open(csv_iter_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'epoch', 'payload_mb', 'c_i', 'sleep_us',
                         'comm_dur_s', 'bw_gbps', 'phase'])
        for r in results:
            writer.writerow([r['iter'], r['epoch'], r['payload_mb'], r['c_i'],
                             r['sleep_us'], r['comm_dur_s'], r['bw_gbps'], r['phase']])
    print(f"[Job{JOB}-{MODE.upper()}-MAIN] Per-iter results saved to {csv_iter_path}")

    if rank == 0:
        csv_epoch_path = f'p4_job{JOB}_reverse_{MODE}_rank0_epoch.csv'
        with open(csv_epoch_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'phase', 'payload_mb', 'c_i', 'sleep_us',
                             'avg_comm_s', 'avg_bw_gbps', 'pi', 'priority',
                             'dscp', 'slowdown', 't_target_ms'])
            for r in epoch_stats:
                writer.writerow([r['epoch'], r['phase'], r['payload_mb'], r['c_i'],
                                 r['sleep_us'], r['avg_comm_s'], r['avg_bw_gbps'],
                                 r['pi'], r['priority'], r['dscp'], r['slowdown'],
                                 r['t_target_ms']])
        print(f"[Job{JOB}-{MODE.upper()}-MAIN] Per-epoch stats saved to {csv_epoch_path}")

        # Summary
        p1 = [r for r in epoch_stats if r['phase'] == 'phase1']
        p2 = [r for r in epoch_stats if r['phase'] == 'phase2']
        if p1:
            print(f"[Job{JOB}-{MODE.upper()}-MAIN] Phase 1 (epochs 0-{REVERSE_EPOCH-1}): "
                  f"c_i={C_I_PHASE1}, "
                  f"avg_comm={sum(r['avg_comm_s'] for r in p1)/len(p1)*1000:.1f}ms, "
                  f"avg_bw={sum(r['avg_bw_gbps'] for r in p1)/len(p1):.2f} Gbps, "
                  f"avg_prio={sum(r['priority'] for r in p1)/len(p1):.1f}")
        if p2:
            print(f"[Job{JOB}-{MODE.upper()}-MAIN] Phase 2 (epochs {REVERSE_EPOCH}-{NUM_EPOCHS-1}): "
                  f"c_i={C_I_PHASE2}, "
                  f"avg_comm={sum(r['avg_comm_s'] for r in p2)/len(p2)*1000:.1f}ms, "
                  f"avg_bw={sum(r['avg_bw_gbps'] for r in p2)/len(p2):.2f} Gbps, "
                  f"avg_prio={sum(r['priority'] for r in p2)/len(p2):.1f}")
        print(f"[Job{JOB}-{MODE.upper()}-MAIN] Total wall time: "
              f"{t_total_end - t_total_start:.1f}s")

    _mc_wrapper.destroy()


def main():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    device_idx = 0
    torch.cuda.set_device(device_idx)
    device = torch.cuda.current_device()

    if PHASE == 'calibrate':
        run_calibration(rank, world_size, device)
    else:  # main
        run_main(rank, world_size, device)


if __name__ == '__main__':
    main()
