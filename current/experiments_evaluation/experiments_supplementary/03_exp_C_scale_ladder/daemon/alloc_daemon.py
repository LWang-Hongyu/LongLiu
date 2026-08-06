#!/usr/bin/env python3
"""
Experiment C — Allocation Daemon (standalone shim logic extraction)
====================================================================
Reads per-epoch stats from emulator processes, runs SLOScheduler (same code
as the NCCL shim), and writes DSCP updates to control files.

Architecture:
  - Emulator processes (epoch_emulator) write stats to /tmp/expC_stats_<job_id>.csv
  - This daemon polls those files for new epoch completions
  - For each completed epoch: SLOScheduler.update(avg_comm_s, data_size) → new priority
  - Writes new DSCP to /tmp/expC_dscp_<job_id> (emulator reads at next iter)

Usage:
  python3 alloc_daemon.py --config scenarios/scenarios.json --regime deep_scarcity

The daemon runs on the CLIENT node (10.1) only — it controls all emulators.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Import SLOScheduler from the existing shim code (same logic = same paper claims)
SLO_SRC = "/home/why/LongLiu_rebuild/multi_comm_slo/src"
sys.path.insert(0, SLO_SRC)
from slo_scheduler import SLOScheduler  # noqa: E402

# DSCP mapping from SLOScheduler (paper §5.3)
# P6→DSCP=8, P4→DSCP=0, P3→DSCP=16, P2→DSCP=24, P1→DSCP=32
# We use DSCP values directly (emulator shifts << 2 for TOS)

STATS_FILE_FMT = "/tmp/expC_stats_{}.csv"
DSCP_FILE_FMT = "/tmp/expC_dscp_{}"
DAEMON_LOG_FMT = "/tmp/expC_daemon_{}.log"


class JobScheduler:
    """Wraps SLOScheduler for one emulator job."""
    def __init__(self, job_id: int, c_i: float, t_target_ms: float,
                 data_size_bytes: int, initial_priority: int = 4,
                 mode: str = "longliu", static_priority: int = 4):
        self.job_id = job_id
        self.c_i = c_i
        self.data_size = data_size_bytes
        self.mode = mode  # "longliu" or "static"
        self.static_priority = static_priority
        self.last_epoch_seen = -1
        self.last_dscp = -1

        if mode == "longliu":
            self.scheduler = SLOScheduler(
                slo_threshold=c_i,
                target_comm_time_ms=t_target_ms,
                preset_target=True,  # T_target from calibration, skip EMA
                initial_priority=initial_priority,
            )
        else:
            # Static mode: no scheduler, fixed priority throughout
            self.scheduler = None
            self.current_priority = static_priority

    def get_dscp(self) -> int:
        if self.mode == "longliu":
            return self.scheduler.get_dscp()
        else:
            return SLOScheduler.PRIORITY_TO_DSCP.get(self.static_priority,
                                                      self.static_priority * 8)

    def get_priority(self) -> int:
        if self.mode == "longliu":
            return self.scheduler.current_priority
        else:
            return self.static_priority

    def get_pi(self) -> float:
        if self.mode == "longliu":
            return self.scheduler.last_pi
        return 0.0

    def on_epoch_complete(self, avg_comm_s: float, epoch: int) -> int:
        """Called when a new epoch completes. Returns new DSCP."""
        if self.mode == "longliu":
            new_prio = self.scheduler.update(avg_comm_s, float(self.data_size))
        # For static mode, priority never changes
        new_dscp = self.get_dscp()
        return new_dscp


def read_new_epochs(stats_path: str, last_seen: int) -> list:
    """Read epoch records newer than last_seen from stats CSV.
    Returns list of (epoch, avg_comm_s, data_bytes, dscp)."""
    if not os.path.exists(stats_path):
        return []
    recs = []
    with open(stats_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["epoch"])
                if ep <= last_seen:
                    continue
                # Aggregate per-epoch: collect all iters in this epoch
                comm_us = float(row["comm_us"])
                recs.append((ep, comm_us, int(row["data_bytes"]), int(row["dscp"])))
            except (ValueError, KeyError):
                continue
    return recs


def compute_epoch_avg(recs: list, target_epoch: int) -> tuple:
    """Compute avg comm time (seconds) for a given epoch from iter records."""
    epoch_recs = [r for r in recs if r[0] == target_epoch]
    if not epoch_recs:
        return None, None
    avg_comm_us = sum(r[1] for r in epoch_recs) / len(epoch_recs)
    data_bytes = epoch_recs[0][2]
    return avg_comm_us / 1e6, data_bytes  # convert to seconds


def write_dscp_file(job_id: int, dscp: int):
    """Write DSCP value to control file."""
    path = DSCP_FILE_FMT.format(job_id)
    with open(path, "w") as f:
        f.write(f"{dscp}\n")


def main():
    ap = argparse.ArgumentParser(description="Experiment C allocation daemon")
    ap.add_argument("--config", required=True, help="Path to scenarios.json")
    ap.add_argument("--regime", required=True,
                    help="Regime name (e.g., deep_scarcity, transition, ample)")
    ap.add_argument("--arm", default="longliu", choices=["longliu", "static", "fair"],
                    help="Scheduling arm")
    ap.add_argument("--static-priority", type=int, default=4,
                    help="Priority for static arm (default P4)")
    ap.add_argument("--runtime-s", type=int, default=600,
                    help="Max runtime in seconds (default 600)")
    ap.add_argument("--poll-interval-s", type=float, default=0.5,
                    help="Poll interval for stats files")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    regime = config.get("regimes", {}).get(args.regime)
    if not regime:
        print(f"[ERR] regime '{args.regime}' not found in config")
        sys.exit(1)

    jobs_config = regime["jobs"]
    link_bw_gbps = config.get("link_bw_gbps", 50.0)

    print(f"[daemon] Regime: {args.regime} ({len(jobs_config)} jobs)")
    print(f"[daemon] Arm: {args.arm}")
    print(f"[daemon] Link BW: {link_bw_gbps} Gbps")

    # Initialize per-job schedulers
    schedulers = {}
    for jc in jobs_config:
        jid = jc["job_id"]
        c_i = jc["c_i"]
        # Read T_target from calibration file, fall back to scenario estimate
        ttarget_file = f"/tmp/expC_ttarget_{jid}.json"
        if os.path.exists(ttarget_file):
            td = json.load(open(ttarget_file))
            t_target_ms = td["target_comm_time_ms"]
        else:
            t_target_ms = jc.get("t_target_ms_est", jc.get("t_target_ms", 1.0))
        data_size = jc.get("payload_kb", 1024) * 1024  # payload_kb → bytes
        init_prio = jc.get("initial_priority", 4)

        schedulers[jid] = JobScheduler(
            job_id=jid, c_i=c_i, t_target_ms=t_target_ms,
            data_size_bytes=data_size,
            initial_priority=init_prio,
            mode=args.arm,
            static_priority=args.static_priority,
        )
        # Write initial DSCP
        dscp = schedulers[jid].get_dscp()
        write_dscp_file(jid, dscp)
        print(f"  Job {jid}: c_i={c_i}, T_target={t_target_ms}ms, "
              f"data={data_size}B, init_prio=P{init_prio}, DSCP={dscp}")

    # Open daemon log
    log_path = DAEMON_LOG_FMT.format(args.regime)
    log_f = open(log_path, "w")
    log_f.write("timestamp,job_id,epoch,avg_comm_s,pi,priority,dscp,changed\n")
    log_f.flush()

    # Main polling loop
    start_time = time.time()
    epoch_counts = {jid: -1 for jid in schedulers}

    print(f"[daemon] Polling started (interval={args.poll_interval_s}s)")

    while time.time() - start_time < args.runtime_s:
        all_done = True
        for jid, js in schedulers.items():
            stats_path = STATS_FILE_FMT.format(jid)
            new_recs = read_new_epochs(stats_path, epoch_counts[jid])
            if not new_recs:
                # Check if emulator is still running
                if not os.path.exists(stats_path):
                    all_done = False
                continue

            # Find the latest completed epoch
            max_epoch = max(r[0] for r in new_recs)
            if max_epoch <= epoch_counts[jid]:
                continue

            all_done = False

            # Process each new epoch
            for target_ep in range(epoch_counts[jid] + 1, max_epoch + 1):
                avg_comm_s, data_bytes = compute_epoch_avg(new_recs, target_ep)
                if avg_comm_s is None:
                    continue

                old_dscp = js.get_dscp()
                new_dscp = js.on_epoch_complete(avg_comm_s, target_ep)
                changed = new_dscp != old_dscp

                if changed:
                    write_dscp_file(jid, new_dscp)

                # Log
                ts = time.time() - start_time
                pi = js.get_pi()
                prio = js.get_priority()
                log_f.write(f"{ts:.1f},{jid},{target_ep},{avg_comm_s:.6f},"
                            f"{pi:.4f},{prio},{new_dscp},{'Y' if changed else 'N'}\n")
                log_f.flush()

                if changed:
                    print(f"[daemon] Job {jid} epoch {target_ep}: "
                          f"π={pi:+.3f} P{prio} DSCP={old_dscp}→{new_dscp} "
                          f"(comm={avg_comm_s*1000:.1f}ms)")

                epoch_counts[jid] = target_ep

        if all_done:
            # Check if all emulators have finished
            max_epochs = max(jc.get("num_epochs", 25) for jc in jobs_config)
            if all(ec >= max_epochs - 1 for ec in epoch_counts.values()):
                print(f"[daemon] All jobs reached epoch {max_epochs - 1}, done.")
                break

        time.sleep(args.poll_interval_s)

    log_f.close()
    print(f"[daemon] Runtime complete. Log: {log_path}")


if __name__ == "__main__":
    main()
