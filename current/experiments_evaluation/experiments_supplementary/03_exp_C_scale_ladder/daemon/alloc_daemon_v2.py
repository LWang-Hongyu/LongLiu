#!/usr/bin/env python3
"""
Experiment C v2 — Allocation Daemon (with c_policy/c_eval split)
================================================================
Reads per-epoch stats from emulator processes, runs SLOScheduler (same code
as the NCCL shim), and writes DSCP updates to control files.

v2 changes from v1:
  - c_policy/c_eval split: scheduler uses c_policy for π; c_eval is stored
    for the analysis script to compute attainment.
  - Static arm: premium→P6 (DSCP=8), standard→P2 (DSCP=24). This replaces
    the old "all P4" static arm and properly models CRUX-like tier mapping.
  - Fair arm: all jobs at P4 (DSCP=0), no differentiation.
  - Supports scenarios_v2.json format with scenario/regime/d_scale.
  - Per-job tier is read from scenario config (premium/standard).

Architecture:
  - Emulator processes (epoch_emulator) write stats to /tmp/expC_stats_<job_id>.csv
  - This daemon polls those files for new epoch completions
  - For each completed epoch: SLOScheduler.update(avg_comm_s, data_size) → new priority
  - Writes new DSCP to /tmp/expC_dscp_<job_id> (emulator reads at next iter)

Usage:
  python3 alloc_daemon_v2.py --config scenarios/scenarios_v2.json \
      --scenario S1 --regime S1_moderate --arm longliu

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

STATS_FILE_FMT = "/tmp/expC_stats_{}.csv"
DSCP_FILE_FMT = "/tmp/expC_dscp_{}"
DAEMON_LOG_FMT = "/tmp/expC_daemon_{}.log"


class JobScheduler:
    """Wraps SLOScheduler for one emulator job."""
    def __init__(self, job_id: int, tier: str, c_policy: float, c_eval: float,
                 t_target_ms: float, data_size_bytes: int,
                 mode: str = "longliu", initial_priority: int = 4,
                 static_premium_priority: int = 6, static_standard_priority: int = 2):
        self.job_id = job_id
        self.tier = tier
        self.c_policy = c_policy  # Used by scheduler for π computation
        self.c_eval = c_eval      # Used by analysis for attainment
        self.data_size = data_size_bytes
        self.mode = mode  # "longliu", "static", or "fair"
        self.last_epoch_seen = -1
        self.last_dscp = -1

        if mode == "longliu":
            self.scheduler = SLOScheduler(
                slo_threshold=c_policy,  # v2: c_policy for π computation
                target_comm_time_ms=t_target_ms,
                preset_target=True,  # T_target from calibration, skip EMA
                initial_priority=initial_priority,
            )
        elif mode == "static":
            # v2 static arm: premium→P6, standard→P2
            if tier == "premium":
                self.static_priority = static_premium_priority
            else:
                self.static_priority = static_standard_priority
            self.scheduler = None
            self.current_priority = self.static_priority
        else:  # fair
            self.static_priority = 4  # All jobs at P4
            self.scheduler = None
            self.current_priority = 4

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
            self.scheduler.update(avg_comm_s, float(self.data_size))
        # For static/fair mode, priority never changes
        return self.get_dscp()


def read_new_epochs(stats_path: str, last_seen: int) -> list:
    """Read epoch records newer than last_seen from stats CSV.
    Returns list of (epoch, comm_us, data_bytes, dscp, sleep_us)."""
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
                comm_us = float(row["comm_us"])
                data_bytes = int(row["data_bytes"])
                dscp = int(row["dscp"])
                sleep_us = int(row.get("sleep_us", 0))
                recs.append((ep, comm_us, data_bytes, dscp, sleep_us))
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
    avg_sleep_us = sum(r[4] for r in epoch_recs) / len(epoch_recs)
    return avg_comm_us / 1e6, data_bytes  # convert to seconds


def write_dscp_file(job_id: int, dscp: int):
    """Write DSCP value to control file."""
    path = DSCP_FILE_FMT.format(job_id)
    with open(path, "w") as f:
        f.write(f"{dscp}\n")


def resolve_jobs(config: dict, scenario_name: str, regime_name: str) -> list:
    """Resolve job list from v2 config, applying d_scale to payloads.
    
    v2.1: Uses payload_kb from scenario (not d_per_epoch_mb which was removed).
    Sleep_us and actual payload_bytes are read from calibration files at init time.
    """
    scenario = config["scenarios"][scenario_name]
    regime = scenario["regimes"][regime_name]
    d_scale = regime.get("d_scale", 1.0)
    iters_per_epoch = config["experiment_params"]["iters_per_epoch"]

    jobs = []
    for bj in scenario["base_jobs"]:
        # v2.1: payload_kb * d_scale, then convert to per-iter bytes
        payload_kb_scaled = int(bj["payload_kb"] * d_scale)
        d_per_iter_bytes = payload_kb_scaled * 1024
        # Sleep_us will be overridden from calibration files
        sleep_us = bj.get("sleep_us", 5000)
        jobs.append({
            "job_id": bj["job_id"],
            "label": bj["label"],
            "tier": bj["tier"],
            "c_policy": bj["c_policy"],
            "c_eval": bj["c_eval"],
            "payload_bytes": d_per_iter_bytes,
            "sleep_us": sleep_us,
            "phi_target": bj["phi_target"],
        })
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Experiment C v2 allocation daemon")
    ap.add_argument("--config", required=True, help="Path to scenarios_v2.json")
    ap.add_argument("--scenario", required=True, help="Scenario name (S1 or S2)")
    ap.add_argument("--regime", required=True,
                    help="Regime name (e.g., S1_moderate, S2_starvation)")
    ap.add_argument("--arm", default="longliu", choices=["longliu", "static", "fair"],
                    help="Scheduling arm")
    ap.add_argument("--runtime-s", type=int, default=900,
                    help="Max runtime in seconds (default 900)")
    ap.add_argument("--poll-interval-s", type=float, default=0.1,
                    help="Poll interval for stats files (default 0.1s)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Resolve jobs with d_scale applied
    jobs_config = resolve_jobs(config, args.scenario, args.regime)
    link_bw_gbps = config.get("link_bw_gbps", 50.0)

    # Read arm configuration
    arm_cfg = config.get("arms", {}).get(args.arm, {})
    static_premium_prio = arm_cfg.get("premium_priority", 6)
    static_standard_prio = arm_cfg.get("standard_priority", 2)
    initial_priority = arm_cfg.get("initial_priority", 4)

    print(f"[daemon] Scenario: {args.scenario} | Regime: {args.regime}")
    print(f"[daemon] Arm: {args.arm} | Jobs: {len(jobs_config)}")
    print(f"[daemon] Link BW: {link_bw_gbps} Gbps")

    # Initialize per-job schedulers
    schedulers = {}
    for jc in jobs_config:
        jid = jc["job_id"]
        c_policy = jc["c_policy"]
        c_eval = jc["c_eval"]
        tier = jc["tier"]
        data_size = jc["payload_bytes"]

        # Read T_target and payload_bytes from calibration file, fall back to estimate
        ttarget_file = f"/tmp/expC_ttarget_{jid}.json"
        if os.path.exists(ttarget_file):
            td = json.load(open(ttarget_file))
            t_target_ms = td["target_comm_time_ms"]
            # Override data_size from calibration (already includes d_scale)
            if "payload_bytes" in td:
                data_size = td["payload_bytes"]
        else:
            # Estimate from theoretical BW = 50Gbps
            tcomm_solo_s = data_size * 8 / (50e9)
            t_target_ms = tcomm_solo_s * 1000
            print(f"  [WARN] Job {jid}: no calibration file, using estimate {t_target_ms:.3f}ms")

        schedulers[jid] = JobScheduler(
            job_id=jid, tier=tier,
            c_policy=c_policy, c_eval=c_eval,
            t_target_ms=t_target_ms,
            data_size_bytes=data_size,
            initial_priority=initial_priority,
            mode=args.arm,
            static_premium_priority=static_premium_prio,
            static_standard_priority=static_standard_prio,
        )
        # Write initial DSCP
        dscp = schedulers[jid].get_dscp()
        write_dscp_file(jid, dscp)
        print(f"  Job {jid} ({tier}): c_policy={c_policy}, c_eval={c_eval}, "
              f"T_target={t_target_ms:.3f}ms, data={data_size}B, "
              f"init_prio=P{schedulers[jid].get_priority()}, DSCP={dscp}")

    # Open daemon log (includes c_eval for analysis)
    log_path = DAEMON_LOG_FMT.format(args.regime)
    log_f = open(log_path, "w")
    log_f.write("timestamp,job_id,epoch,avg_comm_s,pi,priority,dscp,changed,"
                "c_policy,c_eval,tier\n")
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

                # Log (v2: includes c_policy, c_eval, tier)
                ts = time.time() - start_time
                pi = js.get_pi()
                prio = js.get_priority()
                log_f.write(f"{ts:.1f},{jid},{target_ep},{avg_comm_s:.6f},"
                            f"{pi:.4f},{prio},{new_dscp},{'Y' if changed else 'N'},"
                            f"{js.c_policy},{js.c_eval},{js.tier}\n")
                log_f.flush()

                if changed:
                    print(f"[daemon] Job {jid} epoch {target_ep}: "
                          f"π={pi:+.3f} P{prio} DSCP={old_dscp}→{new_dscp} "
                          f"(comm={avg_comm_s*1000:.1f}ms, {js.tier})")

                epoch_counts[jid] = target_ep

        if all_done:
            # Check if all emulators have finished
            max_epochs = config["experiment_params"]["num_epochs"]
            if all(ec >= max_epochs - 1 for ec in epoch_counts.values()):
                print(f"[daemon] All jobs reached epoch {max_epochs - 1}, done.")
                break

        time.sleep(args.poll_interval_s)

    log_f.close()
    print(f"[daemon] Runtime complete. Log: {log_path}")


if __name__ == "__main__":
    main()
