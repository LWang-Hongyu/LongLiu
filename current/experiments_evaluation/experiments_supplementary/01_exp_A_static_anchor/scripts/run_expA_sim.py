#!/usr/bin/env python3
"""
Experiment A — Simulator Arm
============================
Mirrors each HW scenario with a bit-for-bit matching analytical simulation that
uses the LongLiu allocation logic (π → priority → SPQ bandwidth sharing).

Design:
  - Anchor T_target loaded from HW solo calibration (or theoretical for hold-out S5)
  - Per epoch: each job computes π = A / (c × T × k) - 1
  - π → priority mapping (4-tier, per project_memory):
        π > 0.3  → P6
       -0.1 < π ≤ 0.3 → P4
       -0.5 < π ≤ -0.1 → P2
        π ≤ -0.5 → P1
  - Bandwidth sharing (SPQ + equal share within same priority):
        - Jobs at the highest active priority split link BW equally
        - Lower-priority jobs get 0 (strict priority; mirrors SP queue semantics)
  - Background flow: reduces effective link capacity by bg_rate_gbps
  - Per-job comm time = payload / allocated_BW
  - slowdown = comm_time / (c_i × T_target_per_iter)

Outputs per-scenario CSV matching p4_job_reverse.py format:
  epoch, c_i, avg_comm_s, avg_bw_gbps, pi, priority, dscp, slowdown, t_target_ms
"""
import json
import os
import sys
import argparse
import csv
from pathlib import Path

# ---- Constants (must match expA_config.sh) ----
LINK_BW_GBPS = 50.0
ITERS_PER_EPOCH = 20
NUM_EPOCHS = 25
SLEEP_US = 30000  # not used in sim (comm-only), kept for manifest

# π → priority (per project_memory §V6)
def pi_to_priority(pi: float) -> int:
    if pi > 0.3:
        return 6
    elif pi > -0.1:
        return 4
    elif pi > -0.5:
        return 2
    else:
        return 1

PRIORITY_TO_DSCP = {6: 48, 4: 32, 2: 16, 1: 8}

# ---- LongLiu SLO state (mirrors slo_scheduler.py) ----
class SimSLOState:
    def __init__(self, c_i: float, t_target_per_iter_ms: float):
        self.c_i = c_i
        self.t_target_per_iter_ms = t_target_per_iter_ms  # solo ideal per-iter
        self.comm_cycle_count = 0
        self.accum_time_ms = 0.0

    def compute_pi(self) -> float:
        if self.comm_cycle_count == 0:
            return 0.0
        expected = self.c_i * self.t_target_per_iter_ms * self.comm_cycle_count
        if expected <= 0:
            return 0.0
        return (self.accum_time_ms / expected) - 1.0

    def priority(self) -> int:
        return pi_to_priority(self.compute_pi())

    def record_comm(self, comm_time_ms: float):
        self.comm_cycle_count += 1
        self.accum_time_ms += comm_time_ms


def load_ttarget(path: str) -> dict:
    """Load T_target JSON, return dict with target_comm_time_ms, payload_mb, etc."""
    if path == "THEORETICAL":
        return None  # caller handles
    with open(path) as f:
        d = json.load(f)
    assert d.get("unit") == "per_epoch_ms", f"unit mismatch in {path}: {d.get('unit')}"
    assert d["target_comm_time_ms"] > 0
    return d


def theoretical_ttarget(payload_mb: int, link_bw_gbps: float = LINK_BW_GBPS) -> dict:
    """Compute theoretical T_target (hold-out anchor).
    Assumes perfect link utilization: T = payload_bits / (link_bw * 1e9).
    This is the IDEAL, not achievable in practice due to NCCL overhead.
    """
    payload_bits = payload_mb * 1024 * 1024 * 8
    per_iter_s = payload_bits / (link_bw_gbps * 1e9)
    per_iter_ms = per_iter_s * 1000
    return {
        "payload_mb": payload_mb,
        "target_comm_time_ms": per_iter_ms * ITERS_PER_EPOCH,
        "unit": "per_epoch_ms",
        "source": "theoretical",
    }


def solo_bw_from_ttarget(d: dict) -> float:
    """Compute solo BW (Gbps) from T_target. Matches run_solo_bw.sh formula."""
    if d is None:
        return LINK_BW_GBPS
    pay = d["payload_mb"]
    t_epoch_ms = d["target_comm_time_ms"]
    # BW = payload_bits_per_iter / per_iter_sec / 1e9, with 0.5 factor (one-way)
    per_iter_s = (t_epoch_ms / ITERS_PER_EPOCH) / 1000.0
    payload_bits = pay * 1024 * 1024 * 8
    return (payload_bits * 0.5) / per_iter_s / 1e9


def run_scenario(scen: dict, out_dir: Path):
    """Run simulation for one scenario, write per-job per-epoch CSV."""
    sid = scen["id"]
    label = scen["label"]
    payload_mb = scen["payload_mb"]
    c_i = scen["c_i"]
    bg_flow = scen.get("bg_flow", False)
    bg_rate = scen.get("bg_rate_gbps", 0.0) if bg_flow else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load anchors
    if scen["ttarget_a"] == "THEORETICAL":
        ttarget_a = theoretical_ttarget(payload_mb)
        ttarget_b = theoretical_ttarget(payload_mb)
        anchor_src = "theoretical"
    else:
        ttarget_a = load_ttarget(scen["ttarget_a"])
        ttarget_b = load_ttarget(scen["ttarget_b"])
        anchor_src = "measured"

    # Solo BW per job (may differ slightly due to A/B asymmetry)
    solo_bw_a = solo_bw_from_ttarget(ttarget_a)
    solo_bw_b = solo_bw_from_ttarget(ttarget_b)
    # For payload mismatch (S6 uses 768MB but V5 file is 1024MB? No — S6 uses 768MB files)
    # Sanity: T_target payload must match scenario payload
    for tt, who in [(ttarget_a, "A"), (ttarget_b, "B")]:
        if tt["payload_mb"] != payload_mb and anchor_src == "measured":
            print(f"[WARN] {sid}: ttarget_{who} payload={tt['payload_mb']}MB != scenario payload={payload_mb}MB")

    t_target_per_iter_a_ms = ttarget_a["target_comm_time_ms"] / ITERS_PER_EPOCH
    t_target_per_iter_b_ms = ttarget_b["target_comm_time_ms"] / ITERS_PER_EPOCH

    # Effective link capacity
    eff_link_bw = LINK_BW_GBPS - bg_rate

    # SLO states for 2 jobs (both same c_i in static scenarios)
    state_a = SimSLOState(c_i, t_target_per_iter_a_ms)
    state_b = SimSLOState(c_i, t_target_per_iter_b_ms)

    # Per-epoch records
    records_a = []
    records_b = []

    payload_bits = payload_mb * 1024 * 1024 * 8

    for epoch in range(NUM_EPOCHS):
        # Per-epoch: simulate ITERS_PER_EPOCH iterations
        epoch_comm_a_ms = 0.0
        epoch_comm_b_ms = 0.0
        last_prio_a = state_a.priority()
        last_prio_b = state_b.priority()

        for it in range(ITERS_PER_EPOCH):
            # Current priorities
            pa = state_a.priority()
            pb = state_b.priority()

            # SPQ bandwidth allocation:
            #   - Both same priority → equal share of eff_link_bw
            #   - Higher priority gets full BW, lower gets 0
            if pa == pb:
                bw_a = eff_link_bw / 2.0
                bw_b = eff_link_bw / 2.0
            elif pa > pb:
                bw_a = eff_link_bw
                bw_b = 0.0  # strictly preempted
            else:
                bw_a = 0.0
                bw_b = eff_link_bw

            # Comm time = payload_bits / allocated_bw
            # Use per-job solo BW as the "max achievable" cap
            comm_a_ms = (payload_bits / max(bw_a, 1e-6) / 1e9) * 1000 if bw_a > 0 else 1e9
            comm_b_ms = (payload_bits / max(bw_b, 1e-6) / 1e9) * 1000 if bw_b > 0 else 1e9

            # If preempted (bw=0), the job waits — in reality NCCL doesn't truly stall;
            # model: preempted job's iter comm time = payload / solo_bw (no progress)
            # but accumulates as if it ran (deficit grows). For SPQ, this is the
            # "starvation" regime. In our static symmetric case, pa==pb always, so
            # this branch is only hit transiently.
            if bw_a == 0:
                comm_a_ms = (payload_bits / solo_bw_a / 1e9) * 1000  # waits one iter
            if bw_b == 0:
                comm_b_ms = (payload_bits / solo_bw_b / 1e9) * 1000

            state_a.record_comm(comm_a_ms)
            state_b.record_comm(comm_b_ms)
            epoch_comm_a_ms += comm_a_ms
            epoch_comm_b_ms += comm_b_ms

        # per-iter averages (ms and s)
        avg_comm_a_ms = epoch_comm_a_ms / ITERS_PER_EPOCH
        avg_comm_b_ms = epoch_comm_b_ms / ITERS_PER_EPOCH
        avg_comm_a_s = avg_comm_a_ms / 1000.0
        avg_comm_b_s = avg_comm_b_ms / 1000.0
        avg_bw_a = (payload_bits * 0.5) / (avg_comm_a_s) / 1e9 if avg_comm_a_s > 0 else 0
        avg_bw_b = (payload_bits * 0.5) / (avg_comm_b_s) / 1e9 if avg_comm_b_s > 0 else 0

        pi_a = state_a.compute_pi()
        pi_b = state_b.compute_pi()
        prio_a = state_a.priority()
        prio_b = state_b.priority()
        dscp_a = PRIORITY_TO_DSCP.get(prio_a, 0)
        dscp_b = PRIORITY_TO_DSCP.get(prio_b, 0)

        slowdown_a = avg_comm_a_ms / (c_i * t_target_per_iter_a_ms) if t_target_per_iter_a_ms > 0 else 0
        slowdown_b = avg_comm_b_ms / (c_i * t_target_per_iter_b_ms) if t_target_per_iter_b_ms > 0 else 0

        records_a.append({
            "epoch": epoch, "c_i": c_i, "avg_comm_s": round(avg_comm_a_s, 6),
            "avg_bw_gbps": round(avg_bw_a, 3), "pi": round(pi_a, 4),
            "priority": prio_a, "dscp": dscp_a,
            "slowdown": round(slowdown_a, 4),
            "t_target_ms": round(ttarget_a["target_comm_time_ms"], 2),
        })
        records_b.append({
            "epoch": epoch, "c_i": c_i, "avg_comm_s": round(avg_comm_b_s, 6),
            "avg_bw_gbps": round(avg_bw_b, 3), "pi": round(pi_b, 4),
            "priority": prio_b, "dscp": dscp_b,
            "slowdown": round(slowdown_b, 4),
            "t_target_ms": round(ttarget_b["target_comm_time_ms"], 2),
        })

    # Write CSVs (match p4_job_reverse.py epoch CSV format)
    csv_header = ["epoch", "c_i", "avg_comm_s", "avg_bw_gbps", "pi",
                  "priority", "dscp", "slowdown", "t_target_ms"]
    for job, recs in [("A", records_a), ("B", records_b)]:
        csv_path = out_dir / f"job{job}_sim_epoch.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            for r in recs:
                w.writerow([r[c] for c in csv_header])

    # Manifest
    manifest = {
        "scenario_id": sid, "label": label,
        "payload_mb": payload_mb, "c_i": c_i,
        "bg_flow": bg_flow, "bg_rate_gbps": bg_rate,
        "link_bw_gbps": LINK_BW_GBPS, "eff_link_bw_gbps": eff_link_bw,
        "solo_bw_a_gbps": round(solo_bw_a, 2),
        "solo_bw_b_gbps": round(solo_bw_b, 2),
        "t_target_a_epoch_ms": round(ttarget_a["target_comm_time_ms"], 2),
        "t_target_b_epoch_ms": round(ttarget_b["target_comm_time_ms"], 2),
        "anchor_source": anchor_src,
        "holdout": scen.get("holdout", False),
        "num_epochs": NUM_EPOCHS, "iters_per_epoch": ITERS_PER_EPOCH,
    }
    with open(out_dir / "sim_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    import numpy as np
    sd_a = [r["slowdown"] for r in records_a[5:20]]  # skip warmup epochs 0-4
    sd_b = [r["slowdown"] for r in records_b[5:20]]
    print(f"[{sid}] {label}: slowdown A = {np.mean(sd_a):.4f} ± {np.std(sd_a):.4f}, "
          f"B = {np.mean(sd_b):.4f} ± {np.std(sd_b):.4f}  (epochs 5-19)")
    print(f"        solo_bw_a={solo_bw_a:.2f}Gbps, solo_bw_b={solo_bw_b:.2f}Gbps, "
          f"eff_link={eff_link_bw:.1f}Gbps, anchor={anchor_src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=None,
                    help="Scenario ID filter (e.g., S1) or ALL (default)")
    ap.add_argument("--scenarios-file",
                    default=str(Path(__file__).parent / "expA_scenarios.json"))
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: data/sim_<timestamp>)")
    args = ap.parse_args()

    with open(args.scenarios_file) as f:
        scen_def = json.load(f)

    out_root = Path(args.out_dir) if args.out_dir else \
        Path(__file__).resolve().parent.parent / "data" / f"sim_run_{os.getpid()}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[sim] output dir: {out_root}")

    filt = args.scenarios or "ALL"
    for scen in scen_def["scenarios"]:
        if filt != "ALL" and scen["id"] != filt:
            continue
        run_scenario(scen, out_root / f"{scen['id']}_{scen['label']}")

    # Record latest sim dir
    latest = Path(__file__).resolve().parent.parent / "data" / "latest_sim.txt"
    latest.write_text(str(out_root))
    print(f"[sim] all done. latest sim dir: {out_root}")


if __name__ == "__main__":
    main()
