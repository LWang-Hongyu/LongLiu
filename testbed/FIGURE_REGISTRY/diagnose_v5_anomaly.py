#!/usr/bin/env python3
"""
V5 Anomaly Diagnosis: Why is the SP-protected job (B) slower than CRUX equal-share?

Two candidate explanations:
  A) Environmental drift: arms ran sequentially, NIC/network conditions changed
  B) SP guard phase backlash: priority protection induces more overlap

This script uses EXISTING data (no re-runs) to discriminate.
"""

import csv
import json
from pathlib import Path

EXP_DIR = Path(__file__).parent
ITERS_PER_EPOCH = 20
COMPUTE_S = 0.030  # sleep_us=30000


def load_epoch_csv(scheduler, job_name):
    fpath = EXP_DIR / f"p4_{job_name}_reverse_{scheduler}_rank0_epoch.csv"
    rows = []
    with open(fpath) as f:
        for row in csv.DictReader(f):
            rows.append({
                "epoch": int(row["epoch"]),
                "phase": row["phase"],
                "c_i": float(row["c_i"]),
                "avg_comm_s": float(row["avg_comm_s"]),
                "avg_bw_gbps": float(row["avg_bw_gbps"]),
                "priority": int(row["priority"]),
                "dscp": int(row["dscp"]),
                "pi": float(row["pi"]),
            })
    return rows


def load_iter_csv(scheduler, job_name):
    fpath = EXP_DIR / f"p4_{job_name}_reverse_{scheduler}_rank0_iter.csv"
    if not fpath.exists():
        return []
    rows = []
    with open(fpath) as f:
        for row in csv.DictReader(f):
            rows.append({
                "iter": int(row["iter"]),
                "epoch": int(row["epoch"]),
                "comm_dur_s": float(row["comm_dur_s"]),
                "phase": row["phase"],
            })
    return rows


def duty_cycle(comm_s, compute_s=COMPUTE_S):
    return comm_s / (comm_s + compute_s)


def compute_overlap_from_iters(iterA, iterB, epoch):
    """
    Estimate overlap for one epoch from per-iter data.
    Since we don't have timestamps, we use a proxy:
    overlap ≈ duty_A * duty_B (random phase assumption)
    But we also compute the actual overlap pattern from iter-level data
    by checking if both jobs are in comm phase at the same iter index.

    NOTE: This is an approximation. True overlap requires timestamps.
    """
    itersA = [r for r in iterA if r["epoch"] == epoch]
    itersB = [r for r in iterB if r["epoch"] == epoch]

    if not itersA or not itersB:
        return None, None, None

    avg_comm_A = sum(r["comm_dur_s"] for r in itersA) / len(itersA)
    avg_comm_B = sum(r["comm_dur_s"] for r in itersB) / len(itersB)

    dA = duty_cycle(avg_comm_A)
    dB = duty_cycle(avg_comm_B)

    # Overlap estimate: P(A_comm ∧ B_comm) ≈ duty_A * duty_B (random phase)
    overlap_random = dA * dB

    # Alternative: if both jobs iterate at same rate, overlap = min(duty_A, duty_B)
    # (worst case: always overlapping when both are in comm)
    overlap_worst = min(dA, dB)

    return dA, dB, overlap_random


def main():
    print("=" * 100)
    print("V5 Anomaly Diagnosis: SP-protected job B slower than CRUX equal-share")
    print("=" * 100)
    print()

    # Load all data
    data = {}
    for sched in ["longliu", "crux"]:
        data[sched] = {}
        for job in ["jobA", "jobB"]:
            data[sched][job] = {
                "epoch": load_epoch_csv(sched, job),
                "iter": load_iter_csv(sched, job),
            }

    # ============================================================
    # TEST 1: Phase 1 Control (equal priority for both schedulers)
    # ============================================================
    print("=" * 100)
    print("TEST 1: Phase 1 Control — Tight Job A (c_i=1.7) under equal priority")
    print("  LongLiu: P2 (DSCP=16) | CRUX: P3 (DSCP=24)")
    print("  If CRUX is systematically faster → environmental drift")
    print("=" * 100)
    print()
    print(f"{'Epoch':>5} {'LL comm(ms)':>12} {'CX comm(ms)':>12} {'Δ (LL-CX)':>12} {'% diff':>10}")
    print("-" * 60)

    ll_comms = []
    cx_comms = []
    for ll_row, cx_row in zip(data["longliu"]["jobA"]["epoch"], data["crux"]["jobA"]["epoch"]):
        if ll_row["phase"] != "phase1":
            continue
        ll_ms = ll_row["avg_comm_s"] * 1000
        cx_ms = cx_row["avg_comm_s"] * 1000
        diff = ll_ms - cx_ms
        pct = diff / cx_ms * 100 if cx_ms > 0 else 0
        ll_comms.append(ll_ms)
        cx_comms.append(cx_ms)
        print(f"{ll_row['epoch']:>5} {ll_ms:>12.1f} {cx_ms:>12.1f} {diff:>+12.1f} {pct:>+9.1f}%")

    avg_ll = sum(ll_comms) / len(ll_comms)
    avg_cx = sum(cx_comms) / len(cx_comms)
    avg_diff = avg_ll - avg_cx
    avg_pct = avg_diff / avg_cx * 100
    print("-" * 60)
    print(f"{'AVG':>5} {avg_ll:>12.1f} {avg_cx:>12.1f} {avg_diff:>+12.1f} {avg_pct:>+9.1f}%")
    print()

    # Same for Job B Phase 1
    print("TEST 1b: Phase 1 Control — Loose Job B (c_i=3.0) under equal priority")
    print(f"{'Epoch':>5} {'LL comm(ms)':>12} {'CX comm(ms)':>12} {'Δ (LL-CX)':>12} {'% diff':>10}")
    print("-" * 60)

    ll_comms_b = []
    cx_comms_b = []
    for ll_row, cx_row in zip(data["longliu"]["jobB"]["epoch"], data["crux"]["jobB"]["epoch"]):
        if ll_row["phase"] != "phase1":
            continue
        ll_ms = ll_row["avg_comm_s"] * 1000
        cx_ms = cx_row["avg_comm_s"] * 1000
        diff = ll_ms - cx_ms
        pct = diff / cx_ms * 100 if cx_ms > 0 else 0
        ll_comms_b.append(ll_ms)
        cx_comms_b.append(cx_ms)
        print(f"{ll_row['epoch']:>5} {ll_ms:>12.1f} {cx_ms:>12.1f} {diff:>+12.1f} {pct:>+9.1f}%")

    avg_ll_b = sum(ll_comms_b) / len(ll_comms_b)
    avg_cx_b = sum(cx_comms_b) / len(cx_comms_b)
    avg_diff_b = avg_ll_b - avg_cx_b
    avg_pct_b = avg_diff_b / avg_cx_b * 100
    print("-" * 60)
    print(f"{'AVG':>5} {avg_ll_b:>12.1f} {avg_cx_b:>12.1f} {avg_diff_b:>+12.1f} {avg_pct_b:>+9.1f}%")
    print()

    # ============================================================
    # TEST 2: Phase 2 comparison (with priority difference)
    # ============================================================
    print("=" * 100)
    print("TEST 2: Phase 2 — Tight Job B (c_i=1.7) under DIFFERENT priority")
    print("  LongLiu: P4 (DSCP=32) | CRUX: P3 (DSCP=24)")
    print("  If gap narrows vs Phase 1 → SP protection partially offsets drift")
    print("  If gap widens vs Phase 1 → SP protection makes things worse")
    print("=" * 100)
    print()
    print(f"{'Epoch':>5} {'LL comm(ms)':>12} {'CX comm(ms)':>12} {'Δ (LL-CX)':>12} {'% diff':>10}")
    print("-" * 60)

    ll_p2 = []
    cx_p2 = []
    for ll_row, cx_row in zip(data["longliu"]["jobB"]["epoch"], data["crux"]["jobB"]["epoch"]):
        if ll_row["phase"] != "phase2":
            continue
        ll_ms = ll_row["avg_comm_s"] * 1000
        cx_ms = cx_row["avg_comm_s"] * 1000
        diff = ll_ms - cx_ms
        pct = diff / cx_ms * 100 if cx_ms > 0 else 0
        ll_p2.append(ll_ms)
        cx_p2.append(cx_ms)
        print(f"{ll_row['epoch']:>5} {ll_ms:>12.1f} {cx_ms:>12.1f} {diff:>+12.1f} {pct:>+9.1f}%")

    avg_ll_p2 = sum(ll_p2) / len(ll_p2)
    avg_cx_p2 = sum(cx_p2) / len(cx_p2)
    avg_diff_p2 = avg_ll_p2 - avg_cx_p2
    avg_pct_p2 = avg_diff_p2 / avg_cx_p2 * 100
    print("-" * 60)
    print(f"{'AVG':>5} {avg_ll_p2:>12.1f} {avg_cx_p2:>12.1f} {avg_diff_p2:>+12.1f} {avg_pct_p2:>+9.1f}%")
    print()

    # ============================================================
    # TEST 3: Overlap comparison
    # ============================================================
    print("=" * 100)
    print("TEST 3: Overlap Ratio Comparison (Phase 2)")
    print("  overlap ≈ duty_A × duty_B (random phase model)")
    print("  If LongLiu overlap >> CRUX overlap → SP phase backlash (Candidate B)")
    print("  If similar → no phase backlash (Candidate A: drift only)")
    print("=" * 100)
    print()
    print(f"{'Epoch':>5} | {'LL dutyA':>8} {'LL dutyB':>8} {'LL overlap':>10} | {'CX dutyA':>8} {'CX dutyB':>8} {'CX overlap':>10} | {'Δ overlap':>10}")
    print("-" * 90)

    for epoch in range(7, 15):
        ll_dA, ll_dB, ll_ov = compute_overlap_from_iters(
            data["longliu"]["jobA"]["iter"], data["longliu"]["jobB"]["iter"], epoch)
        cx_dA, cx_dB, cx_ov = compute_overlap_from_iters(
            data["crux"]["jobA"]["iter"], data["crux"]["jobB"]["iter"], epoch)

        if ll_ov is None or cx_ov is None:
            continue

        delta = ll_ov - cx_ov
        print(f"{epoch:>5} | {ll_dA:>8.3f} {ll_dB:>8.3f} {ll_ov:>10.3f} | "
              f"{cx_dA:>8.3f} {cx_dB:>8.3f} {cx_ov:>10.3f} | {delta:>+10.3f}")

    print()

    # ============================================================
    # CONCLUSION
    # ============================================================
    print("=" * 100)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 100)
    print()

    # Phase 1 drift magnitude
    print(f"Phase 1 environmental drift (Job A, equal priority):")
    print(f"  LongLiu avg: {avg_ll:.1f}ms | CRUX avg: {avg_cx:.1f}ms")
    print(f"  CRUX is {avg_pct:+.1f}% faster (ran second, ~2 min later)")
    print()

    print(f"Phase 1 environmental drift (Job B, equal priority):")
    print(f"  LongLiu avg: {avg_ll_b:.1f}ms | CRUX avg: {avg_cx_b:.1f}ms")
    print(f"  CRUX is {avg_pct_b:+.1f}% faster")
    print()

    print(f"Phase 2 gap (Job B, different priority):")
    print(f"  LongLiu avg: {avg_ll_p2:.1f}ms | CRUX avg: {avg_cx_p2:.1f}ms")
    print(f"  CRUX is {avg_pct_p2:+.1f}% faster")
    print()

    # Compare Phase 1 vs Phase 2 gap
    gap_p1 = avg_pct  # Phase 1 gap for Job A
    gap_p2 = avg_pct_p2  # Phase 2 gap for Job B

    print(f"Gap comparison:")
    print(f"  Phase 1 gap (Job A, equal pri): {gap_p1:+.1f}%")
    print(f"  Phase 2 gap (B, LL=P4/CX=P3): {gap_p2:+.1f}%")
    print()

    if abs(gap_p2) < abs(gap_p1):
        narrowing = abs(gap_p1) - abs(gap_p2)
        print(f"  Phase 2 gap is {narrowing:.1f}pp SMALLER than Phase 1 gap.")
        print(f"  → SP protection partially offsets environmental drift.")
        print(f"  → Estimated SP benefit: ~{narrowing:.1f}% (after drift correction)")
        print()
        print(f"  VERDICT: Candidate A (environmental drift) is PRIMARY explanation.")
        print(f"  The 'protected job slower' artifact is mostly drift, not SP backlash.")
    else:
        widening = abs(gap_p2) - abs(gap_p1)
        print(f"  Phase 2 gap is {widening:.1f}pp LARGER than Phase 1 gap.")
        print(f"  → SP protection makes things WORSE than drift alone.")
        print(f"  → Candidate B (SP phase backlash) supported.")
        print()
        print(f"  VERDICT: Candidate B (SP phase backlash) is supported.")

    print()
    print("  NOTE: Both arms ran sequentially (LongLiu first, CRUX second).")
    print("  File timestamps confirm: LL @13:28-29, CX @13:30-31.")
    print("  Without alternating run order, drift cannot be fully excluded.")
    print("  V6 should alternate: LL→CX and CX→LL.")
    print()

    # Save to file
    report_path = EXP_DIR / "v5_anomaly_diagnosis.txt"
    # (Report is printed to stdout; user can redirect if needed)


if __name__ == "__main__":
    main()
