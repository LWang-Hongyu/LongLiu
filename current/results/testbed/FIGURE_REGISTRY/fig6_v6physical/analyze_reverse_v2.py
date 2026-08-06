#!/usr/bin/env python3
"""
Analyze P4-1 Role-Reversal V4 results (c_i swap, same payload 512MB, solo pre-learning T_target).

Generates:
  - reverse_v2_comparison_epoch.csv : per-epoch comparison (both modes, both jobs)
  - reverse_v2_summary.txt          : human-readable summary with form description

Slowdown is recomputed correctly using per-epoch c_i:
  slowdown = avg_comm_per_iter / (c_i * T_target_per_iter)
  where T_target_per_iter = T_target_ms / 1000 / iters_per_epoch
  and c_i is the phase-specific SLO relaxation coefficient (from CSV column).
"""

import csv
import os
import json
import glob

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ITERS_PER_EPOCH = 20
REVERSE_EPOCH = 7


def load_ttarget(job):
    """Read T_target from calibration JSON. Searches multiple patterns, most recent first.
    Parameterized: no hardcoded constants. Looks for:
      /tmp/ttarget_v5_job{job}.json, /tmp/ttarget_v4_job{job}.json, etc.
    Uses most recently modified file.
    """
    files = glob.glob(f'/tmp/ttarget_*_job{job}.json')
    if not files:
        raise FileNotFoundError(f"No T_target file found for job {job} in /tmp/")
    # Sort by modification time, most recent first
    files.sort(key=os.path.getmtime, reverse=True)
    with open(files[0]) as f:
        data = json.load(f)
    print(f"  Loaded T_target for job {job}: {data['target_comm_time_ms']:.3f}ms "
          f"(from {os.path.basename(files[0])})")
    return data['target_comm_time_ms']


# Load T_target from JSON (parameterized — no hardcoded values)
TTARGET_A_MS = load_ttarget('A')
TTARGET_B_MS = load_ttarget('B')
TTARGET_A_PER_ITER_S = TTARGET_A_MS / 1000.0 / ITERS_PER_EPOCH
TTARGET_B_PER_ITER_S = TTARGET_B_MS / 1000.0 / ITERS_PER_EPOCH


def load_epoch_csv(path):
    """Load per-epoch CSV, return list of dicts with corrected slowdown."""
    rows = []
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            # Recompute slowdown correctly using per-epoch c_i
            avg_comm = float(r['avg_comm_s'])
            t_target_ms = float(r['t_target_ms'])
            c_i = float(r['c_i'])  # per-epoch c_i (V3: swaps at REVERSE_EPOCH)
            t_target_per_iter = t_target_ms / 1000.0 / ITERS_PER_EPOCH
            if t_target_per_iter > 0:
                correct_slowdown = avg_comm / (c_i * t_target_per_iter)
            else:
                correct_slowdown = float('nan')
            r['slowdown_correct'] = round(correct_slowdown, 4)
            r['slo_met'] = 'YES' if correct_slowdown <= 1.0 else 'NO'
            # Convert pi to float
            try:
                r['pi'] = float(r['pi'])
            except (ValueError, TypeError):
                r['pi'] = float('nan')
            rows.append(r)
    return rows


def main():
    # Load all 4 datasets
    data = {}
    for job in ['A', 'B']:
        for mode in ['longliu', 'crux']:
            path = os.path.join(EXP_DIR, f'p4_job{job}_reverse_{mode}_rank0_epoch.csv')
            data[(job, mode)] = load_epoch_csv(path)

    # ============================================================
    # Write merged comparison CSV
    # ============================================================
    out_path = os.path.join(EXP_DIR, 'reverse_v2_comparison_epoch.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mode', 'job', 'epoch', 'phase', 'payload_mb', 'c_i',
            'avg_comm_s', 'avg_bw_gbps', 'pi', 'priority', 'dscp',
            'slowdown_correct', 'slo_met', 't_target_ms'
        ])
        for (job, mode), rows in sorted(data.items()):
            for r in rows:
                writer.writerow([
                    mode, job, r['epoch'], r['phase'], r['payload_mb'], r['c_i'],
                    r['avg_comm_s'], r['avg_bw_gbps'], r['pi'],
                    r['priority'], r['dscp'],
                    r['slowdown_correct'], r['slo_met'], r['t_target_ms']
                ])
    print(f"Comparison CSV: {out_path}")

    # ============================================================
    # Generate summary with form description
    # ============================================================
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("P4-1 Role-Reversal V4 Analysis (c_i swap, same payload, solo pre-learning)")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"T_target acquisition: SOLO PRE-LEARNING (Phase 0 calibration)")
    summary_lines.append(f"  Job A (512MB): T_target = {TTARGET_A_MS}ms/epoch "
                         f"= {TTARGET_A_PER_ITER_S*1000:.1f}ms/iter")
    summary_lines.append(f"  Job B (512MB): T_target = {TTARGET_B_MS}ms/epoch "
                         f"= {TTARGET_B_PER_ITER_S*1000:.1f}ms/iter")
    summary_lines.append(f"SLO: c_i swaps at epoch {REVERSE_EPOCH}")
    summary_lines.append(f"  Phase 1 (epochs 0-{REVERSE_EPOCH-1}): A c_i=1.6 (tight), B c_i=3.0 (loose)")
    summary_lines.append(f"  Phase 2 (epochs {REVERSE_EPOCH}-14): A c_i=3.0 (loose), B c_i=1.6 (tight)")
    summary_lines.append(f"  SLO target per iter (phase 1): Job A = {1.6*TTARGET_A_PER_ITER_S*1000:.1f}ms, "
                         f"Job B = {3.0*TTARGET_B_PER_ITER_S*1000:.1f}ms")
    summary_lines.append(f"  SLO target per iter (phase 2): Job A = {3.0*TTARGET_A_PER_ITER_S*1000:.1f}ms, "
                         f"Job B = {1.6*TTARGET_B_PER_ITER_S*1000:.1f}ms")
    summary_lines.append(f"Reversal: epoch {REVERSE_EPOCH} (c_i swap, same payload 512MB)")
    summary_lines.append("")

    # Per-mode, per-job phase summary
    for mode in ['longliu', 'crux']:
        scheduler_label = "v1(π)" if mode == "longliu" else "CRUX-static"
        summary_lines.append("-" * 80)
        summary_lines.append(f"Mode: {mode} (scheduler={scheduler_label}, queue=SP)")
        summary_lines.append("-" * 80)
        for job in ['A', 'B']:
            rows = data[(job, mode)]
            if not rows:
                continue
            p1 = [r for r in rows if r['phase'] == 'phase1']
            p2 = [r for r in rows if r['phase'] == 'phase2']

            summary_lines.append(f"\n  Job {job}:")
            for phase_name, phase_rows in [('Phase 1', p1), ('Phase 2', p2)]:
                if not phase_rows:
                    continue
                avg_comm = sum(float(r['avg_comm_s']) for r in phase_rows) / len(phase_rows)
                avg_bw = sum(float(r['avg_bw_gbps']) for r in phase_rows) / len(phase_rows)
                avg_prio = sum(int(r['priority']) for r in phase_rows) / len(phase_rows)
                avg_slowdown = sum(float(r['slowdown_correct']) for r in phase_rows) / len(phase_rows)
                pi_vals = [float(r['pi']) for r in phase_rows
                           if r['pi'] == r['pi']]  # filter NaN
                avg_pi = sum(pi_vals) / len(pi_vals) if pi_vals else float('nan')
                slo_met_count = sum(1 for r in phase_rows if r['slo_met'] == 'YES')
                payload = phase_rows[0]['payload_mb']

                summary_lines.append(f"    {phase_name} (payload={payload}MB, {len(phase_rows)} epochs):")
                summary_lines.append(f"      avg_comm  = {avg_comm*1000:.1f} ms/iter")
                summary_lines.append(f"      avg_bw    = {avg_bw:.2f} Gbps")
                summary_lines.append(f"      avg_prio  = P{avg_prio:.1f}")
                summary_lines.append(f"      avg_π     = {avg_pi:+.3f}" if pi_vals else "      avg_π     = N/A")
                summary_lines.append(f"      avg_slow  = {avg_slowdown:.3f}x (SLO≤1.0)")
                summary_lines.append(f"      SLO met   = {slo_met_count}/{len(phase_rows)} epochs")

            # Priority trajectory
            prios = [int(r['priority']) for r in rows]
            summary_lines.append(f"    Priority trajectory: {prios}")

    # ============================================================
    # Form description (no conclusions)
    # ============================================================
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("FORM DESCRIPTION (data shapes only, no conclusions)")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    # Priority trajectory comparison
    summary_lines.append("1. Priority trajectory (per epoch):")
    summary_lines.append("   LongLiu v1(π):")
    for job in ['A', 'B']:
        rows = data[(job, 'longliu')]
        if rows:
            prios = [int(r['priority']) for r in rows]
            summary_lines.append(f"     Job {job}: {prios}")
    summary_lines.append("   CRUX-static:")
    for job in ['A', 'B']:
        rows = data[(job, 'crux')]
        if rows:
            prios = [int(r['priority']) for r in rows]
            summary_lines.append(f"     Job {job}: {prios}")

    # π trajectory
    summary_lines.append("")
    summary_lines.append("2. π trajectory (per epoch):")
    summary_lines.append("   LongLiu v1(π):")
    for job in ['A', 'B']:
        rows = data[(job, 'longliu')]
        if rows:
            pis = [float(r['pi']) for r in rows]
            pis_str = ', '.join(f'{p:+.2f}' for p in pis)
            summary_lines.append(f"     Job {job}: [{pis_str}]")

    # Slowdown trajectory
    summary_lines.append("")
    summary_lines.append("3. Slowdown vs phase-specific c_i trajectory (per epoch):")
    summary_lines.append("   (slowdown = avg_comm / (c_i × T_target_per_iter); > 1.0 = SLO violated)")
    summary_lines.append(f"   Phase 1 c_i: A=1.6, B=3.0; Phase 2 c_i: A=3.0, B=1.6")
    for mode in ['longliu', 'crux']:
        label = "LongLiu v1(π)" if mode == "longliu" else "CRUX-static"
        summary_lines.append(f"   {label}:")
        for job in ['A', 'B']:
            rows = data[(job, mode)]
            if rows:
                slows = [float(r['slowdown_correct']) for r in rows]
                slows_str = ', '.join(f'{s:.2f}' for s in slows)
                summary_lines.append(f"     Job {job}: [{slows_str}]")

    # Comm time trajectory
    summary_lines.append("")
    summary_lines.append("4. Per-epoch avg comm time (ms/iter):")
    for mode in ['longliu', 'crux']:
        label = "LongLiu v1(π)" if mode == "longliu" else "CRUX-static"
        summary_lines.append(f"   {label}:")
        for job in ['A', 'B']:
            rows = data[(job, mode)]
            if rows:
                comms = [float(r['avg_comm_s'])*1000 for r in rows]
                comms_str = ', '.join(f'{c:.0f}' for c in comms)
                summary_lines.append(f"     Job {job}: [{comms_str}]")

    # Phase 2 SLO violation check
    summary_lines.append("")
    summary_lines.append("5. Phase 2 SLO violation count (slowdown > 1.0):")
    for mode in ['longliu', 'crux']:
        label = "LongLiu v1(π)" if mode == "longliu" else "CRUX-static"
        for job in ['A', 'B']:
            rows = data[(job, mode)]
            p2 = [r for r in rows if r['phase'] == 'phase2']
            if p2:
                violated = sum(1 for r in p2 if float(r['slowdown_correct']) > 1.0)
                summary_lines.append(f"   {label} Job {job}: {violated}/{len(p2)} epochs violated")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(EXP_DIR, 'reverse_v2_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"\nSummary: {summary_path}")


if __name__ == '__main__':
    main()
