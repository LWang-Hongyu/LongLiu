"""
E3' 证据关检查脚本：
  Step 3: 静态复制品 — post-swap ci 从 t=0 跑 CRUX @630G
  Step 4: 窗口过滤修正 — start_ms 过滤 vs end_ms 过滤对比

用法：python3 experiments/_e3p_evidence_check.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_e3_swap import (
    create_jobs_no_shuffle, _compute_mb, compute_window_stats,
    OVERLAP, OVERHEAD, CONFIG_HASH, SEMANTICS_VERSION,
    W1_START, W1_END, W2_START, W2_END, W3_START, W3_END,
    SWAP_TIME_MS, DURATION_MS, SEED,
)
from longliu_sim.policy.crux import CRUX
from longliu_sim.policy.dwrr import LongLiuAllocatorV4
from longliu_sim.core import Simulator, SimulationResult
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V3_PRO_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS, get_comp_ms


# ── E3' post-swap workload (ci mapping from t=0) ──
# Original E2-pro: premium=small models ci≤2.0, standard=large models ci=3.0
# Post-swap: premium=large models ci=1.5, standard=small models ci=3.0
LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}

def build_static_postswap_workload():
    """从 E2-pro workload 构建 post-swap ci 的静态 workload。"""
    result = []
    for model, dp, original_ci in FEAS_BOUNDARY_V3_PRO_WORKLOAD:
        was_premium = original_ci <= 2.0
        if was_premium:
            new_ci = 3.0  # small model → standard
        else:
            new_ci = 1.5  # large model → premium (all dp=8, not dp=4)
        result.append((model, dp, new_ci))
    return result


def run_static_replica():
    """Step 3: 静态复制品 — post-swap ci 从 t=0 运行。"""
    workload = build_static_postswap_workload()
    n_jobs = len(workload)
    sp_bw = 630

    print("=" * 80)
    print("Step 3: Static Replica — post-swap ci from t=0, CRUX @630G")
    print("=" * 80)
    print(f"Workload: {n_jobs} jobs, ci mapping:")
    for i, (m, dp, ci) in enumerate(workload):
        tier = "P" if ci <= 2.0 else "S"
        print(f"  J{i}: {m} dp={dp} ci={ci} [{tier}]")
    print()

    for policy_name in ["v4", "CRUX"]:
        tag = f"_static_postswap_{policy_name}_s{SEED}"
        out_dir = f"outputs/e3_swap/{tag}"
        os.makedirs(out_dir, exist_ok=True)

        topo = FatTreeTopology(k=4, host_bw_bps=100e9, spine_bw_bps=sp_bw * 1e9)

        if policy_name == "v4":
            policy = LongLiuAllocatorV4(
                overhead_factor=OVERHEAD, overlap_factor=OVERLAP,
                trace_file=f"{out_dir}/trace.jsonl"
            )
        else:
            policy = CRUX()

        from longliu_sim.core import Simulator
        sim = Simulator(
            topo, policy, duration_ms=DURATION_MS, seed=SEED,
            overhead_factor=OVERHEAD, overlap_factor=OVERLAP
        )

        jobs = create_jobs_no_shuffle(workload, seed=SEED, num_hosts=16)
        for j in jobs:
            sim.submit(j)

        print(f"[{policy_name}] running...", end=" ", flush=True)
        result = sim.run()
        if hasattr(policy, 'flush_trace'):
            policy.flush_trace()

        # Record ci mapping
        pre_swap_cis = {}
        for i, (_, _, ci) in enumerate(workload):
            pre_swap_cis[f"J{i}"] = ci

        window_stats = compute_window_stats(result, workload, pre_swap_cis)

        w1 = window_stats["w1"]
        w3 = window_stats["w3"]
        print(f"W1 P-attn={w1['p_attn']*100:.1f}% | W3 P-attn={w3['p_attn']*100:.1f}% "
              f"| starv={window_stats['starv_pre_swap']}")

        run_meta = {
            "config_hash": CONFIG_HASH,
            "SEMANTICS_VERSION": SEMANTICS_VERSION,
            "policy": policy_name,
            "seed": SEED,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spine_bw_gbps": sp_bw,
            "static_postswap": True,
            "n_jobs": n_jobs,
            "w1": w1,
            "w3": w3,
            "starv": window_stats["starv_pre_swap"],
            "total_iters": result.total_iterations(),
        }
        with open(f"{out_dir}/run_meta.json", "w") as f:
            json.dump(run_meta, f, indent=2)


def run_window_filter_comparison():
    """Step 4: 窗口过滤方法论修正 — start_ms vs end_ms 对比。"""
    print()
    print("=" * 80)
    print("Step 4: Window Filter Comparison (start_ms vs end_ms)")
    print("=" * 80)

    for arm_tag, workload_raw in [
        ("e3p_swap", FEAS_BOUNDARY_V3_PRO_WORKLOAD),
    ]:
        n_jobs = len(workload_raw)
        for policy_name in ["v4", "CRUX"]:
            tag = f"{arm_tag}_{policy_name}_s{SEED}"
            meta_path = f"outputs/e3_swap/{tag}/run_meta.json"
            if not os.path.exists(meta_path):
                print(f"  Skipping {tag}: no run_meta.json")
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            print(f"\n  [{tag}] spine={meta['spine_bw_gbps']}G")

            # Compare end_ms vs start_ms filtering
            # We need to re-compute from the per_job data using different filter logic
            # The per_job in run_meta already uses end_ms filter.
            # To get start_ms filter, we'd need to re-process trace records.
            # 
            # Alternative: compute using the trace file
            trace_path = f"outputs/e3_swap/{tag}/trace.jsonl"
            
            if not os.path.exists(trace_path):
                print(f"    No trace file, using run_meta comparison only")
                continue

            # Load trace records and re-compute with start_ms filter
            from collections import defaultdict
            job_records = defaultdict(list)
            
            with open(trace_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if "jid" in r and "iter_ms" in r:
                            job_records[r["jid"]].append(r)
                    except:
                        continue

            # Premium tier identification (post-swap)
            workload = list(workload_raw)
            premium_jids_post = set()
            for i, (_, _, ci) in enumerate(workload):
                if ci > 2.0:  # was standard → post-swap premium
                    premium_jids_post.add(f"J{i}")

            # Get target info from run_meta per_job (W3 uses post-swap ci)
            w3_end_ms = meta["w3"]["per_job"]
            
            # Re-compute W3 with start_ms filter
            def recompute_w3_start_ms():
                p_attn = 0
                p_total = 0
                s_sas_vals = []
                result_per_job = {}

                for jid in [f"J{i}" for i in range(n_jobs)]:
                    rs = [r for r in job_records.get(jid, [])
                          if r.get("start_ms", 0) >= W3_START
                          and r.get("end_ms", 0) <= W3_END]
                    if not rs:
                        continue

                    avg_iter = sum(r["iter_ms"] for r in rs) / len(rs)
                    target = w3_end_ms.get(jid, {}).get("target_ms")
                    if target is None:
                        # Need to compute from workload
                        continue

                    sas = target / avg_iter if avg_iter > 0 else 0.0
                    result_per_job[jid] = {
                        "n_iters": len(rs),
                        "avg_iter_ms": round(avg_iter, 1),
                        "target_ms": target,
                        "sas": round(min(sas, 5.0), 4),
                    }

                    if jid in premium_jids_post:
                        p_total += 1
                        if sas >= 0.98:
                            p_attn += 1
                    else:
                        s_sas_vals.append(min(sas, 1.0))

                p_attn_rate = p_attn / p_total if p_total > 0 else 1.0
                s_cap = np.mean(s_sas_vals) if s_sas_vals else 0.0
                return p_attn_rate, s_cap, p_total, p_attn, result_per_job

            try:
                p_attn_s, s_cap_s, p_total_s, p_attn_count_s, per_job_s = recompute_w3_start_ms()
            except Exception as e:
                print(f"    Re-compute error: {e}")
                continue

            # Original end_ms values from run_meta
            w3_orig = meta["w3"]
            p_attn_e = w3_orig["p_attn"]
            s_cap_e = w3_orig["s_cont_cap"]
            p_total_e = w3_orig["n_premium"]
            p_attn_count_e = w3_orig["n_premium_attn"]
            per_job_e = w3_orig["per_job"]

            print(f"    W3 end_ms filter:  P-attn={p_attn_e*100:.1f}% ({p_attn_count_e}/{p_total_e}) "
                  f"S-cap={s_cap_e:.3f}")
            print(f"    W3 start_ms filter: P-attn={p_attn_s*100:.1f}% ({p_attn_count_s}/{p_total_s}) "
                  f"S-cap={s_cap_s:.3f}")

            # Per-job comparison (premium only)
            print(f"    {'JID':<6} {'end_sas':>8} {'start_sas':>8} {'end_n':>6} {'start_n':>6} {'delta_sas':>8} {'end_avg':>8} {'start_avg':>8}")
            any_diff = False
            for jid in sorted(premium_jids_post):
                e = per_job_e.get(jid, {})
                s = per_job_s.get(jid, {})
                if not e and not s:
                    continue
                es = e.get("sas", 0)
                ss = s.get("sas", 0)
                en = e.get("n_iters", 0)
                sn = s.get("n_iters", 0)
                ea = e.get("avg_iter_ms", 0)
                sa = s.get("avg_iter_ms", 0)
                delta = ss - es
                if abs(delta) > 0.005:
                    any_diff = True
                print(f"    {jid:<6} {es:>8.4f} {ss:>8.4f} {en:>6d} {sn:>6d} {delta:>+8.4f} {ea:>8.1f} {sa:>8.1f}")
            
            if not any_diff:
                print(f"    → No significant difference between filters (<0.005 sas delta)")


if __name__ == "__main__":
    print(f"E3' Evidence Check | SEMANTICS_VERSION={SEMANTICS_VERSION} CONFIG_HASH={CONFIG_HASH}")
    print()

    # Step 3: Static replica
    run_static_replica()

    # Step 4: Window filter comparison (uses existing trace files)
    run_window_filter_comparison()

    print()
    print("Done. Check outputs/e3_swap/_static_postswap_* for static replica results.")
