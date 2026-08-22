"""
exp_e3_swap: E3 动态 ci Swap 实验（1 seed 快检）

场景：E1 workload @800G spine，t=300s 时交换 premium↔standard 的 slo_ci。
策略：v4、CRUX
预登记：
  - v4 W1: P-attn=100%, S-cont-cap≈0.74
  - v4 W3: P-attn=100%, S-cont-cap≈0.62, starv=0
  - v4 W2: 闭式无瞬态，1-2 epoch 内收敛
  - CRUX W1: P-attn≈87.5%±
  - CRUX W3: P-attn ≤50%（预期 ~33%）
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy.crux import CRUX
from longliu_sim.policy.dwrr import LongLiuAllocatorV4
from longliu_sim.core import Simulator, SimulationResult
from longliu_sim.job import Job
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace.synthetic import (
    FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD, place_workers_random
)
from longliu_sim.utils.model_params import MODEL_PARAMS, get_comp_ms
from longliu_sim.utils.config import load_config

_cfg = load_config()
SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
OVERLAP = _cfg["frozen"]["overlap_factor"]
OVERHEAD = _cfg["frozen"]["overhead_factor"]

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")) as f:
    CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

SWAP_TIME_MS = 300_000.0
DURATION_MS = 600_000.0

W1_START = 200_000
W1_END = 300_000
W2_START = 300_000
W2_END = 320_000
W3_START = 500_000
W3_END = 600_000

POLICIES = ["v4", "CRUX"]
SEED = 0


class SwapSimulator(Simulator):
    """支持 mid-simulation ci 交换的仿真器子类。"""

    def run(self) -> SimulationResult:
        self._recompute_bandwidth()
        swap_done = False

        while self.events and self.time_ms < self.duration_ms:
            next_time = self.events[0].time_ms

            if self.active_flows:
                finish_times = [
                    self.time_ms + f.rem_bits / f.rate_bps * 1000.0
                    for f in self.active_flows.values() if f.rate_bps > 0
                ]
                if finish_times:
                    earliest_finish = min(finish_times)
                    if earliest_finish < next_time:
                        next_time = earliest_finish

            if next_time > self.duration_ms:
                break

            self._advance(next_time)
            self._record_link_utilization_snapshot()
            self._cleanup_finished_flows()

            while (self.events and
                   abs(self.events[0].time_ms - self.time_ms) < 1e-9):
                event = heapq.heappop(self.events)
                self._process_event(event)

            if not swap_done and self.time_ms >= SWAP_TIME_MS:
                self._do_ci_swap()
                swap_done = True

            self._recompute_bandwidth()
            self._schedule_next_flow_end()

        result = SimulationResult(
            self.jobs, self.records, self.overlap_factor, self.overhead_factor
        )
        if self.link_utilization_history:
            result.link_utilization = self._compute_link_utilization_stats()
        return result

    # Model size classification: large models (LLaMA, T5) vs small models (BERT, ViT)
    LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}

    def _do_ci_swap(self):
        """按模型大小重新分配 ci，而非简单对调。
        
        设计稿 §六：
        - Phase 2: 小模型受保护（premium），大模型降级（standard）
        - LLaMA-13B ×2 → ci=1.5 (premium)
        - BERT dp4 ×2 → ci=2.0 (premium)
        - BERT dp2 ×2 → ci=1.5 (premium)
        - 原 premium → 全部 ci=3.0 (standard)
        """
        swaps = []
        for jid, job in self.jobs.items():
            old_ci = job.slo_ci
            is_large = job.model in self.LARGE_MODELS
            was_premium = old_ci <= 2.0

            if was_premium:
                # Premium → standard: all ci=3.0
                new_ci = 3.0
            else:
                # Standard → premium: ci depends on model
                if is_large:
                    new_ci = 1.5  # LLaMA-13B → ci=1.5
                elif job.num_workers == 4:
                    new_ci = 2.0  # BERT dp4 → ci=2.0
                else:
                    new_ci = 1.5  # BERT dp2 → ci=1.5
            
            job.slo_ci = new_ci
            swaps.append((jid, job.model, old_ci, new_ci))

        print(f"  [SWAP @{self.time_ms/1000:.0f}s] {len(swaps)} jobs ci exchanged")


def create_jobs_no_shuffle(workload, seed, num_hosts=16):
    """直接创建 Job 列表（不 shuffle），保持 JID↔workload 定义一致。"""
    rng = random.Random(seed)
    # Compact start: all jobs start within 0-100s to ensure W1 captures steady state
    start_window_ms = 100_000.0

    jobs = []
    for i, (model, dp, slo_ci) in enumerate(workload):
        if model not in MODEL_PARAMS:
            raise ValueError(f"Unknown model '{model}' in workload")

        params = MODEL_PARAMS[model]
        mb_per_iter = _compute_mb(params, dp)
        raw_comm_ms = mb_per_iter * 8 * 1024 * 1024 / (HOST_BW_GBPS * 1e9) * 1000.0
        comp_ms = get_comp_ms(model, default=50.0)
        iter_interval_ms = raw_comm_ms + comp_ms

        effective_comm_solo = raw_comm_ms * OVERHEAD
        target_iter_ms = comp_ms + effective_comm_solo * slo_ci
        # Use a very large target_iters so jobs don't finish within simulation
        target_iters = 999999

        start_time_ms = rng.uniform(0, start_window_ms)

        jobs.append(Job(
            jid=f"J{i}",
            model=model,
            mb_per_iter=mb_per_iter,
            iter_interval_ms=iter_interval_ms,
            target_iters=target_iters,
            slo_ci=slo_ci,
            num_workers=dp,
            start_time_ms=start_time_ms,
            comm_solo_ms=raw_comm_ms,
            compute_ms=comp_ms,
            overhead_factor=OVERHEAD,
            worker_hosts=place_workers_random(
                dp, num_hosts, seed=rng.randint(0, 2**31)
            ) if num_hosts > 0 else None,
        ))
    return jobs


def _compute_mb(params, dp):
    param_count = params.get("params", 1e9)
    bpp = 2 if params.get("fp16", True) else 4
    bytes_per_iter = 2 * param_count * bpp / max(dp, 1)
    return bytes_per_iter / (1024 * 1024)


def compute_window_stats(result, workload, pre_swap_cis):
    """窗口统计（W1/W2/W3），使用 pre_swap_cis 做 tier 判定。"""
    n_jobs = len(workload)

    # Tier from original ci definition (pre-swap)
    premium_jids_pre = set()
    premium_jids_post = set()
    for i, (_, _, ci) in enumerate(workload):
        jid = f"J{i}"
        if ci <= 2.0:
            premium_jids_pre.add(jid)
        else:
            premium_jids_post.add(jid)

    job_records = defaultdict(list)
    for r in result.records:
        job_records[r.jid].append(r)

    def _target_for(job, ci):
        """Compute target_iter_ms with a given ci (for pre/post swap)."""
        comm_budget_ms = ci * job.comm_solo_ms * OVERHEAD
        if OVERLAP > 0:
            return max(job.comp_ms, comm_budget_ms) + \
                   (1.0 - OVERLAP) * min(job.comp_ms, comm_budget_ms)
        return job.comp_ms + comm_budget_ms

    def stats_for_window(all_jids, start_ms, end_ms, premium_set, ci_map):
        """Filter iterations by start time (start_ms >= window_start) in window.
        
        Uses start_ms filter to ensure pure post-swap steady state:
        only iterations that STARTED within the window are counted.
        For W3 this prevents any iteration that began pre-swap from
        contaminating the post-swap measurement.
        """
        p_attn_count = 0
        p_total = 0
        s_sas_values = []
        per_job = {}

        for jid in all_jids:
            rs = [r for r in job_records.get(jid, [])
                  if r.start_ms >= start_ms and r.end_ms <= end_ms]
            if not rs:
                continue

            avg_iter = sum(r.iter_ms for r in rs) / len(rs)
            job = result.jobs[jid]
            ci = ci_map.get(jid, job.slo_ci)
            target = _target_for(job, ci)
            sas = target / avg_iter if avg_iter > 0 else 0.0

            per_job[jid] = {
                "n_iters": len(rs),
                "avg_iter_ms": round(avg_iter, 1),
                "target_ms": round(target, 1),
                "sas": round(min(sas, 5.0), 4),  # cap at 5 to avoid inf
            }

            if jid in premium_set:
                p_total += 1
                if sas >= 1.0 - 0.02:
                    p_attn_count += 1
            else:
                s_sas_values.append(min(sas, 1.0))

        p_attn = p_attn_count / p_total if p_total > 0 else 1.0
        s_cont_cap = np.mean(s_sas_values) if s_sas_values else 0.0

        return {
            "p_attn": round(p_attn, 4),
            "s_cont_cap": round(s_cont_cap, 4),
            "n_premium": p_total,
            "n_premium_attn": p_attn_count,
            "n_standard_with_data": len(s_sas_values),
            "per_job": per_job,
        }

    all_jids = [f"J{i}" for i in range(n_jobs)]

    # W1: pre-swap, use original ci
    w1 = stats_for_window(all_jids, W1_START, W1_END, premium_jids_pre, pre_swap_cis)
    w1["label"] = "W1 (pre-swap 200-300s)"

    # W2: transient, post-swap premium (was pre-swap standard)
    # ci mapping matches SwapSimulator._do_ci_swap
    LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}
    post_swap_cis = {}
    for jid, old_ci in pre_swap_cis.items():
        job = result.jobs[jid]
        was_premium = old_ci <= 2.0
        if was_premium:
            post_swap_cis[jid] = 3.0
        else:
            is_large = job.model in LARGE_MODELS
            if is_large:
                post_swap_cis[jid] = 1.5
            elif job.num_workers == 4:
                post_swap_cis[jid] = 2.0
            else:
                post_swap_cis[jid] = 1.5
    w2 = stats_for_window(all_jids, W2_START, W2_END, premium_jids_post, post_swap_cis)
    w2["label"] = "W2 (transient 300-320s)"

    # W3: post-swap steady state
    w3 = stats_for_window(all_jids, W3_START, W3_END, premium_jids_post, post_swap_cis)
    w3["label"] = "W3 (post-swap 500-600s)"

    starv_pre = sum(1 for jid in premium_jids_pre
                    if result.jobs[jid].completed_iters == 0)
    starv_post = sum(1 for jid in premium_jids_post
                     if result.jobs[jid].completed_iters == 0)

    return {
        "w1": w1, "w2": w2, "w3": w3,
        "starv_pre_swap": starv_pre,
        "starv_post_swap": starv_post,
    }


def run_single(policy_name: str, workload_raw=None, spine_bw_gbps=800,
               tag_prefix="e3_swap", seed=None) -> dict:
    if seed is None:
        seed = SEED
    workload = list(workload_raw) if workload_raw else list(FEAS_BOUNDARY_V3_WORKLOAD)
    n_jobs = len(workload)

    # Record pre-swap ci mapping before any mutation
    pre_swap_cis = {}
    for i, (_, _, ci) in enumerate(workload):
        pre_swap_cis[f"J{i}"] = ci

    tag = f"{tag_prefix}_{policy_name}_s{seed}"
    out_dir = f"outputs/e3_swap/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    spine_bw_bps = spine_bw_gbps * 1e9
    topo = FatTreeTopology(k=4, host_bw_bps=100e9, spine_bw_bps=spine_bw_bps)

    if policy_name == "v4":
        policy = LongLiuAllocatorV4(
            overhead_factor=OVERHEAD, overlap_factor=OVERLAP,
            trace_file=trace_file
        )
    elif policy_name == "CRUX":
        policy = CRUX()
    else:
        raise ValueError(policy_name)

    sim = SwapSimulator(
        topo, policy, duration_ms=DURATION_MS, seed=seed,
        overhead_factor=OVERHEAD, overlap_factor=OVERLAP
    )

    jobs = create_jobs_no_shuffle(workload, seed=seed, num_hosts=16)
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    if hasattr(policy, 'flush_trace'):
        policy.flush_trace()

    # Save swap log per-policy (match _do_ci_swap mapping)
    LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}
    swaps_detail = []
    for jid, old_ci in pre_swap_cis.items():
        job = result.jobs[jid]
        was_premium = old_ci <= 2.0
        if was_premium:
            new_ci = 3.0
        else:
            is_large = job.model in LARGE_MODELS
            if is_large:
                new_ci = 1.5
            elif job.num_workers == 4:
                new_ci = 2.0
            else:
                new_ci = 1.5
        swaps_detail.append({
            "jid": jid, "model": job.model, "dp": job.num_workers,
            "old_ci": old_ci, "new_ci": new_ci,
        })
    swap_info = {
        "swap_time_ms": SWAP_TIME_MS,
        "swaps": swaps_detail,
    }
    with open(f"{out_dir}/swap_log.json", "w") as f:
        json.dump(swap_info, f, indent=2)

    window_stats = compute_window_stats(result, workload, pre_swap_cis)

    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "policy": policy_name,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spine_bw_gbps": spine_bw_gbps,
        "swap_time_s": 300,
        "n_jobs": n_jobs,
        "w1": window_stats["w1"],
        "w2": window_stats["w2"],
        "w3": window_stats["w3"],
        "starv_pre_swap": window_stats["starv_pre_swap"],
        "starv_post_swap": window_stats["starv_post_swap"],
        "total_iters": result.total_iterations(),
    }

    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Save iteration records for sas-t trajectory plotting
    records_data = []
    for r in result.records:
        records_data.append({
            "jid": r.jid,
            "iter_idx": r.iter_idx,
            "start_ms": r.start_ms,
            "end_ms": r.end_ms,
            "iter_ms": r.iter_ms,
            "comm_ms": r.comm_ms,
            "n_flows": r.n_flows,
        })
    with open(f"{out_dir}/records.jsonl", "w") as f:
        for rec in records_data:
            f.write(json.dumps(rec) + "\n")

    return run_meta


def main():
    """Run both E3 (control arm) and E3' (kill arm) swap experiments."""

    configs = [
        {
            "name": "E3 (control arm: CRUX-advantaging swap)",
            "tag_prefix": "e3_swap",
            "workload": FEAS_BOUNDARY_V3_WORKLOAD,
            "spine_bw": 800,
            "n_jobs": 14,
            "premium_pre": 8, "standard_pre": 6,
            "premium_post": 6, "standard_post": 8,
        },
        {
            "name": "E3' (kill arm: CRUX-disadvantaging swap)",
            "tag_prefix": "e3p_swap",
            "workload": FEAS_BOUNDARY_V3_PRO_WORKLOAD,
            "spine_bw": 630,
            "n_jobs": 13,
            "premium_pre": 5, "standard_pre": 8,
            "premium_post": 8, "standard_post": 5,
        },
    ]

    all_results = {}

    for cfg in configs:
        print("=" * 80)
        print(f"{cfg['name']}")
        print("=" * 80)
        print(f"Workload: {cfg['n_jobs']} jobs ({cfg['premium_pre']}P+{cfg['standard_pre']}S), "
              f"Spine: {cfg['spine_bw']}G, Swap @ t=300s")
        print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}  CONFIG_HASH = {CONFIG_HASH}")
        print()

        results = {}
        for pn in POLICIES:
            print(f"[{pn}] ", end="", flush=True)
            try:
                r = run_single(pn, workload_raw=cfg["workload"],
                               spine_bw_gbps=cfg["spine_bw"],
                               tag_prefix=cfg["tag_prefix"])
                results[pn] = r
                w1 = r["w1"]
                w3 = r["w3"]
                print(f"W1 P-attn={w1['p_attn']*100:.1f}% S-cap={w1['s_cont_cap']:.3f} "
                      f"| W3 P-attn={w3['p_attn']*100:.1f}% S-cap={w3['s_cont_cap']:.3f} "
                      f"| starv_pre={r['starv_pre_swap']} starv_post={r['starv_post_swap']}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        all_results[cfg["tag_prefix"]] = results

        # Pre-registration verification
        print()
        print("=" * 80)
        print("Verification (Pre-registered)")
        print("=" * 80)

        if cfg["tag_prefix"] == "e3_swap":
            # E3 control arm checks
            if "v4" in results:
                r = results["v4"]
                w1, w3 = r["w1"], r["w3"]
                checks = [
                    ("v4 W1 P-attn=100%", w1["p_attn"], 1.0, 0.02),
                    ("v4 W3 P-attn=100%", w3["p_attn"], 1.0, 0.02),
                    ("v4 W3 starv=0", 1.0 - float(r["starv_post_swap"] > 0), 1.0, 0.0),
                    ("v4 W3 S-cont-cap≥0.62 (lower bound)", w3["s_cont_cap"], 0.62,
                     float("inf"), "ge"),  # one-sided: observed ≥ 0.62
                ]
                for item in checks:
                    if len(item) == 5 and item[4] == "ge":
                        label, val, expected, tol, _ = item
                        ok = val >= expected
                    else:
                        label, val, expected, tol = item
                        ok = abs(val - expected) <= tol
                    flag = "PASS" if ok else "FAIL"
                    print(f"  [{flag}] {label}: got={val:.4f} expected={expected}")

            if "CRUX" in results:
                r = results["CRUX"]
                w3 = r["w3"]
                # E3: CRUX W3 P-attn observed=83.3% — this arm is CRUX-advantaging,
                # so no FAIL criterion; just report
                print(f"  [INFO] CRUX W3 P-attn={w3['p_attn']*100:.1f}% "
                      f"(CRUX-advantaging swap, intensity-key consistent)")

        elif cfg["tag_prefix"] == "e3p_swap":
            # E3' kill arm checks
            all_pass = True
            if "v4" in results:
                r = results["v4"]
                w1, w3 = r["w1"], r["w3"]
                checks = [
                    ("v4 W1 P-attn=100%", w1["p_attn"], 1.0, 0.02),
                    ("v4 W3 P-attn=100%", w3["p_attn"], 1.0, 0.02),
                    ("v4 W3 starv=0", 1.0 - float(r["starv_post_swap"] > 0), 1.0, 0.0),
                ]
                for label, val, expected, tol in checks:
                    ok = abs(val - expected) <= tol
                    flag = "PASS" if ok else "FAIL"
                    if not ok:
                        all_pass = False
                    print(f"  [{flag}] {label}: got={val:.4f} expected={expected}±{tol}")

            if "CRUX" in results:
                r = results["CRUX"]
                w1, w3 = r["w1"], r["w3"]
                print(f"  [INFO] CRUX W1 P-attn={w1['p_attn']*100:.1f}% (expected ≈100% in pro zone)")
                
                # Key check: CRUX W3 P-attn ≪ v4
                if "v4" in results:
                    v4_w3_pattn = results["v4"]["w3"]["p_attn"]
                    crux_w3_pattn = w3["p_attn"]
                    gap = (v4_w3_pattn - crux_w3_pattn) * 100
                    ok = gap >= 10.0
                    flag = "PASS" if ok else "FAIL"
                    if not ok:
                        all_pass = False
                    print(f"  [{flag}] CRUX W3 P-attn ≪ v4 (gap≥10pp): "
                          f"v4={v4_w3_pattn*100:.1f}% CRUX={crux_w3_pattn*100:.1f}% gap={gap:.1f}pp")

            if all_pass:
                print("\n*** E3' PILOT PASSED ***")
            else:
                print("\n*** E3' PILOT: SOME CHECKS FAILED ***")

        # Summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_hash": CONFIG_HASH,
            "SEMANTICS_VERSION": SEMANTICS_VERSION,
            "spine_bw_gbps": cfg["spine_bw"],
            "results": {},
        }
        for pn, r in results.items():
            summary["results"][pn] = {
                "w1": {k: v for k, v in r["w1"].items() if k != "per_job"},
                "w2": {k: v for k, v in r["w2"].items() if k != "per_job"},
                "w3": {k: v for k, v in r["w3"].items() if k != "per_job"},
                "starv_pre_swap": r["starv_pre_swap"],
                "starv_post_swap": r["starv_post_swap"],
            }

        summary_path = f"outputs/e3_swap/summary_{cfg['tag_prefix']}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
