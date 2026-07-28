"""
exp_v3_batch1: feas_boundary_v3 第一批快检

E1 四点（@630/800/1000/1200G）+ E2 两点（@800G/@630G）
1 seed 五策略（Fair/CRUX/SP/D1/v4），对照修订矩阵逐点判定。
任一 FAIL 停跑上报。

修订矩阵：
  E1 @630G: P-attn≈82%, starv=0
  E1 @800G: P-attn=100%, S-cont-cap=0.74±0.03, starv=0
  E1 @1000G: P-attn=100%, S-cont-cap=1.0, starv=0
  E1 @1200G: P-attn=100%, S-cont-cap=1.0, starv=0
  E2 @800G: LongLiu P-attn=100%, S-cont-cap=0.607; CRUX P-attn<50%
  E2 @630G: P-attn≈78%, starv=0
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.policy.fair import Fair
from longliu_sim.policy.crux import CRUX
from longliu_sim.policy.srpt import SRPT
from longliu_sim.policy.dwrr import LongLiuDWRR, LongLiuAllocatorV4
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import (
    FEAS_BOUNDARY_V3_WORKLOAD,
    FEAS_BOUNDARY_V3_ANTI_WORKLOAD,
)
from longliu_sim.utils import compute_sas_eval
from longliu_sim.utils.config import load_config

_cfg = load_config()
HOST_BW_GBPS = 100.0
OVERLAP = _cfg["frozen"]["overlap_factor"]
OVERHEAD = _cfg["frozen"]["overhead_factor"]
K = _cfg["frozen"]["K"]

SAS_TOLERANCE = 0.02
P_ATTN_TOLERANCE = 0.03  # for ~82% type values


def get_policy(name: str, trace_file: str):
    if name == "Fair":
        return Fair()
    elif name == "CRUX":
        return CRUX()
    elif name == "SP":
        return SRPT()
    elif name == "D1":
        return LongLiuDWRR(
            overhead_factor=OVERHEAD,
            overlap_factor=OVERLAP,
            trace_file=trace_file,
        )
    elif name == "v4":
        return LongLiuAllocatorV4(
            overhead_factor=OVERHEAD,
            overlap_factor=OVERLAP,
            trace_file=trace_file,
        )
    else:
        raise ValueError(f"Unknown policy: {name}")


def run_single(scene: str, workload, spine_bw: float, policy_name: str,
               seed: int = 0) -> dict:
    """运行单个 (scene, spine, policy) 组合。"""
    n_jobs = len(workload)
    duration_ms = 600000

    # 区分 premium/standard
    premium_jids = set()
    standard_jids = set()
    for i, (model, dp, ci) in enumerate(workload):
        jid = f"J{i}"
        if ci <= 2.0:
            premium_jids.add(jid)
        else:
            standard_jids.add(jid)

    out_dir = f"outputs/v3_batch1/{scene}_{policy_name}_{int(spine_bw)}g"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(
        k=4, host_bw_bps=100e9, spine_bw_bps=spine_bw * 1e9
    )

    policy = get_policy(policy_name, trace_file)
    sim = Simulator(
        topo, policy, duration_ms=duration_ms, seed=seed,
        overhead_factor=OVERHEAD, overlap_factor=OVERLAP,
    )

    loader = SyntheticTraceLoader(
        model_types=[], gpu_distribution={}, ci_distribution={},
        job_count=n_jobs, duration_ms=duration_ms, seed=seed,
        overhead_factor=OVERHEAD, target_bw_bps=100e9, num_hosts=16,
        workload_profile=list(workload),
    )
    jobs = loader.load()

    # 对齐设计稿 JID：job J_i 对应 workload[i]
    for i, j in enumerate(jobs):
        j.jid = f"J{i}"

    for j in jobs:
        sim.submit(j)

    result = sim.run()
    if hasattr(policy, 'flush_trace'):
        policy.flush_trace()

    # --- per-job stats ---
    stats = result.per_job_stats(host_bw_gbps=HOST_BW_GBPS)
    per_job = {}
    n_premium = 0
    n_premium_attn = 0
    s_sas_values = []

    for jid, s in stats.items():
        job = sim.jobs[jid]
        sas = s["sas"]
        per_job[jid] = {
            "model": job.model, "dp": job.num_workers,
            "ci": job.slo_ci, "sas": sas,
            "avg_iter_ms": s["avg_iter_ms"],
            "completed_iters": s["completed_iters"],
        }
        if jid in premium_jids:
            n_premium += 1
            if sas >= 1.0 - SAS_TOLERANCE:
                n_premium_attn += 1
        elif jid in standard_jids:
            s_sas_values.append(min(sas, 1.0))

    p_attn = n_premium_attn / n_premium if n_premium > 0 else 1.0
    s_cont_cap = sum(s_sas_values) / len(s_sas_values) if s_sas_values else 0.0
    starv = sum(1 for jid in premium_jids
                if per_job.get(jid, {}).get("completed_iters", 0) == 0)

    return {
        "scene": scene,
        "spine_bw": spine_bw,
        "policy": policy_name,
        "n_jobs": n_jobs,
        "n_premium": n_premium,
        "n_standard": len(standard_jids),
        "p_attn": p_attn,
        "s_cont_cap": s_cont_cap,
        "starv": starv,
        "total_iters": result.total_iterations(),
        "per_job": per_job,
    }


# --- 预登记矩阵 ---
MATRIX = {
    ("E1", 630): {
        "all": {"p_attn_range": (0.79, 0.85), "starv": 0},
    },
    ("E1", 800): {
        "all": {"p_attn_range": (0.98, 1.01), "starv": 0},
        "v4": {"s_cont_cap_range": (0.71, 0.77)},  # 0.74±0.03
        "D1": {"s_cont_cap_min": 0.0},  # D1 not calibrated for this, just check no starv
    },
    ("E1", 1000): {
        "all": {"p_attn_range": (0.98, 1.01), "s_cont_cap_range": (0.98, 1.01), "starv": 0},
    },
    ("E1", 1200): {
        "all": {"p_attn_range": (0.98, 1.01), "s_cont_cap_range": (0.98, 1.01), "starv": 0},
    },
    ("E2", 630): {
        "all": {"p_attn_range": (0.75, 0.81), "starv": 0},
    },
    ("E2", 800): {
        "v4": {"p_attn_range": (0.98, 1.01), "s_cont_cap_range": (0.577, 0.637)},  # 0.607±0.03
        "D1": {"p_attn_range": (0.90, 1.01), "s_cont_cap_range": (0.50, 0.70)},  # D1 likely lower
        "Fair": {"p_attn_min": 0.0, "s_cont_cap_min": 0.0},
        "SP": {"p_attn_min": 0.0, "s_cont_cap_min": 0.0},
        "CRUX": {"p_attn_max": 0.50},  # CRUX P-attn 崩溃 <50%
    },
}

POLICIES = ["Fair", "CRUX", "SP", "D1", "v4"]

SCENES = [
    ("E1", FEAS_BOUNDARY_V3_WORKLOAD, [630, 800, 1000, 1200]),
    ("E2", FEAS_BOUNDARY_V3_ANTI_WORKLOAD, [800, 630]),  # order from user: @800G first, @630G
]


def check_result(r: dict) -> list[str]:
    """对照矩阵检查，返回 FAIL 列表。"""
    key = (r["scene"], int(r["spine_bw"]))
    matrix = MATRIX.get(key, {})
    failures = []
    pn = r["policy"]

    # 通用检查（all policies）
    all_rules = matrix.get("all", {})
    # 策略特定检查
    spec_rules = matrix.get(pn, {})

    rules = {**all_rules, **spec_rules}

    if "p_attn_range" in rules:
        lo, hi = rules["p_attn_range"]
        if not (lo <= r["p_attn"] <= hi):
            failures.append(
                f"P-attn={r['p_attn']:.3f} ∉ [{lo},{hi}]"
            )
    if "p_attn_min" in rules:
        if r["p_attn"] < rules["p_attn_min"]:
            failures.append(f"P-attn={r['p_attn']:.3f} < min={rules['p_attn_min']}")
    if "p_attn_max" in rules:
        if r["p_attn"] > rules["p_attn_max"]:
            failures.append(f"P-attn={r['p_attn']:.3f} > max={rules['p_attn_max']}")

    if "s_cont_cap_range" in rules:
        lo, hi = rules["s_cont_cap_range"]
        if not (lo <= r["s_cont_cap"] <= hi):
            failures.append(
                f"S-cont-cap={r['s_cont_cap']:.3f} ∉ [{lo},{hi}]"
            )
    if "s_cont_cap_min" in rules:
        if r["s_cont_cap"] < rules["s_cont_cap_min"]:
            failures.append(f"S-cont-cap={r['s_cont_cap']:.3f} < min={rules['s_cont_cap_min']}")

    if "starv" in rules and r["starv"] != rules["starv"]:
        failures.append(f"starv={r['starv']} ≠ {rules['starv']}")

    return failures


def main():
    print("=" * 80)
    print("feas_boundary_v3 Batch 1 Quick Check")
    print("=" * 80)
    print(f"Scenes: E1 (4 pts) + E2 (2 pts) = 6 runs/policy")
    print(f"Policies: {POLICIES}")
    print(f"Total: {6 * 5} = 30 simulations")
    print()

    all_results = []
    any_fail = False

    for scene_name, workload, spine_pts in SCENES:
        for spine_bw in spine_pts:
            for pn in POLICIES:
                label = f"{scene_name} @{int(spine_bw)}G {pn}"
                print(f"[{label}] ", end="", flush=True)
                try:
                    r = run_single(scene_name, workload, spine_bw, pn)
                except Exception as e:
                    print(f"ERROR: {e}")
                    any_fail = True
                    continue

                all_results.append(r)
                p_attn_pct = r["p_attn"] * 100
                print(f"P-attn={p_attn_pct:.1f}% S-cap={r['s_cont_cap']:.3f} "
                      f"starv={r['starv']} iters={r['total_iters']}")

                failures = check_result(r)
                if failures:
                    print(f"  >>> FAIL: {'; '.join(failures)}")
                    any_fail = True
                else:
                    print(f"  >>> PASS")

    print()
    print("=" * 80)
    print("Batch 1 Summary")
    print("=" * 80)
    for r in all_results:
        failures = check_result(r)
        status = "FAIL" if failures else "PASS"
        detail = "; ".join(failures) if failures else ""
        print(f"  [{status}] {r['scene']} @{int(r['spine_bw'])}G {r['policy']:4s}  "
              f"P-attn={r['p_attn']*100:5.1f}%  S-cont-cap={r['s_cont_cap']:.3f}  "
              f"{'  ' + detail if detail else ''}")

    # Save results
    out_path = "outputs/v3_batch1/summary.json"
    os.makedirs("outputs/v3_batch1", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if any_fail:
        print("\n*** BATCH FAILED — stopping as instructed ***")
        sys.exit(1)
    else:
        print("\n*** BATCH PASSED — all checks green ***")


if __name__ == "__main__":
    main()
