"""
exp_v3_batch2: feas_boundary_v3 第二批全量快检（矩阵 v2）

E1 六点（@400/500/630/800/1000/1200G）
  + E2' 两点（@630/800G）
  + E2-pro 两点（@630/800G）
= 10 配置点 × 5 策略 × 1 seed = 50 run

判定规则（矩阵 v2 三节框架）：
  一、v4 保障下界：@800/1000/1200G P-attn<100%(容差2%) = FAIL
  二、基线观测行：全部不设 FAIL，仅记录
  三、机制预测行：
      P1: E2' CRUX P-attn ≪ v4 P-attn（方向性）
      P2: E1 @400/500G v4 ≥ D1 > Fair
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

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
    FEAS_BOUNDARY_V3_PRIME_WORKLOAD,
    FEAS_BOUNDARY_V3_PRO_WORKLOAD,
)
from longliu_sim.utils import compute_sas_eval, compute_target_iter_ms
from longliu_sim.utils.config import load_config

_cfg = load_config()
SEMANTICS_VERSION = "anchor-v2"
HOST_BW_GBPS = 100.0
OVERLAP = _cfg["frozen"]["overlap_factor"]
OVERHEAD = _cfg["frozen"]["overhead_factor"]
K = _cfg["frozen"]["K"]

# 用 config 内容做哈希
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")) as f:
    CONFIG_HASH = hashlib.md5(f.read().encode()).hexdigest()[:8]

# --- 判定容差 ---
V4_GUARANTEE_TOLERANCE = 0.02  # P-attn ≥ 0.98 即为命中
P1_SIGNIFICANCE = 0.10  # CRUX P-attn 比 v4 低至少 10 个百分点


def get_policy(name: str, trace_file: str):
    if name == "Fair":
        return Fair()
    elif name == "CRUX":
        return CRUX()
    elif name == "SP":
        return SRPT()
    elif name == "D1":
        return LongLiuDWRR(
            overhead_factor=OVERHEAD, overlap_factor=OVERLAP,
            trace_file=trace_file,
        )
    elif name == "v4":
        return LongLiuAllocatorV4(
            overhead_factor=OVERHEAD, overlap_factor=OVERLAP,
            trace_file=trace_file,
        )
    raise ValueError(f"Unknown policy: {name}")


def run_single(scene: str, workload: List[Tuple[str, int, float]],
               spine_bw: float, policy_name: str,
               seed: int = 0) -> dict:
    """运行单个 (scene, spine, policy) 组合并返回统计。"""
    n_jobs = len(workload)
    duration_ms = 600000

    # 区分 premium/standard tier
    premium_jids = set()
    for i, (_, _, ci) in enumerate(workload):
        if ci <= 2.0:
            premium_jids.add(f"J{i}")

    tag = f"{scene}_{policy_name}_{int(spine_bw)}g"
    out_dir = f"outputs/v3_batch2/{tag}"
    trace_file = f"{out_dir}/trace.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    topo = FatTreeTopology(k=4, host_bw_bps=100e9, spine_bw_bps=spine_bw * 1e9)
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
            "ci": job.slo_ci, "sas": round(sas, 4),
            "avg_iter_ms": round(s["avg_iter_ms"], 2),
            "completed_iters": s["completed_iters"],
        }
        if jid in premium_jids:
            n_premium += 1
            if sas >= 1.0 - V4_GUARANTEE_TOLERANCE:
                n_premium_attn += 1
        else:
            s_sas_values.append(min(sas, 1.0))

    p_attn = round(n_premium_attn / n_premium, 4) if n_premium > 0 else 1.0
    s_cont_cap = round(sum(s_sas_values) / len(s_sas_values), 4) if s_sas_values else 0.0
    starv_count = sum(1 for jid in premium_jids
                      if per_job.get(jid, {}).get("completed_iters", 0) == 0)

    # --- run_meta ---
    run_meta = {
        "config_hash": CONFIG_HASH,
        "SEMANTICS_VERSION": SEMANTICS_VERSION,
        "scene": scene,
        "spine_bw": int(spine_bw),
        "policy": policy_name,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_premium": n_premium,
        "n_standard": len(stats) - n_premium,
        "p_attn": p_attn,
        "s_cont_cap": s_cont_cap,
        "starv": starv_count,
        "total_iters": result.total_iterations(),
    }
    with open(f"{out_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


# ============================================================
# 矩阵 v2 规则
# ============================================================

POLICIES = ["Fair", "CRUX", "SP", "D1", "v4"]

SCENARIOS = [
    ("E1", FEAS_BOUNDARY_V3_WORKLOAD, [400, 500, 630, 800, 1000, 1200]),
    ("E2'", FEAS_BOUNDARY_V3_PRIME_WORKLOAD, [630, 800]),
    ("E2-pro", FEAS_BOUNDARY_V3_PRO_WORKLOAD, [630, 800]),
]

V4_GUARANTEE_POINTS = {800, 1000, 1200}  # v4 P-attn≥100% 保障下界

# 需要做机制预测的 (scene, spine_bw) 组合
PREDICTION_CHECKS = {
    "P1": [("E2'", 630), ("E2'", 800)],   # CRUX ≪ v4
    "P2": [("E1", 400), ("E1", 500)],      # v4 ≥ D1 > Fair
}


def check_v4_guarantee(results_db: dict) -> list[str]:
    """检查 v4 保障下界：@800/1000/1200G P-attn >= 0.98"""
    failures = []
    for scene, _, spine_pts in SCENARIOS:
        for bw in spine_pts:
            if bw not in V4_GUARANTEE_POINTS:
                continue
            key = (scene, bw, "v4")
            if key not in results_db:
                failures.append(f"[v4保障] {scene} @{bw}G v4 未执行")
                continue
            r = results_db[key]
            if r["p_attn"] < 1.0 - V4_GUARANTEE_TOLERANCE:
                failures.append(
                    f"[v4保障 FAIL] {scene} @{bw}G v4 P-attn={r['p_attn']*100:.1f}% "
                    f"< 98% —— 下界被击穿！"
                )
    return failures


def check_prediction_p1(results_db: dict) -> list[str]:
    """P1: E2' CRUX P-attn 显著低于 v4 P-attn"""
    failures = []
    for scene, bw in PREDICTION_CHECKS["P1"]:
        k_v4 = (scene, bw, "v4")
        k_crux = (scene, bw, "CRUX")
        if k_v4 not in results_db or k_crux not in results_db:
            failures.append(f"[P1] {scene} @{bw}G: 缺少 v4 或 CRUX 数据")
            continue
        v4_attn = results_db[k_v4]["p_attn"]
        crux_attn = results_db[k_crux]["p_attn"]
        diff = v4_attn - crux_attn
        if diff < P1_SIGNIFICANCE:
            failures.append(
                f"[P1 FAIL] {scene} @{bw}G: CRUX P-attn={crux_attn*100:.1f}% "
                f"vs v4={v4_attn*100:.1f}% (diff={diff*100:.1f}pp < {P1_SIGNIFICANCE*100:.0f}pp) "
                f"—— CRUX 未显著低于 v4！"
            )
    return failures


def check_prediction_p2(results_db: dict) -> list[str]:
    """P2: E1 @400/500G v4 ≥ D1 > Fair"""
    failures = []
    for scene, bw in PREDICTION_CHECKS["P2"]:
        k = {pn: (scene, bw, pn) for pn in ["v4", "D1", "Fair"]}
        vals = {}
        for pn, key in k.items():
            if key not in results_db:
                failures.append(f"[P2] {scene} @{bw}G: 缺少 {pn} 数据")
                continue
            vals[pn] = results_db[key]["p_attn"]
        if len(vals) < 3:
            continue
        # v4 ≥ D1 > Fair (allow equality at the ≥ boundary)
        if vals["v4"] < vals["D1"] - 0.01:
            failures.append(
                f"[P2 FAIL] {scene} @{bw}G: v4({vals['v4']*100:.1f}%) "
                f"< D1({vals['D1']*100:.1f}%) —— 排序不成立！"
            )
        if vals["D1"] <= vals["Fair"]:
            failures.append(
                f"[P2 FAIL] {scene} @{bw}G: D1({vals['D1']*100:.1f}%) "
                f"≤ Fair({vals['Fair']*100:.1f}%) —— 排序不成立！"
            )
    return failures


def main():
    print("=" * 80)
    print("feas_boundary_v3 Batch 2 (Matrix v2)")
    print("=" * 80)
    n_total = sum(len(pts) for _, _, pts in SCENARIOS) * len(POLICIES)
    print(f"Scenarios × points × policies = {n_total} runs")
    print(f"SEMANTICS_VERSION = {SEMANTICS_VERSION}")
    print()

    results_db: Dict[tuple, dict] = {}

    # ---- 执行 ----
    for scene_name, workload, spine_pts in SCENARIOS:
        for spine_bw in spine_pts:
            for pn in POLICIES:
                label = f"{scene_name} @{spine_bw}G {pn}"
                print(f"[{label}] ", end="", flush=True)
                try:
                    r = run_single(scene_name, workload, spine_bw, pn)
                except Exception as e:
                    print(f"ERROR: {e}")
                    results_db[(scene_name, spine_bw, pn)] = {"_error": str(e)}
                    continue

                results_db[(scene_name, spine_bw, pn)] = r
                print(f"P-attn={r['p_attn']*100:5.1f}%  "
                      f"S-cap={r['s_cont_cap']:.3f}  "
                      f"starv={r['starv']}  iters={r['total_iters']}")

    # ---- 判定 ----
    print()
    print("=" * 80)
    print("Verification")
    print("=" * 80)

    all_failures = []
    any_fail = False

    # 一、v4 保障下界
    v4_fails = check_v4_guarantee(results_db)
    if v4_fails:
        print("\n[v4 保障下界检查]")
        for f in v4_fails:
            print(f"  FAIL: {f}")
        all_failures.extend(v4_fails)
        any_fail = True
    else:
        print("\n[v4 保障下界] 全部通过 ✓")

    # 二、基线观测行：不设 FAIL，仅打印
    print("\n[基线观测行]")
    obs_table = []
    for scene_name, _, spine_pts in SCENARIOS:
        for spine_bw in spine_pts:
            row = [f"{scene_name} @{spine_bw}G"]
            for pn in POLICIES:
                key = (scene_name, spine_bw, pn)
                r = results_db.get(key)
                if r and "_error" not in r:
                    row.append(f"{r['p_attn']*100:.1f}%")
                else:
                    row.append("ERR")
            obs_table.append(row)

    header = ["Scene"] + POLICIES
    col_w = [max(len(h), max(len(r[i]) for r in obs_table)) for i, h in enumerate(header)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in col_w]))
    for row in obs_table:
        print(fmt.format(*row))

    # 三、机制预测
    print("\n[机制预测行]")
    p1_fails = check_prediction_p1(results_db)
    if p1_fails:
        for f in p1_fails:
            print(f"  FAIL: {f}")
        all_failures.extend(p1_fails)
        any_fail = True
    else:
        print("  P1 (E2' CRUX ≪ v4): 成立 ✓")

    p2_fails = check_prediction_p2(results_db)
    if p2_fails:
        for f in p2_fails:
            print(f"  FAIL: {f}")
        all_failures.extend(p2_fails)
        any_fail = True
    else:
        print("  P2 (E1 @400/500G v4≥D1>Fair): 成立 ✓")

    # ---- 保存汇总 ----
    out_path = "outputs/v3_batch2/summary.json"
    os.makedirs("outputs/v3_batch2", exist_ok=True)
    serializable = {f"{s}_{b}_{p}": r for (s, b, p), r in results_db.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if any_fail:
        print("\n*** BATCH FAILED — stopping as instructed ***")
        for f in all_failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\n*** BATCH PASSED — all checks green ***")


if __name__ == "__main__":
    main()
