"""
从 e3_swap / e3p_swap 的 DF (LongLiuDWRR) records.jsonl 重建 fig4 轨迹 CSV。

输出（figure_registry）：
- fig4_d1_trajectory_e3.csv  （E3, 800G, 5 seeds）
- fig4_d1_trajectory_e3p.csv （E3' kill, 630G, 5 seeds）

格式：time_s,mean,std,seed0..seed4（与 _draw_final_v3.load_df_csv 兼容）。
轨迹语义与 _draw_final_v3.compute_trajectory 一致（regime-truncated sliding window P-attn）。
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE)

from longliu_sim.trace.synthetic import (
    FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD,
)
from longliu_sim.utils.model_params import MODEL_PARAMS
from longliu_sim.utils.config import load_config

_cfg = load_config()
OVERHEAD = _cfg["frozen"]["overhead_factor"]
OVERLAP = _cfg["frozen"]["overlap_factor"]

REG = os.path.join(_BASE, "figure_pipeline", "data", "figure_registry")
E3_BASE = os.path.join(_BASE, "outputs", "e3_swap")

WINDOW_S = 100.0
SWAP_TIME_S = 300.0
TIME_STEP = 0.25
T_START, T_END = 100.0, 600.0
LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}


def get_target(comp_ms, comm_solo_ms, ci):
    comm_budget = ci * comm_solo_ms * OVERHEAD
    if OVERLAP > 0:
        return max(comp_ms, comm_budget) + (1.0 - OVERLAP) * min(comp_ms, comm_budget)
    return comp_ms + comm_budget


def build_job_info(workload_raw):
    info = {}
    for i, (model, dp, orig_ci) in enumerate(workload_raw):
        jid = f"J{i}"
        p = MODEL_PARAMS[model]
        bpp = 2 if p.get("fp16", True) else 4
        mb = 2 * p["params"] * bpp / max(dp, 1) / (1024 * 1024)
        raw_comm = mb * 8 * 1024 * 1024 / 100e9 * 1000.0
        comp = p.get("comp_ms", 50.0)
        was_p = orig_ci <= 2.0
        post_ci = 3.0 if was_p else (1.5 if model in LARGE_MODELS or dp != 4 else 2.0)
        info[jid] = {
            "model": model, "dp": dp,
            "pre_target": get_target(comp, raw_comm, orig_ci),
            "post_target": get_target(comp, raw_comm, post_ci),
            "pre_is_premium": was_p,
            "post_is_premium": not was_p,
        }
    return info


def compute_trajectory(records, job_info):
    records.sort(key=lambda r: r["start_ms"])
    n_pts = int((T_END - T_START) / TIME_STEP) + 1
    time_grid = np.linspace(T_START, T_END, n_pts)
    results_t, results_p = [], []
    left = right = 0
    jsum = {}; jcnt = {}

    for t_s in time_grid:
        t_ms = t_s * 1000.0
        lo = max(0.0, (SWAP_TIME_S if t_s > SWAP_TIME_S else t_s - WINDOW_S) * 1000.0)
        hi = t_ms

        while left < len(records) and records[left]["start_ms"] < lo:
            r = records[left]; jid = r["jid"]
            if jid in jcnt:
                jsum[jid] -= r["iter_ms"]; jcnt[jid] -= 1
                if jcnt[jid] <= 0:
                    jsum.pop(jid, None); jcnt.pop(jid, None)
            left += 1
        while right < len(records) and records[right]["start_ms"] <= hi:
            r = records[right]; jid = r["jid"]
            jsum[jid] = jsum.get(jid, 0.0) + r["iter_ms"]
            jcnt[jid] = jcnt.get(jid, 0) + 1
            right += 1

        if t_s <= SWAP_TIME_S:
            pset = {j for j, info in job_info.items() if info["pre_is_premium"]}
            tkey = "pre_target"
        else:
            pset = {j for j, info in job_info.items() if info["post_is_premium"]}
            tkey = "post_target"

        ptot = patt = 0
        for jid in pset:
            if jid in jcnt and jcnt[jid] > 0:
                sas = job_info[jid][tkey] / (jsum[jid] / jcnt[jid])
                ptot += 1
                if sas >= 0.98:
                    patt += 1
        if ptot > 0:
            results_t.append(t_s)
            results_p.append(patt / ptot)
    return np.array(results_t), np.array(results_p)


def build_common_csv(tag, workload_raw, out_name):
    job_info = build_job_info(workload_raw)
    seed_trajs = []
    for s in range(5):
        path = os.path.join(E3_BASE, f"{tag}_DF_s{s}", "records.jsonl")
        if not os.path.exists(path):
            print(f"  WARN: missing {path}")
            continue
        recs = [json.loads(l) for l in open(path) if l.strip()]
        t, p = compute_trajectory(recs, job_info)
        seed_trajs.append((t, p))
    if not seed_trajs:
        print(f"  SKIP {tag}: no data")
        return

    all_t = sorted(set(t for t_arr, _ in seed_trajs for t in t_arr))
    common = np.array(all_t)
    interp = np.array([np.interp(common, t_arr, p_arr) for t_arr, p_arr in seed_trajs])
    mean = np.mean(interp, axis=0)
    std = np.std(interp, axis=0)

    path = os.path.join(REG, out_name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "mean", "std"] + [f"seed{i}" for i in range(5)])
        for i, t in enumerate(common):
            row = [f"{t:.2f}", f"{mean[i]:.6f}", f"{std[i]:.6f}"]
            row += [f"{interp[s, i]:.6f}" for s in range(5)]
            w.writerow(row)
    print(f"  -> {path} ({len(common)} points)")


if __name__ == "__main__":
    os.makedirs(REG, exist_ok=True)
    workload = FEAS_BOUNDARY_V3_WORKLOAD
    build_common_csv("e3_swap", workload, "fig4_d1_trajectory_e3.csv")
    workload = FEAS_BOUNDARY_V3_PRO_WORKLOAD
    build_common_csv("e3p_swap", workload, "fig4_d1_trajectory_e3p.csv")
    print("Done.")
