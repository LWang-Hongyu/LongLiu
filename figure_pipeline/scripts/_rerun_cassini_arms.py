"""重跑受 comm_offset_ms 重复应用 bug 影响的 CASSINI 臂（fig2/fig3/E11/E15/E17/S3）。

背景：simulator._create_collective_flows 此前在每轮迭代都重复应用 comm_offset_ms，
导致 CASSINI 的相位偏移被逐轮累加平移。修复后 offset 仅在首轮 collective flow
发射时应用（job._comm_offset_pending 标志）。

隔离验证：30-seed trace replay 重跑显示 7 策略中仅 CASSINI 一行变化，其余 6 策略
run_meta 逐字节一致 → 其他实验只需重跑 CASSINI 臂（s3 另含 LongLiu_Staggered）。

本脚本复用原实验脚本的 run_single / 配置循环，仅重跑受影响臂：
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py backup      # 备份旧数据
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py e11         # 50 runs
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py e15         # 20 runs
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py e17         # 40 runs
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py batch3      # 120 runs
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py s3          # 20 runs
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py quarantine  # 隔离残留目录
    python3 figure_pipeline/scripts/_rerun_cassini_arms.py rebuild     # 重建 summary

必须在 sim-nextgen/ 根目录下运行（run_single 使用相对路径 outputs/...）。
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # sim-nextgen/
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "outputs")
BK = os.path.join(OUT, "cassini_offset_fix_backup")

E11_E15_SEEDS = [0, 1, 2, 3, 4]


def _config_hash() -> str:
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return hashlib.md5(f.read().encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════════
# backup：重跑前归档受影响臂的旧数据（仅小 JSON/CSV，不含 trace）
# ═══════════════════════════════════════════════════════════════

def step_backup() -> None:
    n = 0

    def cp(src: str, dst: str) -> None:
        nonlocal n
        if not os.path.exists(src):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        n += 1

    # 聚合产物
    cp(f"{OUT}/e11_overlap/summary.csv", f"{BK}/e11_overlap/summary.csv")
    cp(f"{OUT}/e15_straggler/summary.csv", f"{BK}/e15_straggler/summary.csv")
    cp(f"{OUT}/e17_mixed_collective/e17_summary.csv",
       f"{BK}/e17_mixed_collective/e17_summary.csv")
    cp(f"{OUT}/s3_component_ablation/s3_component_ablation.csv",
       f"{BK}/s3_component_ablation/s3_component_ablation.csv")
    cp(f"{OUT}/s3_component_ablation/raw_results.json",
       f"{BK}/s3_component_ablation/raw_results.json")
    cp(f"{OUT}/v3_batch3_formal/summary.json",
       f"{BK}/v3_batch3_formal/summary.json")

    # 受影响 run 目录的 run_meta.json（保留目录名）
    for sub, pat in [("e11_overlap", "E1_CASSINI_*"),
                     ("e15_straggler", "sf*_CASSINI_*"),
                     ("e17_mixed_collective", "E17_CASSINI_*"),
                     ("v3_batch3_formal", "*_CASSINI_*")]:
        for d in sorted(glob.glob(os.path.join(OUT, sub, pat))):
            cp(os.path.join(d, "run_meta.json"),
               os.path.join(BK, sub, os.path.basename(d), "run_meta.json"))

    print(f"backup -> {BK} ({n} files)")


# ═══════════════════════════════════════════════════════════════
# e11 / e15：复用 exp_e11_overlap / exp_e15_straggler.run_single
# ═══════════════════════════════════════════════════════════════

def step_e11() -> None:
    from experiments import exp_e11_overlap as e11
    from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V3_WORKLOAD

    e11.CONFIG_HASH = _config_hash()
    cfg = e11.load_e11_config()
    frozen = e11.load_frozen()

    t0 = time.time()
    done = 0
    for overlap in cfg["overlap_factors"]:
        for bw in cfg["spine_bw_gbps"]:
            for seed in E11_E15_SEEDS:
                r = e11.run_single("E1", FEAS_BOUNDARY_V3_WORKLOAD, bw,
                                   overlap, "CASSINI", seed, frozen)
                done += 1
                print(f"[e11 {done}/50] ov={overlap} @{bw}G s{seed} "
                      f"P-attn={r['p_attn']*100:.1f}%", flush=True)
    print(f"*** E11 CASSINI done: {done} runs in {time.time()-t0:.0f}s ***")


def step_e15() -> None:
    from experiments import exp_e15_straggler as e15
    from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V3_WORKLOAD

    e15.CONFIG_HASH = _config_hash()
    cfg = e15.load_e15_config()
    frozen = e15.load_frozen()
    spine_bw = cfg["topology"]["spine_bw_bps"] / 1e9

    t0 = time.time()
    done = 0
    total = len(cfg["straggler_factors"]) * len(E11_E15_SEEDS)
    for sf in cfg["straggler_factors"]:
        for seed in E11_E15_SEEDS:
            r = e15.run_single(sf, FEAS_BOUNDARY_V3_WORKLOAD, spine_bw,
                               "CASSINI", seed, frozen)
            done += 1
            print(f"[e15 {done}/{total}] sf={sf} s{seed} "
                  f"P-attn={r['p_attn']*100:.1f}%", flush=True)
    print(f"*** E15 CASSINI done: {done} runs in {time.time()-t0:.0f}s ***")


# ═══════════════════════════════════════════════════════════════
# e17 / batch3：复用 exp_e17_mixed_collective / exp_v3_batch3_formal.run_single
# ═══════════════════════════════════════════════════════════════

def step_e17() -> None:
    from experiments import exp_e17_mixed_collective as e17

    t0 = time.time()
    done = 0
    total = len(e17.SPINES) * len(e17.SEEDS)
    for spine in e17.SPINES:
        for seed in e17.SEEDS:
            r = e17.run_single(spine, "CASSINI", seed)
            done += 1
            print(f"[e17 {done}/{total}] @{spine}G s{seed} "
                  f"P-attn={r['p_attn']*100:.1f}%", flush=True)
    print(f"*** E17 CASSINI done: {done} runs in {time.time()-t0:.0f}s ***")


def step_batch3() -> None:
    from experiments import exp_v3_batch3_formal as b3

    n_configs = sum(len(pts) for _, _, pts in b3.SCENARIOS)
    total = n_configs * len(b3.SEEDS)

    t0 = time.time()
    done = 0
    for scene, workload, spine_pts in b3.SCENARIOS:
        for bw in spine_pts:
            for seed in b3.SEEDS:
                r = b3.run_single(scene, workload, bw, "CASSINI", seed)
                done += 1
                print(f"[batch3 {done}/{total}] {scene} @{bw}G s{seed} "
                      f"P-attn={r['p_attn']*100:.1f}%", flush=True)
    print(f"*** BATCH3 CASSINI done: {done} runs in {time.time()-t0:.0f}s ***")


# ═══════════════════════════════════════════════════════════════
# s3：重跑 CASSINI + LongLiu_Staggered 两臂（复用 exp_s3 的构造与指标块）
# ═══════════════════════════════════════════════════════════════

def step_s3() -> None:
    import experiments.exp_s3_component_ablation as s3
    from experiments.exp_ablation import (
        load_config, build_topology, generate_jobs,
        compute_fairness_metrics, _validate_tiers, run_experiment,
        _HAS_SCIPY,
    )
    from longliu_sim.core import Simulator
    from longliu_sim.policy import CASSINI

    cfg_path = os.path.join(ROOT, "configs", "fatree_16host.yaml")
    cfg = load_config(cfg_path)
    overhead, overlap = 1.3, 0.85  # 脚本默认参数
    seeds = list(range(10))
    longliu_kwargs = {"K": 2.0, "use_dynamic_T_target": True}

    policies = {
        "CASSINI": CASSINI(),
        "LongLiu_Staggered": s3.LongLiuStaggered(**longliu_kwargs),
    }

    raw_path = os.path.join(OUT, "s3_component_ablation", "raw_results.json")
    with open(raw_path) as f:
        raw = json.load(f)

    actual_tiers = _validate_tiers(s3.DEFAULT_TIERED_WORKLOAD)
    t0 = time.time()
    done = 0
    for name, policy in policies.items():
        new_results = []
        for seed in seeds:
            jobs = generate_jobs(cfg, seed, overhead)
            s3.apply_cassini_offsets(jobs)
            topo = build_topology(cfg)
            sim = Simulator(topo, policy, duration_ms=cfg["duration_ms"],
                            seed=seed, overhead_factor=overhead,
                            overlap_factor=overlap)
            for j in jobs:
                sim.submit(j)
            result = sim.run()
            host_bw_gbps = cfg["topology"].get("host_bw_bps", 100e9) / 1e9
            stats = result.per_job_stats(host_bw_gbps=host_bw_gbps)

            tier_meets = {c: [] for c in actual_tiers}
            tier_sas = {c: [] for c in actual_tiers}
            all_sas = []
            for jid, st in stats.items():
                ci = sim.jobs[jid].slo_ci
                tier_meets[ci].append(st["meets_slo"])
                tier_sas[ci].append(st["sas"])
                all_sas.append(st["sas"])

            r = {
                "total_iters": result.total_iterations(),
                "slo_attainment_overall":
                    sum(1 for st in stats.values() if st["meets_slo"]) / len(stats)
                    if stats else 0.0,
                "sas_mean_overall":
                    sum(all_sas) / len(all_sas) if all_sas else 0.0,
            }
            for ci in actual_tiers:
                r[f"slo_attainment_ci{ci}"] = (
                    sum(tier_meets[ci]) / len(tier_meets[ci]) if tier_meets[ci] else 0.0)
                r[f"sas_mean_ci{ci}"] = (
                    sum(tier_sas[ci]) / len(tier_sas[ci]) if tier_sas[ci] else 0.0)
            r.update(compute_fairness_metrics(all_sas))

            r["per_job"] = []
            for jid, st in stats.items():
                job = sim.jobs[jid]
                r["per_job"].append({
                    "jid": jid, "model": job.model, "dp": job.num_workers,
                    "ci": job.slo_ci, "iter_solo_ms": job.iter_solo_ms,
                    "comp_ms": job.comp_ms, "comm_solo_ms": job.comm_solo_ms,
                    "avg_iter_ms": st["avg_iter_ms"],
                    "avg_comm_ms": st["avg_comm_ms"],
                    "target_iter_ms": st["target_iter_ms"],
                    "meets_slo": st["meets_slo"], "sas": st["sas"],
                })

            new_results.append(r)
            done += 1
            print(f"[s3 {done}/20] {name} s{seed} "
                  f"SAS={r['sas_mean_overall']:.3f} "
                  f"SLO={r['slo_attainment_overall']*100:.1f}%", flush=True)
        raw[name] = new_results

    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"raw_results.json updated ({time.time()-t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════
# quarantine：隔离 outputs 中的历史残留 run 目录
# ═══════════════════════════════════════════════════════════════
# e11/e15 磁盘上混有早期脚本迭代遗留的 D1/SP/v4 目录（e12/e11b 实验别名）
# 以及 e15 的 spine=800 旧时代目录（当前 e15 固定 400G）。原 exp main()
# 聚合时按 cfg["policies"]（7 策略，见 configs/e1*.yaml）过滤，这些残留
# 从不进入 summary；此处物理隔离以免未来误吸收。

def step_quarantine() -> None:
    import yaml

    with open(os.path.join(ROOT, "configs", "e11_overlap.yaml")) as f:
        e11_keep = set(yaml.safe_load(f)["policies"])
    with open(os.path.join(ROOT, "configs", "e15_straggler.yaml")) as f:
        e15_keep = set(yaml.safe_load(f)["policies"])

    def is_stale(m: dict, sub: str) -> bool:
        keep = e11_keep if sub == "e11_overlap" else e15_keep
        if m.get("policy") not in keep:
            return True
        if sub == "e15_straggler" and m.get("spine_bw") != 400:
            return True  # 当前 e15 固定 400G
        return False

    qroot = os.path.join(OUT, "_quarantine_stale")
    n = 0
    for sub in ["e11_overlap", "e15_straggler"]:
        for d in sorted(glob.glob(os.path.join(OUT, sub, "*"))):
            if not os.path.isdir(d):
                continue
            meta = os.path.join(d, "run_meta.json")
            stale = True
            if os.path.exists(meta):
                with open(meta) as f:
                    stale = is_stale(json.load(f), sub)
            if stale:
                dst = os.path.join(qroot, sub, os.path.basename(d))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(d, dst)
                n += 1
    print(f"quarantine -> {qroot} ({n} dirs moved)")


# ═══════════════════════════════════════════════════════════════
# rebuild：从磁盘 run_meta / raw_results 重建各 summary
# ═══════════════════════════════════════════════════════════════

def _load_metas(sub: str) -> list[dict]:
    metas = []
    for p in sorted(glob.glob(os.path.join(OUT, sub, "*", "run_meta.json"))):
        with open(p) as f:
            metas.append(json.load(f))
    return metas


def _compare_old(new_path: str, old_path: str, key_cols: list[int],
                 skip_rows: set[str], row_policy_col: int) -> None:
    """非 CASSINI 行应与旧文件逐字节一致（完整性自检）。"""
    if not os.path.exists(old_path):
        print(f"  [check] {old_path} 不存在，跳过")
        return
    with open(new_path) as f:
        new_rows = list(csv.reader(f))
    with open(old_path) as f:
        old_rows = list(csv.reader(f))
    old_map = {}
    for r in old_rows[1:]:
        old_map[tuple(r[i] for i in key_cols)] = r
    diffs = 0
    for r in new_rows[1:]:
        pol = r[row_policy_col]
        if pol in skip_rows:
            continue
        k = tuple(r[i] for i in key_cols)
        if k not in old_map:
            diffs += 1
            print(f"  [check] MISSING in old: {k}")
        elif old_map[k] != r:
            diffs += 1
            print(f"  [check] DIFF {k}:\n    old={old_map[k]}\n    new={r}")
    print(f"  [check] {os.path.basename(new_path)}: "
          f"{'OK — 非 CASSINI 行与旧数据一致' if diffs == 0 else f'{diffs} 处不一致!'}")


def step_rebuild() -> None:
    import numpy as np
    import yaml
    from experiments import exp_v3_batch3_formal as b3

    with open(os.path.join(ROOT, "configs", "e11_overlap.yaml")) as f:
        e11_policies = set(yaml.safe_load(f)["policies"])
    with open(os.path.join(ROOT, "configs", "e15_straggler.yaml")) as f:
        e15_policies = set(yaml.safe_load(f)["policies"])

    # ---- e11 summary.csv（np.std 即 ddof=0，与原 main() 一致）----
    groups = {}
    for m in _load_metas("e11_overlap"):
        if m["policy"] not in e11_policies:
            continue
        groups.setdefault((m["spine_bw"], m["overlap_factor"], m["policy"]),
                          []).append(m)
    path = f"{OUT}/e11_overlap/summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spine_bw", "overlap_factor", "policy", "n_seeds",
                    "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean"])
        for k in sorted(groups):
            runs = groups[k]
            w.writerow([k[0], k[1], k[2], len(runs),
                        round(float(np.mean([r["p_attn"] for r in runs])), 4),
                        round(float(np.std([r["p_attn"] for r in runs])), 4),
                        round(float(np.mean([r["p_cap"] for r in runs])), 4),
                        round(float(np.mean([r["s_cont_cap"] for r in runs])), 4)])
    print(f"rebuild -> {path}")
    _compare_old(path, f"{BK}/e11_overlap/summary.csv",
                 key_cols=[0, 1, 2], skip_rows={"CASSINI"}, row_policy_col=2)

    # ---- e15 summary.csv ----
    groups = {}
    for m in _load_metas("e15_straggler"):
        if m["policy"] not in e15_policies:
            continue
        groups.setdefault((m["straggler_factor"], m["policy"]), []).append(m)
    path = f"{OUT}/e15_straggler/summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["straggler_factor", "policy", "n_seeds", "seeds",
                    "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean"])
        for k in sorted(groups):
            runs = sorted(groups[k], key=lambda r: r["seed"])
            w.writerow([k[0], k[1], len(runs),
                        f"[{', '.join(str(r['seed']) for r in runs)}]",
                        round(float(np.mean([r["p_attn"] for r in runs])), 4),
                        round(float(np.std([r["p_attn"] for r in runs])), 4),
                        round(float(np.mean([r["p_cap"] for r in runs])), 4),
                        round(float(np.mean([r["s_cont_cap"] for r in runs])), 4)])
    print(f"rebuild -> {path}")
    _compare_old(path, f"{BK}/e15_straggler/summary.csv",
                 key_cols=[0, 1], skip_rows={"CASSINI"}, row_policy_col=1)

    # ---- e17_summary.csv（按 SPINES × POLICIES × SEEDS 原序）----
    from experiments import exp_e17_mixed_collective as e17
    metas = _load_metas("e17_mixed_collective")
    order_p = {p: i for i, p in enumerate(e17.POLICIES)}
    order_s = {s: i for i, s in enumerate(e17.SPINES)}
    metas.sort(key=lambda m: (order_s[m["spine_bw"]], order_p[m["policy"]],
                              m["seed"]))
    path = f"{OUT}/e17_mixed_collective/e17_summary.csv"
    cols = ["scene", "spine_bw", "policy", "seed",
            "p_attn", "p_cap", "s_cont_cap", "starv", "total_iters"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for m in metas:
            f.write(",".join(str(m[c]) for c in cols) + "\n")
    print(f"rebuild -> {path} ({len(metas)} rows)")
    _compare_old(path, f"{BK}/e17_mixed_collective/e17_summary.csv",
                 key_cols=[1, 2, 3], skip_rows={"CASSINI"}, row_policy_col=2)

    # ---- fig7_e17_mixed.csv（无 header，std 为 ddof=1，与旧文件约定一致）----
    reg = os.path.join(ROOT, "figure_pipeline", "data", "figure_registry")
    groups = {}
    for m in metas:
        groups.setdefault((m["spine_bw"], m["policy"]), []).append(m["p_attn"])
    path = os.path.join(reg, "fig7_e17_mixed.csv")
    draw_order = ["LongLiu", "LL-S", "DF", "SRPT", "CRUX", "CASSINI", "Fair"]
    with open(path, "w") as f:
        for bw in e17.SPINES:
            for pol in draw_order:
                v = groups.get((bw, pol))
                if not v:
                    continue
                seeds_str = "|".join(str(s) for s in e17.SEEDS)
                f.write(f"E17,{bw},{pol},10,{seeds_str},"
                        f"{np.mean(v):.4f},{np.std(v, ddof=1):.4f}\n")
    print(f"rebuild -> {path}")
    _compare_old(path, f"{BK}/e17_mixed_collective/fig7_e17_mixed_old.csv",
                 key_cols=[1, 2], skip_rows={"CASSINI"}, row_policy_col=2)

    # ---- batch3 summary.json（从全量 run_meta 聚合）----
    all_runs = [m for m in _load_metas("v3_batch3_formal")
                if m.get("policy") in b3.POLICIES]
    agg = b3.aggregate(all_runs)
    serializable = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in agg.items()}
    path = f"{OUT}/v3_batch3_formal/summary.json"
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"rebuild -> {path} ({len(serializable)} groups)")
    failures = b3.verify(agg)
    if failures:
        for msg in failures:
            print(f"  [verify] FAIL: {msg}")
    else:
        print("  [verify] All checks passed (v4 guarantee + P1a + P1b + P2)")

    # ---- s3_component_ablation.csv（从 raw_results.json 重算）----
    _rebuild_s3_csv()


def _rebuild_s3_csv() -> None:
    import numpy as np
    import experiments.exp_s3_component_ablation as s3

    raw_path = f"{OUT}/s3_component_ablation/raw_results.json"
    with open(raw_path) as f:
        results = json.load(f)

    first = next(iter(results))
    actual_tiers = sorted({float(k.split("ci")[1])
                           for k in results[first][0]
                           if k.startswith("slo_attainment_ci")})
    metrics = (["total_iters"]
               + [f"slo_attainment_ci{ci}" for ci in actual_tiers]
               + ["slo_attainment_overall"])
    sas_metrics = ["sas_mean_overall"] + [f"sas_mean_ci{ci}" for ci in actual_tiers]
    fairness_metrics = ["jain_index", "gini_coeff", "handoff_rate"]
    all_metric_names = metrics + sas_metrics + fairness_metrics

    order = ["Fair", "CRUX", "LongLiu", "LongLiu_StaticT",
             "CASSINI", "LongLiu_Staggered"]

    # paired t-test vs LongLiu（与原脚本一致）
    p_vals = {}
    try:
        from scipy import stats as scipy_stats
        base = [r["sas_mean_overall"] for r in results["LongLiu"]]
        for name in results:
            if name == "LongLiu":
                continue
            tgt = [r["sas_mean_overall"] for r in results[name]]
            if len(tgt) == len(base) and len(base) >= 2:
                _, p = scipy_stats.ttest_rel(tgt, base)
                p_vals[name] = p
    except Exception as e:
        print(f"  [s3] t-test failed: {e}")

    path = f"{OUT}/s3_component_ablation/s3_component_ablation.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Policy"] + [f"{m}(mean±std)" for m in all_metric_names]
                   + ["p_vs_longliu"])
        for name in order:
            rows = results[name]
            row = [name]
            for m in all_metric_names:
                vals = [r[m] for r in rows]
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals)
                       / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
                row.append(f"{mean:.4f}±{std:.4f}")
            p = p_vals.get(name)
            row.append(f"{p:.4e}" if isinstance(p, float) else "")
            w.writerow(row)
    print(f"rebuild -> {path}")
    _compare_old(path, f"{BK}/s3_component_ablation/s3_component_ablation.csv",
                 key_cols=[0], skip_rows={"CASSINI", "LongLiu_Staggered"},
                 row_policy_col=0)


STEPS = {
    "backup": step_backup,
    "e11": step_e11,
    "e15": step_e15,
    "e17": step_e17,
    "batch3": step_batch3,
    "s3": step_s3,
    "quarantine": step_quarantine,
    "rebuild": step_rebuild,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(__doc__)
        sys.exit(1)
    STEPS[sys.argv[1]]()
