"""
生成 figure_registry 权威 CSV（7 策略 / 10-seed 版本）+ 备份实验数据到 figure_pipeline/data。

用途：在 P0-P2 实验全部完成后运行一次：
    python3 figure_pipeline/scripts/_make_registry_csvs.py

产出：
- figure_pipeline/data/figure_registry/fig2_e1_ladder_10seed.csv
- figure_pipeline/data/figure_registry/fig3_e2_ladder_10seed.csv
- figure_pipeline/data/figure_registry/fig6_trace_compare.csv（含每 seed P-attn）
- 备份 outputs/{e3_swap,e11_overlap,e15_straggler,trace_replay} → figure_pipeline/data/
- 备份 anchor per_policy_results.json → figure_pipeline/data/anchor/
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJ, "outputs")
REG = os.path.join(PROJ, "figure_pipeline", "data", "figure_registry")
os.makedirs(REG, exist_ok=True)

MAIN_DIR = os.path.join(OUT, "v3_batch3_formal")
E3_DIR = os.path.join(OUT, "e3_swap")
E11_DIR = os.path.join(OUT, "e11_overlap")
E15_DIR = os.path.join(OUT, "e15_straggler")
TRACE_DIR = os.path.join(OUT, "trace_replay")
ANCHOR_OUT = os.path.join(OUT, "anchor_regen_v1")

# figure_pipeline/data 目标
PIPE_DATA = os.path.join(PROJ, "figure_pipeline", "data")
PIPE_E3 = os.path.join(PIPE_DATA, "e3_swap")
PIPE_E11 = os.path.join(PIPE_DATA, "e11_overlap")
PIPE_E15 = os.path.join(PIPE_DATA, "e15_straggler")
PIPE_TRACE = os.path.join(PIPE_DATA, "trace_replay")
PIPE_ANCHOR = os.path.join(PIPE_DATA, "anchor")

POLICIES = ["Fair", "SRPT", "CRUX", "CASSINI", "DF", "LL-S", "LongLiu"]


def load_main_metas():
    """遍历 outputs/v3_batch3_formal/*/run_meta.json。

    仅保留 2026-08-24 重新运行的结果（旧 2026-07-27 残留按 timestamp 过滤，
    避免 D1/v4/SP 旧名目录与旧 Fair/CRUX 混入 CSV）。
    """
    metas = []
    for p in glob.glob(os.path.join(MAIN_DIR, "*", "run_meta.json")):
        with open(p) as f:
            m = json.load(f)
        if not m.get("timestamp", "").startswith("2026-08-24"):
            continue
        if m.get("policy") not in POLICIES:
            continue
        metas.append(m)
    return metas


def agg(rows):
    """rows: list of dicts -> (mean, std) 各列, ddof=1（sample std，与 summary.csv 一致）。"""
    out = {}
    keys = list(rows[0].keys())
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1))
    return out


def write_ladder_csv(scene_names, fname):
    metas = load_main_metas()
    rows_by_key = {}
    for m in metas:
        sc = m["scene"]
        if sc not in scene_names:
            continue
        key = (sc, int(m["spine_bw"]), m["policy"])
        rows_by_key.setdefault(key, []).append(m)

    path = os.path.join(REG, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # 先收集所有 seed 列
        all_seeds = sorted({m["seed"] for m in metas if m["scene"] in scene_names})
        header = ["scene", "spine_bw", "policy", "n_seeds", "seeds",
                  "p_attn_mean", "p_attn_std", "p_cap_mean", "s_cont_cap_mean"]
        header += [f"p_attn_s{s}" for s in all_seeds]
        w.writerow(header)
        for key in sorted(rows_by_key):
            rows = sorted(rows_by_key[key], key=lambda r: r["seed"])
            sc, bw, pol = key
            p_attns = [r["p_attn"] for r in rows]
            p_caps = [r["p_cap"] for r in rows]
            s_caps = [r["s_cont_cap"] for r in rows]
            row = [sc, bw, pol, len(rows), str([r["seed"] for r in rows]),
                   f"{np.mean(p_attns):.4f}", f"{np.std(p_attns, ddof=1):.4f}",
                   f"{np.mean(p_caps):.4f}", f"{np.mean(s_caps):.4f}"]
            row += [f"{v:.4f}" for v in p_attns]
            w.writerow(row)
    print(f"  -> {path} ({len(rows_by_key)} rows)")
    return path


def write_trace_csv():
    """从 outputs/trace_replay/run_meta_*.json 生成 fig6_trace_compare.csv。"""
    metas = {}
    for p in glob.glob(os.path.join(TRACE_DIR, "run_meta_*.json")):
        base = os.path.basename(p)
        # run_meta_<POLICY>_s<SEED>.json
        rest = base[len("run_meta_"):-len(".json")]
        pol, _, seed = rest.rpartition("_s")
        metas.setdefault(pol, {})[int(seed)] = json.load(open(p))

    path = os.path.join(REG, "fig6_trace_compare.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        all_seeds = sorted({s for d in metas.values() for s in d})
        header = ["scene", "spine_bw", "policy", "n_seeds", "seeds",
                  "p_attn_mean", "p_attn_std"] + [f"p_attn_s{s}" for s in all_seeds]
        w.writerow(header)
        for pol in POLICIES:
            if pol not in metas:
                print(f"  WARN: trace missing {pol}")
                continue
            d = metas[pol]
            seeds = sorted(d)
            p_attns = [d[s]["p_attn"] for s in seeds]
            w.writerow(["TRACE", "-", pol, len(seeds), str(seeds),
                        f"{np.mean(p_attns):.4f}", f"{np.std(p_attns, ddof=1):.4f}"]
                       + [f"{v:.4f}" for v in p_attns])
    print(f"  -> {path}")
    return path


def backup(src_dir, dst_dir, patterns):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for pat in patterns:
        for p in glob.glob(os.path.join(src_dir, pat)):
            rel = os.path.relpath(p, src_dir)
            dest = os.path.join(dst_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.isdir(p):
                if not os.path.exists(dest):
                    shutil.copytree(p, dest)
                    n += 1
            else:
                shutil.copy2(p, dest)
                n += 1
    print(f"  backup {os.path.basename(src_dir)} -> {os.path.basename(dst_dir)} ({n} items)")


def main():
    print("=" * 60)
    print("生成 figure_registry CSV + 备份实验数据")
    print("=" * 60)

    # 1) 主表 ladder CSV（E1 / E2' / E2-pro）
    write_ladder_csv(["E1"], "fig2_e1_ladder_10seed.csv")
    write_ladder_csv(["E2'", "E2-pro"], "fig3_e2_ladder_10seed.csv")

    # 2) trace compare CSV
    write_trace_csv()

    # 3) 备份实验数据到 figure_pipeline/data/
    backup(E3_DIR, PIPE_E3,
           ["e3_swap_*", "e3p_swap_*", "summary_*.json", "e3_swap_5seed_v2.log"])
    backup(E11_DIR, PIPE_E11, ["summary.csv", "e11_5seed_v2.log"])
    backup(E15_DIR, PIPE_E15, ["summary.csv", "e15_5seed_v2.log"])
    backup(TRACE_DIR, PIPE_TRACE, ["trace_replay_summary.csv", "run_meta_*.json",
                                   "trace_replay_v2.log"])
    # anchor: per_policy_results.json + run_meta.json
    backup(ANCHOR_OUT, PIPE_ANCHOR, ["per_policy_results.json", "run_meta.json"])

    print("\nDone.")


if __name__ == "__main__":
    main()
