"""复现门禁：两段检查。

Phase 1 — 代码同一性：
    对 longliu_sim/core/ longliu_sim/policy/ longliu_sim/job/
    longliu_sim/network/ longliu_sim/trace/ 下所有 .py 文件计算 SHA256，
    与锚点哈希表比对，差异列表即为"代码漂移"清单。

Phase 2 — 结果复现：
    使用锚点 workload profile + 同 seed 同配置重跑仿真，
    逐 job SAS 对比锚点值，容差 0（bit-exact 断言）。

首次运行（--init）时建立锚点；后续运行（默认）时对比锚点。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# 添加项目根目录到 path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from longliu_sim.utils.config import (
    load_config, config_hash, get_topology,
    get_simulation, get_v2_anchor_workload,
)


# ── 配置（从 config.yaml 读取，禁止硬编码）──

ANCHOR_DIR = os.path.join(_project_root, "outputs", "gatekeeper")
CODE_ANCHOR_FILE = os.path.join(ANCHOR_DIR, "code_anchor.json")
RESULT_ANCHOR_FILE = os.path.join(ANCHOR_DIR, "result_anchor.json")

# Phase 1：纳入哈希检查的源码目录
CODE_DIRS = [
    "longliu_sim/core",
    "longliu_sim/policy",
    "longliu_sim/job",
    "longliu_sim/network",
    "longliu_sim/trace",
    "longliu_sim/utils",
]

# Phase 2 配置源：config.yaml
_cfg = load_config()
_ANCHOR_CONFIG_CACHE = None


def _get_anchor_config():
    """从 config.yaml 构建 Phase 2 仿真配置（缓存）。"""
    global _ANCHOR_CONFIG_CACHE
    if _ANCHOR_CONFIG_CACHE is not None:
        return _ANCHOR_CONFIG_CACHE
    _cfg = load_config()
    frozen = _cfg["frozen"]
    topo = get_topology()
    sim = get_simulation()
    _ANCHOR_CONFIG_CACHE = {
        "topology": {
            "type": topo.get("type", "fatree"),
            "k": topo["k"],
            "host_bw_bps": topo["host_bw_bps"],
            "spine_bw_bps": topo["spine_bw_bps"],
        },
        "duration_ms": sim["duration_ms"],
        "overhead_factor": frozen["overhead_factor"],
        "overlap_factor": frozen["overlap_factor"],
        "K": frozen["K"],
        "seeds": [0, 1, 2],  # 三 seed 锚点（匹配 anchor-regen-v1）
    }
    return _ANCHOR_CONFIG_CACHE


# Phase 2：锚点 workload（从 config.yaml 读取，冻结不变）
def _get_v2_anchor_workload():
    """从 config.yaml 读取 v2 锚点 workload，转换为 (model, dp, ci) 元组列表。"""
    raw = get_v2_anchor_workload()
    if not raw:
        raise RuntimeError("config.yaml 中 v2_anchor_workload 为空")
    return [tuple(item) for item in raw]


# ── Phase 1: Code Identity ────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_code_hashes() -> dict:
    """遍历 CODE_DIRS 下所有 .py 文件，返回 {relpath: sha256}。"""
    hashes = {}
    for dir_rel in CODE_DIRS:
        dir_abs = os.path.join(_project_root, dir_rel)
        if not os.path.isdir(dir_abs):
            print(f"[WARN] 目录不存在，跳过: {dir_rel}")
            continue
        for root, _, files in os.walk(dir_abs):
            for fname in sorted(files):
                if fname.endswith(".py") and "__pycache__" not in root:
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, _project_root)
                    hashes[rel] = compute_file_hash(fpath)
    return hashes


def check_code_identity() -> dict:
    """Phase 1: 对比当前代码哈希与锚点。

    返回 {"pass": bool, "added": [...], "removed": [...], "changed": [...]}
    """
    if not os.path.exists(CODE_ANCHOR_FILE):
        return {"pass": False, "error": "锚点文件不存在，请先 --init"}

    with open(CODE_ANCHOR_FILE) as f:
        anchor = json.load(f)

    current = build_code_hashes()

    added = sorted(set(current) - set(anchor))
    removed = sorted(set(anchor) - set(current))
    changed = [k for k in (set(current) & set(anchor)) if current[k] != anchor[k]]

    return {
        "pass": not (added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": changed,
        "anchor_files": len(anchor),
        "current_files": len(current),
    }


# ── Phase 2: Result Reproduction ───────────────────────

def run_seed(cfg: dict, seed: int):
    """运行单个 seed 的仿真并返回 per-job SAS。"""
    from longliu_sim.policy import LongLiu
    from longliu_sim.core import Simulator
    from longliu_sim.network import FatTreeTopology
    from longliu_sim.trace import SyntheticTraceLoader

    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
    policy = LongLiu(K=cfg["K"], use_dynamic_T_target=True)
    sim = Simulator(
        topo,
        policy,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        overlap_factor=cfg["overlap_factor"],
    )

    loader = SyntheticTraceLoader(
        model_types=[
            "ResNet-18", "ResNet-50-fp16", "BERT-Base", "BERT-Large-fp16",
            "ViT-Base", "ViT-Large", "LLaMA-2-1B", "LLaMA-2-7B", "T5-1B",
        ],
        gpu_distribution={1: 0.2, 2: 0.2, 4: 0.3, 8: 0.3},
        ci_distribution={1.5: 0.3, 2.0: 0.35, 3.0: 0.35},
        job_count=24,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=_get_v2_anchor_workload(),
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    stats = result.per_job_stats()

    per_job = {}
    for jid, s in stats.items():
        per_job[jid] = {
            "sas": s["sas"],
            "avg_iter_ms": s["avg_iter_ms"],
            "meets_slo": s["meets_slo"],
        }

    return {
        "seed": seed,
        "per_job": per_job,
        "mean_sas": sum(s["sas"] for s in stats.values()) / len(stats) if stats else 0.0,
        "slo_rate": sum(1 for s in stats.values() if s["meets_slo"]) / len(stats) if stats else 0.0,
    }


def check_result_reproduction(tolerance: float = 0.0) -> dict:
    """Phase 2: 同 seed 重跑并对比锚点 SAS 值。

    返回 {"pass": bool, "seed_results": {...}}
    """
    if not os.path.exists(RESULT_ANCHOR_FILE):
        return {"pass": False, "error": "锚点文件不存在，请先 --init"}

    with open(RESULT_ANCHOR_FILE) as f:
        anchor = json.load(f)

    all_pass = True
    seed_results = {}

    anchor_cfg = _get_anchor_config()
    for seed in anchor_cfg["seeds"]:
        print(f"  [Phase 2] 运行 seed={seed}...")
        current = run_seed(anchor_cfg, seed)

        seed_key = str(seed)
        if seed_key not in anchor:
            print(f"  [FAIL] seed={seed} 锚点数据缺失")
            all_pass = False
            continue

        anchor_seed = anchor[seed_key]
        mismatches = []

        for jid, a in anchor_seed["per_job"].items():
            if jid not in current["per_job"]:
                mismatches.append(f"{jid}: 锚点存在但当前结果缺失")
                continue
            c = current["per_job"][jid]
            if abs(a["sas"] - c["sas"]) > tolerance:
                mismatches.append(
                    f"{jid}: anchor SAS={a['sas']:.6f} vs current={c['sas']:.6f} (delta={abs(a['sas']-c['sas']):.2e})"
                )

        seed_pass = len(mismatches) == 0
        seed_results[seed_key] = {
            "pass": seed_pass,
            "mismatches": mismatches,
            "anchor_mean_sas": anchor_seed.get("mean_sas"),
            "current_mean_sas": current["mean_sas"],
        }

        if not seed_pass:
            print(f"  [FAIL] seed={seed}: {len(mismatches)} mismatches")
            for m in mismatches:
                print(f"    {m}")
            all_pass = False
        else:
            print(f"  [PASS] seed={seed}: mean SAS={current['mean_sas']:.6f}")

    return {
        "pass": all_pass,
        "seed_results": seed_results,
        "tolerance": tolerance,
    }


# ── Init: 建立锚点 ─────────────────────────────────────

def init_anchor():
    """首次建立锚点：保存当前代码哈希 + 仿真结果。"""
    os.makedirs(ANCHOR_DIR, exist_ok=True)

    anchor_cfg = _get_anchor_config()
    _cfg = load_config()

    # Phase 1 锚点
    code_hashes = build_code_hashes()
    with open(CODE_ANCHOR_FILE, "w") as f:
        json.dump(code_hashes, f, indent=2, sort_keys=True)
    print(f"[INIT] Phase 1 锚点已保存: {CODE_ANCHOR_FILE} ({len(code_hashes)} files)")

    # Phase 2 锚点
    result_anchor = {}
    for seed in anchor_cfg["seeds"]:
        print(f"[INIT] Phase 2 运行 seed={seed}...")
        r = run_seed(anchor_cfg, seed)
        result_anchor[str(seed)] = r
        print(f"  seed={seed}: mean SAS={r['mean_sas']:.6f}, SLO rate={r['slo_rate']:.1%}")

    with open(RESULT_ANCHOR_FILE, "w") as f:
        json.dump(result_anchor, f, indent=2, sort_keys=True)
    print(f"[INIT] Phase 2 锚点已保存: {RESULT_ANCHOR_FILE}")

    # 保存 meta（含 config_hash + SEMANTICS_VERSION）
    meta = {
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "config": anchor_cfg,
        "code_files_count": len(code_hashes),
        "seeds": anchor_cfg["seeds"],
        "semantics_version": _cfg.get("semantics_version", "unknown"),
        "config_hash": config_hash(),
    }
    meta_path = os.path.join(ANCHOR_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INIT] meta 已保存: {meta_path}")
    print(f"  semantics_version: {meta['semantics_version']}")
    print(f"  config_hash: {meta['config_hash']}")


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="复现门禁：代码同一性 + 结果复现")
    parser.add_argument("--init", action="store_true", help="首次建立锚点基线")
    parser.add_argument("--phase1-only", action="store_true", help="仅运行 Phase 1（代码同一性）")
    parser.add_argument("--phase2-only", action="store_true", help="仅运行 Phase 2（结果复现）")
    parser.add_argument("--tolerance", type=float, default=0.0, help="Phase 2 SAS 容差（默认 0）")
    args = parser.parse_args()

    if args.init:
        print("=" * 60)
        print("建立锚点基线")
        print("=" * 60)
        init_anchor()
        print("\n锚点建立完成。后续运行不带 --init 进行门禁检查。")
        return

    overall_pass = True

    if not args.phase2_only:
        print("=" * 60)
        print("Phase 1: 代码同一性检查")
        print("=" * 60)
        result = check_code_identity()
        if "error" in result:
            print(f"[SKIP] {result['error']}")
        elif result["pass"]:
            print(f"[PASS] {result['current_files']} files, no changes from anchor")
        else:
            print(f"[FAIL] {result['anchor_files']} anchor files, {result['current_files']} current files")
            if result["added"]:
                print(f"  NEW: {result['added']}")
            if result["removed"]:
                print(f"  DELETED: {result['removed']}")
            if result["changed"]:
                print(f"  CHANGED ({len(result['changed'])}):")
                for f in result["changed"]:
                    print(f"    {f}")
            overall_pass = False
        print()

    if not args.phase1_only:
        print("=" * 60)
        print("Phase 2: 结果复现检查")
        print("=" * 60)
        result = check_result_reproduction(tolerance=args.tolerance)
        if "error" in result:
            print(f"[SKIP] {result['error']}")
        elif result["pass"]:
            print(f"[PASS] 所有 seed SAS 值完全一致（容差 {result['tolerance']}）")
        else:
            print(f"[FAIL] SAS 值不一致（容差 {result['tolerance']}）")
            overall_pass = False
        print()

    print("=" * 60)
    print(f"门禁结果: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
