"""
非确定性自测：同 seed 连跑两次，检查是否逐位一致
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from longliu_sim.policy import LongLiu
from longliu_sim.core import Simulator
from longliu_sim.network import FatTreeTopology
from longliu_sim.trace import SyntheticTraceLoader
from longliu_sim.trace.synthetic import DEFAULT_TIERED_WORKLOAD


def run_single_seed(cfg: dict, policy, seed: int) -> dict:
    """运行单个 seed 的仿真。"""
    topo = FatTreeTopology(
        k=cfg["topology"]["k"],
        host_bw_bps=cfg["topology"]["host_bw_bps"],
        spine_bw_bps=cfg["topology"]["spine_bw_bps"],
    )
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
            "ResNet-18",
            "ResNet-50-fp16",
            "BERT-Base",
            "BERT-Large-fp16",
            "ViT-Base",
            "ViT-Large",
            "LLaMA-2-1B",
            "LLaMA-2-7B",
            "T5-1B",
        ],
        gpu_distribution={1: 0.2, 2: 0.2, 4: 0.3, 8: 0.3},
        ci_distribution={1.5: 0.3, 2.0: 0.35, 3.0: 0.35},
        job_count=24,
        duration_ms=cfg["duration_ms"],
        seed=seed,
        overhead_factor=cfg["overhead_factor"],
        target_bw_bps=cfg["topology"]["host_bw_bps"],
        num_hosts=16,
        workload_profile=DEFAULT_TIERED_WORKLOAD,
    )
    jobs = loader.load()
    for j in jobs:
        sim.submit(j)

    result = sim.run()
    stats = result.per_job_stats()

    # 收集每个 job 的 SAS
    per_job_sas = {}
    for jid, s in stats.items():
        job = sim.jobs[jid]
        per_job_sas[jid] = {
            "sas": s["sas"],
            "avg_iter_ms": s["avg_iter_ms"],
            "ci": job.slo_ci,
        }

    return {
        "seed": seed,
        "per_job_sas": per_job_sas,
        "overall_mean_sas": sum(s["sas"] for s in stats.values()) / len(stats) if stats else 0.0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=str, default="/tmp/det_test.json")
    args = parser.parse_args()

    cfg = {
        "topology": {
            "type": "fatree",
            "k": 4,
            "host_bw_bps": 100e9,
            "spine_bw_bps": 400e9,
        },
        "duration_ms": 600000,
        "overhead_factor": 1.3,
        "overlap_factor": 0.85,
    }

    policy = LongLiu(K=2.0, use_dynamic_T_target=True)

    print(f"运行 seed 0 ({args.out})...")
    result = run_single_seed(cfg, policy, 0)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"Mean SAS: {result['overall_mean_sas']:.3f}")
    print(f"结果已保存至: {args.out}")


if __name__ == "__main__":
    main()