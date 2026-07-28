#!/usr/bin/env python3
"""验证 feas_boundary_v1 的 placement：每个 job 必须跨 ≥2 个 pod。"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.network import FatTreeTopology


def assign_workers(num_workers: int, num_hosts: int, seed: int, force_cross_pod: bool = False) -> list[int]:
    """为 job 分配 worker hosts（随机放置）。
    
    Args:
        num_workers: worker 数量
        num_hosts: 总 host 数
        seed: 随机种子
        force_cross_pod: 强制跨 pod 放置
    """
    random.seed(seed)
    workers = []
    
    if force_cross_pod and num_workers >= 2:
        # 强制跨 pod：第一个 worker 在 pod 0，第二个在 pod 1
        workers.append(random.randint(0, 3))  # pod 0
        workers.append(random.randint(4, 7))  # pod 1
        
        # 剩余 worker 随机分配
        for _ in range(num_workers - 2):
            host = random.randint(0, num_hosts - 1)
            workers.append(host)
    else:
        # 默认随机放置
        for _ in range(num_workers):
            host = random.randint(0, num_hosts - 1)
            workers.append(host)
    
    return workers


def get_pod_id(host_id: int, hosts_per_pod: int = 4) -> int:
    """计算 host 所属的 pod（假设 k=4 FatTree，每个 pod 4 个 hosts）。"""
    return host_id // hosts_per_pod


def verify_pod_span(workers: list[int], hosts_per_pod: int = 4) -> tuple[bool, set[int]]:
    """验证 job 是否跨 ≥2 个 pod。"""
    pod_ids = set(get_pod_id(h, hosts_per_pod) for h in workers)
    return len(pod_ids) >= 2, pod_ids


def main():
    print("=" * 80)
    print("Placement 验证：每个 job 必须跨 ≥2 个 pod")
    print("=" * 80)

    # 配置
    num_hosts = 16
    hosts_per_pod = 4
    seed = 0

    # 主场景构型
    jobs_config = [
        {"jid": "P1", "model": "LLaMA-2-13B", "dp": 8},
        {"jid": "P2", "model": "LLaMA-2-7B", "dp": 8},
        {"jid": "P3", "model": "BERT-Large-fp16", "dp": 2},
        {"jid": "S1", "model": "LLaMA-2-13B", "dp": 8},
        {"jid": "S2", "model": "T5-11B-fp16", "dp": 8},
        {"jid": "S3", "model": "BERT-Large-fp16", "dp": 4},
        {"jid": "S4", "model": "ViT-Base", "dp": 2},
    ]

    # 拓扑（用于 ECMP 路径）
    topo = FatTreeTopology(k=4, host_bw_bps=100e9, spine_bw_bps=800e9)

    # 分配 workers 并验证
    print("\n逐 job 验证：")
    print("-" * 80)

    verification_table = []
    all_valid = True

    for i, job_cfg in enumerate(jobs_config):
        jid = job_cfg["jid"]
        dp = job_cfg["dp"]

        # 分配 workers（每个 job 使用不同的 seed）
        # dp=2 的 job 强制跨 pod
        force_cross_pod = (dp == 2)
        workers = assign_workers(dp, num_hosts, seed + i, force_cross_pod)

        # 验证 pod 跨度
        is_cross_pod, pod_ids = verify_pod_span(workers, hosts_per_pod)

        # 计算 ECMP 所属 spine link
        if is_cross_pod:
            src = workers[0]
            dst = workers[1]
            spine_idx = topo._ecmp_path(src, dst)
        else:
            spine_idx = None

        # 输出
        status = "✓" if is_cross_pod else "✗"
        print(f"{jid}: workers={workers} → pods={sorted(pod_ids)} → {status} cross_pod={is_cross_pod}")

        if not is_cross_pod:
            all_valid = False
            print(f"  ⚠️  警告：{jid} 未跨 pod，需要调整 placement")

        verification_table.append({
            "jid": jid,
            "num_workers": dp,
            "worker_hosts": workers,
            "pod_ids": sorted(pod_ids),
            "num_pods": len(pod_ids),
            "is_cross_pod": is_cross_pod,
            "spine_link_idx": spine_idx,
        })

    # 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)

    cross_pod_count = sum(1 for v in verification_table if v["is_cross_pod"])
    print(f"\n跨 pod job 数：{cross_pod_count} / {len(jobs_config)}")

    if all_valid:
        print(f"✓ 所有 job 都跨 ≥2 个 pod，placement 验证通过")
    else:
        print(f"✗ 存在未跨 pod 的 job，需要调整 placement")

    # 保存验证表
    import csv
    os.makedirs("outputs/feas_boundary_v1_design", exist_ok=True)

    csv_path = "outputs/feas_boundary_v1_design/placement_verification.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "jid", "num_workers", "worker_hosts", "pod_ids",
            "num_pods", "is_cross_pod", "spine_link_idx"
        ])
        for row in verification_table:
            writer.writerow([
                row["jid"],
                row["num_workers"],
                ",".join(map(str, row["worker_hosts"])),
                ",".join(map(str, row["pod_ids"])),
                row["num_pods"],
                row["is_cross_pod"],
                row["spine_link_idx"] if row["spine_link_idx"] is not None else "",
            ])

    print(f"\n✓ 验证表已保存: {csv_path}")

    # 保存 JSON
    summary = {
        "num_hosts": num_hosts,
        "hosts_per_pod": hosts_per_pod,
        "seed": seed,
        "all_valid": all_valid,
        "cross_pod_count": cross_pod_count,
        "total_jobs": len(jobs_config),
        "verification_table": verification_table,
    }

    with open("outputs/feas_boundary_v1_design/placement_verification.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ 验证摘要已保存: outputs/feas_boundary_v1_design/placement_verification.json")


if __name__ == "__main__":
    main()