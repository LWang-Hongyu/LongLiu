#!/usr/bin/env python3
"""Dump 全部链路清单：ID、端点、容量。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.network import FatTreeTopology


def main():
    print("=" * 120)
    print("拓扑链路清单（k=4 FatTree）")
    print("=" * 120)

    # 创建拓扑（与实验配置一致）
    topo = FatTreeTopology(
        k=4,
        host_bw_bps=100e9,
        spine_bw_bps=400e9,
    )

    print(f"拓扑参数：k={topo.k}, num_spine_links={topo.num_spine_links}")
    print()

    # 遍历所有链路
    print("链路分类统计：")
    print("-" * 120)

    # Spine links
    print(f"Spine links（核心链路）：{len(topo.spine_links)} 条")
    total_spine_bw = sum(link.bw_bps for link in topo.spine_links) / 1e9
    print(f"  总容量：{total_spine_bw:.0f} Gbps")

    for link in topo.spine_links:
        print(f"  {link.lid}: {link.bw_bps/1e9:.0f} Gbps")

    print()

    # 链路总览
    print("=" * 120)
    print("链路容量汇总：")
    print("-" * 120)

    # Spine links
    spine_capacity = sum(link.bw_bps for link in topo.spine_links) / 1e9
    print(f"Spine links 总容量：{spine_capacity:.0f} Gbps ({len(topo.spine_links)} 条 × {topo.spine_links[0].bw_bps/1e9:.0f} Gbps)")

    print()

    # 关键结论
    print("=" * 120)
    print("关键结论：")
    print("-" * 120)
    print(f"1. Spine links 总容量 = {spine_capacity:.0f} Gbps（{len(topo.spine_links)} 条 × {topo.spine_links[0].bw_bps/1e9:.0f} Gbps）")
    print(f"2. 瓶颈链路 = Spine links（总容量 {spine_capacity:.0f} Gbps）")
    print()


if __name__ == "__main__":
    main()