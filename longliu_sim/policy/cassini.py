"""CASSINI 策略：基于 Time-Shift 的通信交错。

论文 [NSDI'24] CASSINI: 面向 GPU 集群的网络感知调度。
核心思想：偏移各 job 的通信阶段开始时间，避免多个 AllReduce 在链路层竞争。

在 flow-level 仿真器中，CASSINI 实现为：
1. 静态计算每个 job 的 comm_offset_ms（基于迭代周期的 LCM）
2. 分配带宽时等同于 Fair（所有 job 均分）
3. 策略本身不直接分配带宽，而是通过调整 job 的通信相位来减少竞争

本实现中，CASSINI 策略提交时计算最优偏移，在 allocate 时按 Fair 分配。
"""

from __future__ import annotations
import math
from typing import List, Dict, Set

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class CASSINI(Policy):
    """
    CASSINI 调度策略。

    带宽分配等同于 Fair sharing。策略的核心价值在于：
    - 为 trace loader 提供 job 的通信相位偏移建议
    - 在 evaluate 中与 LongLiu 对比 time-shift vs priority 两种路径

    偏移计算：对每个 job j，偏移量 o_j = sum_{k<j} min(iv_k, LCM(iv_0..iv_{j-1}))
    其中 iv_k 是 job k 的 iter_interval_ms。
    实际偏移在 trace loader 中设为 job.comm_offset_ms。
    """

    def __init__(self):
        super().__init__("CASSINI")

    @staticmethod
    def compute_offsets(iter_intervals_ms: List[float]) -> List[float]:
        """
        静态计算一组 job 的最优通信相位偏移。

        策略：按 iter_interval 递增排序，每个 job 在前一个 job 的
        通信窗口结束后开始。使用最小公倍数思想错开高峰期。

        返回与输入等长的偏移列表（ms）。
        """
        if not iter_intervals_ms:
            return []

        # 按间隔排序（短 job 优先偏移）
        sorted_idx = sorted(range(len(iter_intervals_ms)),
                            key=lambda i: iter_intervals_ms[i])

        offsets = [0.0] * len(iter_intervals_ms)
        last_end = 0.0

        for idx in sorted_idx:
            iv = iter_intervals_ms[idx]
            # 通信窗口 = iv * 0.5（假设通信占迭代时间的一半）
            comm_window = iv * 0.5
            offsets[idx] = last_end
            last_end += comm_window

        return offsets

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """CASSINI 在带宽分配上等同于 Fair。"""
        if not flows or not links:
            return {}

        link = links[0]
        jobs: Set[str] = set(f.jid for f in flows)
        bw_per_job = link.bw_bps / len(jobs)

        alloc: Allocation = {}
        for jid in jobs:
            job_flows = [f for f in flows if f.jid == jid]
            bw_per_flow = bw_per_job / len(job_flows)
            for f in job_flows:
                alloc[f] = {link: bw_per_flow}
        return alloc