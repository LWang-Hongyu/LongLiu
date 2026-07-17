"""SRPT 策略：优先剩余数据量小的 flow。

Shortest Remaining Processing Time — 经典调度基线。
适用于 flow-level 仿真：剩余数据量最小的 flow 获得最大带宽。
"""

from __future__ import annotations
from typing import List, Dict

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class SRPT(Policy):
    """
    SRPT 调度策略。

    按 flow 的剩余数据量（rem_bits）排序，剩余越少的 flow 获得越多带宽。
    每个 flow 的带宽与 1 / rem_bits 成正比。
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__("SRPT")
        self.eps = eps

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        if not flows or not links:
            return {}

        link = links[0]

        # 每个 flow 权重 = 1 / (rem_bits + eps)
        total_weight = sum(1.0 / (f.rem_bits + self.eps) for f in flows)
        if total_weight <= 0:
            return {}

        alloc: Allocation = {}
        for f in flows:
            w = 1.0 / (f.rem_bits + self.eps)
            bw = link.bw_bps * w / total_weight
            alloc[f] = {link: bw}
        return alloc