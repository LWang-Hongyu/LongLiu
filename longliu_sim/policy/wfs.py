"""Weighted Fair Sharing (WFS) 策略。

权重 = 1 / ci（SLO 松弛系数越小 → 权重越高 → 带宽越多）。
同一 job 的多个 flow 均分该 job 的份额。

与 Fair 的区别：Fair 对所有 job 均分带宽，WFS 按 ci 倒数加权。
与 LongLiu 的区别：WFS 是静态权重（不随进度变化），无法表达
"demand ceiling"（attain_bw 上限），在资源稀缺且作业异构时
会因过度分配给 tight-SLO job 而导致其他 premium job 饿死。
"""

from __future__ import annotations
from typing import List, Dict, Set

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class WFS(Policy):
    """Weighted Fair Sharing: weight_j = 1 / ci_j。"""

    def __init__(self):
        super().__init__("WFS")

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        if not flows or not links:
            return {}

        link = links[0]
        jobs: Set[str] = set(f.jid for f in flows)

        # 权重 = 1/ci（tighter SLO → higher weight）
        weights: Dict[str, float] = {}
        for jid in jobs:
            ci = job_stats[jid].slo_ci
            weights[jid] = 1.0 / ci if ci > 0 else 1.0

        total_w = sum(weights.values())
        total_bw = link.bw_bps

        alloc: Allocation = {}
        for jid in jobs:
            job_flows = [f for f in flows if f.jid == jid]
            bw_per_job = total_bw * weights[jid] / total_w
            bw_per_flow = bw_per_job / len(job_flows)
            for f in job_flows:
                alloc[f] = {link: bw_per_flow}
        return alloc
