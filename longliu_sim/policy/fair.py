"""Fair Sharing 策略。"""

from __future__ import annotations
from typing import List, Dict, Set

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class Fair(Policy):
    """每 job 均分带宽，同一 job 的多个 flow 再均分该 job 份额。"""

    def __init__(self):
        super().__init__("Fair")

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        if not flows or not links:
            return {}

        # 单链路简化：所有 flow 共享瓶颈链路
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
