"""CRUX 策略：基于 GPU Intensity 的权重分配。

论文 [SIGCOMM'24] CRUX: 面向 GPU 集群的负载感知调度。
核心思想：分配带宽与 GPU intensity Ij = W_j / t_j 成正比。
W_j ≈ params (FLOPS ∝ 参数量)，t_j ∝ bits_per_iter / B_e，
即 Ij ∝ params / bits_per_iter。
大模型计算量大 → Intensity 高 → 获得更多带宽以释放更多 GPU 计算。
"""

from __future__ import annotations
from typing import List, Dict, Set

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class CRUX(Policy):
    """
    CRUX 调度策略。

    权重公式：w_j = I_j^alpha，其中 Ij = comp_ms / comm_solo_ms。
    高 GPU intensity 的 job 获得更多带宽（通信占比小，快速释放链路）。

    包含 profiling 阶段：前 profile_iters 个迭代所有 job 获得相同权重，
    以便收集准确的 intensity 信息。
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-6, profile_iters: int = 3):
        """
        参数：
            alpha: GPU intensity 的指数权重（论文默认 1.0）
            eps: 防零除的小量
            profile_iters: profiling 阶段的迭代次数（默认 3）
        """
        super().__init__("CRUX")
        self.alpha = alpha
        self.eps = eps
        self.profile_iters = profile_iters

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        if not flows or not links:
            return {}

        link = links[0]
        jobs: Set[str] = set(f.jid for f in flows)

        # 检查是否仍在 profiling 阶段
        in_profile = any(job_stats[jid].completed_iters < self.profile_iters for jid in jobs)

        # 按 job 的 GPU intensity 分配权重
        weights: Dict[str, float] = {}
        for jid in jobs:
            job = job_stats[jid]
            if in_profile:
                weights[jid] = 1.0
            else:
                intensity = job.gpu_intensity
                weights[jid] = (intensity + self.eps) ** self.alpha

        total_weight = sum(weights.values())
        if total_weight <= 0:
            return {}

        alloc: Allocation = {}
        for jid in jobs:
            job_flows = [f for f in flows if f.jid == jid]
            bw_for_job = link.bw_bps * weights[jid] / total_weight
            bw_per_flow = bw_for_job / len(job_flows)
            for f in job_flows:
                alloc[f] = {link: bw_per_flow}
        return alloc