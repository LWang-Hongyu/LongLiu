"""CRUX 策略：基于 GPU Intensity 的严格优先级调度。

论文 [SIGCOMM'24] CRUX: GPU-Efficient Communication Scheduling for DL Training。

核心思想（Theorem 1）：最大化 GPU 利用率等价于最大化链路上传输的 GPU Intensity 总和。
因此，GPU Intensity 越高的 job 应获得越高的通信优先级。

I_j = W_j / t_j（FLOPs / 瓶颈链路传输时间）
本实现用 comp_ms / comm_solo_ms 作为 I_j 的代理。

优先级分配：将 I_j 映射到硬件 DSCP 优先级，严格优先级执行。
高 Intensity → 高优先级 → 独占带宽 → 快速释放 GPU 计算。
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Dict, Set

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class CRUX(Policy):
    """
    CRUX 调度策略（严格优先级版）。

    GPU Intensity I_j = comp_ms / comm_solo_ms 映射到 7 级 DSCP 优先级，
    使用硬件严格优先级队列执行调度。

    包含 profiling 阶段：前 profile_iters 个迭代所有 job 获得相同优先级。
    """

    # Intensity → DSCP 映射（I_j 越大优先级越高）
    # 典型 intensity 范围：10~10000+（对数分布）
    DSCP_MAP = [
        (1000.0, 38),  # I > 1000  → P6 (DSCP 38) 极高计算强度
        (500.0, 34),   # I > 500   → P5 (DSCP 34)
        (200.0, 36),   # I > 200   → P4 (DSCP 36)
        (100.0, 26),   # I > 100   → P3 (DSCP 26)
        (50.0, 28),    # I > 50    → P2 (DSCP 28)
        (20.0, 18),    # I > 20    → P1 (DSCP 18)
    ]                  # I ≤ 20    → P0 (DSCP 0) 最低优先级
    DSCP_DEFAULT = 0
    _DSCP_PRIORITY_ORDER = [38, 34, 36, 26, 28, 18, 0]

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

    def _get_dscp(self, intensity: float) -> int:
        """将 GPU intensity 映射到 DSCP 优先级。"""
        for threshold, dscp in self.DSCP_MAP:
            if intensity > threshold:
                return dscp
        return self.DSCP_DEFAULT

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """
        严格优先级调度：按 GPU Intensity 的 DSCP 级别分配带宽。

        P6 > P5 > P4 > P3 > P2 > P1 > P0
        最高优先级类独占全部带宽，同级别内均分。
        """
        if not flows or not links:
            return {}

        link = links[0]
        jobs: Set[str] = set(f.jid for f in flows)

        # 检查是否仍在 profiling 阶段
        in_profile = any(job_stats[jid].completed_iters < self.profile_iters for jid in jobs)

        # 按 GPU intensity 计算每个 job 的 DSCP
        job_dscp: Dict[str, int] = {}
        for jid in jobs:
            job = job_stats[jid]
            if in_profile:
                # profiling 阶段：所有 job 获得最高优先级（与论文一致）
                job_dscp[jid] = 38
            else:
                intensity = job.gpu_intensity
                job_dscp[jid] = self._get_dscp(intensity)

        # 按 DSCP 分组 flows
        flows_by_dscp: Dict[int, List[Flow]] = defaultdict(list)
        for f in flows:
            flows_by_dscp[job_dscp[f.jid]].append(f)

        # 严格优先级分配：最高优先级类独占全部带宽
        alloc: Allocation = {}
        remaining_bw = link.bw_bps
        for dscp in self._DSCP_PRIORITY_ORDER:
            if dscp not in flows_by_dscp:
                continue
            flows_at_level = flows_by_dscp[dscp]
            if not flows_at_level:
                continue
            # 同级别内均分带宽
            bw_per_flow = remaining_bw / len(flows_at_level)
            for f in flows_at_level:
                alloc[f] = {link: bw_per_flow}
            remaining_bw = 0.0
            break

        return alloc
