"""LongLiu Continuous: 理想连续分配（无 DSCP 量化）。

继承 LongLiu，重写 allocate() 跳过 DSCP bucket 步骤，
用连续的 pi 值排序，使用 strict priority（严格优先级）分配带宽。
用作 E12 DSCP 量化实验的"理想连续"对照。
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Dict, Set

from .longliu import LongLiu
from .base import Allocation
from ..network.flow import Flow
from ..network.link import Link


class LongLiuContinuous(LongLiu):
    """
    LongLiu 理想连续分配器（Strict Priority，无量化）。

    与 LongLiu 的区别：
    - 跳过 DSCP 映射（不将 pi 离散化为 7 级）
    - 直接用连续的 pi 值排序，使用 strict priority 分配
    - 高 pi job 优先发送，低 pi job 借用剩余带宽（work-conserving）

    用途：E12 DSCP 量化实验的"理想连续"对照基线。
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('n_dscp_levels', 7)
        super().__init__(**kwargs)
        self.name = "LongLiuContinuous"

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """连续 pi 值 + proportional sharing（work-conserving）。
        
        与 v4 的区别：
        - v4: pi 量化到 7 级 DSCP + 类内 exp(pi*K) 加权
        - Continuous: pi 连续值 + 直接用 exp(pi*K) 作为权重
        
        两者都是 work-conserving（按权重比例分配所有带宽）。
        """
        if not flows or not links:
            return {}

        link = links[0]
        jobs: Set[str] = set(f.jid for f in flows)

        # 计算每个 job 的 pi（复用父类的 deficit 计算逻辑）
        job_pi: Dict[str, float] = {}
        for jid in jobs:
            job = job_stats[jid]
            if job.is_first_iter:
                job_pi[jid] = 1.0  # 第一轮给高优先级
                continue
            pi = job.compute_deficit()
            if self.use_dynamic_T_target:
                T_target = job.get_T_target(has_highest_priority=False)
                if job.completed_iters > 0:
                    if self.window_size > 0:
                        sw_avg = job.sliding_avg_iter_ms
                        if sw_avg is not None:
                            avg_iter_ms = sw_avg
                        else:
                            avg_iter_ms = job.accumulated_iter_ms / job.completed_iters
                    else:
                        avg_iter_ms = job.accumulated_iter_ms / job.completed_iters
                    pi = avg_iter_ms / T_target - 1.0
                else:
                    pi = 0.0
            job_pi[jid] = pi

        # 计算每个 flow 的连续权重（不量化）
        weights: Dict[str, float] = {}
        for f in flows:
            pi = job_pi.get(f.jid, 0.0)
            pi_clipped = max(-2.0, min(3.0, pi))
            w = max(self.MIN_W, math.exp(self.K * pi_clipped)) * self.BASE_W
            weights[f.fid] = w

        # Proportional sharing（work-conserving）
        total_weight = sum(weights.values())
        total_bw = link.bw_bps

        alloc: Allocation = {}
        for f in flows:
            alloc[f] = {link: total_bw * weights[f.fid] / total_weight}

        return alloc
