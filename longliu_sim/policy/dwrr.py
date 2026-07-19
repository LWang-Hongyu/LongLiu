"""DWRR (Deficit Weighted Round Robin) 有界加权策略。

跨类 DWRR：按权重分配带宽，work-conserving（空类份额让给其他类）。
类内分配：exp(pi·K) 加权或公平分配，可选限幅。
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Dict

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class LongLiuDWRR(Policy):
    """
    LongLiu DWRR 调度策略。

    跨类 DWRR：
    - DSCP 映射将 pi 离散化为 P0-P6 七个优先级类
    - 每个类按权重分配带宽：class_weight = {P0:1, P1:2, P2:4, P3:8, P4:16, P5:32, P6:64}
    - work-conserving：空类份额按权重比例让给其他类

    类内分配：
    - 模式 1（D1）：exp(pi·K) 加权，clip ≤10×（防止对称饿死）
    - 模式 2（D2）：公平分配（所有 job 平分）
    - 模式 3（D3）：软权重表 {1,2,3,4,6,8,12}，其余同 D1
    """

    # 标准 DWRR 权重表（D1/D2）
    CLASS_WEIGHTS_STD = [1, 2, 4, 8, 16, 32, 64]  # P0-P6

    # 软权重表（D3）
    CLASS_WEIGHTS_SOFT = [1, 2, 3, 4, 6, 8, 12]  # P0-P6

    # DSCP 映射（与 LongLiu-SP 一致）
    DSCP_MAP = [
        (0.6, 38),   # pi > 0.6  → P6 (DSCP 38)
        (0.4, 34),   # pi > 0.4  → P5 (DSCP 34)
        (0.2, 36),   # pi > 0.2  → P4 (DSCP 36)
        (0.0, 26),   # pi > 0.0  → P3 (DSCP 26)
        (-0.2, 28),  # pi > -0.2 → P2 (DSCP 28)
        (-0.4, 18),  # pi > -0.4 → P1 (DSCP 18)
    ]
    DSCP_DEFAULT = 0
    DSCP_PRIORITY_ORDER = [38, 34, 36, 26, 28, 18, 0]  # P6→P0
    DSCP_LEVEL_MAP = {38: 6, 34: 5, 36: 4, 26: 3, 28: 2, 18: 1, 0: 0}

    def __init__(self,
                 K: float = 2.0,
                 use_soft_weights: bool = False,
                 intra_class_fair: bool = False,
                 clip_ratio: float = 10.0):
        """
        参数：
            K: deficit 指数增益
            use_soft_weights: 是否使用软权重表（D3）
            intra_class_fair: 类内是否公平分配（D2），False 则使用 exp(pi·K) 加权（D1）
            clip_ratio: 类内权重最大比值（D1/D3），防止对称饿死
        """
        super().__init__("LongLiuDWRR")
        self.K = K
        self.use_soft_weights = use_soft_weights
        self.intra_class_fair = intra_class_fair
        self.clip_ratio = clip_ratio

        # 权重表
        self.class_weights = self.CLASS_WEIGHTS_SOFT if use_soft_weights else self.CLASS_WEIGHTS_STD

    def get_dscp(self, pi: float) -> int:
        """根据 pi 映射到 DSCP。"""
        for threshold, dscp in self.DSCP_MAP:
            if pi > threshold:
                return dscp
        return self.DSCP_DEFAULT

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """
        DWRR 带宽分配。

        参数：
            flows: 当前活跃的 flow 列表
            links: 涉及的所有链路（DWRR 假设单瓶颈）
            time_ms: 当前仿真时间（ms）
            job_stats: job 的运行时统计信息

        返回：
            Allocation: flow -> link -> bps
        """
        if not flows or not links:
            return {}

        # DWRR 假设单瓶颈链路（取第一个）
        link = links[0]

        # 1. 按 job 分组 flows
        job_flows = defaultdict(list)
        for flow in flows:
            job_flows[flow.jid].append(flow)

        # 2. 计算每个 job 的 pi 和 DSCP
        job_pi = {}
        job_dscp = {}
        for jid in job_flows:
            if jid not in job_stats:
                continue

            job = job_stats[jid]
            # 计算静态 T_target（ci * comm_solo_ms * overhead_factor）
            T_target = job.slo_ci * job.comm_solo_ms * job.overhead_factor

            if job.completed_iters > 0:
                avg_iter_ms = job.accumulated_iter_ms / job.completed_iters
            else:
                avg_iter_ms = T_target  # 未完成迭代时，假设刚好达标

            if T_target > 0:
                pi = avg_iter_ms / T_target - 1.0
            else:
                pi = 0.0

            job_pi[jid] = pi
            job_dscp[jid] = self.get_dscp(pi)

        # 3. 按 DSCP 类分组
        class_jobs = defaultdict(list)
        for jid in job_flows:
            dscp = job_dscp.get(jid, 0)
            level = self.DSCP_LEVEL_MAP[dscp]
            class_jobs[level].append(jid)

        # 4. 跨类 DWRR 分配
        active_classes = [lvl for lvl in range(7) if class_jobs[lvl]]
        if not active_classes:
            return {}

        total_weight = sum(self.class_weights[lvl] for lvl in active_classes)
        class_bw_share = {}
        for lvl in active_classes:
            class_bw_share[lvl] = (self.class_weights[lvl] / total_weight) * link.bw_bps

        # 5. 类内分配
        allocations = {}
        for lvl in active_classes:
            bw_budget = class_bw_share[lvl]
            jobs_in_class = class_jobs[lvl]

            if self.intra_class_fair:
                # D2: 公平分配
                per_job_bw = bw_budget / len(jobs_in_class)
                for jid in jobs_in_class:
                    for flow in job_flows[jid]:
                        allocations[flow] = {link: per_job_bw / len(job_flows[jid])}
            else:
                # D1/D3: exp(pi·K) 加权，clip ≤10×
                job_weights = {}
                for jid in jobs_in_class:
                    pi = job_pi.get(jid, 0.0)
                    weight = math.exp(self.K * pi)
                    job_weights[jid] = weight

                # Clip 权重比 ≤ clip_ratio
                if job_weights:
                    max_weight = max(job_weights.values())
                    min_weight = min(job_weights.values())
                    if min_weight > 0 and max_weight / min_weight > self.clip_ratio:
                        scale = self.clip_ratio * min_weight
                        for jid in job_weights:
                            if job_weights[jid] > scale:
                                job_weights[jid] = scale

                # 按权重分配
                total_job_weight = sum(job_weights.values()) if job_weights else 1.0
                for jid in jobs_in_class:
                    job_weight_share = job_weights.get(jid, 1.0) / total_job_weight * bw_budget
                    for flow in job_flows[jid]:
                        allocations[flow] = {link: job_weight_share / len(job_flows[jid])}

        return allocations


class LongLiuDWRRFair(LongLiuDWRR):
    """D2: DWRR + 类内公平分配。"""

    def __init__(self, K: float = 2.0):
        super().__init__(K=K, intra_class_fair=True)