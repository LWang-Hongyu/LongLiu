"""DWRR (Deficit Weighted Round Robin) 有界加权策略。

跨类 DWRR：按权重分配带宽，work-conserving（空类份额让给其他类）。
类内分配：exp(pi·K) 加权或公平分配，可选限幅。
"""

from __future__ import annotations
import json
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

    # 陡权重表（D5）
    CLASS_WEIGHTS_STEEP = [1, 2, 8, 32, 128, 512, 1024]  # P0-P6

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
                 clip_ratio: float = 10.0,
                 class_weights: list = None,
                 trace_file: str = None,
                 overlap_factor: float = None,
                 overhead_factor: float = None):
        """
        参数：
            K: deficit 指数增益
            use_soft_weights: 是否使用软权重表（D3）
            intra_class_fair: 类内是否公平分配（D2），False 则使用 exp(pi·K) 加权（D1）
            clip_ratio: 类内权重最大比值（D1/D3），防止对称饿死
            class_weights: 自定义类权重表（None 则默认 STD/SOFT）
            trace_file: 轨迹日志文件路径（JSONL，用于机制调试）
            overlap_factor: compute-comm 重叠度（用于计算 T_target）
            overhead_factor: wire 因子（默认 1.3，锚点冻结值）
        """
        if overlap_factor is None:
            raise ValueError(
                "LongLiuDWRR 必须显式传入 overlap_factor，禁止静默默认值"
            )
        if overhead_factor is None:
            raise ValueError(
                "LongLiuDWRR 必须显式传入 overhead_factor，禁止静默默认值"
            )
        super().__init__("LongLiuDWRR")
        self.K = K
        self.use_soft_weights = use_soft_weights
        self.intra_class_fair = intra_class_fair
        self.clip_ratio = clip_ratio
        self.overlap_factor = max(0.0, min(1.0, overlap_factor))
        self.overhead_factor = overhead_factor
        self._trace_epoch = 0  # epoch counter
        self._trace_handle = None
        if trace_file:
            self._trace_handle = open(trace_file, "w")

        # 权重表
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = self.CLASS_WEIGHTS_SOFT if use_soft_weights else self.CLASS_WEIGHTS_STD

    def flush_trace(self):
        """关闭轨迹文件（仿真结束后调用）。"""
        if self._trace_handle:
            self._trace_handle.close()
            self._trace_handle = None

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
            # 锚点公式 T_target = max(comp, ci×comm_solo×overhead) + (1-overlap)×min(...)
            # 与 metrics.compute_target_iter_ms 同源，SEMANTICS_VERSION="anchor-v2"
            comm_budget_ms = job.slo_ci * job.comm_solo_ms * self.overhead_factor
            if self.overlap_factor > 0:
                T_target = max(job.comp_ms, comm_budget_ms) + \
                           (1.0 - self.overlap_factor) * min(job.comp_ms, comm_budget_ms)
            else:
                T_target = job.comp_ms + comm_budget_ms

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

        # Trace: 记录每次分配的 π/DSCP/类/份额（机制调试用）
        if self._trace_handle:
            # 计算每个 job 在这次分配中获得的总带宽
            job_bw = defaultdict(float)
            for flow, link_alloc in allocations.items():
                job_bw[flow.jid] += sum(link_alloc.values())

            row = {"epoch": self._trace_epoch, "time_ms": time_ms,
                   "link_bw_gbps": link.bw_bps / 1e9}
            for jid in job_pi:
                bw = job_bw.get(jid, 0.0)
                dscp = job_dscp.get(jid, 0)
                lvl = self.DSCP_LEVEL_MAP[dscp]
                row[f"{jid}_pi"] = round(job_pi[jid], 4)
                row[f"{jid}_dscp"] = dscp
                row[f"{jid}_level"] = lvl
                row[f"{jid}_bw_gbps"] = round(bw / 1e9, 4)
                row[f"{jid}_share"] = round(bw / link.bw_bps, 4) if link.bw_bps > 0 else 0.0
            self._trace_handle.write(json.dumps(row) + "\n")
            self._trace_epoch += 1

        return allocations


class LongLiuDWRRGap(LongLiuDWRR):
    """D1G: DWRR + gap 比例权重。

    主键：绝对带宽缺口 gap_i = max(0, demand_i − bw_i)
    - demand_i = bits_per_iter / (ci·iter_solo)  [Gbps]
    - bw_i = 分配器上一轮分配给 job i 的瓶颈带宽 [Gbps]

    权重映射：w_i = max(floor_w, gap_i / G0)
    - floor_w ∈ {1, 2}: gap=0 保底权重（starvation-free 地板）
    - G0 ∈ {10, 25, 50}: 缺口归一化因子

    类量化（DSCP 对数带映射）：
        gap ≤ 0 → P0; gap∈(0,G0/8) → P1; (G0/8,G0/4) → P2;
        (G0/4,G0/2) → P3; (G0/2,G0) → P4; (G0,2G0) → P5; >2G0 → P6
    """

    def __init__(self,
                 K: float = 2.0,
                 floor_w: float = 2.0,
                 G0_gbps: float = 25.0,
                 class_weights: list = None,
                 trace_file: str = None,
                 overlap_factor: float = None,
                 overhead_factor: float = None):
        """
        参数：
            K: 保留（与 D1 接口兼容，D1G 不使用 exp(pi·K)）
            floor_w: gap=0 保底权重
            G0_gbps: 缺口归一化因子（Gbps）
            class_weights: 自定义类权重表（None 则默认 STD）
            trace_file: 轨迹日志文件路径
            overlap_factor: compute-comm 重叠度
            overhead_factor: wire 因子（默认 1.3，锚点冻结值）
        """
        if overlap_factor is None:
            raise ValueError(
                "LongLiuDWRRGap 必须显式传入 overlap_factor，禁止静默默认值"
            )
        if overhead_factor is None:
            raise ValueError(
                "LongLiuDWRRGap 必须显式传入 overhead_factor，禁止静默默认值"
            )
        super().__init__(
            K=K, use_soft_weights=False, intra_class_fair=False,
            clip_ratio=100.0, class_weights=class_weights,
            trace_file=trace_file, overlap_factor=overlap_factor,
            overhead_factor=overhead_factor,
        )
        self.name = "LongLiuDWRRGap"
        self.floor_w = float(floor_w)
        self.G0_gbps = float(G0_gbps)

        # 上一轮分配的瓶颈带宽跟踪：jid -> bps
        self._prev_job_bw: Dict[str, float] = {}

    def _gap_to_level(self, gap_gbps: float) -> int:
        """将 gap 映射到 7 个 DSCP 类级 (0=P0, 6=P6)。"""
        G0 = self.G0_gbps
        if gap_gbps <= 0.0:
            return 0
        if gap_gbps <= G0 / 8.0:
            return 1
        if gap_gbps <= G0 / 4.0:
            return 2
        if gap_gbps <= G0 / 2.0:
            return 3
        if gap_gbps <= G0:
            return 4
        if gap_gbps <= 2.0 * G0:
            return 5
        return 6

    def _level_to_dscp(self, level: int) -> int:
        """将类级 (0-6) 映射到 DSCP 值。"""
        mapping = {0: 0, 1: 18, 2: 28, 3: 26, 4: 36, 5: 34, 6: 38}
        return mapping.get(level, 0)

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """
        D1G 带宽分配：gap 比例权重 + 7 类量化 DWRR。

        gap_i = max(0, demand_bps − bw_i)
        w_i   = max(floor_w, gap_i / G0)
        """
        if not flows or not links:
            return {}

        link = links[0]
        link_bw_bps = link.bw_bps

        # 1. 按 job 分组 flows
        job_flows = defaultdict(list)
        for flow in flows:
            job_flows[flow.jid].append(flow)

        # 2. 计算每个 job 的 demand、gap、weight、DSCP 类
        job_gap = {}
        job_demand = {}
        job_weight = {}
        job_level = {}
        job_dscp = {}

        for jid in job_flows:
            if jid not in job_stats:
                continue

            job = job_stats[jid]

            # Demand: bits_per_iter / T_target [bps]
            # 锚点公式（与 metrics.compute_target_iter_ms 同源）
            comm_budget_ms = job.slo_ci * job.comm_solo_ms * self.overhead_factor
            if self.overlap_factor > 0:
                T_target = max(job.comp_ms, comm_budget_ms) + \
                           (1.0 - self.overlap_factor) * min(job.comp_ms, comm_budget_ms)
            else:
                T_target = job.comp_ms + comm_budget_ms
            demand_bps = job.bits_per_iter / (T_target / 1000.0)  # bits/sec

            # bw_i: 上一轮分配的带宽（首次分配用 demand_bps 作为初始值）
            bw_bps = self._prev_job_bw.get(jid, demand_bps)

            # gap [bps]
            gap_bps = max(0.0, demand_bps - bw_bps)
            gap_gbps = gap_bps / 1e9

            # weight: max(floor_w, gap_i / G0)
            w = self.floor_w if gap_bps <= 0 else max(self.floor_w, gap_gbps / self.G0_gbps)

            # DSCP 类
            lvl = self._gap_to_level(gap_gbps)
            dscp = self._level_to_dscp(lvl)

            job_gap[jid] = gap_bps
            job_demand[jid] = demand_bps
            job_weight[jid] = w
            job_level[jid] = lvl
            job_dscp[jid] = dscp

        # 3. 按 DSCP 类分组
        class_jobs = defaultdict(list)
        for jid in job_flows:
            lvl = job_level.get(jid, 0)
            class_jobs[lvl].append(jid)

        # 4. 跨类 DWRR 分配
        active_classes = [lvl for lvl in range(7) if class_jobs[lvl]]
        if not active_classes:
            return {}

        total_weight = sum(self.class_weights[lvl] for lvl in active_classes)
        class_bw_share = {}
        for lvl in active_classes:
            class_bw_share[lvl] = (self.class_weights[lvl] / total_weight) * link_bw_bps

        # 5. 类内分配（按 gap 比例权重）
        allocations = {}
        new_job_bw = defaultdict(float)

        for lvl in active_classes:
            bw_budget = class_bw_share[lvl]
            jobs_in_class = class_jobs[lvl]

            total_job_weight = sum(job_weight.get(jid, self.floor_w) for jid in jobs_in_class)
            if total_job_weight <= 0:
                total_job_weight = 1.0

            for jid in jobs_in_class:
                job_weight_share = job_weight.get(jid, self.floor_w) / total_job_weight * bw_budget
                n_flows = len(job_flows[jid])
                for flow in job_flows[jid]:
                    allocations[flow] = {link: job_weight_share / n_flows}
                new_job_bw[jid] = job_weight_share

        # 更新上一轮带宽跟踪
        self._prev_job_bw = dict(new_job_bw)

        # 6. Trace 记录
        if self._trace_handle:
            row = {"epoch": self._trace_epoch, "time_ms": time_ms,
                   "link_bw_gbps": link_bw_bps / 1e9,
                   "G0_gbps": self.G0_gbps, "floor_w": self.floor_w}
            for jid in job_gap:
                bw = new_job_bw.get(jid, 0.0)
                gap = job_gap.get(jid, 0.0)
                demand = job_demand.get(jid, 0.0)
                w = job_weight.get(jid, self.floor_w)
                lvl = job_level.get(jid, 0)
                row[f"{jid}_demand_gbps"] = round(demand / 1e9, 4)
                row[f"{jid}_gap_gbps"] = round(gap / 1e9, 4)
                row[f"{jid}_weight"] = round(w, 4)
                row[f"{jid}_level"] = lvl
                row[f"{jid}_bw_gbps"] = round(bw / 1e9, 4)
                row[f"{jid}_share"] = round(bw / link_bw_bps, 4) if link_bw_bps > 0 else 0.0
            self._trace_handle.write(json.dumps(row) + "\n")
            self._trace_epoch += 1

        return allocations


class LongLiuDWRRFair(LongLiuDWRR):
    """D2: DWRR + 类内公平分配。"""

    def __init__(self, K: float = 2.0,
                 overlap_factor: float = None,
                 overhead_factor: float = None):
        super().__init__(K=K, intra_class_fair=True,
                         overlap_factor=overlap_factor,
                         overhead_factor=overhead_factor)


class LongLiuDWRRGapV3(LongLiuDWRR):
    """D1G-v3: DWRR + attain_bw gap 键 + 封顶 + 水填充。

    核心改进（相对于 D1G）：
    1. gap_i = max(0, attain_bw_i − bw_i)
       - attain_bw = bits / comm_budget（仅通信时间，不含 comp）
       - comm_budget = ci × iter_solo − comp
    2. 类映射：固定 log2 带（1,2,4,8,16,32,64 Gbps → P0-P6）
    3. 分配封顶：min(share, attain_bw)，盈余水填充
    4. starvation-free：floor_w 保底
    """

    # 固定 log2 带：gap > 1,2,4,8,16,32,64 Gbps → P0-P6
    GAP_THRESHOLDS = [1, 2, 4, 8, 16, 32, 64]  # Gbps

    def __init__(self,
                 floor_w: float = 2.0,
                 class_weights: list = None,
                 trace_file: str = None,
                 overlap_factor: float = None,
                 overhead_factor: float = None):
        """
        参数：
            floor_w: gap=0 保底权重
            class_weights: 自定义类权重表（None 则默认 STD）
            trace_file: 轨迹日志文件路径
            overlap_factor: compute-comm 重叠度
            overhead_factor: wire 因子（默认 1.3，锚点冻结值）
        """
        if overlap_factor is None:
            raise ValueError(
                "LongLiuDWRRGapV3 必须显式传入 overlap_factor"
            )
        if overhead_factor is None:
            raise ValueError(
                "LongLiuDWRRGapV3 必须显式传入 overhead_factor"
            )
        super().__init__(
            K=2.0, use_soft_weights=False, intra_class_fair=False,
            clip_ratio=100.0, class_weights=class_weights,
            trace_file=trace_file, overlap_factor=overlap_factor,
            overhead_factor=overhead_factor,
        )
        self.name = "LongLiuDWRRGapV3"
        self.floor_w = float(floor_w)

        # 上一轮分配的瓶颈带宽跟踪：jid -> bps
        self._prev_job_bw: Dict[str, float] = {}

    def _gap_to_level(self, gap_gbps: float) -> int:
        """将 gap 映射到 7 个类级 (0=P0, 6=P6)。

        固定 log2 带：gap > 1,2,4,8,16,32,64 Gbps → P0-P6
        """
        for lvl, thresh in enumerate(self.GAP_THRESHOLDS):
            if gap_gbps <= thresh:
                return lvl
        return 6  # > 64 Gbps

    def _level_to_dscp(self, level: int) -> int:
        """将类级 (0-6) 映射到 DSCP 值。"""
        mapping = {0: 0, 1: 18, 2: 28, 3: 26, 4: 36, 5: 34, 6: 38}
        return mapping.get(level, 0)

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """
        v3 带宽分配：gap 键 + 封顶 + 水填充。

        gap_i = max(0, attain_bw_i − bw_i)
        w_i = max(floor_w, gap_i / G0)  # G0 固定为 1 Gbps（log2 带已归一化）
        分配：bw_i = min(share_i, attain_bw_i)
        盈余：按 gap 权重水填充
        """
        if not flows or not links:
            return {}

        link = links[0]
        link_bw_bps = link.bw_bps

        # 1. 按 job 分组 flows
        job_flows = defaultdict(list)
        for flow in flows:
            job_flows[flow.jid].append(flow)

        # 2. 计算每个 job 的 attain_bw、gap、weight、DSCP 类
        job_gap = {}
        job_attain_bw = {}
        job_weight = {}
        job_level = {}
        job_dscp = {}

        for jid in job_flows:
            if jid not in job_stats:
                continue

            job = job_stats[jid]

            # attain_bw = bits / comm_budget（锚点语义）
            # comm_budget = ci × comm_solo × overhead
            # target = max(comp, comm_budget) + (1-overlap) × min(...)
            # eff_budget = target - comp → comm only
            comm_budget_ms = job.slo_ci * job.comm_solo_ms * self.overhead_factor
            if self.overlap_factor > 0:
                target_ms = max(job.comp_ms, comm_budget_ms) + \
                            (1.0 - self.overlap_factor) * min(job.comp_ms, comm_budget_ms)
            else:
                target_ms = job.comp_ms + comm_budget_ms
            eff_budget_ms = target_ms - job.comp_ms

            # 调试：检查 eff_budget_ms <= 0 的情况
            if eff_budget_ms <= 0:
                # 无通信预算（comp 占满迭代），attain_bw 设为 +inf（无上限）
                attain_bw_bps = float('inf')
                import warnings
                warnings.warn(
                    f"[v3] jid={jid} eff_budget_ms={eff_budget_ms:.1f} <= 0, "
                    f"setting attain_bw=inf (comp={job.comp_ms}, target={target_ms:.1f}, ci={job.slo_ci})"
                )
            else:
                attain_bw_bps = job.bits_per_iter / (eff_budget_ms * 1e-3)

            # bw_i: 上一轮分配的带宽（首次用 attain_bw 作为初始值）
            bw_bps = self._prev_job_bw.get(jid, attain_bw_bps if attain_bw_bps != float('inf') else 0.0)

            # gap [bps]
            gap_bps = max(0.0, attain_bw_bps - bw_bps)
            gap_gbps = gap_bps / 1e9

            # weight: max(floor_w, gap_i)（G0 固定为 1 Gbps）
            w = self.floor_w if gap_bps <= 0 else max(self.floor_w, gap_gbps)

            # DSCP 类（固定 log2 带）
            lvl = self._gap_to_level(gap_gbps)
            dscp = self._level_to_dscp(lvl)

            job_gap[jid] = gap_bps
            job_attain_bw[jid] = attain_bw_bps
            job_weight[jid] = w
            job_level[jid] = lvl
            job_dscp[jid] = dscp

        # 3. 按 DSCP 类分组
        class_jobs = defaultdict(list)
        for jid in job_flows:
            lvl = job_level.get(jid, 0)
            class_jobs[lvl].append(jid)

        # 4. 跨类 DWRR 分配
        active_classes = [lvl for lvl in range(7) if class_jobs[lvl]]
        if not active_classes:
            return {}

        total_weight = sum(self.class_weights[lvl] for lvl in active_classes)
        class_bw_share = {}
        for lvl in active_classes:
            class_bw_share[lvl] = (self.class_weights[lvl] / total_weight) * link_bw_bps

        # 5. 类内分配 + 封顶 + 水填充
        allocations = {}
        surplus_bw = 0.0
        new_job_bw = {}

        for lvl in active_classes:
            bw_budget = class_bw_share[lvl]
            jobs_in_class = class_jobs[lvl]

            # 按权重分配
            total_job_weight = sum(job_weight.get(jid, self.floor_w) for jid in jobs_in_class)
            if total_job_weight <= 0:
                total_job_weight = 1.0

            for jid in jobs_in_class:
                # share: 按权重分配
                job_weight_share = job_weight.get(jid, self.floor_w) / total_job_weight * bw_budget

                # 封顶：min(share, attain_bw)，attain_bw=inf 时不封顶
                attain_bw = job_attain_bw.get(jid, 0.0)
                if attain_bw == float('inf'):
                    actual_bw = job_weight_share  # 无上限，分配全部 share
                else:
                    actual_bw = min(job_weight_share, attain_bw)

                # 累积盈余
                surplus_bw += job_weight_share - actual_bw

                # 分配给 flow
                n_flows = len(job_flows[jid])
                for flow in job_flows[jid]:
                    allocations[flow] = {link: actual_bw / n_flows}

                new_job_bw[jid] = actual_bw

        # 6. 盈余水填充（按 gap 权重）
        if surplus_bw > 0:
            # 找出还有 gap 的 job（未达到 attain_bw）
            gap_jobs = [(jid, job_gap.get(jid, 0), job_weight.get(jid, self.floor_w))
                        for jid in new_job_bw
                        if job_gap.get(jid, 0) > 0]
            if gap_jobs:
                total_gap_weight = sum(w for _, _, w in gap_jobs)
                for jid, gap, w in gap_jobs:
                    if total_gap_weight > 0:
                        extra_bw = surplus_bw * w / total_gap_weight
                        new_job_bw[jid] += extra_bw
                        # 更新 allocation
                        n_flows = len(job_flows[jid])
                        for flow in job_flows[jid]:
                            allocations[flow] = {link: new_job_bw[jid] / n_flows}

        # 更新上一轮带宽跟踪
        self._prev_job_bw = dict(new_job_bw)

        # 7. Trace 记录
        if self._trace_handle:
            row = {"epoch": self._trace_epoch, "time_ms": time_ms,
                   "link_bw_gbps": link_bw_bps / 1e9, "floor_w": self.floor_w}
            for jid in job_gap:
                bw = new_job_bw.get(jid, 0.0)
                gap = job_gap.get(jid, 0.0)
                attain = job_attain_bw.get(jid, 0.0)
                w = job_weight.get(jid, self.floor_w)
                lvl = job_level.get(jid, 0)
                # 处理 inf 的情况
                attain_str = "inf" if attain == float('inf') else f"{round(attain / 1e9, 1)}"
                row[f"{jid}_attain_gbps"] = attain_str
                row[f"{jid}_gap_gbps"] = round(gap / 1e9, 2)
                row[f"{jid}_weight"] = round(w, 2)
                row[f"{jid}_level"] = lvl
                row[f"{jid}_bw_gbps"] = round(bw / 1e9, 1)
                row[f"{jid}_share"] = round(bw / link_bw_bps, 4) if link_bw_bps > 0 else 0.0
            self._trace_handle.write(json.dumps(row) + "\n")
            self._trace_epoch += 1

        return allocations

class LongLiuDWRRGapV31(LongLiuDWRR):
    """D1G-v3.1: 分层水填充（premium 池 + standard 池）。

    核心改进（相对于 v3）：
    1. 两层分配：premium 池优先，standard 池吃 residual
    2. premium 池上限 = min(Σattain_premium, capacity - standard_floor)
    3. gap 初始化 bw=0（修掉 attain_bw 初始化）
    4. 池内：连续 gap 权重 + 封顶 + water-filling 迭代

    tier 归属即 SLO 语义（ci 划分），非系数。
    """

    # Premium ci 阈值：ci <= PREMIUM_CI_THRESHOLD 为 premium
    PREMIUM_CI_THRESHOLD = 2.0

    def __init__(self,
                 floor_w: float = 2.0,
                 standard_floor_ratio: float = 0.2,
                 trace_file: str = None,
                 overlap_factor: float = None,
                 overhead_factor: float = None):
        if overlap_factor is None:
            raise ValueError("LongLiuDWRRGapV31 必须显式传入 overlap_factor")
        if overhead_factor is None:
            raise ValueError("LongLiuDWRRGapV31 必须显式传入 overhead_factor")
        super().__init__(
            K=2.0, use_soft_weights=False, intra_class_fair=False,
            clip_ratio=100.0, class_weights=None,
            trace_file=trace_file, overlap_factor=overlap_factor,
            overhead_factor=overhead_factor,
        )
        self.name = "LongLiuDWRRGapV31"
        self.floor_w = float(floor_w)
        self.standard_floor_ratio = standard_floor_ratio
        self._prev_job_bw: Dict[str, float] = {}

    def _compute_attain_bw(self, job) -> float:
        """锚点语义 attain_bw（与 metrics.compute_target_iter_ms 同源）。"""
        comm_budget_ms = job.slo_ci * job.comm_solo_ms * self.overhead_factor
        if self.overlap_factor > 0:
            target_ms = max(job.comp_ms, comm_budget_ms) + \
                        (1.0 - self.overlap_factor) * min(job.comp_ms, comm_budget_ms)
        else:
            target_ms = job.comp_ms + comm_budget_ms
        eff_budget_ms = target_ms - job.comp_ms
        if eff_budget_ms <= 0:
            return float('inf')
        else:
            return job.bits_per_iter / (eff_budget_ms * 1e-3)

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        if not flows or not links:
            return {}

        link = links[0]
        link_bw_bps = link.bw_bps

        job_flows = defaultdict(list)
        for flow in flows:
            job_flows[flow.jid].append(flow)

        job_attain_bw = {}
        job_tier = {}

        for jid in job_flows:
            if jid not in job_stats:
                continue
            job = job_stats[jid]
            job_attain_bw[jid] = self._compute_attain_bw(job)
            job_tier[jid] = 'premium' if job.slo_ci <= self.PREMIUM_CI_THRESHOLD else 'standard'

        premium_jids = [jid for jid in job_flows if job_tier.get(jid) == 'premium']
        standard_jids = [jid for jid in job_flows if job_tier.get(jid) == 'standard']

        sigma_attain_premium = sum(
            job_attain_bw.get(jid, 0) for jid in premium_jids
            if job_attain_bw.get(jid) != float('inf')
        )
        standard_floor = link_bw_bps * self.standard_floor_ratio
        premium_pool_max = min(sigma_attain_premium, link_bw_bps - standard_floor)

        infeasible = sigma_attain_premium > premium_pool_max
        if infeasible:
            import warnings
            warnings.warn(f"[v3.1] infeasible: Σattain_premium={sigma_attain_premium/1e9:.0f}G > premium_pool_max={premium_pool_max/1e9:.0f}G")

        allocations = {}
        new_job_bw = {}

        if premium_jids:
            premium_alloc = self._water_filling_allocate(
                premium_jids, premium_pool_max, job_attain_bw
            )
            new_job_bw.update(premium_alloc)

        if standard_jids:
            used_by_premium = sum(new_job_bw.get(jid, 0) for jid in premium_jids)
            standard_pool_bw = max(link_bw_bps - used_by_premium, standard_floor)
            standard_alloc = self._water_filling_allocate(
                standard_jids, standard_pool_bw, job_attain_bw
            )
            new_job_bw.update(standard_alloc)

        for jid, bw_bps in new_job_bw.items():
            n_flows = len(job_flows[jid])
            for flow in job_flows[jid]:
                allocations[flow] = {link: bw_bps / n_flows}

        self._prev_job_bw = dict(new_job_bw)

        if self._trace_handle:
            row = {"epoch": self._trace_epoch, "time_ms": time_ms,
                   "link_bw_gbps": link_bw_bps / 1e9,
                   "premium_pool_max_gbps": premium_pool_max / 1e9,
                   "infeasible": infeasible}
            for jid in job_attain_bw:
                bw = new_job_bw.get(jid, 0.0)
                attain = job_attain_bw.get(jid, 0.0)
                tier = job_tier.get(jid, "unknown")
                attain_str = "inf" if attain == float('inf') else f"{attain / 1e9:.1f}"
                row[f"{jid}_tier"] = tier
                row[f"{jid}_attain_gbps"] = attain_str
                row[f"{jid}_bw_gbps"] = round(bw / 1e9, 1)
            self._trace_handle.write(json.dumps(row) + "\n")
            self._trace_epoch += 1

        return allocations

    def _water_filling_allocate(self, jids: List[str], pool_bw: float,
                                 job_attain_bw: Dict[str, float]) -> Dict[str, float]:
        max_iter = 10
        converge_threshold = 0.01

        bw = {jid: 0.0 for jid in jids}

        for iteration in range(max_iter):
            gap = {}
            for jid in jids:
                attain = job_attain_bw.get(jid, 0.0)
                if attain == float('inf'):
                    attain = 1e18
                gap[jid] = max(0.0, attain - bw.get(jid, 0.0))

            weight = {}
            for jid in jids:
                g = gap.get(jid, 0.0) / 1e9
                weight[jid] = max(self.floor_w, g)

            total_weight = sum(weight.values())
            if total_weight <= 0:
                total_weight = 1.0

            new_bw = {}
            surplus = 0.0
            for jid in jids:
                share = weight.get(jid, self.floor_w) / total_weight * pool_bw
                attain = job_attain_bw.get(jid, 0.0)
                if attain == float('inf'):
                    actual = share
                else:
                    actual = min(share, attain)
                new_bw[jid] = actual
                surplus += share - actual

            if surplus > 0:
                gap_jobs = [jid for jid in jids if gap.get(jid, 0) > 0]
                if gap_jobs:
                    gap_weight = sum(weight.get(jid, self.floor_w) for jid in gap_jobs)
                    if gap_weight > 0:
                        for jid in gap_jobs:
                            extra = surplus * weight.get(jid, self.floor_w) / gap_weight
                            attain = job_attain_bw.get(jid, 0.0)
                            if attain == float('inf'):
                                new_bw[jid] += extra
                            else:
                                new_bw[jid] = min(new_bw[jid] + extra, attain)

            if bw:
                max_change = max(
                    abs(new_bw.get(jid, 0) - bw.get(jid, 0)) / max(bw.get(jid, 1), 1)
                    for jid in jids
                )
            else:
                max_change = 0
            bw = new_bw

            if max_change < converge_threshold:
                break

        return bw


class LongLiuAllocatorV4(LongLiuDWRR):
    """v4: 闭式 SLO 分配器（无控制回路）。
    
    三公理：
    1. SLO 优先：若 Σattain_P ≤ C − β·Σattain_S，所有 premium 精确拿到 attain_i
    2. 有界降级：standard 至少拿到 β·Σattain_S（β=0.5，预登记降级界限）
    3. work-conserving：任一未封顶 job 还要带宽时不许闲置容量
    
    闭式解：
    - 可行域：premium 精确 attain，standard 在残余上等 sas 水位线 capped-filling
    - 不可行域：premium 在 C−β·Σattain_S 上等 sas 水位线，standard 在 β·Σattain_S 上
    """

    PREMIUM_CI_THRESHOLD = 2.0
    BETA = 0.5  # standard 降级界限（预登记 S-cont-cap≥0.5）

    def __init__(self, overlap_factor: float = None, overhead_factor: float = None, trace_file: str = None):
        if overlap_factor is None:
            raise ValueError("LongLiuAllocatorV4 必须显式传入 overlap_factor")
        if overhead_factor is None:
            raise ValueError("LongLiuAllocatorV4 必须显式传入 overhead_factor")
        super().__init__(
            K=2.0, use_soft_weights=False, intra_class_fair=False,
            clip_ratio=100.0, class_weights=None,
            trace_file=trace_file, overlap_factor=overlap_factor,
            overhead_factor=overhead_factor,
        )
        self.name = "LongLiuAllocatorV4"
        self.overhead_factor = overhead_factor
        self._trace_epoch = 0

    def _compute_attain_bw(self, job) -> float:
        """计算 job 的 attain_bw (bps) —— 锚点语义（SEMANTICS_VERSION="anchor-v2"）。

        wire_bits = bits_per_iter × overhead_factor
        comm_budget = ci × comm_solo × overhead_factor
        target = max(comp, comm_budget) + (1−overlap) × min(comp, comm_budget)
        attain_bw = wire_bits / (target − comp)

        overhead_factor 和 overlap_factor 从 config 读取（禁止硬编码）。
        """
        wire_bits = job.bits_per_iter * self.overhead_factor
        comm_budget_ms = job.slo_ci * job.comm_solo_ms * self.overhead_factor

        if self.overlap_factor > 0:
            target_ms = max(job.comp_ms, comm_budget_ms) + \
                        (1.0 - self.overlap_factor) * min(job.comp_ms, comm_budget_ms)
        else:
            target_ms = job.comp_ms + comm_budget_ms

        effective_budget_ms = target_ms - job.comp_ms

        if effective_budget_ms <= 0:
            raise ValueError(
                f"job {job.jid}: effective_budget_ms={effective_budget_ms:.1f} <= 0, "
                f"comp={job.comp_ms}, target={target_ms:.1f}, ci={job.slo_ci}"
            )

        return wire_bits / (effective_budget_ms * 1e-3)

    def _capped_filling(self, jids: List[str], pool_bw: float,
                        job_attain_bw: Dict[str, float]) -> Dict[str, float]:
        """等 sas 水位线 capped-filling（闭式解）。
        
        算法：按 attain 排序，渐进填充直到达到水位线 λ。
        """
        if not jids:
            return {}

        # 按 attain 排序（小到大）
        sorted_jids = sorted(jids, key=lambda jid: job_attain_bw.get(jid, float('inf')))
        
        # 渐进填充
        remaining_pool = pool_bw
        allocations = {}
        
        for i, jid in enumerate(sorted_jids):
            attain = job_attain_bw.get(jid, 0.0)
            n_remaining = len(sorted_jids) - i  # 剩余 job 数
            
            # 水位线：每个 job 最多拿 attain，但总和不超过 remaining_pool
            # 理想情况：每个剩余 job 拿 λ·attain，Σ = remaining_pool
            # 但 λ·attain ≤ attain，所以 λ ≤ 1
            
            if attain == float('inf'):
                attain = 1e18
            
            # 计算水位线
            total_attain_remaining = sum(
                job_attain_bw.get(j, 0) for j in sorted_jids[i:]
                if job_attain_bw.get(j, 0) != float('inf')
            )
            if total_attain_remaining <= 0:
                total_attain_remaining = 1.0
            
            lambda_factor = min(1.0, remaining_pool / total_attain_remaining)
            
            # 分配
            alloc = min(attain, lambda_factor * attain)
            allocations[jid] = alloc
            remaining_pool -= alloc

        return allocations

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """v4 闭式分配。"""
        if not flows or not links:
            return {}

        # 容量 C：全部 spine 链路带宽之和（修 links[0] 砍半 bug）
        # 优先使用注入的 topology 引用
        if self._topology and hasattr(self._topology, 'spine_links'):
            capacity_bps = sum(link.bw_bps for link in self._topology.spine_links)
        else:
            # fallback：假设所有 links 都是 spine links（向后兼容单链路场景）
            capacity_bps = sum(link.bw_bps for link in links)

        link = links[0]  # 用于构建 allocation

        # 1. 按 job 分组 flows
        job_flows = defaultdict(list)
        for flow in flows:
            job_flows[flow.jid].append(flow)

        # 2. 计算 attain_bw 和 tier
        job_attain_bw = {}
        job_tier = {}

        for jid in job_flows:
            if jid not in job_stats:
                continue
            job = job_stats[jid]
            job_attain_bw[jid] = self._compute_attain_bw(job)
            job_tier[jid] = 'premium' if job.slo_ci <= self.PREMIUM_CI_THRESHOLD else 'standard'

        # 3. 分层
        premium_jids = [jid for jid in job_flows if job_tier.get(jid) == 'premium']
        standard_jids = [jid for jid in job_flows if job_tier.get(jid) == 'standard']

        # 4. 计算 Σattain
        sigma_attain_premium = sum(
            job_attain_bw.get(jid, 0) for jid in premium_jids
        )
        sigma_attain_standard = sum(
            job_attain_bw.get(jid, 0) for jid in standard_jids
        )

        # 5. 可行性判定
        standard_floor = self.BETA * sigma_attain_standard
        feasible = sigma_attain_premium <= capacity_bps - standard_floor

        allocations = {}
        job_bw = {}

        # 6. 分配
        if feasible:
            # 可行域：premium 精确 attain
            for jid in premium_jids:
                job_bw[jid] = job_attain_bw.get(jid, 0)
            
            # standard 在残余上等 sas 水位线
            residual = capacity_bps - sum(job_bw.get(jid, 0) for jid in premium_jids)
            standard_alloc = self._capped_filling(standard_jids, residual, job_attain_bw)
            job_bw.update(standard_alloc)
        else:
            # 不可行域：premium 在 C−β·Σattain_S 上等 sas 水位线
            premium_pool = capacity_bps - standard_floor
            premium_alloc = self._capped_filling(premium_jids, premium_pool, job_attain_bw)
            job_bw.update(premium_alloc)
            
            # standard 在 β·Σattain_S 上等 sas 水位线
            standard_alloc = self._capped_filling(standard_jids, standard_floor, job_attain_bw)
            job_bw.update(standard_alloc)

        # 7. 盈余再分配（分给 premium）
        used = sum(job_bw.values())
        surplus = capacity_bps - used
        if surplus > 0 and premium_jids:
            # 分给未封顶的 premium
            uncapped_premium = [
                jid for jid in premium_jids
                if job_bw.get(jid, 0) < job_attain_bw.get(jid, float('inf'))
            ]
            if uncapped_premium:
                per_job_surplus = surplus / len(uncapped_premium)
                for jid in uncapped_premium:
                    attain = job_attain_bw.get(jid, float('inf'))
                    job_bw[jid] = min(job_bw.get(jid, 0) + per_job_surplus, attain)

        # 8. 构建 allocations
        for jid, bw_bps in job_bw.items():
            n_flows = len(job_flows[jid])
            for flow in job_flows[jid]:
                allocations[flow] = {link: bw_bps / n_flows}

        # 9. Trace 记录
        if self._trace_handle:
            row = {
                "epoch": self._trace_epoch,
                "time_ms": time_ms,
                "capacity_gbps": capacity_bps / 1e9,
                "sigma_attain_premium_gbps": sigma_attain_premium / 1e9,
                "sigma_attain_standard_gbps": sigma_attain_standard / 1e9,
                "standard_floor_gbps": standard_floor / 1e9,
                "feasible": feasible,
                "regime": "feasible" if feasible else "infeasible",
            }
            for jid in job_attain_bw:
                bw = job_bw.get(jid, 0.0)
                attain = job_attain_bw.get(jid, 0.0)
                tier = job_tier.get(jid, "unknown")
                row[f"{jid}_tier"] = tier
                row[f"{jid}_attain_gbps"] = round(attain / 1e9, 1)
                row[f"{jid}_bw_gbps"] = round(bw / 1e9, 1)
                row[f"{jid}_sas"] = round(bw / attain, 3) if attain > 0 else 0.0
            self._trace_handle.write(json.dumps(row) + "\n")
            self._trace_epoch += 1

        return allocations
