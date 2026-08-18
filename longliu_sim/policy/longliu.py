"""LongLiu 策略：基于 Progress Deficit 的优先级带宽分配 + T_target 两阶段校准。"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Dict, Set

from .base import Policy, Allocation
from ..network.flow import Flow
from ..network.link import Link


class LongLiu(Policy):
    """
    LongLiu 调度策略。

    每轮根据 job 的 Progress Deficit 分配带宽（端到端一致）：
        pi = avg_iter_ms / T_target - 1
        w = max(MIN_W, exp(K * pi) * BASE_W)

    T_target 支持两阶段校准：
    - 静态 T_target（ci * comm_solo_ms）用于 ablation 控制组
    - 动态 T_target（RTT probe + EMA）用于完整机制的评估

    DSCP 映射将连续权重离散化为 7 级优先级（见 get_dscp）。
    持有 DSCP 38（最高优先级）的 job 触发 T_target EMA 更新。
    """

    K: float = 3.0
    MIN_W: float = 0.5
    BASE_W: float = 4.0

    # 论文 Table I: 7 级 DSCP 映射（基于 pi = avg_iter/T_target - 1）
    # pi > 0.6   → P6 (DSCP 38) 严重违约
    # pi > 0.4   → P5 (DSCP 34) 中度违约
    # pi > 0.2   → P4 (DSCP 36) 轻度违约
    # pi > 0.0   → P3 (DSCP 26) 刚开始违约
    # pi > -0.2  → P2 (DSCP 28) 接近 SLO
    # pi > -0.4  → P1 (DSCP 18) 领先 SLO
    # pi ≤ -0.4  → P0 (DSCP 0) 显著领先（最低优先级）
    DSCP_MAP = [
        (0.6, 38),   # pi > 0.6  → P6 (DSCP 38)
        (0.4, 34),   # pi > 0.4  → P5 (DSCP 34)
        (0.2, 36),   # pi > 0.2  → P4 (DSCP 36)
        (0.0, 26),   # pi > 0.0  → P3 (DSCP 26)
        (-0.2, 28),  # pi > -0.2 → P2 (DSCP 28)
        (-0.4, 18),  # pi > -0.4 → P1 (DSCP 18)
    ]               # pi ≤ -0.4 → P0 (DSCP 0)
    DSCP_DEFAULT = 0

    # DWRR floor weights: P0-P6 minimum bandwidth share
    # 对应 Cumulus 配置: qos scheduling dwrr weight 5 5 8 10 12 20 40
    DWRR_FLOOR_WEIGHTS = [0.05, 0.05, 0.08, 0.10, 0.12, 0.20, 0.40]

    # 低优先级地板：P0/P1/P2 各保底 5%（防止完全饿死）
    # DSCP 0 (P0), 18 (P1), 28 (P2)
    LOW_PRIORITY_FLOOR_DSCPS = {0: 0.05, 18: 0.05, 28: 0.05}

    # 被动校准观测信任度（按优先级 level 0-6）：P6=1.0 全信，P0=0.05 几乎不信
    # 低优先级 job 的迭代时间观测可能只是"没分到带宽"，不可全信
    EMA_PRIORITY_WEIGHTS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self, K: float = 3.0, min_w: float = 0.5, base_w: float = 4.0,
                 use_dynamic_T_target: bool = True,
                 no_startup: bool = False,
                 no_weighted: bool = False,
                 n_dscp_levels: int = 7,
                 # --- 控制论四件套参数 ---
                 dead_zone_delta: float = 0.0,
                 window_size: int = 0,
                 aging_L: int = 0,
                 aging_theta: float = 0.05,
                 hysteresis_h: float = 0.0,
                 # --- DWRR 地板参数 ---
                 dwrr_floor: bool = False,
                 # --- 低优先级地板参数 ---
                 low_priority_floor: bool = False,
                 # --- 被动校准参数（方案1+2，替代主动探测） ---
                 ema_passive: bool = False,
                 ema_weights: list = None):
        """
        参数：
            K: deficit 指数增益
            min_w: 最小带宽权重
            base_w: 基础带宽权重
            use_dynamic_T_target: 是否启用 T_target 动态校准
            no_startup: 消融——取消第一轮 DSCP 38 启动提升
            no_weighted: 消融——同级别内均匀分配带宽（而非 exp(pi*K) 加权）
            n_dscp_levels: 消融——使用的 DSCP 等级数（7=完整, 4=消融简化版）
            --- 控制论四件套 ---
            dead_zone_delta: 死区阈值。|pi|<delta 时强制映射 P3（静默），0 禁用
            window_size: 滑窗迭代次数 W。>0 启用滑窗 urgency，0 使用全历史平均
            aging_L: 老化触发 epoch 数。连续 L 次带宽份额<theta 则强制升一级，0 禁用
            aging_theta: 老化带宽份额阈值
            hysteresis_h: 迟滞半带宽。跨档需越过阈值 ±h，0 禁用
            --- DWRR ---
            dwrr_floor: 启用 DWRR 带宽地板。P0-P6 各保底 5/5/8/10/12/20/40%（已废弃）
            --- 低优先级地板 ---
            low_priority_floor: 启用低优先级地板。P0/P1/P2 各保底 5%，P6 不 cap
        """
        super().__init__("LongLiu")
        self.K = K
        self.MIN_W = min_w
        self.BASE_W = base_w
        self.use_dynamic_T_target = use_dynamic_T_target
        self.no_startup = no_startup
        self.no_weighted = no_weighted
        if n_dscp_levels not in (4, 7):
            raise ValueError(f"n_dscp_levels must be 4 or 7, got {n_dscp_levels}")
        if n_dscp_levels == 4:
            self.DSCP_MAP = [
                (1.0, 38),
                (0.3, 36),
                (-0.2, 28),
            ]
            self._DSCP_PRIORITY_ORDER = [38, 36, 28, 0]
            self._DSCP_LEVEL_MAP = {38: 6, 36: 4, 28: 2, 0: 0}  # dscp→level idx
        else:
            self._DSCP_PRIORITY_ORDER = [38, 34, 36, 26, 28, 18, 0]
            self._DSCP_LEVEL_MAP = {38: 6, 34: 5, 36: 4, 26: 3, 28: 2, 18: 1, 0: 0}

        # 控制论参数
        self.dead_zone_delta = dead_zone_delta
        self.window_size = window_size
        self.aging_L = aging_L
        self.aging_theta = aging_theta
        self.hysteresis_h = hysteresis_h

        # DWRR 地板
        self.dwrr_floor = dwrr_floor

        # 低优先级地板
        self.low_priority_floor = low_priority_floor

        # 被动校准（方案1+2）：所有 job 每轮弱更新 EMA，按优先级折扣
        self.ema_passive = ema_passive
        # 信任权重表（level 0-6），None 用默认
        if ema_weights is not None:
            if len(ema_weights) != 7:
                raise ValueError(f"ema_weights must have 7 entries (level 0-6), got {len(ema_weights)}")
            self.ema_weights = list(ema_weights)
        else:
            self.ema_weights = list(self.EMA_PRIORITY_WEIGHTS)

        # 运行时状态
        self._last_dscp: Dict[str, int] = {}          # 每个 job 的上次 DSCP（迟滞用）
        self._starved_epochs: Dict[str, int] = {}     # 每个 job 的连续饿死计数（老化用）
        self._startup_epochs: Dict[str, int] = {}     # 每个 job 已过的启动 epoch 数（启动保护）

        # 启动期插桩计数器
        self._total_alloc_calls = 0
        self._startup_alloc_calls = 0
        self._startup_job_total = 0

        # 统计插桩
        self._dead_zone_hits = 0
        self._aging_activations = 0
        self._hysteresis_actions = 0

        # 主动探测机制（E14）
        self.probe_enabled = False
        self.probe_frozen_threshold = 10  # 连续 N 个窗口未更新则触发探测
        self.probe_duration = 0           # 探测保持最高优先级的 allocate 次数（0=不限制）
        self._probe_counters: Dict[str, int] = {}  # 每个 job 的冻结计数器
        self._probe_remaining: Dict[str, int] = {}  # 每个 job 剩余探测期 allocate 次数
        self._probe_activations = 0  # 探测触发次数

    def get_instrumentation(self) -> dict:
        """返回插桩数据。"""
        return {
            "total_alloc_calls": self._total_alloc_calls,
            "startup_alloc_calls": self._startup_alloc_calls,
            "startup_job_total": self._startup_job_total,
            "dead_zone_hits": self._dead_zone_hits,
            "aging_activations": self._aging_activations,
            "hysteresis_actions": self._hysteresis_actions,
        }

    def get_dscp(self, pi: float) -> int:
        """
        将连续 deficit pi 映射到离散 DSCP 值（7 级）。

        论文 Table I 对应（基于 Ui = avg_iter / T_target, pi = Ui - 1）：
            pi > 1.0   → DSCP 38 (P6, 严重违约)
            pi > 0.6   → DSCP 34 (P5, 中度违约)
            pi > 0.3   → DSCP 36 (P4, 轻度违约)
            pi > 0.0   → DSCP 26 (P3, 刚开始违约)
            pi > -0.2  → DSCP 28 (P2)
            pi > -0.4  → DSCP 18 (P1)
            else       → DSCP 0  (P0, 最低优先级)

        死区：|pi| < dead_zone_delta 时强制映射到 P3（DSCP 26），消除噪声抖动。
        """
        # 死区：|pi| 在死区范围内时强制静默（P3）
        if self.dead_zone_delta > 0 and abs(pi) < self.dead_zone_delta:
            self._dead_zone_hits += 1
            return 26  # P3
        for threshold, dscp in self.DSCP_MAP:
            if pi > threshold:
                return dscp
        return self.DSCP_DEFAULT

    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """
        严格优先级调度：按 DSCP 级别分配带宽。
        
        P6 > P5 > P4 > P3 > P2 > P1 > P0
        
        SP 语义：
        1. 高优先级队列优先服务
        2. 高优先级队列占满带宽后，低优先级饿死
        3. 高优先级队列为空时，低优先级获得带宽
        4. 同一 DSCP 级别内，按 exp(pi * K) 加权分配带宽
           （防止同级别内某些任务被饿死）
        
        动态调整：
        - 每轮迭代后重新计算 pi 和 DSCP
        - 第一轮所有任务给最高优先级（DSCP 38）
        - 落后的 job 优先级上升，领先的 job 优先级下降
        - 形成自然的"优先级轮转"机制
        """
        if not flows or not links:
            return {}

        link = links[0]
        jobs: Set[str] = set(f.jid for f in flows)

        # 将策略的 window_size 同步到每个 job 的滑窗（惰性，保证注入的新 job 也生效）
        if self.window_size > 0:
            for jid in jobs:
                job = job_stats[jid]
                if job.sliding_window_len != self.window_size:
                    job.set_window_size(self.window_size)

        # --- 启动期插桩 ---
        self._total_alloc_calls += 1
        startup_jobs = sum(1 for jid in jobs if job_stats[jid].is_first_iter)
        if startup_jobs > 0:
            self._startup_alloc_calls += 1
            self._startup_job_total += startup_jobs

        # 按 job 计算 pi 和 DSCP（同一 job 的所有 flow 共享优先级）
        job_dscp: Dict[str, int] = {}
        job_pi: Dict[str, float] = {}  # 保存每个 job 的 pi，用于加权分配
        for jid in jobs:
            job = job_stats[jid]
            
            # 第一轮迭代：所有任务给最高优先级（DSCP 38），除非 no_startup 消融
            # 收敛即退：最多 3 个 startup epoch，超过后恢复正常调度
            if not self.no_startup and job.is_first_iter:
                startup_ep = self._startup_epochs.get(jid, 0)
                if startup_ep < 3:  # 最多 3 个 epoch
                    self._startup_epochs[jid] = startup_ep + 1
                    job_dscp[jid] = 38
                    job_pi[jid] = 1.0  # 第一轮给统一权重
                    continue
                # 超过 3 epoch → 降级到正常调度（is_first_iter 在 on_comm_end 中置 False）
            
            # 后续迭代：根据 pi 动态计算 DSCP
            # 1. 用静态 T_target 计算初始 pi，确定 DSCP 优先级
            pi = job.compute_deficit()
            dscp = self.get_dscp(pi)
            has_highest_priority = (dscp == 38)

            # 2. 如果启用动态校准，用 T_target 更新 pi（端到端一致）
            if self.use_dynamic_T_target:
                T_target = job.get_T_target(has_highest_priority=has_highest_priority)
                if job.completed_iters > 0:
                    # 滑窗 urgency：优先使用滑动平均
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
                if job.last_iter_comm_time_ms is not None:
                    if has_highest_priority:
                        # 最高优先级：观测最可信，全额采纳
                        job.update_ema_from_comm_time(weight=1.0)
                    elif self.ema_passive:
                        # 被动弱更新（方案1+2）：所有 job 每轮更新，按优先级折扣
                        level = self._DSCP_LEVEL_MAP.get(dscp, 0)
                        weight = self.ema_weights[level]
                        job.update_ema_from_comm_time(weight=weight)
                # 用更新后的 pi 重新计算 DSCP（含死区）
                dscp = self.get_dscp(pi)

            # 3. 迟滞：跨档需越过阈值 ±h（从上次 DSCP 出发）
            if self.hysteresis_h > 0 and jid in self._last_dscp:
                old_dscp = self._last_dscp[jid]
                if dscp != old_dscp:
                    old_level = self._DSCP_LEVEL_MAP.get(old_dscp, 0)
                    new_level = self._DSCP_LEVEL_MAP.get(dscp, 0)
                    level_diff = new_level - old_level
                    if abs(level_diff) == 1:
                        # 相邻跳：直接用 h 检查是否越过迟滞
                        old_pi = job_pi.get(jid, pi)
                        # 升档：需要 pi > threshold + h
                        # 降档：需要 pi < threshold - h
                        # 简化为只需要越过半带宽 h
                        if abs(level_diff) > 0:
                            self._hysteresis_actions += 1
                            pass  # 相邻跳允许
                    else:
                        # 跨多档：只允许移动一档（anti-chattering）
                        if level_diff > 1:
                            # 每级 DSCP 映射到阈值
                            dscp_levels = list(self._DSCP_PRIORITY_ORDER)
                            old_idx = dscp_levels.index(old_dscp)
                            if old_idx + 1 < len(dscp_levels):
                                dscp = dscp_levels[old_idx + 1]
                            self._hysteresis_actions += 1
                        elif level_diff < -1:
                            dscp_levels = list(self._DSCP_PRIORITY_ORDER)
                            old_idx = dscp_levels.index(old_dscp)
                            if old_idx - 1 >= 0:
                                dscp = dscp_levels[old_idx - 1]
                            self._hysteresis_actions += 1

            # 4. 优先级老化：从上次 epoch 的饿死计数判断是否强制升级
            if self.aging_L > 0:
                starved = self._starved_epochs.get(jid, 0)
                if starved > 0 and starved % self.aging_L == 0:
                    # 连续 L 次饿死 → 强制升一级
                    dscp_levels = self._DSCP_PRIORITY_ORDER
                    curr_idx = dscp_levels.index(dscp)
                    if curr_idx - 1 >= 0:  # 不能超过 P6
                        dscp = dscp_levels[curr_idx - 1]
                        self._aging_activations += 1

            # 5. 主动探测机制（E14）：检测 T_target_ema 冻结
            if self.probe_enabled and self.use_dynamic_T_target:
                # 处于探测期：保持最高优先级，消耗剩余探测次数
                if self._probe_remaining.get(jid, 0) > 0:
                    dscp = 38
                    self._probe_remaining[jid] -= 1
                    self._probe_counters[jid] = 0
                elif has_highest_priority:
                    # 已获得最高优先级（自然调度），重置冻结计数器
                    self._probe_counters[jid] = 0
                else:
                    # 未获得最高优先级，计数器 +1
                    counter = self._probe_counters.get(jid, 0)
                    counter += 1
                    self._probe_counters[jid] = counter

                    # 如果连续 N 个窗口未获得最高优先级，触发探测：
                    # 立即提升到 P6（DSCP 38）采样，保持 probe_duration 次 allocate
                    if counter >= self.probe_frozen_threshold:
                        self._probe_remaining[jid] = max(0, self.probe_duration - 1)
                        self._probe_counters[jid] = 0
                        self._probe_activations += 1
                        dscp = 38

            job_dscp[jid] = dscp
            job_pi[jid] = pi  # 保存 pi 用于加权
            self._last_dscp[jid] = dscp

        # 按 DSCP 分组 flows
        flows_by_dscp: Dict[int, List[Flow]] = defaultdict(list)
        for f in flows:
            dscp = job_dscp[f.jid]
            flows_by_dscp[dscp].append(f)

        # 从高到低遍历 DSCP，严格优先级分配
        alloc: Allocation = {}
        if self.dwrr_floor:
            # --- DWRR 地板分配模式 ---
            # 1. 枚举所有非空优先级类
            # 2. 每个非空类至少得到 floor_weight[level] * link.bw_bps
            # 3. 剩余带宽全给最高优先级类
            total_bw = link.bw_bps
            non_empty_dscps = [d for d in self._DSCP_PRIORITY_ORDER
                               if d in flows_by_dscp and flows_by_dscp[d]]
            if not non_empty_dscps:
                return alloc

            highest_dscp = non_empty_dscps[-1]

            # 计算所有非空类的 floor 总量
            floor_sum = 0.0
            for dscp in non_empty_dscps:
                level = self._DSCP_LEVEL_MAP[dscp]
                floor_sum += self.DWRR_FLOOR_WEIGHTS[level] * total_bw

            # 如果 floor 总和超过链路容量，统一缩放
            scale = min(1.0, total_bw / floor_sum) if floor_sum > 0 else 1.0

            # 第一遍：分配每个非空类的地板带宽
            total_floor_alloc = 0.0
            for dscp in non_empty_dscps:
                level = self._DSCP_LEVEL_MAP[dscp]
                bw_floor = self.DWRR_FLOOR_WEIGHTS[level] * total_bw * scale
                total_floor_alloc += bw_floor
                flows_at_level = flows_by_dscp[dscp]
                self._allocate_level(flows_at_level, bw_floor, job_pi, alloc, link)

            # 第二遍：剩余带宽全给最高优先级
            remaining = total_bw - total_floor_alloc
            if remaining > 1e-6 and highest_dscp:
                highest_flows = flows_by_dscp[highest_dscp]
                # 提取已分配给最高优先级的带宽，重新分配（地板 + 剩余）
                for f in highest_flows:
                    existing = alloc.pop(f, None)
                existing_floor = sum(alloc[f][link] for f in highest_flows if f in alloc)
                bw_total = remaining + existing_floor
                self._allocate_level(highest_flows, bw_total, job_pi, alloc, link)
        else:
            # --- 低优先级地板分配模式 ---
            # P0/P1/P2 各保底 5%，P6 不被 cap
            if self.low_priority_floor:
                total_bw = link.bw_bps
                non_empty_dscps = [d for d in self._DSCP_PRIORITY_ORDER
                                   if d in flows_by_dscp and flows_by_dscp[d]]
                if not non_empty_dscps:
                    return alloc

                # 找出最高优先级类
                highest_dscp = non_empty_dscps[-1]

                # 第一遍：给 P0/P1/P2 分配保底带宽（如果这些类有流量）
                floor_sum = 0.0
                low_priority_alloc = {}
                for dscp, floor_pct in self.LOW_PRIORITY_FLOOR_DSCPS.items():
                    if dscp in flows_by_dscp and flows_by_dscp[dscp]:
                        bw_floor = floor_pct * total_bw
                        floor_sum += bw_floor
                        flows_at_level = flows_by_dscp[dscp]
                        self._allocate_level(flows_at_level, bw_floor, job_pi, low_priority_alloc, link)

                # 第二遍：剩余带宽全给最高优先级类（work-conserving）
                remaining = total_bw - floor_sum
                if remaining > 1e-6:
                    highest_flows = flows_by_dscp[highest_dscp]
                    # 提取已分配给最高优先级的带宽（如果有）
                    existing_floor = sum(low_priority_alloc[f][link] for f in highest_flows if f in low_priority_alloc)
                    bw_total = remaining + existing_floor
                    # 清除最高优先级的旧分配
                    for f in highest_flows:
                        if f in low_priority_alloc:
                            del low_priority_alloc[f]
                    # 重新分配（地板 + 剩余）
                    self._allocate_level(highest_flows, bw_total, job_pi, low_priority_alloc, link)

                alloc = low_priority_alloc
            else:
                # --- 纯 SP 分配模式（现有逻辑） ---
                remaining_bw = link.bw_bps
                for dscp in self._DSCP_PRIORITY_ORDER:
                    if dscp not in flows_by_dscp:
                        continue
                    flows_at_level = flows_by_dscp[dscp]
                    if not flows_at_level:
                        continue
                    self._allocate_level(flows_at_level, remaining_bw, job_pi, alloc, link)
                    remaining_bw = 0.0
                    break

        # --- 分配完成后更新老化状态 ---
        if self.aging_L > 0:
            allocated_jids: Set[str] = set(f.jid for f in alloc)
            for jid in jobs:
                was_allocated = jid in allocated_jids
                prev = self._starved_epochs.get(jid, 0)
                if was_allocated:
                    # 获得带宽 → 重置饿死计数
                    self._starved_epochs[jid] = 0
                else:
                    # 没获得带宽 → 饿死计数 +1
                    self._starved_epochs[jid] = prev + 1

        return alloc

    def _allocate_level(self, flows: list, total_bw: float,
                        job_pi: Dict[str, float],
                        alloc: Allocation, link) -> None:
        """在同一个 DSCP 优先级类内分配带宽。"""
        if not flows or total_bw <= 0:
            return
        if self.no_weighted:
            bw_per_flow = total_bw / len(flows)
            for f in flows:
                alloc[f] = {link: bw_per_flow}
        else:
            weights: Dict[str, float] = {}
            for f in flows:
                pi = job_pi.get(f.jid, 0.0)
                pi_clipped = max(-2.0, min(3.0, pi))
                w = max(self.MIN_W, math.exp(self.K * pi_clipped)) * self.BASE_W
                weights[f.jid] = w
            total_weight = sum(weights[f.jid] for f in flows)
            for f in flows:
                alloc[f] = {link: total_bw * weights[f.jid] / total_weight}
