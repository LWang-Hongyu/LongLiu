"""Job 模型：支持 AllReduce 多流 + Barrier 语义 + T_target 两阶段校准。"""

from __future__ import annotations
import math
from collections import deque

from ..utils.model_params import get_comp_ms


class Job:
    """
    一个训练任务，按迭代产生 compute + communication 事件。

    AllReduce Barrier 语义：
    - 每轮迭代生成 N 个 flow（num_workers=N），共享同一瓶颈
    - 迭代完成时间由最慢的 flow 决定（tail flow）
    - 默认 num_workers=1（aggregate flow），向后兼容旧 sim.py
    - allreduce_algo="aggregate" | "ring" | "tree" 指定集合通信算法

    T_target 两阶段校准（论文核心机制）：
    - Stage 1: RTT probing 获取网络延迟基准 → 设置保守初始值
    - Stage 2: 当 job 持有最高 DSCP 优先级时，用 EMA 更新 T_target
    - 校准后的 T_target 替代静态的 ci * comm_solo_ms 用于 deficit 计算
    """

    def __init__(self, jid: str, model: str,
                 mb_per_iter: float, iter_interval_ms: float,
                 target_iters: int, slo_ci: float = 1.5,
                 num_workers: int = 1,
                 allreduce_algo: str = "aggregate",
                 compute_ms: float | None = None,
                 comm_solo_ms: float | None = None,
                 start_time_ms: float = 0.0,
                 comm_offset_ms: float = 0.0,
                 T_target: float | None = None,
                 rtt_probe_ms: float | None = None,
                 alpha: float = 0.3,
                 overhead_factor: float = 2.0,
                 worker_hosts: list[int] | None = None):
        """
        参数：
            jid: job 唯一标识
            model: 模型名称
            mb_per_iter: 每轮迭代 AllReduce 总数据量（MB）
            iter_interval_ms: 无竞争时单轮迭代时间（ms）
            target_iters: 目标迭代次数
            slo_ci: SLO 松弛系数 c_i
            num_workers: DDP worker 数量
            allreduce_algo: 集合通信算法
                "aggregate" — 单流 aggregate（默认，向后兼容）
                "ring" — Ring AllReduce，每轮 2(N-1) 个阶段串行
                "tree" — Tree AllReduce
            compute_ms: 显式计算时间（ms），为 None 时从 iter_interval 推导
            comm_solo_ms: 无竞争通信时间（ms），为 None 时从带宽推导
            start_time_ms: job 开始时间（ms）
            comm_offset_ms: CASSINI time-shift 偏移（ms）
            T_target: 静态 T_target 值（用于 ablation 控制组，为 None 则启用动态校准）
            rtt_probe_ms: Stage 1 RTT probing 结果，保守估计
            alpha: EMA 增益（默认 0.3）
            overhead_factor: NCCL/PCIe 协议开销因子（默认 2.0）
        """
        self.jid = jid
        self.model = model
        self.mb_per_iter = mb_per_iter
        self.iter_interval_ms = iter_interval_ms
        self.target_iters = target_iters
        self.slo_ci = slo_ci
        self.num_workers = num_workers
        self.allreduce_algo = allreduce_algo
        self.comm_offset_ms = comm_offset_ms
        self.overhead_factor = overhead_factor
        self.worker_hosts = worker_hosts  # None = legacy (src=0,dst=1)

        # 通信和计算时间拆分
        if comm_solo_ms is not None:
            self.comm_solo_ms = comm_solo_ms
        else:
            self.comm_solo_ms = self._mb_to_bits(mb_per_iter) / 40e9 * 1000.0

        if compute_ms is not None:
            self.comp_ms = compute_ms
        else:
            self.comp_ms = get_comp_ms(model, default=50.0)

        self.start_time_ms = start_time_ms

        # --- T_target 两阶段校准状态 ---
        self.T_target_static = T_target
        self.T_target_probed = rtt_probe_ms
        self.T_target_ema: float | None = None
        self.alpha = alpha
        self.last_iter_comm_time_ms: float | None = None
        self.ema_initialized = False
        self.ema_update_count = 0  # EMA 更新次数（插桩，验证硬冻结）

        # 运行时状态
        self.completed_iters = 0
        self.accumulated_comm_ms = 0.0
        self.accumulated_iter_ms = 0.0  # 端到端迭代时间（comp + comm）
        self.iter_start_time_ms = 0.0
        self.comm_start_time_ms = 0.0
        self.compute_end_time_ms: float | None = None
        self.slo_violations = 0
        self.is_first_iter: bool = True  # 第一轮标记（所有任务第一轮给最高优先级）

        # 滑窗 urgency：存储最近 W 次迭代时间，用于计算滑动平均
        self._iter_times_deque: deque = deque(maxlen=8)  # 默认 W=8

        # AllReduce 多流 barrier 计数
        self._outstanding_flows: int = 0
        self._iter_version: int = 0  # 迭代版本号，防止重叠迭代的 flow 互相干扰

    @staticmethod
    def _mb_to_bits(mb: float) -> float:
        return mb * 8 * 1024 * 1024

    @property
    def bits_per_iter(self) -> float:
        """每轮迭代总比特数。"""
        return self._mb_to_bits(self.mb_per_iter)

    @property
    def bits_per_flow(self) -> float:
        """每个 flow 承载的比特数（多流时均分）。"""
        n = max(1, self.num_workers)
        return self.bits_per_iter / n

    @property
    def gpu_intensity(self) -> float:
        """CRUX 的 GPU intensity Ij = compute / communication。"""
        return self.comp_ms / self.comm_solo_ms if self.comm_solo_ms > 0 else 0.0

    @property
    def default_T_target(self) -> float:
        """静态默认 T_target = ci * comm_solo_ms。"""
        return self.slo_ci * self.comm_solo_ms

    @property
    def iter_solo_ms(self) -> float:
        """无竞争时的端到端迭代时间（comp + comm × overhead_factor）。"""
        return self.comp_ms + self.comm_solo_ms * self.overhead_factor

    @property
    def sliding_avg_iter_ms(self) -> float | None:
        """
        滑窗平均迭代时间（最近 W 次）。
        当 deque 为空时返回 None。
        """
        if not self._iter_times_deque:
            return None
        return sum(self._iter_times_deque) / len(self._iter_times_deque)

    @property
    def sliding_window_len(self) -> int:
        """当前滑窗长度 W（deque maxlen）。"""
        return self._iter_times_deque.maxlen

    def set_window_size(self, w: int) -> None:
        """动态调整滑窗大小。"""
        self._iter_times_deque = deque(self._iter_times_deque, maxlen=w)

    # --- T_target 两阶段校准 ---

    def get_T_target(self, has_highest_priority: bool = False) -> float:
        """
        两阶段校准获取当前 T_target。

        策略：
        1. 如果静态值存在（T_target_static），直接返回（用于 control 组）
        2. 如果 EMA 已收敛，返回 EMA 值
        3. 如果当前 job 持有最高优先级，用本次迭代实际通信时间更新 EMA
        4. 否则返回 RTT probe 估算值（保守估计）

        参数：
            has_highest_priority: 当前 job 是否分配了最高 DSCP 优先级

        返回：
            T_target (ms)
        """
        # 1. 静态值 → 跳过校准
        if self.T_target_static is not None:
            return self.T_target_static

        # 2. EMA 已收敛 → 直接使用
        if self.T_target_ema is not None:
            return self.T_target_ema

        # 3. 持有最高优先级 → 用实际通信时间更新 EMA
        if has_highest_priority and self.last_iter_comm_time_ms is not None:
            if not self.ema_initialized:
                self.T_target_ema = self.last_iter_comm_time_ms
                self.ema_initialized = True
            else:
                self.T_target_ema = (
                    self.alpha * self.last_iter_comm_time_ms +
                    (1.0 - self.alpha) * self.T_target_ema
                )
            return self.T_target_ema

        # 4. 回退：RTT probe 或静态默认值
        return self.T_target_probed or self.default_T_target

    def update_ema_from_comm_time(self, weight: float = 1.0) -> None:
        """
        根据上次迭代的实际通信时间更新 T_target EMA。

        参数：
            weight: 观测信任度（0-1）。最高优先级 job 传 1.0，
                    被动校准模式下低优先级 job 按优先级折扣（P6=1.0 → P0=0.05）。

        单边更新（防止冻结 + 防止被饥饿观测带偏）：
        - 观测值低于当前 EMA（实际更快）→ 物理上必可达，全额采纳（安全方向）
        - 观测值高于当前 EMA（实际更慢）→ 可能只是没分到带宽而非容量不足，
          按 weight 折扣，避免 T_target 被系统性拉高。
        """
        if self.T_target_static is not None:
            return
        if self.last_iter_comm_time_ms is None:
            return
        obs = self.last_iter_comm_time_ms
        if not self.ema_initialized:
            self.T_target_ema = obs
            self.ema_initialized = True
            self.ema_update_count += 1
            return
        if obs < self.T_target_ema:
            # 安全方向：观测更快 → 全额采纳，防止 EMA 冻结/被拉高
            eff_alpha = max(self.alpha, 0.5)
        else:
            # 观测更慢 → 可能是带宽不足，按信任度折扣
            eff_alpha = self.alpha * max(0.0, min(1.0, weight))
        self.T_target_ema = (
            eff_alpha * obs + (1.0 - eff_alpha) * self.T_target_ema
        )
        self.ema_update_count += 1

    # --- Deficit 计算 ---

    def compute_deficit(self) -> float:
        """
        计算 Progress Deficit pi。

        默认使用静态 T_target = ci * comm_solo_ms。
        LongLiu 策略可调用 get_T_target 获取动态 T_target 后覆盖此值。

        定义：pi = avg_comm_ms / T_target - 1
             pi > 0 → job 落后于 SLO，需提升优先级
             pi < 0 → job 领先于 SLO，可降低优先级
        """
        if self.completed_iters == 0:
            return 0.0
        avg_comm_ms = self.accumulated_comm_ms / self.completed_iters
        return avg_comm_ms / self.default_T_target - 1.0

    # --- 事件回调 ---

    def on_iter_start(self, time_ms: float) -> None:
        """新一轮迭代开始（包括 compute 阶段）。"""
        self.iter_start_time_ms = time_ms

    def on_comm_start(self, time_ms: float) -> None:
        """进入通信阶段。"""
        self.comm_start_time_ms = time_ms

    def on_comm_end(self, time_ms: float) -> None:
        """
        通信阶段结束（barrier 通过）。
        记录通信时间、端到端迭代时间，更新 completed_iters。
        记录 last_iter_comm_time_ms 用于 T_target EMA 更新。
        """
        comm_ms = time_ms - self.comm_start_time_ms
        iter_ms = time_ms - self.iter_start_time_ms  # 端到端迭代时间
        self.last_iter_comm_time_ms = comm_ms
        self.accumulated_comm_ms += comm_ms
        self.accumulated_iter_ms += iter_ms
        self._iter_times_deque.append(iter_ms)  # 滑窗记录
        self.completed_iters += 1
        self.is_first_iter = False

        pi = self.compute_deficit()
        if pi > 0:
            self.slo_violations += 1

    # --- AllReduce 多流控制 ---

    def start_allreduce(self, n_flows: int) -> None:
        """开始一轮 AllReduce，设置期待完成的 flow 数。"""
        self._iter_version += 1
        self._outstanding_flows = n_flows

    def on_flow_complete(self, iter_version: int = 0) -> bool:
        """一个 flow 完成。返回 True 当所有 flow 都完成（barrier 通过）。

        iter_version: flow 所属的迭代版本号，防止重叠迭代的旧 flow 干扰当前 barrier。
        """
        if iter_version > 0 and iter_version != self._iter_version:
            return False  # 旧迭代的 flow，忽略
        self._outstanding_flows -= 1
        return self._outstanding_flows <= 0

    def __repr__(self) -> str:
        return (f"Job({self.jid}, {self.model}, {self.mb_per_iter}MB, "
                f"c={self.slo_ci}, nw={self.num_workers})")