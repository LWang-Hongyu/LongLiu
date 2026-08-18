"""
SLO Scheduler for Multi-Communicator Dynamic Priority Scheduling

调度粒度：WINDOW（论文 Section IV-F：W=20 个迭代组成一个窗口，
调度器在每个 window boundary 重新计算分配并切换 DSCP）。

Usage:
    from slo_scheduler import MultiCommWrapper, SLOScheduler

    scheduler = SLOScheduler(slo_threshold=1.5)
    mc = MultiCommWrapper(scheduler, rank=0, world_size=2, device_list="0",
                          master_addr="192.10.10.110", port=29500)

    for window in range(num_windows):
        mc.window_start(window)
        # ... allreduce with mc.allreduce(...) ...  (内部累计窗口纯通信时间)
        mc.window_end(window, data_size=2048*1024*1024)
"""

import ctypes
import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# SLO Scheduler — Progress Deficit based priority decision (LongLiu algorithm)
# 调度决策粒度 = 1 window（W 个迭代）。决策信号：
#   1) π（总时间尺度）: π_i = A_i / (c_i × T_target × k_i) - 1
#   2) 窗口通信信号（新增）: comm_ratio = 当前窗口纯通信时间 / 基线窗口通信时间
#      当通信时间膨胀超过阈值（默认 1.5 倍）时提升 urgency，确保在计算主导的
#      负载（如真实模型训练）中通信恶化也能触发 P6。
# ---------------------------------------------------------------------------

class SLOScheduler:
    """
    Computes priority from Progress Deficit (π_i) based on LongLiu algorithm
    (paper-aligned formula, scheduler v1(π), queue=SP).

    Paper formula (LongLiu_INFOCOM_Draft.md §4.2):
        π_i(t) = A_i(t) / (c_i × T_target × k_i(t)) - 1

    where:
        A_i(t)        = accumulated actual communication time
        T_target      = auto-calibrated solo window time (EMA sliding)
        k_i(t)        = completed windows
        c_i           = SLO relaxation coefficient (>1, provided by tenant)

    When π_i > 0: task is behind SLO → increase priority
    When π_i < 0: task is ahead of SLO → can decrease priority

    T_target calibration:
        T_target is EMA-updated only during the solo warmup phase (first
        SOLO_WARMUP_WINDOWS windows) and FIXED afterwards — this matches the
        paper's design where T_target represents the uncongested solo window
        time, measured once and held constant for π computation. Updating
        during contention would create a vicious cycle (contention inflates
        T_target → π drops → priority drops → more contention).

    Window communication signal (comm_ratio):
        π uses the window WALL time (compute + comm). On compute-dominated
        loads (e.g. real model training) a 1.8× comm slowdown only moves the
        wall time a few percent, so π stays negative and P6 is never
        triggered. We therefore add a pure-communication signal measured
        inside MultiCommWrapper.allreduce():
            comm_ratio = window_comm_time / baseline_comm_time
        baseline_comm_time is EMA-calibrated during solo warmup windows.
        When comm_ratio > COMM_RATIO_EMERGENCY (default 1.5, i.e. comm
        inflated >50%), urgency is raised to max(π, comm_ratio × factor)
        which forces the scheduler into the high-priority tier (P6).
    """

    # EMA smoothing factor for T_target sliding baseline
    EMA_ALPHA = 0.3
    # First N windows treated as solo warmup → T_target / baseline_comm EMA updates here
    SOLO_WARMUP_WINDOWS = 5

    # Window communication emergency signal
    # comm_ratio = 当前窗口纯通信时间 / 基线窗口纯通信时间
    # 阈值 1.3 依据：10.1 测试床实测两作业共享 100G 链路时自然竞争膨胀为
    # 1.3~1.4×（solo ~21 Gbps 算法带宽 / 竞争 ~16 Gbps），链路远未饱和，
    # 1.5× 阈值在本测试床上物理不可达。1.3× 高于 solo 稳态噪声（±15%），
    # 且对 SLO 保护足够敏感。
    COMM_RATIO_EMERGENCY = 1.3
    # urgency = comm_ratio × COMM_RATIO_FACTOR（放大后远超 0.3 → P6）
    COMM_RATIO_FACTOR = 1.5

    # Paper §5.3 priority mapping (4-tier discrete)
    #   π > 0.3        → P6 (DSCP=8)    — tc:0 (strict highest)
    #  -0.1 < π ≤ 0.3  → P4 (DSCP=0)    — tc:1
    #  -0.5 < π ≤ -0.1 → P2 (DSCP=24)   — tc:3
    #   π ≤ -0.5       → P1 (DSCP=32)   — tc:4
    #
    # DSCP mapping follows the testbed's MEASURED DSCP→TC map (NOT the
    # class-selector name ordering). Verified on 10.1 via `mlnx_qos
    # --trust dscp` + probe experiments: DSCP=8→tc:0 (highest), DSCP=0→tc:1,
    # DSCP=16→tc:2, DSCP=24→tc:3, DSCP=32→tc:4, DSCP=40→tc:5.
    # The naive "higher DSCP = higher priority" mapping was INVERTED on
    # this testbed (P6→DSCP=56→tc:7 was the LOWEST queue), so P6 never
    # preempted P3. Mapping now targets the correct TC per priority.
    # Note P4 (DSCP=0→tc:1) and P3/P5 (DSCP=16→tc:2): P4 outranks P3.
    PRIORITY_TO_DSCP = {0: 40, 1: 32, 2: 24, 3: 16, 4: 0, 5: 16, 6: 8}
    PI_THRESHOLDS = [
        (0.3,  6),   # significantly behind
        (-0.1, 4),   # at SLO boundary
        (-0.5, 2),   # slightly ahead
    ]

    def __init__(self, slo_threshold: float = 1.5, target_comm_time_ms: float = None,
                 ema_alpha: float = None, solo_warmup_windows: int = None,
                 preset_target: bool = False, initial_priority: int = 4,
                 max_priority: int = None):
        """
        Args:
            slo_threshold: SLO relaxation factor c_i (e.g., 1.2 means 20% slack)
            target_comm_time_ms: initial baseline window time in ms
                                 (if None, learned from first window via EMA)
            ema_alpha: EMA smoothing factor for T_target (default 0.3)
            solo_warmup_windows: number of initial windows treated as solo (default 5)
            preset_target: if True, treat target_comm_time_ms as a pre-calibrated
                           solo baseline and skip ALL EMA updates (warmup too).
                           Used when T_target is established via a dedicated
                           Phase-0 solo calibration run, avoiding warmup-phase
                           contention pollution. Requires target_comm_time_ms != None.
            initial_priority: starting priority level (default 4=P4, V6 uses 3=P3).
                              LongLiu dynamically adjusts from this starting point
                              based on π; CRUX holds it static throughout.
            max_priority: maximum allowed priority (e.g., 4 caps at P4).
                          If None, no cap is applied (allow P6).
        """
        self.slo_threshold = slo_threshold
        self.max_priority = max_priority
        self.target_comm_time_ms = target_comm_time_ms
        self.target_comm_time_s = target_comm_time_ms / 1000.0 if target_comm_time_ms else None
        self.preset_target = preset_target
        if preset_target and target_comm_time_ms is None:
            raise ValueError("preset_target=True requires target_comm_time_ms to be set")

        # EMA parameters
        self.ema_alpha = ema_alpha if ema_alpha is not None else self.EMA_ALPHA
        self.solo_warmup_windows = (solo_warmup_windows if solo_warmup_windows is not None
                                    else self.SOLO_WARMUP_WINDOWS)

        # Priority state — start at configurable level, can go up or down based on π
        self.current_priority = initial_priority
        self.window_count = 0
        self.comm_times = []
        self.priority_history = [self.current_priority]

        # Progress Deficit tracking (paper formula)
        self.cumulative_actual_s = 0.0      # A_i(t)
        self.completed_windows = 0          # k_i(t)
        self.last_pi = 0.0                  # last computed π value (for logging/CSV)

        # Window communication signal (pure comm time inside a window)
        self.baseline_comm_time_s = None    # solo 窗口纯通信时间基线（EMA 校准）
        self.last_comm_ratio = 1.0          # 最近窗口通信膨胀比（监控/日志）
        self.last_window_comm_s = 0.0       # 最近窗口纯通信时间

        # EMA bandwidth estimation (for monitoring only)
        self.ema_bandwidth = 0.0
        self.ema_bw_initialized = False

    def compute_priority(self, urgency: float) -> int:
        """
        Map urgency (π 或 comm_ratio×factor) to priority level
        (paper §5.3 4-tier discrete mapping).
        The lowest tier maps to P1 (DSCP=32/tc:4) instead of P0 (DSCP=40/tc:5)
        so the traffic class stays distinguishable on the wire.
        If max_priority is set, the result is capped (never exceeds max_priority).
        """
        for threshold, prio in self.PI_THRESHOLDS:
            if urgency > threshold:
                result = prio
                if self.max_priority is not None:
                    result = min(result, self.max_priority)
                return result
        result = 1   # far ahead of SLO
        if self.max_priority is not None:
            result = min(result, self.max_priority)
        return result

    def update(self, window_wall_time: float, data_size: float,
               window_comm_time: float = 0.0) -> int:
        """
        每个 window 边界调用一次。window_wall_time 为该窗口墙钟时间（含计算），
        window_comm_time 为该窗口内累计纯通信时间（由 MultiCommWrapper 测量）。

        Returns: new priority.
        """
        self.window_count += 1
        self.completed_windows += 1   # k_i(t)

        # --- T_target EMA sliding baseline (paper §3) ---------------------
        # T_target represents the UNCONGESTED solo window time.
        # EMA updates ONLY during solo warmup phase (first SOLO_WARMUP_WINDOWS
        # windows) to smooth NCCL ramp-up noise. After warmup, T_target is
        # FIXED — this matches the paper's design where T_target is measured
        # once and held constant for π computation.
        if not self.preset_target:
            if self.target_comm_time_s is None:
                # First ever measurement → initialize T_target
                self.target_comm_time_s = window_wall_time
                print(f"[SLO] T_target initialized: {self.target_comm_time_s*1000:.1f}ms "
                      f"(c_i={self.slo_threshold}, SLO target="
                      f"{self.target_comm_time_s*self.slo_threshold*1000:.1f}ms)")
            elif self.window_count <= self.solo_warmup_windows:
                # Solo warmup phase: EMA-smooth T_target to reduce ramp-up noise
                new_target = (self.ema_alpha * window_wall_time +
                              (1 - self.ema_alpha) * self.target_comm_time_s)
                delta_ms = (new_target - self.target_comm_time_s) * 1000
                self.target_comm_time_s = new_target
                print(f"[SLO] T_target EMA update (warmup): "
                      f"{self.target_comm_time_s*1000:.1f}ms "
                      f"(delta={delta_ms:+.1f}ms, window={self.window_count})")
        else:
            # Preset mode: T_target is fixed from calibration, just log on first window
            if self.window_count == 1:
                print(f"[SLO] T_target PRESET (calibrated): "
                      f"{self.target_comm_time_s*1000:.1f}ms "
                      f"(c_i={self.slo_threshold}, SLO target="
                      f"{self.target_comm_time_s*self.slo_threshold*1000:.1f}ms) "
                      f"— EMA updates SKIPPED")

        # --- Update EMA bandwidth (monitoring only) -----------------------
        actual_bw = data_size / window_wall_time if window_wall_time > 0 else 0.0
        if not self.ema_bw_initialized:
            self.ema_bandwidth = actual_bw
            self.ema_bw_initialized = True
        else:
            self.ema_bandwidth = (self.ema_alpha * actual_bw +
                                  (1 - self.ema_alpha) * self.ema_bandwidth)

        # --- Window pure-communication baseline (solo warmup 校准) ---------
        # 注意：window_count==1 的窗口包含 NCCL 首次 allreduce 初始化开销
        # （iter0 可达数秒，实测 8.9s），会严重污染 comm baseline，故跳过
        # 第一个窗口，从第二个窗口开始校准。
        # warmup 期间用 min 聚合而非 EMA：EMA 会被 iter0 型突发与 warmup 期
        # 竞争污染拉高；min 对偶发大值鲁棒，且 warmup 期竞争只会让窗口 comm
        # 增大（min 不会取到污染值）。
        self.last_window_comm_s = window_comm_time
        if window_comm_time > 0 and self.window_count > 1:
            if self.baseline_comm_time_s is None:
                # 第一个可用窗口 → 初始化 baseline
                self.baseline_comm_time_s = window_comm_time
                self.last_comm_ratio = 1.0
                print(f"[SLO] comm baseline initialized: "
                      f"{self.baseline_comm_time_s*1000:.1f}ms/window "
                      f"(window {self.window_count})")
            elif self.window_count <= self.solo_warmup_windows + 1:
                # Solo warmup (窗口 2 ~ warmup+1): 取 min
                self.baseline_comm_time_s = min(self.baseline_comm_time_s,
                                                window_comm_time)
                self.last_comm_ratio = 1.0
            else:
                # Post-warmup: baseline FROZEN; ratio reflects comm inflation
                self.last_comm_ratio = (window_comm_time / self.baseline_comm_time_s
                                        if self.baseline_comm_time_s > 0 else 1.0)

        # --- Progress Deficit (paper formula, wall-time scale) ------------
        # π_i(t) = A_i(t) / (c_i × T_target × k_i(t)) - 1
        self.cumulative_actual_s += window_wall_time     # A_i(t)
        expected_total = (self.slo_threshold *
                          self.target_comm_time_s *
                          self.completed_windows)
        if expected_total > 0:
            pi_ratio = self.cumulative_actual_s / expected_total - 1.0
        else:
            pi_ratio = 0.0
        self.last_pi = pi_ratio

        # --- Priority decision (max of π and comm emergency signal) -------
        urgency = pi_ratio
        comm_emergency = False
        if self.last_comm_ratio > self.COMM_RATIO_EMERGENCY:
            comm_emergency = True
            urgency = max(urgency, self.last_comm_ratio * self.COMM_RATIO_FACTOR)

        new_priority = self.compute_priority(urgency)

        if new_priority != self.current_priority:
            print(f"[SLO] Window {self.window_count}: "
                  f"π={pi_ratio:+.3f} "
                  f"(A={self.cumulative_actual_s:.2f}s, "
                  f"expected={expected_total:.2f}s, "
                  f"T_target={self.target_comm_time_s*1000:.1f}ms, "
                  f"k={self.completed_windows}), "
                  f"comm_ratio={self.last_comm_ratio:.2f}"
                  f"{' [EMERGENCY]' if comm_emergency else ''}, "
                  f"urgency={urgency:+.3f}, "
                  f"Priority P{self.current_priority} -> P{new_priority} "
                  f"(DSCP {self.PRIORITY_TO_DSCP.get(self.current_priority, self.current_priority*8)} "
                  f"-> {self.PRIORITY_TO_DSCP.get(new_priority, new_priority*8)})")
            self.current_priority = new_priority

        self.priority_history.append(self.current_priority)
        self.comm_times.append(window_wall_time)

        return self.current_priority

    def set_slo_threshold(self, new_threshold: float):
        """Update c_i (SLO relaxation coefficient) at runtime.

        Used in c_i-swap experiments where the tenant's SLO tightness changes
        mid-run (e.g., window 7). The Progress Deficit formula uses the new
        threshold for all subsequent windows; cumulative actual and
        completed_windows are NOT reset (they reflect total progress so far).
        """
        old = self.slo_threshold
        self.slo_threshold = new_threshold
        print(f"[SLO] c_i updated: {old:.2f} → {new_threshold:.2f}")

    def get_dscp(self) -> int:
        return self.PRIORITY_TO_DSCP.get(self.current_priority, self.current_priority * 8)

    def get_stats(self) -> dict:
        return {
            'comm_times': self.comm_times,
            'priority_history': self.priority_history,
            'current_priority': self.current_priority,
            'current_dscp': self.get_dscp(),
            'ema_bandwidth': self.ema_bandwidth,
            'last_pi': self.last_pi,
            'last_comm_ratio': self.last_comm_ratio,
            'last_window_comm_s': self.last_window_comm_s,
            'target_comm_time_s': self.target_comm_time_s,
            'cumulative_actual_s': self.cumulative_actual_s,
            'completed_windows': self.completed_windows,
        }


# ---------------------------------------------------------------------------
# Multi-Communicator ctypes wrapper
# ---------------------------------------------------------------------------

class MultiCommWrapper:
    def __init__(self, scheduler: SLOScheduler, rank: int, world_size: int,
                 device_list: str = "0", master_addr: str = "192.10.10.110",
                 port: int = 29500, lib_path: Optional[str] = None):
        self.scheduler = scheduler
        self.rank = rank
        self.world_size = world_size
        self.device_list = device_list
        self.master_addr = master_addr
        self.port = port

        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libmulti_comm.so')
        lib_path = os.path.abspath(lib_path)

        print(f"[MultiComm] Loading library from {lib_path}")
        self._lib = ctypes.CDLL(lib_path)

        # multi_comm_init(rank, world_size, device_list, master_addr, port)
        self._lib.multi_comm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                               ctypes.c_char_p, ctypes.c_int]
        self._lib.multi_comm_init.restype = ctypes.c_int

        self._lib.multi_comm_set_priority.argtypes = [ctypes.c_int]
        self._lib.multi_comm_set_priority.restype = ctypes.c_int

        self._lib.multi_comm_allreduce.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self._lib.multi_comm_allreduce.restype = ctypes.c_int

        self._lib.multi_comm_allgather.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int
        ]
        self._lib.multi_comm_allgather.restype = ctypes.c_int

        self._lib.multi_comm_destroy.argtypes = []
        self._lib.multi_comm_destroy.restype = None

        # Initialize
        ret = self._lib.multi_comm_init(rank, world_size, device_list.encode(),
                                         master_addr.encode(), port)
        if ret != 0:
            raise RuntimeError(f"multi_comm_init failed with code {ret}")

        self._window_start_time = None
        self._window_comm_s = 0.0     # 当前窗口累计纯通信时间
        self._initialized = True
        print(f"[MultiComm] Initialized: rank={rank}, world_size={world_size}, "
              f"master={master_addr}:{port}, priority=P{scheduler.current_priority}")

    def window_start(self, window: int):
        """Window 边界：记录起始墙钟时间，清零窗口通信累计，应用当前优先级。"""
        self._window_start_time = time.time()
        self._window_comm_s = 0.0
        self._lib.multi_comm_set_priority(self.scheduler.current_priority)

    def window_end(self, window: int, data_size: float = 0):
        """Window 边界：计算窗口墙钟时间，调度器决策（π + comm_ratio），切换优先级。"""
        if self._window_start_time is None:
            return
        wall_time = time.time() - self._window_start_time
        new_priority = self.scheduler.update(wall_time, data_size,
                                             self._window_comm_s)
        self._lib.multi_comm_set_priority(new_priority)
        self._window_start_time = None

    # 向后兼容别名：旧脚本使用 epoch_start/epoch_end，语义即 window 粒度
    def epoch_start(self, epoch: int):
        self.window_start(epoch)

    def epoch_end(self, epoch: int, data_size: float = 0):
        self.window_end(epoch, data_size=data_size)

    @staticmethod
    def _as_ptr(buf):
        """兼容 Tensor 与 int/None 指针：Tensor → data_ptr()，其余原样。"""
        if buf is not None and hasattr(buf, 'data_ptr'):
            return buf.data_ptr()
        return buf

    def allreduce(self, sendbuf, recvbuf, count, datatype=0, op=0, device_idx=0):
        # 计时累计窗口内纯通信时间（NCCL allreduce 为阻塞调用，返回即完成）
        t0 = time.time()
        ret = self._lib.multi_comm_allreduce(
            self._as_ptr(sendbuf), self._as_ptr(recvbuf), count,
            datatype, op, device_idx
        )
        self._window_comm_s += time.time() - t0
        return ret

    def allgather(self, sendbuf, recvbuf, sendcount, datatype=0, device_idx=0):
        t0 = time.time()
        ret = self._lib.multi_comm_allgather(
            self._as_ptr(sendbuf), self._as_ptr(recvbuf), sendcount,
            datatype, device_idx
        )
        self._window_comm_s += time.time() - t0
        return ret

    def destroy(self):
        if self._initialized:
            self._lib.multi_comm_destroy()
            self._initialized = False

    def __del__(self):
        try:
            self.destroy()
        except:
            pass
