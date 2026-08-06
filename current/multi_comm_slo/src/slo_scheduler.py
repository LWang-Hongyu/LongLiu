"""
SLO Scheduler for Multi-Communicator Dynamic Priority Scheduling

Usage:
    from slo_scheduler import MultiCommWrapper, SLOScheduler
    
    scheduler = SLOScheduler(slo_threshold=1.5)
    mc = MultiCommWrapper(scheduler, rank=0, world_size=2, device_list="0",
                          master_addr="192.10.10.110", port=29500)
    
    for epoch in range(num_epochs):
        mc.epoch_start(epoch)
        # ... allreduce with mc.allreduce(...) ...
        mc.epoch_end(epoch, data_size=2048*1024*1024)
"""

import ctypes
import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# SLO Scheduler — Progress Deficit based priority decision (LongLiu algorithm)
# ---------------------------------------------------------------------------

class SLOScheduler:
    """
    Computes priority from Progress Deficit (π_i) based on LongLiu algorithm
    (paper-aligned formula, scheduler v1(π), queue=SP).

    Paper formula (LongLiu_INFOCOM_Draft.md §4.2):
        π_i(t) = A_i(t) / (c_i × T_target × k_i(t)) - 1

    where:
        A_i(t)        = accumulated actual communication time
        T_target      = auto-calibrated solo iteration time (EMA sliding)
        k_i(t)        = completed iterations
        c_i           = SLO relaxation coefficient (>1, provided by tenant)

    When π_i > 0: task is behind SLO → increase priority
    When π_i < 0: task is ahead of SLO → can decrease priority

    T_target calibration:
        Originally learned from first epoch only. Now updated via per-epoch EMA
        sliding to align with paper §3 (two-stage T_target measurement).
        To avoid the vicious cycle (contention inflates T_target → masks deficit),
        T_target EMA only updates during the solo warmup phase (first
        SOLO_WARMUP_EPOCHS epochs). After warmup, T_target is FIXED — this
        matches the paper's design where T_target represents the uncongested
        solo iteration time, measured once and held constant for π computation.
        EMA during warmup smooths NCCL ramp-up noise (cold cache, first-iter
        setup) that would otherwise pollute the baseline.
    """

    # EMA smoothing factor for T_target sliding baseline
    EMA_ALPHA = 0.3
    # First N epochs treated as solo warmup → T_target EMA updates here
    SOLO_WARMUP_EPOCHS = 5

    # Paper §5.3 priority mapping (4-tier discrete)
    #   π > 0.3        → P6 (DSCP=8)    — hw tc:0 (highest strict priority)
    #  -0.1 < π ≤ 0.3  → P4 (DSCP=0)    — hw tc:1 (second)
    #  -0.5 < π ≤ -0.1 → P2 (DSCP=24)   — hw tc:3 (fourth)
    #   π ≤ -0.5       → P1 (DSCP=32)   — hw tc:4 (fifth)
    #
    # Hardware TC order (mlnx_qos enp130s0f0np0):
    #   tc:0(prio1,dscp8-15) > tc:1(prio0,dscp0-7) > tc:2(prio2,dscp16-23) >
    #   tc:3(prio3,dscp24-31) > tc:4(prio4,dscp32-39) > ... > tc:7(prio7,dscp56-63)
    # Previous mapping `DSCP = priority * 8` was wrong — mapped P6 to tc:6 (lowest).
    #
    # Corrected mapping: higher software priority → higher hardware TC
    PRIORITY_TO_DSCP = {6: 8, 4: 0, 3: 16, 2: 24, 1: 32, 0: 40}
    PI_THRESHOLDS = [
        (0.3,  6),   # significantly behind
        (-0.1, 4),   # at SLO boundary
        (-0.5, 2),   # slightly ahead
    ]

    def __init__(self, slo_threshold: float = 1.5, target_comm_time_ms: float = None,
                 ema_alpha: float = None, solo_warmup_epochs: int = None,
                 preset_target: bool = False, initial_priority: int = 4,
                 max_priority: int = None):
        """
        Args:
            slo_threshold: SLO relaxation factor c_i (e.g., 1.2 means 20% slack)
            target_comm_time_ms: initial baseline communication time in ms
                                 (if None, learned from first epoch via EMA)
            ema_alpha: EMA smoothing factor for T_target (default 0.3)
            solo_warmup_epochs: number of initial epochs treated as solo (default 5)
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
        self.solo_warmup_epochs = (solo_warmup_epochs if solo_warmup_epochs is not None
                                   else self.SOLO_WARMUP_EPOCHS)

        # Priority state — start at configurable level, can go up or down based on π
        self.current_priority = initial_priority
        self.epoch_count = 0
        self.comm_times = []
        self.priority_history = [self.current_priority]

        # Progress Deficit tracking (paper formula)
        self.cumulative_actual_s = 0.0      # A_i(t)
        self.completed_iters = 0            # k_i(t)
        self.last_pi = 0.0                  # last computed π value (for logging/CSV)

        # EMA bandwidth estimation (for monitoring only)
        self.ema_bandwidth = 0.0
        self.ema_bw_initialized = False

    def compute_priority(self, pi_ratio: float) -> int:
        """
        Map π ratio to priority level (paper §5.3 4-tier discrete mapping).
        We use P1 (DSCP=8) instead of P0 (DSCP=0) for the lowest tier to keep
        DSCP non-zero so the traffic class is visible on the wire.
        If max_priority is set, the result is capped (never exceeds max_priority).
        """
        for threshold, prio in self.PI_THRESHOLDS:
            if pi_ratio > threshold:
                result = prio
                if self.max_priority is not None:
                    result = min(result, self.max_priority)
                return result
        result = 1   # far ahead of SLO
        if self.max_priority is not None:
            result = min(result, self.max_priority)
        return result

    def update(self, actual_comm_time: float, data_size: float) -> int:
        self.epoch_count += 1
        self.completed_iters += 1   # k_i(t)

        # --- T_target EMA sliding baseline (paper §3) ---------------------
        # T_target represents the UNCONGESTED solo iteration time.
        # EMA updates ONLY during solo warmup phase (first SOLO_WARMUP_EPOCHS
        # epochs) to smooth NCCL ramp-up noise. After warmup, T_target is FIXED
        # — this matches the paper's design where T_target is measured once
        # and held constant for π computation. Updating T_target during
        # contention would create a vicious cycle (contention inflates
        # T_target → expected_total grows → π drops → priority drops →
        # more contention).
        #
        # If preset_target=True, T_target was established via a dedicated
        # Phase-0 solo calibration run → skip ALL EMA updates (warmup too).
        # This is the recommended mode for contested experiments to avoid
        # warmup-phase contention polluting the baseline.
        if not self.preset_target:
            if self.target_comm_time_s is None:
                # First ever measurement → initialize T_target
                self.target_comm_time_s = actual_comm_time
                print(f"[SLO] T_target initialized: {self.target_comm_time_s*1000:.1f}ms "
                      f"(c_i={self.slo_threshold}, SLO target="
                      f"{self.target_comm_time_s*self.slo_threshold*1000:.1f}ms)")
            elif self.epoch_count <= self.solo_warmup_epochs:
                # Solo warmup phase: EMA-smooth T_target to reduce ramp-up noise
                new_target = (self.ema_alpha * actual_comm_time +
                              (1 - self.ema_alpha) * self.target_comm_time_s)
                delta_ms = (new_target - self.target_comm_time_s) * 1000
                self.target_comm_time_s = new_target
                print(f"[SLO] T_target EMA update (warmup): "
                      f"{self.target_comm_time_s*1000:.1f}ms "
                      f"(delta={delta_ms:+.1f}ms, epoch={self.epoch_count})")
        else:
            # Preset mode: T_target is fixed from calibration, just log on first epoch
            if self.epoch_count == 1:
                print(f"[SLO] T_target PRESET (calibrated): "
                      f"{self.target_comm_time_s*1000:.1f}ms "
                      f"(c_i={self.slo_threshold}, SLO target="
                      f"{self.target_comm_time_s*self.slo_threshold*1000:.1f}ms) "
                      f"— EMA updates SKIPPED")

        # --- Update EMA bandwidth (monitoring only) -----------------------
        actual_bw = data_size / actual_comm_time if actual_comm_time > 0 else 0.0
        if not self.ema_bw_initialized:
            self.ema_bandwidth = actual_bw
            self.ema_bw_initialized = True
        else:
            self.ema_bandwidth = (self.ema_alpha * actual_bw +
                                  (1 - self.ema_alpha) * self.ema_bandwidth)

        # --- Progress Deficit (paper formula) -----------------------------
        # π_i(t) = A_i(t) / (c_i × T_target × k_i(t)) - 1
        self.cumulative_actual_s += actual_comm_time     # A_i(t)
        expected_total = (self.slo_threshold *
                          self.target_comm_time_s *
                          self.completed_iters)
        if expected_total > 0:
            pi_ratio = self.cumulative_actual_s / expected_total - 1.0
        else:
            pi_ratio = 0.0
        self.last_pi = pi_ratio

        # --- Priority decision --------------------------------------------
        new_priority = self.compute_priority(pi_ratio)

        if new_priority != self.current_priority:
            print(f"[SLO] Epoch {self.epoch_count}: "
                  f"π={pi_ratio:+.3f} "
                  f"(A={self.cumulative_actual_s:.2f}s, "
                  f"expected={expected_total:.2f}s, "
                  f"T_target={self.target_comm_time_s*1000:.1f}ms, "
                  f"k={self.completed_iters}), "
                  f"Priority P{self.current_priority} -> P{new_priority} "
                  f"(DSCP {self.PRIORITY_TO_DSCP.get(self.current_priority, self.current_priority*8)} "
                  f"-> {self.PRIORITY_TO_DSCP.get(new_priority, new_priority*8)})")
            self.current_priority = new_priority

        self.priority_history.append(self.current_priority)
        self.comm_times.append(actual_comm_time)

        return self.current_priority

    def set_slo_threshold(self, new_threshold: float):
        """Update c_i (SLO relaxation coefficient) at runtime.

        Used in c_i-swap experiments where the tenant's SLO tightness changes
        mid-run (e.g., epoch 7). The Progress Deficit formula uses the new
        threshold for all subsequent iterations; cumulative actual and
        completed_iters are NOT reset (they reflect total progress so far).
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
            'target_comm_time_s': self.target_comm_time_s,
            'cumulative_actual_s': self.cumulative_actual_s,
            'completed_iters': self.completed_iters,
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
        
        self._lib.multi_comm_destroy.argtypes = []
        self._lib.multi_comm_destroy.restype = None
        
        # Initialize
        ret = self._lib.multi_comm_init(rank, world_size, device_list.encode(),
                                         master_addr.encode(), port)
        if ret != 0:
            raise RuntimeError(f"multi_comm_init failed with code {ret}")
        
        self._epoch_start_time = None
        self._initialized = True
        print(f"[MultiComm] Initialized: rank={rank}, world_size={world_size}, "
              f"master={master_addr}:{port}, priority=P{scheduler.current_priority}")
    
    def epoch_start(self, epoch: int):
        self._epoch_start_time = time.time()
        self._lib.multi_comm_set_priority(self.scheduler.current_priority)
    
    def epoch_end(self, epoch: int, data_size: float = 0):
        if self._epoch_start_time is None:
            return
        comm_time = time.time() - self._epoch_start_time
        new_priority = self.scheduler.update(comm_time, data_size)
        self._lib.multi_comm_set_priority(new_priority)
        self._epoch_start_time = None
    
    def allreduce(self, sendbuf, recvbuf, count, datatype=0, op=0, device_idx=0):
        return self._lib.multi_comm_allreduce(
            sendbuf, recvbuf, count, datatype, op, device_idx
        )
    
    def destroy(self):
        if self._initialized:
            self._lib.multi_comm_destroy()
            self._initialized = False
    
    def __del__(self):
        try:
            self.destroy()
        except:
            pass
