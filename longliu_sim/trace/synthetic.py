"""
合成 workload 生成器，作为 Lingjun trace 的 fallback。

支持两种生成模式：
1. 随机模式：从给定列表按 weight 采样模型/GPU/ci
2. 分层模式（workload_profile）：显式指定 (model, dp, ci) 列表，
   用于论文 Table 3 需要的"大模型+tight SLO"结构性 workload。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from longliu_sim.job import Job
from longliu_sim.utils.model_params import MODEL_PARAMS, get_comp_ms


def place_workers_random(num_workers: int, num_hosts: int,
                         seed: int | None = None) -> List[int]:
    """将 num_workers 个 worker 均匀随机放置在 num_hosts 个主机上。

    参数：
        num_workers: worker 数量
        num_hosts: 可用主机数量
        seed: 随机种子（None 时使用全局 random）

    返回：
        worker 所在的主机 ID 列表，长度 = num_workers
    """
    rng = random.Random(seed)
    return [rng.randint(0, num_hosts - 1) for _ in range(num_workers)]

# 论文 Table 3 默认分层 workload（v4 异构化）：
# - 大模型（>1000MB, ≥4 GPUs）占 50%（12/24），拆分为：
#   - 6 个 premium（ci=1.2，紧租户）
#   - 6 个 standard（ci=2.0，松租户）
# - 中模型（100-1000MB）占 33%（8/24），medium SLO (ci=2.0)
# - 小模型（<100MB）占 17%（4/24），loose SLO (ci=3.0)
#
# 异构化目的：验证 SLO 感知策略的价值（premium vs standard 的差异化调度）
DEFAULT_TIERED_WORKLOAD: List[Tuple[str, int, float]] = [
    # 大模型 premium：6 个，ci=1.2（紧租户）
    ("LLaMA-2-13B", 8, 1.2),
    ("LLaMA-2-13B", 8, 1.2),
    ("LLaMA-2-7B", 4, 1.2),
    ("LLaMA-2-7B", 4, 1.2),
    ("T5-11B-fp16", 8, 1.2),
    ("LLaMA-2-7B", 8, 1.2),
    # 大模型 standard：6 个，ci=2.0（松租户）
    ("LLaMA-2-13B", 8, 2.0),
    ("LLaMA-2-13B", 8, 2.0),
    ("LLaMA-2-7B", 4, 2.0),
    ("LLaMA-2-7B", 4, 2.0),
    ("T5-11B-fp16", 8, 2.0),
    ("LLaMA-2-7B", 8, 2.0),
    # 中模型：8 个，ci=2.0
    ("BERT-Large-fp16", 2, 2.0),
    ("BERT-Large-fp16", 2, 2.0),
    ("BERT-Large-fp16", 4, 2.0),
    ("BERT-Large-fp16", 4, 2.0),
    ("ViT-Large", 8, 2.0),
    ("ViT-Large", 8, 2.0),
    ("ViT-Base", 2, 2.0),
    ("ViT-Base", 2, 2.0),
    # 小模型：4 个，ci=3.0
    ("ResNet-18", 1, 3.0),
    ("ResNet-18", 1, 3.0),
    ("ResNet-50-fp16", 2, 3.0),
    ("ResNet-50-fp16", 2, 3.0),
]

# feas_boundary_v1 主场景（定量定标后）：
# - 目标 Σ∈[920,1040] Gbps @ 800G spine → 1.15-1.30×
# - 7 job 全跨 pod，≥3 premium(ci=1.2) + ≥4 standard(ci=2.0)
# - 战斗场：4 个 contested job（2 premium + 2 standard）
# - 设计原则：区分度只存在于需求>公平份额的job身上
FEAS_BOUNDARY_V1_WORKLOAD: List[Tuple[str, int, float]] = [
    # Premium tier：3 个，ci=1.5（放宽后 P1 需求 219 Gbps）
    ("LLaMA-2-13B", 8, 1.5),       # P1: 219.41 Gbps, contested ✓
    ("LLaMA-2-7B", 8, 1.5),        # P2: 227.64 Gbps, contested ✓
    ("BERT-Large-fp16", 2, 1.5),    # P3: 62.36 Gbps, non-contested
    # Standard tier：4 个，ci=2.5（放宽给 premium 让渡空间）
    ("LLaMA-2-13B", 8, 2.5),       # S1: 131.65 Gbps, contested
    ("T5-11B-fp16", 8, 2.5),       # S2: 139.68 Gbps, contested
    ("BERT-Large-fp16", 4, 2.5),    # S3: 20.12 Gbps, non-contested
    ("ViT-Base", 2, 2.5),           # S4: 13.09 Gbps, non-contested
]


class SyntheticTraceLoader:
    """生成合成 trace 的 Job 列表。"""

    def __init__(
        self,
        model_types: Optional[List[str]] = None,
        gpu_distribution: Optional[Dict[int, float]] = None,
        ci_distribution: Optional[Dict[float, float]] = None,
        job_count: int = 24,
        duration_ms: float = 600000.0,
        seed: int = 0,
        overhead_factor: float = 2.0,
        target_bw_bps: float = 40e9,
        workload_profile: Optional[List[Tuple[str, int, float]]] = None,
        num_hosts: int = 0,
    ):
        """
        Args:
            model_types: 候选模型名列表；None 时使用 MODEL_PARAMS 所有 key
            gpu_distribution: {gpu_count: probability}
            ci_distribution: {slo_ci: probability}
            job_count: 生成的 job 数量
            duration_ms: 仿真总时长（毫秒）
            seed: 随机种子
            overhead_factor: NCCL/PCIe 协议开销因子（论文默认 2.0）
            target_bw_bps: 目标链路带宽，用于计算无竞争迭代间隔
            workload_profile: 显式 (model, dp, ci) 列表；提供时优先使用
            num_hosts: 拓扑主机数；>0 时为每个 job 分配 worker host（随机放置）
        """
        self.model_types = model_types or list(MODEL_PARAMS.keys())
        self.gpu_distribution = gpu_distribution or {1: 0.3, 2: 0.2, 4: 0.3, 8: 0.2}
        self.ci_distribution = ci_distribution or {1.5: 0.33, 2.0: 0.33, 3.0: 0.34}
        self.job_count = job_count
        self.duration_ms = duration_ms
        self.rng = random.Random(seed)
        self.overhead_factor = overhead_factor
        self.target_bw_bps = target_bw_bps
        self.workload_profile = workload_profile
        self.num_hosts = num_hosts

    def load(self) -> List[Job]:
        """生成并返回 Job 列表。"""
        if self.workload_profile is not None:
            return self._load_from_profile()
        return self._load_random()

    def _load_from_profile(self) -> List[Job]:
        """按 workload_profile 显式生成分层 workload。
        
        任务开始时间使用 Poisson 过程（Exponential 间隔）。
        平均间隔 = 2 * duration_ms / job_count（增加间隔，避免过于紧凑）
        """
        profile = list(self.workload_profile)
        self.rng.shuffle(profile)

        # Poisson 过程：任务到达时间间隔服从 Exponential 分布
        # 平均间隔 = 2 * duration_ms / job_count（增加间隔）
        mean_interval_ms = 2.0 * self.duration_ms / len(profile)
        
        jobs: List[Job] = []
        current_time = 0.0
        
        for i, (model, dp, slo_ci) in enumerate(profile):
            if model not in MODEL_PARAMS:
                raise ValueError(f"Unknown model '{model}' in workload_profile")

            params = MODEL_PARAMS[model]
            mb_per_iter = self._compute_mb_per_iter(params, dp)
            raw_comm_ms = mb_per_iter * 8 * 1024 * 1024 / self.target_bw_bps * 1000.0
            comp_ms = get_comp_ms(model, default=50.0)
            iter_interval_ms = raw_comm_ms + comp_ms

            effective_comm_solo = raw_comm_ms * self.overhead_factor
            target_iter_ms = comp_ms + effective_comm_solo * slo_ci
            target_iters = max(1, int(self.duration_ms / target_iter_ms))

            # Exponential 分布的任务到达间隔
            interval = self.rng.expovariate(1.0 / mean_interval_ms)
            current_time += interval
            start_time_ms = min(current_time, self.duration_ms * 0.9)  # 不超过仿真时长的 90%

            jobs.append(
                Job(
                    jid=f"J{i}",
                    model=model,
                    mb_per_iter=mb_per_iter,
                    iter_interval_ms=iter_interval_ms,
                    target_iters=target_iters,
                    slo_ci=slo_ci,
                    num_workers=dp,
                    start_time_ms=start_time_ms,
                    comm_solo_ms=raw_comm_ms,
                    compute_ms=comp_ms,
                    overhead_factor=self.overhead_factor,
                    worker_hosts=(
                        place_workers_random(dp, self.num_hosts, seed=self.rng.randint(0, 2**31))
                        if self.num_hosts > 0 else None
                    ),
                )
            )
        return jobs

    def _load_random(self) -> List[Job]:
        """生成随机 workload（向后兼容）。"""
        ci_list = self._generate_ci_list()
        self.rng.shuffle(ci_list)

        jobs: List[Job] = []
        for i in range(self.job_count):
            model = self.rng.choice(self.model_types)
            gpu_count = self._sample_gpu_count()
            dp = gpu_count
            slo_ci = ci_list[i]

            params = MODEL_PARAMS.get(model, {"params": 1e9, "fp16": True, "weight": 1.0})
            mb_per_iter = self._compute_mb_per_iter(params, dp)
            raw_comm_ms = mb_per_iter * 8 * 1024 * 1024 / self.target_bw_bps * 1000.0
            comp_ms = get_comp_ms(model, default=50.0)
            iter_interval_ms = raw_comm_ms + comp_ms

            effective_comm_solo = raw_comm_ms * self.overhead_factor
            target_iter_ms = comp_ms + effective_comm_solo * slo_ci
            target_iters = max(1, int(self.duration_ms / target_iter_ms))

            start_time_ms = self.rng.uniform(0, 10000)

            jobs.append(
                Job(
                    jid=f"J{i}",
                    model=model,
                    mb_per_iter=mb_per_iter,
                    iter_interval_ms=iter_interval_ms,
                    target_iters=target_iters,
                    slo_ci=slo_ci,
                    num_workers=dp,
                    start_time_ms=start_time_ms,
                    comm_solo_ms=raw_comm_ms,
                    compute_ms=comp_ms,
                    overhead_factor=self.overhead_factor,
                    worker_hosts=(
                        place_workers_random(dp, self.num_hosts, seed=self.rng.randint(0, 2**31))
                        if self.num_hosts > 0 else None
                    ),
                )
            )
        return jobs

    def _generate_ci_list(self) -> List[float]:
        """生成确定性的 ci 列表（精确计数，保证可复现）。"""
        cis = sorted(self.ci_distribution.keys())
        total_weight = sum(self.ci_distribution.values())
        counts = []
        allocated = 0
        for ci in cis[:-1]:
            c = round(self.ci_distribution[ci] / total_weight * self.job_count)
            counts.append((ci, c))
            allocated += c
        counts.append((cis[-1], self.job_count - allocated))

        result = []
        for ci, cnt in counts:
            result.extend([ci] * cnt)
        return result

    def _sample_gpu_count(self) -> int:
        """按 gpu_distribution 采样 GPU 数量。"""
        items = sorted(self.gpu_distribution.items())
        counts, weights = zip(*items)
        total = sum(weights)
        r = self.rng.uniform(0, total)
        acc = 0.0
        for c, w in zip(counts, weights):
            acc += w
            if r <= acc:
                return int(c)
        return int(counts[-1])

    def _sample_ci(self) -> float:
        """按 ci_distribution 采样 slo_ci。"""
        items = list(self.ci_distribution.items())
        cis, weights = zip(*items)
        total = sum(weights)
        r = self.rng.uniform(0, total)
        acc = 0.0
        for ci, w in zip(cis, weights):
            acc += w
            if r <= acc:
                return float(ci)
        return float(cis[-1])

    def _compute_mb_per_iter(self, params: dict, dp: int) -> float:
        """
        计算 per-worker 通信需求（MB/iter）。

        物理口径：
        D_i = overlap 后的暴露通信需求，约等于 per-worker ring AllReduce 流量。
        当前公式：bytes_per_iter = 2 * params * bpp / dp
        其中：
        - 2 * params * bpp：total AllReduce 数据量（gradient sync, fp16/bf16）
        - 除以 dp：per-worker 通信量（ring AllReduce 每个 worker 承担 1/dp）
        - overlap_factor 由 Simulator 在运行时处理，此处不乘 (1 - overlap)

        典型值（paper-baseline-v2）：
        - 13B job, dp=8: ~6200 MB/iter per-worker
        - 7B job, dp=4:  ~3500 MB/iter per-worker
        """
        param_count = params.get("params", 1e9)
        bpp = 2 if params.get("fp16", True) else 4
        bytes_per_iter = 2 * param_count * bpp / max(dp, 1)
        return bytes_per_iter / (1024 * 1024)

    def _estimate_iter_interval_ms(self, model: str, dp: int) -> float:
        params = MODEL_PARAMS.get(model, {"params": 1e9, "fp16": True, "weight": 1.0})
        mb_per_iter = self._compute_mb_per_iter(params, dp)
        bits = mb_per_iter * 8 * 1024 * 1024
        comm_ms = bits / 40e9 * 1000.0
        return comm_ms + 50.0