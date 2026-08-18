"""
Alibaba Lingjun 数据集 Loader - 时段重放（真实到达模式）。

数据集格式（来自 alibaba-lingjun-dataset-2023-main.zip）：
- job.csv: id, uid, job_name, kind, namespace, model, status, reason, ...
          gmt_job_submitted (如 "2023/7/15 02:19"), gmt_job_finished (如 "2023/7/15 02:23")
- worker.csv: id, job_name, worker_name, replica_type, host_ip, gmt_created, ...
- topo.csv: ip, DSW, PSW, ASW

重放语义（与合成 workload 的关键区别）：
- 只保留 model 字段可映射到参数规模的 job（丢弃 unknown/cognitive/rl 等任务类型）
- job 到达时刻 = trace 中真实 gmt_job_submitted 相对所选窗口起点的偏移（等比压缩到仿真时长）
- 保留真实到达间隔分布，而非 Poisson/均匀随机
- dp = worker 数（replica_type=worker），过滤 dp>max_gpus
- 窗口默认取到达最密集的 window_hours 小时时段

如果数据集不存在，初始化时抛出 FileNotFoundError，应改用 SyntheticTraceLoader。
"""

from __future__ import annotations

import csv
import io
import os
import random
import zipfile
from datetime import datetime, timedelta
from typing import List

from longliu_sim.job import Job
from longliu_sim.utils.model_params import MODEL_PARAMS, get_comp_ms
from longliu_sim.utils.metrics import compute_target_iter_ms
from longliu_sim.trace.synthetic import place_workers_random

# 数据集中的模型名 → MODEL_PARAMS 标准键名（白名单，只保留可映射规模的模型）
# 不在白名单的模型（unknown/cognitive/rl/preprocess/sidecar/DLM/multigen/dota 等）一律丢弃
_MODEL_ALIASES = {
    "llama": "LLaMA-2-7B",
    "llama_30b": "LLaMA-2-13B",
    "gpt-13b": "LLaMA-2-13B",
    "gpt-13B": "LLaMA-2-13B",
    "gpt-7b": "LLaMA-2-7B",
    "gpt-7B": "LLaMA-2-7B",
    "gpt-3b": "GPT-3B",
    "gpt-3B": "GPT-3B",
    "gpt-1b": "LLaMA-2-1B",
    "gpt-1B": "LLaMA-2-1B",
    "65b": "LLaMA-2-65B",
    "resnet": "ResNet-50-fp16",
    "mnist": "ResNet-18",
}

# 按参数规模分层的 SLO CI 阈值
_LARGE_MODEL_PARAMS = 7e9      # >= 7B → ci=1.5
_MEDIUM_MODEL_PARAMS = 340e6   # >= 340M → ci=2.0
# < 340M → ci=3.0

# 持续时间过滤（分钟）
_MIN_DURATION_MIN = 2.0    # 太短（<2min）无训练意义
_MAX_DURATION_MIN = 1440.0 # 24h，排除长尾


class LingjunTraceLoader:
    """从 Alibaba Lingjun trace ZIP 加载时段重放的 Job 列表。"""

    def __init__(
        self,
        zip_path: str,
        max_gpus: int = 8,
        min_gpus: int = 2,
        duration_ms: float = 600000.0,
        seed: int = 0,
        target_bw_bps: float = 100e9,
        overhead_factor: float = 1.3,
        overlap_factor: float = 0.85,
        window_hours: float = 6.0,
        window_start: str | None = None,
        max_jobs: int | None = None,
        num_hosts: int = 16,
    ):
        """
        Args:
            zip_path: 指向 alibaba-lingjun-dataset-2023-main.zip 的路径
            max_gpus: dp（worker 数）上限，过滤过大的 job（默认 8，与仿真器 16 节点适配）
            min_gpus: dp 下限（默认 2）。dp=1 无 AllReduce、不产生网络流量，
                且锚点公式对其 target 按"有通信"计算导致 SAS 虚高，应排除于网络调度对照
            duration_ms: 仿真总时长（毫秒），重放窗口等比压缩到此时长
            seed: 随机种子（用于 worker 放置）
            target_bw_bps: 目标链路带宽，用于计算无竞争迭代间隔（默认 100Gbps）
            overhead_factor: NCCL/PCIe 协议开销因子（默认 1.3）
            overlap_factor: compute-comm 重叠度（默认 0.85，锚点冻结值）
            window_hours: 重放窗口长度（小时），自动选择到达最密集的窗口
            window_start: 固定窗口起点（"YYYY/MM/DD HH:MM"）；None 时自动选最密集窗口
            max_jobs: 窗口内最多保留的 job 数（按到达顺序）；None 时全部保留
            num_hosts: 仿真器拓扑主机数（worker 放置范围，必须与拓扑一致，非 trace 集群规模）
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Lingjun dataset not found: {zip_path}")
        self.zip_path = zip_path
        self.max_gpus = max_gpus
        self.min_gpus = min_gpus
        self.duration_ms = duration_ms
        self.rng = random.Random(seed)
        self.target_bw_bps = target_bw_bps
        self.overhead_factor = overhead_factor
        self.overlap_factor = overlap_factor
        self.window_hours = window_hours
        self.window_start = window_start
        self.max_jobs = max_jobs
        self.num_hosts = num_hosts

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _map_model(model_name: str) -> str | None:
        """将数据集中的模型名映射到 MODEL_PARAMS 标准名（白名单）。

        返回 None 表示该模型不可映射（unknown/cognitive/rl/任务类型等），应丢弃。
        """
        # 1) 直接命中
        if model_name in MODEL_PARAMS:
            return model_name
        # 2) 白名单别名表
        key = model_name.lower()
        for alias, mapped in _MODEL_ALIASES.items():
            if key == alias.lower():
                return mapped
        # 3) 不可映射 → None
        return None

    @staticmethod
    def _compute_mb_per_iter(params: dict, dp: int) -> float:
        """计算每轮迭代的 AllReduce 数据量（MB，十进制）。

        dp=1 时不需要网络通信（单 worker 无 AllReduce），返回 0。
        """
        if dp <= 1:
            return 0.0
        param_count = params.get("params", 1e9)
        bpp = 2 if params.get("fp16", True) else 4
        bytes_per_iter = 2 * param_count * bpp / max(dp, 1)
        return bytes_per_iter / 1e6

    @staticmethod
    def _get_slo_ci(mapped_model: str) -> float:
        """按模型参数规模分层 SLO CI。

        大模型（>= 7B params）→ 1.5
        中模型（>= 340M params）→ 2.0
        小模型（< 340M params）→ 3.0
        """
        params = MODEL_PARAMS.get(mapped_model, {})
        param_count = params.get("params", 0)
        if param_count >= _LARGE_MODEL_PARAMS:
            return 1.5
        elif param_count >= _MEDIUM_MODEL_PARAMS:
            return 2.0
        else:
            return 3.0

    # ------------------------------------------------------------------
    # 核心加载逻辑
    # ------------------------------------------------------------------

    def load(self) -> List[Job]:
        """加载并返回时段重放的 Job 列表。"""
        z = zipfile.ZipFile(self.zip_path)
        prefix = "alibaba-lingjun-dataset-2023-main/data/"

        # ---- 1. 读取 worker.csv，统计每个 job 的 worker 数（= dp）----
        worker_count: dict[str, int] = {}
        with z.open(prefix + "worker.csv") as f:
            raw = f.read().decode("utf-8")
        for row in csv.DictReader(io.StringIO(raw)):
            if row.get("replica_type") == "worker":
                jn = row["job_name"]
                worker_count[jn] = worker_count.get(jn, 0) + 1

        # ---- 3. 读取 job.csv，过滤有效 job ----
        # 注意：worker 放置使用 self.num_hosts（仿真器拓扑主机数），
        # 而非 trace 集群规模（topo.csv 847 节点），避免 host id 越界。
        with z.open(prefix + "job.csv") as f:
            raw = f.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))

        candidates: list[dict] = []
        for row in reader:
            # 只取 Succeeded（有完整生命周期）
            if row.get("status") != "Succeeded":
                continue
            # 模型必须可映射（白名单），否则丢弃
            model = row.get("model", "")
            mapped = self._map_model(model)
            if mapped is None:
                continue

            job_name = row.get("job_name", "")
            dp = worker_count.get(job_name, 0)
            # dp 下限过滤（默认 2）：dp=1 无 AllReduce、不产生网络流量
            if dp < self.min_gpus or dp > self.max_gpus:
                continue

            # 解析提交/完成时间
            t_sub_str = row.get("gmt_job_submitted", "")
            t_fin_str = row.get("gmt_job_finished", "")
            if not t_sub_str or not t_fin_str:
                continue
            try:
                t_sub = datetime.strptime(t_sub_str, "%Y/%m/%d %H:%M")
                t_fin = datetime.strptime(t_fin_str, "%Y/%m/%d %H:%M")
            except ValueError:
                continue

            duration_min = (t_fin - t_sub).total_seconds() / 60.0
            if duration_min < _MIN_DURATION_MIN or duration_min > _MAX_DURATION_MIN:
                continue

            candidates.append({
                "job_name": job_name,
                "model": mapped,
                "dp": dp,
                "t_sub": t_sub,
                "t_fin": t_fin,
            })

        if not candidates:
            return []

        # ---- 4. 选择重放窗口（固定起点或自动选最密集窗口）----
        candidates.sort(key=lambda x: x["t_sub"])

        if self.window_start is not None:
            try:
                ws = datetime.strptime(self.window_start, "%Y/%m/%d %H:%M")
            except ValueError:
                raise ValueError(
                    f"window_start must be 'YYYY/MM/DD HH:MM', got '{self.window_start}'"
                )
            we = ws + timedelta(hours=self.window_hours)
            window = [c for c in candidates if ws <= c["t_sub"] < we]
        else:
            # 扫描：找 (window_hours 内到达数) 最大的窗口
            times = [c["t_sub"] for c in candidates]
            best_n = -1
            best_i = 0
            for i, t in enumerate(times):
                end = t + timedelta(hours=self.window_hours)
                n = sum(1 for tt in times[i:] if tt < end)
                if n > best_n:
                    best_n = n
                    best_i = i
            ws = times[best_i]
            we = ws + timedelta(hours=self.window_hours)
            window = [c for c in candidates if ws <= c["t_sub"] < we]

        if self.max_jobs is not None:
            window = window[: self.max_jobs]

        # ---- 5. 时间压缩：窗口 [ws, we) → [0, duration_ms) ----
        window_ms = (we - ws).total_seconds() * 1000.0
        scale = self.duration_ms / window_ms if window_ms > 0 else 1.0

        # ---- 6. 构建 Job 对象 ----
        jobs: List[Job] = []
        overhead_factor = self.overhead_factor

        for i, cand in enumerate(window):
            model = cand["model"]
            dp = cand["dp"]
            slo_ci = self._get_slo_ci(model)

            params = MODEL_PARAMS[model]
            mb_per_iter = self._compute_mb_per_iter(params, dp)
            raw_comm_ms = mb_per_iter * 8 * 1e6 / self.target_bw_bps * 1000.0
            # dp=1 时 comm_solo_ms = 0，但为了数值稳定性至少保留 0.1ms
            comm_solo = max(raw_comm_ms, 0.1)
            comp_ms = get_comp_ms(model, default=50.0)
            iter_interval_ms = comm_solo + comp_ms

            # 锚点公式（与 metrics.compute_target_iter_ms 同源，SEMANTICS_VERSION="anchor-v2"）
            target_iter_ms = compute_target_iter_ms(
                model, dp, slo_ci,
                host_bw_gbps=self.target_bw_bps / 1e9,
                overhead_factor=overhead_factor,
                overlap_factor=self.overlap_factor,
            )
            target_iters = max(1, int(self.duration_ms / target_iter_ms))

            # 真实到达时刻（相对窗口起点），等比压缩到仿真时长
            start_time_ms = (cand["t_sub"] - ws).total_seconds() * 1000.0 * scale
            start_time_ms = min(start_time_ms, self.duration_ms * 0.95)

            jobs.append(
                Job(
                    jid=f"LJ{i}",
                    model=model,
                    mb_per_iter=mb_per_iter,
                    iter_interval_ms=iter_interval_ms,
                    target_iters=target_iters,
                    slo_ci=slo_ci,
                    num_workers=dp,
                    start_time_ms=start_time_ms,
                    comm_solo_ms=comm_solo,
                    compute_ms=comp_ms,
                    overhead_factor=overhead_factor,
                    worker_hosts=(
                        place_workers_random(
                            dp, self.num_hosts, seed=self.rng.randint(0, 2**31)
                        )
                        if self.num_hosts > 0
                        else None
                    ),
                )
            )

        return jobs
