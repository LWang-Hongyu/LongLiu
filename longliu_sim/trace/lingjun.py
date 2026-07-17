"""
Alibaba Lingjun 数据集 Loader - 解析真实的 Alibaba Lingjun trace。

数据集格式（来自 alibaba-lingjun-dataset-2023-main.zip）：
- job.csv: id, uid, job_name, kind, namespace, model, status, reason, ...
          gmt_job_submitted (如 "2023/7/15 02:19"), gmt_job_finished (如 "2023/7/15 02:23")
- worker.csv: id, job_name, worker_name, replica_type, host_ip, gmt_created, ...
- topo.csv: ip, DSW, PSW, ASW

如果数据集不存在，初始化时抛出 FileNotFoundError，应改用 SyntheticTraceLoader。
"""

from __future__ import annotations

import csv
import io
import os
import random
import zipfile
from datetime import datetime
from typing import List

from longliu_sim.job import Job
from longliu_sim.utils.model_params import MODEL_PARAMS, get_comp_ms
from longliu_sim.trace.synthetic import place_workers_random

# 数据集中的模型名 → MODEL_PARAMS 标准键名的映射表（大小写不敏感）
_MODEL_ALIASES = {
    "llama": "LLaMA-2-7B",
    "llama_30b": "LLaMA-2-13B",
    "gpt-13b": "LLaMA-2-13B",
    "gpt-7b": "LLaMA-2-7B",
    "gpt-3b": "LLaMA-2-1B",
    "resnet": "ResNet-50-fp16",
    "cognitive": "BERT-Large-fp16",
    "rl": "BERT-Base",
    "65b": "LLaMA-2-70B",
    "word_embedding": "ResNet-50-fp16",
    # 未显式列出的模型（如 multigen, dota, DLM 等）将回退到 BERT-Base
}

# 按参数规模分层的 SLO CI 阈值
_LARGE_MODEL_PARAMS = 7e9      # >= 7B → ci=1.5
_MEDIUM_MODEL_PARAMS = 340e6   # >= 340M → ci=2.0
# < 340M → ci=3.0


class LingjunTraceLoader:
    """从 Alibaba Lingjun trace ZIP 文件加载 Job 列表。"""

    def __init__(
        self,
        zip_path: str,
        max_gpus: int = 128,
        duration_ms: float = 600000.0,
        seed: int = 0,
        target_bw_bps: float = 100e9,
        overhead_factor: float = 1.3,
    ):
        """
        Args:
            zip_path: 指向 alibaba-lingjun-dataset-2023-main.zip 的路径
            max_gpus: 集群最大 GPU 数，用于过滤过大的 job
            duration_ms: 仿真总时长（毫秒）
            seed: 随机种子
            target_bw_bps: 目标链路带宽，用于计算无竞争迭代间隔（默认 100Gbps）
            overhead_factor: NCCL/PCIe 协议开销因子（默认 1.3）
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Lingjun dataset not found: {zip_path}")
        self.zip_path = zip_path
        self.max_gpus = max_gpus
        self.duration_ms = duration_ms
        self.rng = random.Random(seed)
        self.target_bw_bps = target_bw_bps
        self.overhead_factor = overhead_factor

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _map_model(model_name: str) -> str:
        """将数据集中的模型名映射到 MODEL_PARAMS 中的标准名（大小写不敏感）。

        返回的键名保证在 MODEL_PARAMS 中存在；未知模型回退到 "BERT-Base"。
        """
        # 1) 直接命中
        if model_name in MODEL_PARAMS:
            return model_name
        # 2) 大小写不敏感查别名表
        key = model_name.lower()
        mapped = _MODEL_ALIASES.get(key)
        if mapped is not None:
            return mapped
        # 3) fallback
        return "BERT-Base"

    @staticmethod
    def _compute_mb_per_iter(params: dict, dp: int) -> float:
        """计算每轮迭代的 AllReduce 数据量（MB）。
        
        dp=1 时不需要网络通信（单 worker 无 AllReduce），返回 0。
        """
        if dp <= 1:
            return 0.0
        param_count = params.get("params", 1e9)
        bpp = 2 if params.get("fp16", True) else 4
        bytes_per_iter = 2 * param_count * bpp / max(dp, 1)
        return bytes_per_iter / (1024 * 1024)

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
        """加载并返回 Job 列表。"""
        z = zipfile.ZipFile(self.zip_path)
        prefix = "alibaba-lingjun-dataset-2023-main/data/"

        # ---- 1. 读取 worker.csv，统计每个 job 的 worker 数量 ----
        worker_count: dict[str, int] = {}
        with z.open(prefix + "worker.csv") as f:
            raw = f.read().decode("utf-8")
        for row in csv.DictReader(io.StringIO(raw)):
            if row.get("replica_type") == "worker":
                jn = row["job_name"]
                worker_count[jn] = worker_count.get(jn, 0) + 1

        # ---- 2. 读取 topo.csv，获取集群主机数 ----
        num_hosts = 0
        with z.open(prefix + "topo.csv") as f:
            raw = f.read().decode("utf-8")
        for _ in csv.DictReader(io.StringIO(raw)):
            num_hosts += 1

        # ---- 3. 读取 job.csv，过滤有效 job ----
        with z.open(prefix + "job.csv") as f:
            raw = f.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))

        candidates: list[dict] = []
        for row in reader:
            # 只取 Succeeded
            if row.get("status") != "Succeeded":
                continue
            # 排除无意义模型
            model = row.get("model", "")
            if model in ("unknown", "preprocess", "sidecar"):
                continue

            job_name = row.get("job_name", "")
            ngpus = worker_count.get(job_name, 0)
            if ngpus == 0 or ngpus > self.max_gpus:
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
            if duration_min <= 5:  # 太短的训练无意义
                continue

            # 映射模型
            mapped = self._map_model(model)
            if mapped not in MODEL_PARAMS:
                continue

            candidates.append({
                "job_name": job_name,
                "model": mapped,
                "num_workers": ngpus,
                "duration_min": duration_min,
            })

        # ---- 4. 按 duration 降序取最多 24 个 job ----
        candidates.sort(key=lambda x: x["duration_min"], reverse=True)
        candidates = candidates[:24]

        if not candidates:
            return []

        # ---- 5. 构建 Job 对象 ----
        jobs: List[Job] = []
        overhead_factor = self.overhead_factor

        for i, cand in enumerate(candidates):
            model = cand["model"]
            dp = cand["num_workers"]
            slo_ci = self._get_slo_ci(model)

            params = MODEL_PARAMS[model]
            mb_per_iter = self._compute_mb_per_iter(params, dp)
            raw_comm_ms = mb_per_iter * 8 * 1024 * 1024 / self.target_bw_bps * 1000.0
            # dp=1 时 comm_solo_ms = 0，但为了数值稳定性至少保留 0.1ms
            comm_solo = max(raw_comm_ms, 0.1)
            comp_ms = get_comp_ms(model, default=50.0)
            iter_interval_ms = comm_solo + comp_ms

            effective_comm_solo = comm_solo * overhead_factor
            target_iter_ms = comp_ms + effective_comm_solo * slo_ci
            target_iters = max(1, int(self.duration_ms / target_iter_ms))

            # 均匀分布的任务开始时间：在仿真时长的 0%-80% 之间均匀分布
            # 确保所有任务有充分的运行时间
            start_time_ms = self.rng.uniform(0, self.duration_ms * 0.8)

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
                            dp, num_hosts, seed=self.rng.randint(0, 2**31)
                        )
                        if num_hosts > 0
                        else None
                    ),
                )
            )

        return jobs
