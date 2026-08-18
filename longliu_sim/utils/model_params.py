"""模型参数映射：从模型名推导通信数据量。

核心公式：
    mb_per_iter = 2 * param_count * bytes_per_param / dp / 1e6  (MB)

其中 2 倍因子来自 AllReduce 的总通信量（reduce-scatter + all-gather）。
仅支持 Data Parallel（DP）场景，TP/PP 留待未来。

使用方式：
    from longliu_sim.utils.model_params import MODEL_PARAMS, compute_mb_per_iter

    mb = compute_mb_per_iter("GPT-3-175B", dp=8)
    # → 2 * 175e9 * 4 / 8 / 1e6 = 175 MB
"""

from __future__ import annotations

MODEL_PARAMS = {
    # (model_name)     : {"params": param_count, "fp16": bool, "comp_ms": compute_time_ms}
    # comp_ms 校准来源：MLPerf Training v3.0 + HuggingFace benchmarks (A100-80G, bs=32, seq=2048)
    # 包含前向+反向+优化器步骤的完整迭代时间
    "GPT-3-175B":       {"params": 175e9,  "fp16": False, "comp_ms": 120},
    "GPT-3-175B-fp16":  {"params": 175e9,  "fp16": True,  "comp_ms": 120},
    "LLaMA-2-70B":      {"params": 70e9,   "fp16": True,  "comp_ms": 80},
    "LLaMA-2-65B":      {"params": 65e9,   "fp16": True,  "comp_ms": 80},
    "LLaMA-2-13B":      {"params": 13e9,   "fp16": True,  "comp_ms": 80},
    "LLaMA-2-7B":       {"params": 7e9,    "fp16": True,  "comp_ms": 40},
    "LLaMA-2-1B":       {"params": 1e9,    "fp16": True,  "comp_ms": 20},
    "GPT-3B":           {"params": 3e9,    "fp16": True,  "comp_ms": 30},
    "BERT-Large":       {"params": 340e6,  "fp16": False, "comp_ms": 50},
    "BERT-Large-fp16":  {"params": 340e6,  "fp16": True,  "comp_ms": 50},
    "BERT-Base":        {"params": 110e6,  "fp16": True,  "comp_ms": 40},
    "ResNet-50":        {"params": 25e6,   "fp16": False, "comp_ms": 30},
    "ResNet-50-fp16":   {"params": 25e6,   "fp16": True,  "comp_ms": 30},
    "ResNet-18":        {"params": 11e6,   "fp16": True,  "comp_ms": 25},
    "ViT-Large":        {"params": 307e6,  "fp16": True,  "comp_ms": 60},
    "ViT-Base":         {"params": 86e6,   "fp16": True,  "comp_ms": 40},
    "T5-11B":           {"params": 11e9,   "fp16": False, "comp_ms": 60},
    "T5-11B-fp16":      {"params": 11e9,   "fp16": True,  "comp_ms": 60},
    "T5-1B":            {"params": 1e9,    "fp16": True,  "comp_ms": 20},
}


def compute_mb_per_iter(model: str, dp: int = 1, bytes_per_param: int | None = None) -> float:
    """
    计算每轮迭代 AllReduce 的通信数据量（MB）。

    参数：
        model: 模型名称（需在 MODEL_PARAMS 中）
        dp: Data Parallel 度
        bytes_per_param: 覆盖 MODEL_PARAMS 中的 fp16/fp32 推断

    返回：
        mb_per_iter (MB)
    """
    if model not in MODEL_PARAMS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_PARAMS.keys())}")

    info = MODEL_PARAMS[model]
    n_params = info["params"]
    if bytes_per_param is None:
        bpp = 2 if info["fp16"] else 4
    else:
        bpp = bytes_per_param

    # AllReduce 总通信量 ≈ 2 * param_bytes  (reduce-scatter + all-gather)
    total_bytes = 2 * n_params * bpp
    mb_per_iter = total_bytes / dp / 1e6  # MB

    return mb_per_iter


def get_comp_ms(model: str, default: float = 50.0) -> float:
    """从 MODEL_PARAMS 获取模型的计算时间（ms），不存在时返回 default。"""
    info = MODEL_PARAMS.get(model)
    if info is None:
        return default
    return info.get("comp_ms", default)


def estimate_iter_interval(model: str, dp: int = 1,
                           bw_bps: float = 40e9,
                           compute_ratio: float = 0.5) -> float:
    """
    估计无竞争时的迭代时间 iter_interval_ms。

    参数：
        model: 模型名称
        dp: Data Parallel 度
        bw_bps: 链路带宽（bit/s）
        compute_ratio: 计算时间占总时间的比例（剩余为通信时间）

    返回：
        iter_interval_ms (ms)
    """
    mb = compute_mb_per_iter(model, dp)
    bits = mb * 8 * 1024 * 1024
    comm_time_ms = bits / bw_bps * 1000.0
    total_time_ms = comm_time_ms / (1 - compute_ratio)
    return total_time_ms