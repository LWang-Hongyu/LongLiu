"""配置加载器。

从 config.yaml 加载唯一配置源，禁止代码内硬编码冻结参数。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import yaml


# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

# 全局缓存
_config_cache: Optional[Dict[str, Any]] = None
_config_hash_cache: Optional[str] = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载 config.yaml（带缓存）。"""
    global _config_cache
    path = config_path or _CONFIG_PATH
    if _config_cache is not None:
        return _config_cache
    with open(path, "r") as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache


def config_hash(config_path: str = None) -> str:
    """计算 config.yaml 的 SHA256 哈希。"""
    global _config_hash_cache
    if _config_hash_cache is not None:
        return _config_hash_cache
    path = config_path or _CONFIG_PATH
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    _config_hash_cache = h.hexdigest()[:16]
    return _config_hash_cache


def get_frozen(key: str, default: Any = None) -> Any:
    """读取冻结参数。"""
    cfg = load_config()
    return cfg.get("frozen", {}).get(key, default)


def get_topology(key: str = None, default: Any = None) -> Any:
    """读取拓扑参数。key=None 返回整个 topology dict。"""
    cfg = load_config()
    topo = cfg.get("topology", {})
    if key is None:
        return topo
    return topo.get(key, default)


def get_simulation(key: str = None, default: Any = None) -> Any:
    """读取仿真参数。"""
    cfg = load_config()
    sim = cfg.get("simulation", {})
    if key is None:
        return sim
    return sim.get(key, default)


def get_tiered_workload(key: str = None, default: Any = None) -> Any:
    """读取分层 workload 参数。"""
    cfg = load_config()
    tw = cfg.get("tiered_workload", {})
    if key is None:
        return tw
    return tw.get(key, default)


def get_model_types() -> list:
    """读取模型列表。"""
    cfg = load_config()
    return cfg.get("model_types", [])


def get_gpu_distribution() -> dict:
    """读取 GPU 分布。"""
    cfg = load_config()
    return cfg.get("gpu_distribution", {})


def get_v2_anchor_workload() -> list:
    """读取 v2 锚点 workload。"""
    cfg = load_config()
    return cfg.get("v2_anchor_workload", [])


def get_v4_config(key: str = None, default: Any = None) -> Any:
    """读取 v4 分配器参数。"""
    cfg = load_config()
    v4 = cfg.get("v4", {})
    if key is None:
        return v4
    return v4.get(key, default)


def get_feas_boundary_v3() -> dict:
    """读取 feas_boundary_v3 场景配置。"""
    cfg = load_config()
    return cfg.get("feas_boundary_v3", {})


def build_run_meta(extra: dict = None) -> dict:
    """构建 run_meta.json 内容。

    包含 config 哈希 + SEMANTICS_VERSION。
    """
    cfg = load_config()
    meta = {
        "semantics_version": cfg.get("semantics_version", "unknown"),
        "config_hash": config_hash(),
        "frozen": cfg.get("frozen", {}),
        "topology": cfg.get("topology", {}),
    }
    if extra:
        meta.update(extra)
    return meta


def validate_config() -> list:
    """CI lint：检查 config.yaml 完整性。

    返回错误列表，空列表表示通过。
    """
    errors = []
    try:
        cfg = load_config()
    except Exception as e:
        return [f"无法加载 config.yaml: {e}"]

    # 必需键检查
    required_top = ["semantics_version", "frozen", "topology", "simulation"]
    for key in required_top:
        if key not in cfg:
            errors.append(f"缺少顶层键: {key}")

    # frozen 参数检查
    frozen_keys = ["overhead_factor", "overlap_factor", "K"]
    frozen = cfg.get("frozen", {})
    for key in frozen_keys:
        if key not in frozen:
            errors.append(f"frozen 缺少: {key}")

    # topology 参数检查
    topo_keys = ["k", "host_bw_bps", "spine_bw_bps"]
    topo = cfg.get("topology", {})
    for key in topo_keys:
        if key not in topo:
            errors.append(f"topology 缺少: {key}")

    return errors


if __name__ == "__main__":
    # 自检
    errors = validate_config()
    if errors:
        print("config.yaml 验证失败:")
        for e in errors:
            print(f"  - {e}")
    else:
        cfg = load_config()
        print("config.yaml 验证通过")
        print(f"  semantics_version: {cfg['semantics_version']}")
        print(f"  config_hash: {config_hash()}")
        print(f"  frozen: {cfg.get('frozen', {})}")
        print(f"  model_types: {get_model_types()}")
        print(f"  v2_anchor_workload: {len(get_v2_anchor_workload())} jobs")
