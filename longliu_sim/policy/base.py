"""调度策略基类。"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List

from ..network.flow import Flow
from ..network.link import Link


# 分配结果：flow -> link -> bps
Allocation = Dict[Flow, Dict[Link, float]]


class Policy(ABC):
    """调度策略基类。"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: dict) -> Allocation:
        """
        为当前活跃 flow 分配带宽。

        参数：
            flows: 当前活跃的 flow 列表
            links: 涉及的所有链路
            time_ms: 当前仿真时间（ms）
            job_stats: job 的运行时统计信息，用于 LongLiu 等策略

        返回：
            Allocation: flow -> link -> bps
        """
        pass

    def __repr__(self) -> str:
        return f"Policy({self.name})"
