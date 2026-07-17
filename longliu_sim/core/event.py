"""离散事件定义。"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


class EventType(Enum):
    JOB_START = auto()      # job 开始第一次迭代
    COMPUTE_END = auto()    # 计算阶段结束，开始通信
    FLOW_END = auto()       # 某个 flow 完成
    ITERATION_COMPLETE = auto()  # dp=1 时跳过 AllReduce，直接完成迭代


@dataclass
class Event:
    time_ms: float
    typ: EventType
    payload: Any = None

    def __lt__(self, other: Event) -> bool:
        return self.time_ms < other.time_ms
