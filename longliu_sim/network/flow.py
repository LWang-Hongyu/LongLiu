"""Flow 模型。"""

from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .link import Link


class Flow:
    """一个需要传输的数据流，对应某 job 某次迭代中的通信阶段。"""

    def __init__(self, fid: str, jid: str, src: int, dst: int,
                 size_bits: float, links: List[Link],
                 iter_version: int = 0):
        """
        参数：
            fid: flow 唯一标识
            jid: 所属 job 标识
            src: 源节点编号
            dst: 目的节点编号
            size_bits: 数据量（bit）
            links: 该 flow 经过的链路列表
            iter_version: 所属迭代版本号（用于重叠迭代的 barrier 隔离）
        """
        self.fid = fid
        self.jid = jid
        self.src = src
        self.dst = dst
        self.size_bits = size_bits
        self.rem_bits = size_bits
        self.links = links
        self.iter_version = iter_version
        self.start_time_ms: float = 0.0
        self.end_time_ms: float = 0.0
        self.rate_bps: float = 0.0
        self.finished: bool = False

    @property
    def is_finished(self) -> bool:
        return self.rem_bits <= 1.0

    def advance(self, dt_ms: float) -> None:
        """按当前速率推进 dt_ms 时间。"""
        self.rem_bits -= self.rate_bps * (dt_ms / 1000.0)
        if self.rem_bits < 0:
            self.rem_bits = 0.0

    def __repr__(self) -> str:
        return f"Flow({self.fid}, {self.jid}, {self.rem_bits / 1e6:.1f}Mb/{self.size_bits / 1e6:.1f}Mb)"
