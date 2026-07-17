"""链路模型。"""

from __future__ import annotations
from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .flow import Flow


class Link:
    """一条有向链路，承载若干 flow 的带宽竞争。"""

    def __init__(self, lid: str, bw_bps: float):
        """
        参数：
            lid: 链路唯一标识
            bw_bps: 链路带宽（bit/s）
        """
        self.lid = lid
        self.bw_bps = bw_bps
        self.active_flows: Set[Flow] = set()

    def add_flow(self, flow: Flow) -> None:
        self.active_flows.add(flow)

    def remove_flow(self, flow: Flow) -> None:
        self.active_flows.discard(flow)

    @property
    def available_bw_bps(self) -> float:
        return self.bw_bps

    def __repr__(self) -> str:
        return f"Link({self.lid}, {self.bw_bps / 1e9:.1f}Gbps)"
