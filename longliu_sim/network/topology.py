"""拓扑模型：基类、单链路、Fat-Tree、TwoTier Spine-TOR。"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from .link import Link
from .flow import Flow


class Topology(ABC):
    """拓扑基类。"""

    @abstractmethod
    def get_path(self, src: int, dst: int) -> List[Link]:
        """返回从 src 到 dst 的链路路径。"""
        pass

    def get_links_for_flow(self, flow: Flow) -> List[Link]:
        """返回 flow 经过的所有链路。"""
        return self.get_path(flow.src, flow.dst)


class SingleLinkTopology(Topology):
    """最简单的单瓶颈链路拓扑：所有节点共享一条链路。"""

    def __init__(self, num_hosts: int, bw_bps: float):
        self.num_hosts = num_hosts
        self.link = Link("bottleneck", bw_bps)

    def get_path(self, src: int, dst: int) -> List[Link]:
        return [self.link]


class FatTreeTopology(Topology):
    """Fat-Tree 拓扑，支持 ECMP 多路径。

    k/2 条等价 spine link，跨 pod 流量通过 flow ID hash 选择路径。
    同 pod 内流量不经过 spine。
    """

    def __init__(self, k: int, host_bw_bps: float, spine_bw_bps: float):
        if k % 2 != 0:
            raise ValueError("k must be even")
        self.k = k
        self.host_bw_bps = host_bw_bps
        self.spine_bw_bps = spine_bw_bps
        self.num_spine_links = k // 2
        per_link_bw = spine_bw_bps / self.num_spine_links
        self.spine_links = [
            Link(f"spine-{i}", per_link_bw)
            for i in range(self.num_spine_links)
        ]
        # 向后兼容：单一 link 属性
        self.link = self.spine_links[0]

    def _ecmp_path(self, src: int, dst: int) -> int:
        """ECMP hash：选择 spine link index。"""
        return (src ^ dst) % self.num_spine_links

    def get_path(self, src: int, dst: int) -> List[Link]:
        if src == dst:
            return []
        idx = self._ecmp_path(src, dst)
        return [self.spine_links[idx]]


class TwoTierTopology(Topology):
    """2-tier Spine-TOR 拓扑，支持 ECMP 多路径。

    同 rack 内流量走 rack link（无竞争）。
    跨 rack 流量走 src_rack → spine → dst_rack，spine 通过 ECMP 分散到多条等价链路。
    """

    def __init__(self, num_hosts: int, hosts_per_rack: int,
                 host_bw_bps: float, spine_bw_bps: float,
                 num_spine_links: int = 8):
        self.num_hosts = num_hosts
        self.hosts_per_rack = hosts_per_rack
        self.num_racks = num_hosts // hosts_per_rack
        self.host_bw_bps = host_bw_bps

        # 每个 rack 一条链路（16 × 40G = 640G）
        self.rack_links = [
            Link(f"rack-{i}", hosts_per_rack * host_bw_bps)
            for i in range(self.num_racks)
        ]
        # ECMP spine 链路：总带宽 spine_bw_bps 均分到 num_spine_links 条
        self.num_spine_links = num_spine_links
        per_link_bw = spine_bw_bps / num_spine_links
        self.spine_links = [
            Link(f"spine-{i}", per_link_bw)
            for i in range(num_spine_links)
        ]
        # 向后兼容
        self.spine_link = self.spine_links[0]

    def _host_to_rack(self, host_id: int) -> int:
        return host_id // self.hosts_per_rack

    def _ecmp_path(self, src: int, dst: int) -> int:
        """ECMP hash：选择 spine link index。"""
        return (src ^ dst) % self.num_spine_links

    def get_path(self, src: int, dst: int) -> List[Link]:
        src_rack = self._host_to_rack(src)
        dst_rack = self._host_to_rack(dst)
        if src_rack == dst_rack:
            return [self.rack_links[src_rack]]
        spine_idx = self._ecmp_path(src, dst)
        return [self.rack_links[src_rack], self.spine_links[spine_idx], self.rack_links[dst_rack]]

    @property
    def bottleneck_link(self) -> Link:
        """向后兼容：返回第一条 spine link。"""
        return self.spine_links[0]
