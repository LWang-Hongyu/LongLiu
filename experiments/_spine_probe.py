"""
_spine_probe: BwProbeSimulator —— 子类化 Simulator 提供：

1. per-job 实际带宽采样（bw_samples）
2. per-spine 物理利用率统计（cap 100%）

背景：FatTree k=4 有 2 条 spine link，Simulator 对每条 spine 独立调用
policy.allocate()。v4 的 allocate() 以 sum(spine_links) 为容量，导致单条 spine
上被分配了超过其物理带宽的流量（聚合利用率可 >100%）。物理上单链路利用率
不可能超过 100%，故这里按每条 spine 独立统计并 cap 到 1.0。

与 _recompute_bandwidth_twotier 的 ECMP 归并一致：每条 flow 只计入其路径上
第一条 spine link，避免同一条 flow 的带宽被重复计入多条 spine。
"""

from __future__ import annotations

from collections import defaultdict

from longliu_sim.core import Simulator


class BwProbeSimulator(Simulator):
    """子类化 Simulator：per-job 带宽采样 + per-spine 物理利用率（cap 100%）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bw_samples: list = []  # [(t_ms, {jid: total_rate_bps})]
        self.spine_util_samples: list = []  # [(t_ms, {spine_lid: capped_util_01})]

    def _record_link_utilization_snapshot(self):
        super()._record_link_utilization_snapshot()
        # per-job rate 采样
        per_job = {}
        for f in self.active_flows.values():
            if f.start_time_ms <= self.time_ms:
                per_job[f.jid] = per_job.get(f.jid, 0.0) + f.rate_bps
        self.bw_samples.append((self.time_ms, per_job))

        # per-spine 物理利用率（cap 100%）
        topo = self.topology
        spine_bw = {link.lid: link.bw_bps for link in topo.spine_links}
        spine_rate = {lid: 0.0 for lid in spine_bw}
        for f in self.active_flows.values():
            if f.start_time_ms > self.time_ms:
                continue
            for link in f.links:
                if link.lid in spine_rate:
                    spine_rate[link.lid] += f.rate_bps
                    break  # ECMP：每条 flow 只计入第一条 spine
        self.spine_util_samples.append((
            self.time_ms,
            {lid: min(spine_rate[lid] / spine_bw[lid], 1.0)
             for lid in spine_bw},
        ))

    def time_avg_spine_util(self, duration_ms: float) -> float:
        """per-spine 时间平均利用率（每条 ≤1.0）再取均值。"""
        if not self.spine_util_samples:
            return 0.0
        spine_total = defaultdict(float)
        for i in range(1, len(self.spine_util_samples)):
            t_prev, u_prev = self.spine_util_samples[i - 1]
            t_cur, _ = self.spine_util_samples[i]
            dt = t_cur - t_prev
            for lid, u in u_prev.items():
                spine_total[lid] += u * dt
        t_last, u_last = self.spine_util_samples[-1]
        dt = max(duration_ms - t_last, 0.0)
        for lid, u in u_last.items():
            spine_total[lid] += u * dt
        if not spine_total:
            return 0.0
        return sum(spine_total.values()) / (duration_ms * len(spine_total))
