"""主仿真器：事件驱动，支持拓扑感知、多策略调度、AllReduce 多流 + Barrier 语义。"""

from __future__ import annotations
import heapq
from typing import Dict, List, Set

from ..network.topology import Topology, TwoTierTopology, FatTreeTopology
from ..network.flow import Flow
from ..network.link import Link
from ..job.job import Job
from ..policy.base import Policy
from .event import Event, EventType


class IterationRecord:
    """记录单次迭代的结果。"""

    def __init__(self, jid: str, iter_idx: int,
                 start_ms: float, end_ms: float,
                 comm_ms: float, n_flows: int = 1,
                 compute_end_ms: float | None = None):
        self.jid = jid
        self.iter_idx = iter_idx
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.comm_ms = comm_ms
        self.n_flows = n_flows
        self.compute_end_ms = compute_end_ms

    @property
    def iter_ms(self) -> float:
        return self.end_ms - self.start_ms


class SimulationResult:
    """仿真结果。"""

    def __init__(self, jobs: Dict[str, Job],
                 records: List[IterationRecord],
                 overlap_factor: float = 1.0):
        self.jobs = jobs
        self.records = records
        self.overlap_factor = overlap_factor
        self.link_utilization: Dict[str, Dict] = {}  # link_id -> {mean, max, min, samples}

    def total_iterations(self) -> int:
        return len(self.records)

    def avg_iteration_ms(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.iter_ms for r in self.records) / len(self.records)

    def slo_attainment(self) -> float:
        """达到目标迭代次数的 job 比例。"""
        if not self.jobs:
            return 0.0
        ok = sum(1 for j in self.jobs.values()
                 if j.completed_iters >= j.target_iters)
        return ok / len(self.jobs)

    def per_job_stats(self) -> Dict[str, dict]:
        stats = {}
        for jid, job in self.jobs.items():
            rs = [r for r in self.records if r.jid == jid]
            avg_iter_ms = sum(r.iter_ms for r in rs) / len(rs) if rs else 0.0
            avg_comm_ms = sum(r.comm_ms for r in rs) / len(rs) if rs else 0.0
            comm_budget = job.slo_ci * job.comm_solo_ms * job.overhead_factor
            if self.overlap_factor > 0:
                # 重叠模式：target = max(comp, comm_budget) + 串行开销
                serial_overhead = (1.0 - self.overlap_factor) * min(job.comp_ms, comm_budget)
                target_iter_ms = max(job.comp_ms, comm_budget) + serial_overhead
            else:
                target_iter_ms = job.comp_ms + comm_budget
            meets_slo = avg_iter_ms <= target_iter_ms if rs else False
            sas = target_iter_ms / avg_iter_ms if avg_iter_ms > 0 else 0.0
            stats[jid] = {
                "completed_iters": job.completed_iters,
                "target_iters": job.target_iters,
                "slo_violations": job.slo_violations,
                "avg_iter_ms": avg_iter_ms,
                "avg_comm_ms": avg_comm_ms,
                "target_iter_ms": target_iter_ms,
                "meets_slo": meets_slo,
                "sas": sas,
            }
        return stats

    def __repr__(self) -> str:
        return (f"SimulationResult(iters={self.total_iterations()}, "
                f"avg_iter={self.avg_iteration_ms():.1f}ms, "
                f"slo_attainment={self.slo_attainment()*100:.1f}%)")


class Simulator:
    """flow-level 离散事件仿真器，支持 AllReduce 多流 + Barrier 语义。"""

    def __init__(self, topology: Topology, policy: Policy,
                 duration_ms: float, seed: int = 0,
                 overhead_factor: float = 1.0,
                 overlap_factor: float = 1.0):
        """
        参数：
            topology: 网络拓扑
            policy: 调度策略
            duration_ms: 仿真时长（ms）
            seed: 随机种子（预留）
            overhead_factor: NCCL/PCIe 协议开销因子（论文默认 2.0）
            overlap_factor: compute-comm 重叠度（0=完全串行, 1=完全重叠, 默认 1.0）
        """
        self.topology = topology
        self.policy = policy
        self.duration_ms = duration_ms
        self.overhead_factor = overhead_factor
        self.overlap_factor = max(0.0, min(1.0, overlap_factor))

        self.jobs: Dict[str, Job] = {}
        self.active_flows: Dict[str, Flow] = {}  # fid -> Flow, O(1) removal
        self.events: List[Event] = []
        self.records: List[IterationRecord] = []
        self.finished_jobs: Set[str] = set()

        self.time_ms: float = 0.0
        self._flow_counter: int = 0
        self.link_utilization_history: Dict[str, List[float]] = {}  # link_id -> [utilization_samples]
        self._last_util_sample_time: float = 0.0  # 上次采样时间

    def submit(self, job: Job) -> None:
        """提交一个 job。"""
        self.jobs[job.jid] = job
        heapq.heappush(
            self.events,
            Event(job.start_time_ms, EventType.JOB_START, job.jid)
        )

    def _next_flow_id(self) -> str:
        self._flow_counter += 1
        return f"f{self._flow_counter}"

    def _advance(self, target_time_ms: float) -> None:
        """将仿真时间推进到 target_time_ms，并推进所有已开始的活跃 flow。"""
        dt_ms = target_time_ms - self.time_ms
        if dt_ms < 0:
            return
        for flow in self.active_flows.values():
            # 跳过尚未到 start_time_ms 的 flow（CASSINI time-shift）
            if flow.start_time_ms > self.time_ms:
                continue
            flow.advance(dt_ms)
        self.time_ms = target_time_ms

    def _collect_links(self) -> List[Link]:
        """收集所有链路（包括 ToR 上行、spine、host NIC）。

        从所有活跃流经过的链路中收集，而不是只收集 spine_links。
        """
        links: List[Link] = []
        seen = set()

        # 从所有活跃流收集链路
        for f in self.active_flows.values():
            for link in f.links:
                if id(link) not in seen:
                    seen.add(id(link))
                    links.append(link)

        # 如果没有活跃流，返回拓扑中定义的 spine_links（向后兼容）
        if not links:
            if isinstance(self.topology, (TwoTierTopology, FatTreeTopology)):
                return list(self.topology.spine_links)
            if hasattr(self.topology, "link"):
                return [self.topology.link]

        return links

    def _recompute_bandwidth(self) -> None:
        """根据策略重新分配活跃 flow 的带宽。"""
        if not self.active_flows:
            return

        if isinstance(self.topology, (TwoTierTopology, FatTreeTopology)):
            self._recompute_bandwidth_twotier()
            return

        # 已 ready 的 flow 才参与带宽分配
        ready_flows = [f for f in self.active_flows.values()
                       if f.start_time_ms <= self.time_ms]

        links = self._collect_links()
        alloc = {}
        if ready_flows:
            alloc = self.policy.allocate(
                ready_flows, links, self.time_ms, self.jobs
            )

        for flow in self.active_flows.values():
            if flow.start_time_ms > self.time_ms:
                flow.rate_bps = 0.0
                continue
            flow.rate_bps = 0.0
            flow_alloc = alloc.get(flow, {})
            for link, bw in flow_alloc.items():
                flow.rate_bps += bw

    def _recompute_bandwidth_twotier(self) -> None:
        """TwoTier/FatTree 拓扑带宽分配：ECMP spine + rack link 竞争。"""
        topo = self.topology
        ready_flows = [f for f in self.active_flows.values()
                       if f.start_time_ms <= self.time_ms]

        is_twotier = isinstance(topo, TwoTierTopology)

        # 按 rack link 分组所有 ready flow（仅 TwoTier）
        rack_flow_map: Dict[int, List[Flow]] = {}
        if is_twotier:
            for f in ready_flows:
                for link in f.links:
                    for i in range(topo.num_racks):
                        if link is topo.rack_links[i]:
                            rack_flow_map.setdefault(i, []).append(f)
                            break

        # 按 spine link 分组跨 rack flow
        spine_flow_map: Dict[int, List[Flow]] = {}
        for f in ready_flows:
            for link in f.links:
                if link in topo.spine_links:
                    spine_idx = topo.spine_links.index(link)
                    spine_flow_map.setdefault(spine_idx, []).append(f)
                    break

        alloc: Dict[Flow, Dict[Link, float]] = {}

        # 每条 spine link 独立分配带宽
        for spine_idx, flows in spine_flow_map.items():
            link = topo.spine_links[spine_idx]
            if flows:
                a = self.policy.allocate(flows, [link], self.time_ms, self.jobs)
                for f, link_alloc in a.items():
                    alloc.setdefault(f, {}).update(link_alloc)

        # 每个 rack link 独立分配带宽（仅 TwoTier）
        if is_twotier:
            for rack_id, flows in rack_flow_map.items():
                link = topo.rack_links[rack_id]
                if flows:
                    a = self.policy.allocate(flows, [link], self.time_ms, self.jobs)
                    for f, link_alloc in a.items():
                        alloc.setdefault(f, {}).update(link_alloc)

        # 设置 flow 速率 = 所有链路分配之和
        for f in self.active_flows.values():
            if f.start_time_ms > self.time_ms:
                f.rate_bps = 0.0
            else:
                f.rate_bps = sum(alloc.get(f, {}).values())

    def _schedule_next_flow_end(self) -> None:
        """将最早的 flow 完成时间作为事件入队。"""
        if not self.active_flows:
            return
        candidates = []
        for f in self.active_flows.values():
            if f.rate_bps > 0:
                finish_ms = self.time_ms + f.rem_bits / f.rate_bps * 1000.0
                candidates.append(finish_ms)
        # 若当前无可完成 flow，但有未 ready 的 flow，则推进到其 start_time
        if not candidates:
            future_starts = [f.start_time_ms for f in self.active_flows.values()
                             if f.start_time_ms > self.time_ms]
            if future_starts:
                earliest = min(future_starts)
                heapq.heappush(
                    self.events,
                    Event(earliest, EventType.FLOW_END, None)
                )
            # 严格优先级调度可能导致所有 flows 都 rate_bps=0（被饿死）
            # 此时依赖事件队列中的其他事件（JOB_START, COMPUTE_END）推进时间
            return
        earliest = min(candidates)
        if earliest <= self.time_ms:
            earliest = self.time_ms + 1e-9
        heapq.heappush(
            self.events,
            Event(earliest, EventType.FLOW_END, None)
        )

    def _create_allreduce_flows(self, job: Job) -> None:
        """
        为 job 创建本轮 AllReduce 的所有 flow。

        num_workers=1 → 1 个 aggregate flow（向后兼容）
        num_workers>1 → N 个 parallel flow，每个承载 bits_per_iter / N 数据

        如果有 worker_hosts，使用实际拓扑路径（TwoTier）；否则使用 legacy src=0,dst=1。
        """
        n = job.num_workers if job.num_workers > 1 else 1
        workers = job.worker_hosts if job.worker_hosts else [0] * n

        # 计算实际需要创建的 flow 数（跳过 src == dst 的本地 segment）
        actual_flows = sum(1 for i in range(n) if workers[i] != workers[(i + 1) % n])
        job.start_allreduce(actual_flows if actual_flows > 0 else n)

        for i in range(n):
            src = workers[i]
            dst = workers[(i + 1) % n]

            # 跳过同主机的 segment（本地内存拷贝，无需网络传输）
            if src == dst:
                continue

            links = self.topology.get_path(src, dst)

            flow = Flow(
                fid=self._next_flow_id(),
                jid=job.jid,
                src=src,
                dst=dst,
                size_bits=job.bits_per_flow * self.overhead_factor,
                links=links,
                iter_version=job._iter_version
            )
            # CASSINI time-shift：偏移通信开始时间
            flow.start_time_ms = self.time_ms + job.comm_offset_ms
            self.active_flows[flow.fid] = flow
            for link in links:
                link.add_flow(flow)

    def _handle_flow_end(self, flow: Flow) -> None:
        """
        处理某个 flow 完成。

        如果 job 还有其他未完成的 flow（barrier），不做处理。
        当最后一个 flow 完成时（barrier 通过），记录迭代并安排下一轮计算。
        """
        job = self.jobs[flow.jid]

        # 从 active_flows 和链路中移除
        del self.active_flows[flow.fid]
        for link in flow.links:
            link.remove_flow(flow)
        flow.finished = True

        # Barrier 语义：等待所有同版本 flow 完成
        if not job.on_flow_complete(iter_version=flow.iter_version):
            return

        # 所有 flow 完成 → barrier 通过
        self._handle_iteration_complete(job, compute_end_ms=job.compute_end_time_ms)

    def _cleanup_finished_flows(self) -> None:
        """批量清理已完成的 flow（避免在迭代中修改 dict）。"""
        finished = [f for f in self.active_flows.values() if f.is_finished]
        for f in finished:
            self._handle_flow_end(f)

    def _handle_iteration_complete(self, job: Job,
                                    compute_end_ms: float | None = None) -> None:
        """记录迭代完成并安排下一轮计算（dp>1 由 barrier 触发，dp=1 直接触发）。

        overlap 模型：下一轮 compute 从 max(compute_end, barrier) 开始。
        dp=1 优化：首次迭代后批量计算剩余迭代，避免逐事件模拟。
        """
        job.on_comm_end(self.time_ms)

        # 记录本轮迭代
        iter_idx = job.completed_iters
        self.records.append(IterationRecord(
            job.jid, iter_idx,
            job.iter_start_time_ms,
            self.time_ms,
            self.time_ms - job.comm_start_time_ms,
            n_flows=job.num_workers if job.num_workers > 1 else 1,
            compute_end_ms=compute_end_ms,
        ))

        # 判断是否完成全部目标迭代
        if job.completed_iters >= job.target_iters:
            self.finished_jobs.add(job.jid)
            return

        # dp=1 优化：无通信的 job 只模拟首次迭代，剩余批量计算
        if job.num_workers <= 1 and job.completed_iters >= 1:
            effective_iter_ms = self.time_ms - job.iter_start_time_ms
            remaining = job.target_iters - job.completed_iters
            for i in range(remaining):
                t = self.time_ms + (i + 1) * effective_iter_ms
                if t > self.duration_ms:
                    break
                self.records.append(IterationRecord(
                    job.jid, job.completed_iters + i + 1,
                    t - effective_iter_ms, t, 0.0,
                    n_flows=1, compute_end_ms=t,
                ))
                job.completed_iters += 1
            self.finished_jobs.add(job.jid)
            return

        # 计算下一轮迭代的开始时间（overlap 模型）
        if compute_end_ms is not None and self.overlap_factor > 0:
            next_start = max(compute_end_ms, self.time_ms)
            next_start = (1.0 - self.overlap_factor) * self.time_ms + \
                          self.overlap_factor * next_start
        else:
            next_start = self.time_ms

        job.on_iter_start(next_start)
        next_comp_end = next_start + job.comp_ms

        heapq.heappush(
            self.events,
            Event(next_comp_end, EventType.COMPUTE_END, job.jid)
        )

    def _process_event(self, event: Event) -> None:
        """处理单个离散事件。"""
        if event.typ == EventType.JOB_START:
            jid = event.payload
            job = self.jobs[jid]
            job.on_iter_start(self.time_ms)
            # 先计算，再通信
            heapq.heappush(
                self.events,
                Event(self.time_ms + job.comp_ms,
                      EventType.COMPUTE_END, jid)
            )

        elif event.typ == EventType.COMPUTE_END:
            jid = event.payload
            job = self.jobs[jid]
            job.on_comm_start(self.time_ms)
            job.compute_end_time_ms = self.time_ms  # 记录 compute 结束时刻
            if job.num_workers <= 1:
                # dp=1：无 AllReduce，直接完成迭代（comm_ms=0）
                heapq.heappush(
                    self.events,
                    Event(self.time_ms, EventType.ITERATION_COMPLETE, jid)
                )
            else:
                # 创建本轮 AllReduce 的所有 flow（start_time_ms 会在内部应用 comm_offset_ms）
                self._create_allreduce_flows(job)

        elif event.typ == EventType.ITERATION_COMPLETE:
            jid = event.payload
            job = self.jobs[jid]
            self._handle_iteration_complete(job, compute_end_ms=job.compute_end_time_ms)

        elif event.typ == EventType.FLOW_END:
            # 处理所有在当前时间点刚好完成的 flow
            self._cleanup_finished_flows()

    def run(self) -> SimulationResult:
        """运行仿真并返回结果。"""
        self._recompute_bandwidth()

        while self.events and self.time_ms < self.duration_ms:
            next_time = self.events[0].time_ms

            # 检查是否有活跃 flow 会先完成
            if self.active_flows:
                finish_times = [
                    self.time_ms + f.rem_bits / f.rate_bps * 1000.0
                    for f in self.active_flows.values() if f.rate_bps > 0
                ]
                if finish_times:
                    earliest_finish = min(finish_times)
                    if earliest_finish < next_time:
                        next_time = earliest_finish

            if next_time > self.duration_ms:
                break

            # 推进时间
            self._advance(next_time)

            # 记录链路利用率快照（每次 bandwidth 重分配时采样）
            self._record_link_utilization_snapshot()

            # 处理在当前时间点完成的 flow
            self._cleanup_finished_flows()

            # 处理事件队列中时间 <= 当前时间的事件
            while (self.events and
                   abs(self.events[0].time_ms - self.time_ms) < 1e-9):
                event = heapq.heappop(self.events)
                self._process_event(event)

            # 重新分配带宽并安排下一次 flow 完成
            self._recompute_bandwidth()
            self._schedule_next_flow_end()

        result = SimulationResult(self.jobs, self.records, self.overlap_factor)

        # 计算链路利用率统计
        if self.link_utilization_history:
            result.link_utilization = self._compute_link_utilization_stats()

        return result

    def _record_link_utilization_snapshot(self) -> None:
        """记录链路利用率快照。"""
        # 计算总活跃带宽
        total_active_bw = sum(f.rate_bps for f in self.active_flows.values())

        # 计算总 spine 带宽
        if hasattr(self.topology, 'spine_links'):
            total_spine_bw = sum(link.bw_bps for link in self.topology.spine_links)

            # 计算平均 spine link 利用率
            avg_utilization = total_active_bw / total_spine_bw if total_spine_bw > 0 else 0.0

            # 记录到每个 spine link（简化：所有 spine links 共享相同利用率）
            for link in self.topology.spine_links:
                if link.lid not in self.link_utilization_history:
                    self.link_utilization_history[link.lid] = []
                self.link_utilization_history[link.lid].append(avg_utilization)

    def _flow_uses_link(self, flow: Flow, link_id: str) -> bool:
        """检查 flow 是否使用指定链路。"""
        # 简化实现：假设所有 cross-pod flow 都使用所有 spine links（ECMP）
        # 更准确的实现需要知道 flow 的实际路径
        return True  # 保守估计：所有 flow 都使用所有 spine links

    def _compute_link_utilization_stats(self) -> Dict[str, float]:
        """计算链路利用率统计。"""
        stats = {}
        for link_id, samples in self.link_utilization_history.items():
            if samples:
                stats[link_id] = {
                    "mean": sum(samples) / len(samples),
                    "max": max(samples),
                    "min": min(samples),
                    "samples": len(samples)
                }
        return stats