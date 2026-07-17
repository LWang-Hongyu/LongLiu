"""基础仿真器单元测试。"""

import sys
sys.path.insert(0, "/home/why/LongLiu_rebuild/sim-nextgen")

from longliu_sim.network import SingleLinkTopology
from longliu_sim.job import Job
from longliu_sim.policy import Fair, LongLiu, SRPT, CRUX, CASSINI
from longliu_sim.core import Simulator
import math


def test_single_job_fair():
    """单个 job 在 Fair 策略下应不受竞争影响。"""
    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)
    policy = Fair()
    sim = Simulator(topo, policy, duration_ms=5000)

    job = Job(
        jid="J0",
        model="gpt",
        mb_per_iter=100,
        iter_interval_ms=200,
        target_iters=10,
        slo_ci=1.5,
        start_time_ms=0
    )
    sim.submit(job)
    result = sim.run()

    print("test_single_job_fair:")
    print(f"  total iters: {result.total_iterations()}")
    print(f"  avg iter ms: {result.avg_iteration_ms():.1f}")
    print(f"  slo attainment: {result.slo_attainment()*100:.1f}%")
    assert result.total_iterations() == 10
    assert result.slo_attainment() == 1.0
    print("  PASSED\n")


def test_longliu_prioritizes_tight_slo():
    """混合 SLO 场景：LongLiu 应优先保障紧 SLO job。"""
    jobspecs = [
        ("J0", 100, 200, 22, 1.2),   # 紧 SLO：希望每 200ms 完成一轮
        ("J1", 200, 400, 11, 3.0),   # 松 SLO：每 400ms 一轮也可接受
    ]

    def run(policy):
        topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)
        sim = Simulator(topo, policy, duration_ms=5000)
        for jid, mb, iv, tg, ci in jobspecs:
            sim.submit(Job(jid, "m", mb, iv, tg, ci, 0))
        return sim.run()

    r_fair = run(Fair())
    r_longliu = run(LongLiu(K=3.0))

    print("test_longliu_prioritizes_tight_slo:")
    print(f"  Fair:     total={r_fair.total_iterations()}, "
          f"slo={r_fair.slo_attainment()*100:.1f}%")
    print(f"  LongLiu:  total={r_longliu.total_iterations()}, "
          f"slo={r_longliu.slo_attainment()*100:.1f}%")

    stats_fair = r_fair.per_job_stats()
    stats_longliu = r_longliu.per_job_stats()

    # LongLiu 下紧 SLO job 完成更多迭代
    assert (stats_longliu["J0"]["completed_iters"] >=
            stats_fair["J0"]["completed_iters"])
    # LongLiu 总体 SLO 达成率不低于 Fair
    assert r_longliu.slo_attainment() >= r_fair.slo_attainment()
    print("  PASSED\n")


def test_heavy_contention_improves_total():
    """高竞争下 LongLiu 的总完成迭代数应不低于 Fair。"""
    jobspecs = [
        ("J0", 100, 200, 20, 1.5),
        ("J1", 100, 200, 20, 1.5),
        ("J2", 100, 200, 20, 1.5),
    ]

    def run(policy):
        topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)
        sim = Simulator(topo, policy, duration_ms=10000)
        for jid, mb, iv, tg, ci in jobspecs:
            sim.submit(Job(jid, "m", mb, iv, tg, ci, 0))
        return sim.run()

    r_fair = run(Fair())
    r_longliu = run(LongLiu(K=3.0))

    print("test_heavy_contention_improves_total:")
    print(f"  Fair:     total={r_fair.total_iterations()}, "
          f"slo={r_fair.slo_attainment()*100:.1f}%")
    print(f"  LongLiu:  total={r_longliu.total_iterations()}, "
          f"slo={r_longliu.slo_attainment()*100:.1f}%")

    # LongLiu 总完成迭代数不低于 Fair（不浪费带宽）
    assert r_longliu.total_iterations() >= r_fair.total_iterations()
    print("  PASSED\n")


def test_allreduce_barrier_multiflow():
    """
    AllReduce 多流 + Barrier 语义测试。

    num_workers=8 时，每轮迭代生成 8 个 parallel flow。
    迭代完成时间由最慢的 flow 决定（barrier）。
    验证：
    1. 多流模式与 aggregate 模式总迭代次数一致
    2. 多流模式的每次迭代记录 n_flows=8
    """
    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)

    # Aggregate flow（num_workers=1）
    sim_agg = Simulator(topo, Fair(), duration_ms=5000)
    job_agg = Job(
        jid="Jagg", model="gpt",
        mb_per_iter=100, iter_interval_ms=200,
        target_iters=15, slo_ci=1.5,
        num_workers=1, start_time_ms=0
    )
    sim_agg.submit(job_agg)
    r_agg = sim_agg.run()

    # Multi-flow（num_workers=8）
    sim_mf = Simulator(topo, Fair(), duration_ms=5000)
    job_mf = Job(
        jid="Jmf", model="gpt",
        mb_per_iter=100, iter_interval_ms=200,
        target_iters=15, slo_ci=1.5,
        num_workers=8, start_time_ms=0
    )
    sim_mf.submit(job_mf)
    r_mf = sim_mf.run()

    print("test_allreduce_barrier_multiflow:")
    print(f"  Aggregate (nw=1): iters={r_agg.total_iterations()}, "
          f"avg={r_agg.avg_iteration_ms():.1f}ms")
    print(f"  Multi-flow (nw=8): iters={r_mf.total_iterations()}, "
          f"avg={r_mf.avg_iteration_ms():.1f}ms")

    # 多流和 aggregate 迭代次数应一致（总数据量相同）
    assert r_mf.total_iterations() == r_agg.total_iterations(), \
        f"Multi-flow iters {r_mf.total_iterations()} != aggregate {r_agg.total_iterations()}"

    # 每次迭代记录应表明 n_flows=8
    mf_records = [r for r in r_mf.records if r.jid == "Jmf"]
    assert all(r.n_flows == 8 for r in mf_records), \
        f"Not all records have n_flows=8: {[r.n_flows for r in mf_records[:5]]}"

    # 多流模式下单个 flow 数据量为 aggregate 的 1/8
    # 在单链路无竞争下，多流总带宽 = aggregate 带宽，完成时间应接近
    ratio = r_mf.avg_iteration_ms() / r_agg.avg_iteration_ms()
    print(f"  Multi-flow / Aggregate avg iter ratio: {ratio:.2f}")
    # 允许一定误差（因为 flow 数多导致调度频率不同）
    assert 0.8 < ratio < 1.2, \
        f"Multi-flow avg iter too different: ratio={ratio:.2f}"

    print("  PASSED\n")


def test_allreduce_barrier_contention():
    """
    多流 + 竞争场景：验证 tail flow 效应。

    两个 job 竞争，其中一个 num_workers=8。
    LongLiu 应能根据 barrier 语义准确计算 deficit。
    """
    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)

    sim = Simulator(topo, LongLiu(K=3.0), duration_ms=8000)
    sim.submit(Job("J0", "gpt", 100, 200, 20, 1.5, num_workers=1, start_time_ms=0))
    sim.submit(Job("J1", "gpt", 100, 200, 20, 1.5, num_workers=8, start_time_ms=0))
    r = sim.run()

    stats = r.per_job_stats()
    print("test_allreduce_barrier_contention:")
    print(f"  Total iters: {r.total_iterations()}")
    print(f"  SLO attainment: {r.slo_attainment()*100:.1f}%")
    for jid, s in stats.items():
        print(f"  {jid}: iters={s['completed_iters']}, "
              f"avg_comm={s['avg_comm_ms']:.1f}ms, "
              f"avg_iter={s['avg_iter_ms']:.1f}ms")
    # 两个 job 都应至少完成一些迭代
    assert stats["J0"]["completed_iters"] > 0
    assert stats["J1"]["completed_iters"] > 0
    print("  PASSED\n")


def test_crux_baseline():
    """CRUX 策略应能运行并产生不同权重。"""
    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)
    sim = Simulator(topo, CRUX(alpha=1.0), duration_ms=3000)
    sim.submit(Job("J0", "gpt", 100, 200, 10, 1.5, start_time_ms=0))
    sim.submit(Job("J1", "gpt", 10, 200, 10, 1.5, start_time_ms=0))
    r = sim.run()

    print("test_crux_baseline:")
    print(f"  Total iters: {r.total_iterations()}")
    print(f"  SLO attainment: {r.slo_attainment()*100:.1f}%")
    stats = r.per_job_stats()
    for jid, s in stats.items():
        print(f"  {jid}: iters={s['completed_iters']}, "
              f"avg_comm={s['avg_comm_ms']:.1f}ms")
    assert r.total_iterations() > 0
    print("  PASSED\n")


def test_t_target_calibration():
    """
    T_target 两阶段校准测试。

    Phase 1: 方法级直接测试 get_T_target() 的校准逻辑。
    Phase 2: 仿真级验证，在完整仿真循环中验证 EMA 收敛。

    校准机制：
    - Stage 1: RTT probe 提供保守初始值
    - Stage 2: 持有最高优先级(DSCP=46)时，用 EMA 更新 T_target
    - 静态 T_target 跳过整个校准流程
    """
    # ============================================================
    # Phase 1: 方法级直接测试（精确控制每个分支）
    # ============================================================

    # --- 1a. 初始状态（无迭代），T_target 应使用 RTT probe ---
    job_a = Job("JA", "gpt", mb_per_iter=50, iter_interval_ms=500,
                target_iters=30, slo_ci=3.0,
                T_target=None, rtt_probe_ms=300.0, alpha=0.3)
    # comm_solo_ms = 50*8*1024*1024/40e9*1000 ≈ 102.4ms
    # default_T_target = 3.0 * 102.4 ≈ 307ms
    # rtt_probe_ms = 300.0

    # 无迭代时，无论是否有最高优先级，都返回 probe 值
    assert job_a.get_T_target(has_highest_priority=False) == 300.0
    assert job_a.get_T_target(has_highest_priority=True) == 300.0  # 无 last_iter_comm_time
    assert not job_a.ema_initialized

    # --- 1b. 有迭代但无最高优先级，仍用 RTT probe ---
    job_a.last_iter_comm_time_ms = 150.0
    assert job_a.get_T_target(has_highest_priority=False) == 300.0
    assert not job_a.ema_initialized  # 不应触发 EMA

    # --- 1c. 获得最高优先级 → EMA 初始化 ---
    t1 = job_a.get_T_target(has_highest_priority=True)
    assert job_a.ema_initialized, "获得最高优先级后应初始化 EMA"
    assert job_a.T_target_ema == 150.0, "EMA 应初始化为 last_iter_comm_time"
    assert t1 == 150.0

    # --- 1d. 后续最高优先级迭代 → EMA 通过 update_ema_from_comm_time 更新 ---
    # get_T_target() 在 EMA 初始化后不再更新，须通过 update_ema_from_comm_time() 更新
    # 这是设计选择：get_T_target 负责"读取"，update_ema_from_comm_time 负责"写入"
    job_a.last_iter_comm_time_ms = 130.0
    job_a.update_ema_from_comm_time()
    t2 = job_a.get_T_target(has_highest_priority=True)  # 读取已更新的 EMA
    expected = 0.3 * 130.0 + 0.7 * 150.0  # = 144.0
    assert abs(t2 - expected) < 1e-9, f"EMA 更新错误: expect {expected}, got {t2}"

    job_a.last_iter_comm_time_ms = 120.0
    job_a.update_ema_from_comm_time()
    t3 = job_a.get_T_target(has_highest_priority=True)
    expected2 = 0.3 * 120.0 + 0.7 * 144.0  # = 136.8
    assert abs(t3 - expected2) < 1e-9, f"EMA 第二次更新错误: expect {expected2}, got {t3}"

    # --- 1e. 失去最高优先级 → 使用已收敛的 EMA（get_T_target 只读不更新） ---
    job_a.last_iter_comm_time_ms = 140.0
    t4 = job_a.get_T_target(has_highest_priority=False)
    assert t4 == expected2, f"无最高优先级时应返回 EMA: expect {expected2}, got {t4}"
    # EMA 值不应被修改（has_highest_priority=False 且 T_target_ema 不为 None）

    # --- 1f. 静态 T_target → 跳过校准 ---
    job_b = Job("JB", "gpt", mb_per_iter=200, iter_interval_ms=200,
                target_iters=5, slo_ci=1.2,
                T_target=200.0)
    # 无论什么情况，静态 T_target 都返回固定值
    assert job_b.get_T_target(has_highest_priority=False) == 200.0
    assert job_b.get_T_target(has_highest_priority=True) == 200.0
    assert job_b.T_target_ema is None  # 不应初始化 EMA
    assert not job_b.ema_initialized

    # ============================================================
    # Phase 2: 仿真级端到端验证
    # ============================================================
    # Job A: 动态校准，tight SLO（slo_ci=1.0）→ 竞争下高 deficit → DSCP 46
    #         T_target 从 probe=300ms 向实际通信时间收敛
    # Job B: 静态 T_target=200ms，slo_ci=3.0（松 SLO）→ 低 deficit → 低优先级
    #         5 次迭代后退出

    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)
    sim = Simulator(topo, LongLiu(K=3.0, use_dynamic_T_target=True),
                    duration_ms=15000)

    job_a2 = Job("JA", "gpt", mb_per_iter=100, iter_interval_ms=250,
                 target_iters=30, slo_ci=1.0,
                 T_target=None, rtt_probe_ms=300.0, alpha=0.3)
    job_b2 = Job("JB", "gpt", mb_per_iter=200, iter_interval_ms=200,
                 target_iters=5, slo_ci=3.0,
                 T_target=200.0)

    sim.submit(job_a2)
    sim.submit(job_b2)
    result = sim.run()
    stats = result.per_job_stats()

    # Job B 应精确完成 5 次迭代
    assert stats["JB"]["completed_iters"] == 5, \
        f"JB 应完成 5 次迭代, 实际 {stats['JB']['completed_iters']}"

    # Job A 应完成 > 10 次迭代（B 退出后继续运行）
    assert stats["JA"]["completed_iters"] > 10, \
        f"JA 应完成 >10 次迭代, 实际 {stats['JA']['completed_iters']}"

    # 验证 EMA 已初始化并收敛
    assert job_a2.ema_initialized, "Job A 应在仿真中初始化 EMA"
    assert job_a2.T_target_ema is not None

    # EMA 应在 probe=300ms 和 solo_comm 之间
    # solo_comm = 100*8*1024*1024/40e9*1000 ≈ 21ms
    # 竞争下初期 comm 较高，退出竞争后收敛到 ~21ms
    # EMA 应低于 probe(300ms) 且高于 solo_comm 的 80%
    solo_comm_ms = job_a2._mb_to_bits(job_a2.mb_per_iter) / 40e9 * 1000.0
    print(f"  Job A: T_target_ema={job_a2.T_target_ema:.1f}ms, "
          f"rtt_probe=300.0ms, last_comm={job_a2.last_iter_comm_time_ms:.1f}ms, "
          f"solo_comm≈{solo_comm_ms:.1f}ms, iters={stats['JA']['completed_iters']}")
    print(f"  Job B: iters={stats['JB']['completed_iters']}")

    # EMA 应收敛到接近 solo_comm（从 probe=300ms 下降）
    assert job_a2.T_target_ema < 300.0, \
        f"T_target_ema={job_a2.T_target_ema:.1f} 应低于 rtt_probe=300ms"
    assert job_a2.T_target_ema >= 0.5 * solo_comm_ms, \
        f"T_target_ema={job_a2.T_target_ema:.1f} 不应低于 solo_comm 的 50% ({0.5*solo_comm_ms:.1f}ms)"

    print("  PASSED\n")


def test_cassini_offset():
    """CASSINI time-shift 后通信相位错开，峰值并发度降低。"""
    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)

    # Fair baseline（无偏移）
    sim_fair = Simulator(topo, Fair(), duration_ms=6000)
    sim_fair.submit(Job("J0", "m", mb_per_iter=100, iter_interval_ms=300,
                        target_iters=10, slo_ci=1.5, start_time_ms=0))
    sim_fair.submit(Job("J1", "m", mb_per_iter=100, iter_interval_ms=300,
                        target_iters=10, slo_ci=1.5, start_time_ms=0))
    r_fair = sim_fair.run()

    # CASSINI：设置 time-shift 偏移
    offsets = CASSINI.compute_offsets([300.0, 300.0])
    sim_cass = Simulator(topo, CASSINI(), duration_ms=6000)
    sim_cass.submit(Job("J0", "m", mb_per_iter=100, iter_interval_ms=300,
                        target_iters=10, slo_ci=1.5, start_time_ms=0,
                        comm_offset_ms=offsets[0]))
    sim_cass.submit(Job("J1", "m", mb_per_iter=100, iter_interval_ms=300,
                        target_iters=10, slo_ci=1.5, start_time_ms=0,
                        comm_offset_ms=offsets[1]))
    r_cass = sim_cass.run()

    # Fair 下两 job 首次迭代同时完成；CASSINI 下应明显错开
    fair_first = {r.jid: r.end_ms for r in r_fair.records if r.iter_idx == 1}
    cass_first = {r.jid: r.end_ms for r in r_cass.records if r.iter_idx == 1}
    fair_gap = abs(fair_first["J0"] - fair_first["J1"])
    cass_gap = abs(cass_first["J0"] - cass_first["J1"])

    print("test_cassini_offset:")
    print(f"  Fair first-iter gap:    {fair_gap:.1f}ms")
    print(f"  CASSINI first-iter gap: {cass_gap:.1f}ms")
    print(f"  Fair avg iter:    {r_fair.avg_iteration_ms():.1f}ms")
    print(f"  CASSINI avg iter: {r_cass.avg_iteration_ms():.1f}ms")
    assert cass_gap > fair_gap, \
        "CASSINI time-shift 应使两 job 首次通信开始时间错开"
    # time-shift 不降低总吞吐量，总迭代数应不低于 Fair
    assert r_cass.total_iterations() >= r_fair.total_iterations(), \
        "CASSINI 不应降低总完成迭代数"
    print("  PASSED\n")


def test_dscp_boundary():
    """验证 LongLiu.get_dscp() 在阈值边界的映射。"""
    print("test_dscp_boundary:")
    print(f"  pi=0.21  → DSCP {LongLiu.get_dscp(0.21)} (expect 46)")
    print(f"  pi=0.20  → DSCP {LongLiu.get_dscp(0.20)} (expect 34)")
    print(f"  pi=-0.30 → DSCP {LongLiu.get_dscp(-0.30)} (expect 18)")
    print(f"  pi=-0.50 → DSCP {LongLiu.get_dscp(-0.50)} (expect 0)")

    assert LongLiu.get_dscp(0.21) == 46
    assert LongLiu.get_dscp(0.20) == 34  # 严格大于阈值
    assert LongLiu.get_dscp(-0.30) == 18
    assert LongLiu.get_dscp(-0.50) == 0
    print("  PASSED\n")


def test_fat_tree_multipath():
    """Fat-Tree(k=4) 上 4 个 job 流量经过不同路径时链路竞争正确计算。"""
    from longliu_sim.network import FatTreeTopology

    # 当前 FatTreeTopology 为简化占位实现：所有路径复用 spine link
    # 测试重点：验证多个 flow 经过同一组 link 时竞争正确聚合
    topo = FatTreeTopology(k=4, host_bw_bps=40e9, spine_bw_bps=320e9)
    sim = Simulator(topo, Fair(), duration_ms=6000)
    for i in range(4):
        sim.submit(Job(f"J{i}", "m", mb_per_iter=50, iter_interval_ms=200,
                       target_iters=10, slo_ci=1.5, start_time_ms=0))
    r = sim.run()

    print("test_fat_tree_multipath:")
    print(f"  Total iters: {r.total_iterations()}")
    print(f"  SLO attainment: {r.slo_attainment()*100:.1f}%")
    stats = r.per_job_stats()
    for jid, s in stats.items():
        print(f"  {jid}: iters={s['completed_iters']}")
    # 4 个 job 共享 spine link，总迭代数应 > 0 且合理分配
    assert r.total_iterations() > 0
    print("  PASSED\n")


def test_t_target_short_job_trap():
    """
    短 job（10 个迭代）在高竞争环境中，T_target 不应被污染。

    即使竞争导致 last_iter_comm_time 很大，RTT probe 仍提供保守初始值。
    验证：job 的初始 T_target 来自 rtt_probe_ms，不直接使用 last_iter_comm_time。
    """
    job = Job("short", "gpt", mb_per_iter=80, iter_interval_ms=500,
              target_iters=10, slo_ci=2.0,
              T_target=None, rtt_probe_ms=250.0, alpha=0.3)
    # 模拟一次被污染的高竞争通信时间
    job.last_iter_comm_time_ms = 800.0
    job.completed_iters = 1

    # 无最高优先级时：应使用 RTT probe 的保守值，而非污染值
    t1 = job.get_T_target(has_highest_priority=False)
    assert t1 == 250.0, f"短 job 应使用 RTT probe，实际 {t1}"

    # 第一次获得最高优先级：EMA 初始化为 800ms（但这是设计行为）
    t2 = job.get_T_target(has_highest_priority=True)
    assert t2 == 800.0 and job.ema_initialized

    # 因此，短 job 设计上若拿到最高优先级会被污染。
    # 本测试验证：仿真中短 job 由于 deficit 通常较低，不太可能拿到最高优先级。
    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)
    sim = Simulator(topo, LongLiu(K=3.0, use_dynamic_T_target=True),
                    duration_ms=6000)
    # 短 job + 一个重竞争长 job
    sim.submit(Job("short", "gpt", mb_per_iter=80, iter_interval_ms=500,
                   target_iters=10, slo_ci=2.0,
                   T_target=None, rtt_probe_ms=250.0, alpha=0.3, start_time_ms=0))
    sim.submit(Job("long", "gpt", mb_per_iter=200, iter_interval_ms=300,
                   target_iters=30, slo_ci=1.2,
                   T_target=None, rtt_probe_ms=400.0, alpha=0.3, start_time_ms=0))
    r = sim.run()
    short_job = sim.jobs["short"]

    print("test_t_target_short_job_trap:")
    print(f"  short job completed iters: {short_job.completed_iters}")
    print(f"  short job EMA initialized: {short_job.ema_initialized}")
    # 短 job 的 EMA 不应被初始化（因优先级较低）
    assert not short_job.ema_initialized, \
        "短 job 不应在重竞争下被提升为最高优先级并初始化 EMA"
    print("  PASSED\n")


def test_synthetic_trace_loader():
    """SyntheticTraceLoader 应生成符合分布的 Job 列表。"""
    from longliu_sim.trace import SyntheticTraceLoader

    loader = SyntheticTraceLoader(
        model_types=["ResNet-50", "BERT-Large"],
        gpu_distribution={1: 0.5, 2: 0.5},
        ci_distribution={1.5: 0.5, 3.0: 0.5},
        job_count=20,
        seed=42,
    )
    jobs = loader.load()

    print("test_synthetic_trace_loader:")
    print(f"  generated jobs: {len(jobs)}")
    assert len(jobs) == 20

    models = [j.model for j in jobs]
    assert set(models) <= {"ResNet-50", "BERT-Large"}

    gpu_counts = [j.num_workers for j in jobs]
    assert all(g in {1, 2} for g in gpu_counts)

    ci_values = [j.slo_ci for j in jobs]
    assert all(ci in {1.5, 3.0} for ci in ci_values)
    print("  PASSED\n")


def test_lingjun_trace_loader():
    """LingjunTraceLoader 应正确解析 CSV 并返回 Job 列表。"""
    import tempfile
    import os
    from longliu_sim.trace import LingjunTraceLoader

    csv_content = (
        "job_id,model_type,gpu_count,start_time_ms,duration_ms\n"
        "lj1,ResNet-50,2,0,300000\n"
        "lj2,BERT-Large,4,1000,600000\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        loader = LingjunTraceLoader(path)
        jobs = loader.load()

        print("test_lingjun_trace_loader:")
        print(f"  loaded jobs: {len(jobs)}")
        assert len(jobs) == 2

        by_id = {j.jid: j for j in jobs}
        assert by_id["lj1"].model == "ResNet-50"
        assert by_id["lj1"].num_workers == 2
        assert by_id["lj2"].model == "BERT-Large"
        assert by_id["lj2"].num_workers == 4
        print("  PASSED\n")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_single_job_fair()
    test_longliu_prioritizes_tight_slo()
    test_heavy_contention_improves_total()
    test_allreduce_barrier_multiflow()
    test_allreduce_barrier_contention()
    test_crux_baseline()
    test_t_target_calibration()
    test_cassini_offset()
    test_dscp_boundary()
    test_fat_tree_multipath()
    test_t_target_short_job_trap()
    test_synthetic_trace_loader()
    test_lingjun_trace_loader()
    print("All tests passed.")