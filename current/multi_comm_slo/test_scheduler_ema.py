#!/usr/bin/env python3
"""单元测试：验证 SLOScheduler v1(π) 的 EMA 滑动基线、论文公式与窗口通信紧急信号"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from slo_scheduler import SLOScheduler

def test_ema_sliding_baseline():
    """测试 EMA 滑动基线：solo phase 持续更新，contested phase 不更新"""
    print("=" * 60)
    print("Test 1: EMA sliding baseline (solo vs contested)")
    print("=" * 60)
    sched = SLOScheduler(slo_threshold=1.2, ema_alpha=0.3, solo_warmup_windows=5)

    # Solo phase: 100ms wall / 10ms comm per window for 5 windows
    print("\n--- Solo phase (5 windows @ wall=100ms, comm=10ms) ---")
    for i in range(5):
        sched.update(0.100, data_size=2048*1024*1024, window_comm_time=0.010)

    print(f"\nAfter solo: T_target={sched.target_comm_time_s*1000:.2f}ms, "
          f"π={sched.last_pi:+.3f}, P{sched.current_priority}, "
          f"comm_baseline={sched.baseline_comm_time_s*1000:.2f}ms")

    # Contested phase: 200ms wall / 200ms comm per window (priority should rise)
    print("\n--- Contested phase (5 windows @ wall=200ms, comm=200ms) ---")
    for i in range(5):
        sched.update(0.200, data_size=2048*1024*1024, window_comm_time=0.200)
        print(f"  -> After window {sched.window_count}: "
              f"T_target={sched.target_comm_time_s*1000:.2f}ms, "
              f"π={sched.last_pi:+.3f}, comm_ratio={sched.last_comm_ratio:.2f}, "
              f"P{sched.current_priority}")

    # Verify T_target didn't inflate during contested phase
    assert sched.target_comm_time_s * 1000 < 120, \
        f"T_target should stay near 100ms during contention, got {sched.target_comm_time_s*1000:.2f}ms"
    print("\n✅ T_target did NOT inflate during contested phase")


def test_paper_formula():
    """测试论文公式：π = A / (c_i × T_target × k) - 1"""
    print("\n" + "=" * 60)
    print("Test 2: Paper formula π = A / (c_i × T_target × k) - 1")
    print("=" * 60)
    sched = SLOScheduler(slo_threshold=1.2, ema_alpha=1.0, solo_warmup_windows=10)

    # Force T_target = 100ms via first window
    sched.update(0.100, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"\nT_target = {sched.target_comm_time_s*1000:.2f}ms (after 1st window)")

    # 9 more solo windows at exactly 100ms
    for i in range(9):
        sched.update(0.100, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"After 10 solo windows @ 100ms: π={sched.last_pi:+.4f} (should be ~0)")

    # Now 1 contested window at 200ms (T_target won't update since not P6 and past warmup)
    # c_i=1.2, T_target=100ms, k=11, A = 10*0.1 + 0.2 = 1.2s
    # expected = 1.2 * 0.1 * 11 = 1.32s
    # π = 1.2/1.32 - 1 = -0.091
    sched.update(0.200, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"After 1 contested window @ 200ms: π={sched.last_pi:+.4f}")
    print(f"  (Expected: A=1.2s, c_i*T_target*k=1.32s, π≈-0.091)")

    # More contested windows
    for i in range(5):
        sched.update(0.200, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"After 6 contested windows @ 200ms: π={sched.last_pi:+.4f}, P{sched.current_priority}")


def test_priority_transitions():
    """测试优先级映射：P1, P2, P4, P6"""
    print("\n" + "=" * 60)
    print("Test 3: Priority transitions (P1→P2→P4→P6)")
    print("=" * 60)
    sched = SLOScheduler(slo_threshold=1.2, ema_alpha=1.0, solo_warmup_windows=10)

    # Solo phase to establish baseline
    for i in range(10):
        sched.update(0.100, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"\nBaseline: T_target=100ms, π={sched.last_pi:+.4f}, P{sched.current_priority}")

    # Heavily contested - should rise to P6
    print("\n--- 20 windows @ wall=300ms (heavy contention) ---")
    for i in range(20):
        sched.update(0.300, data_size=2048*1024*1024, window_comm_time=0.300)
    print(f"Result: π={sched.last_pi:+.4f}, P{sched.current_priority}")
    assert sched.current_priority == 6, f"Expected P6, got P{sched.current_priority}"
    pi_at_p6 = sched.last_pi

    # Back to fast - π should decrease (recovery), but cumulative deficit
    # means priority may not drop immediately (paper formula is cumulative)
    print("\n--- 30 windows @ 80ms (fast, recovering) ---")
    for i in range(30):
        sched.update(0.080, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"Result: π={sched.last_pi:+.4f}, P{sched.current_priority}")
    # Verify π is recovering (decreasing toward 0 and below)
    assert sched.last_pi < pi_at_p6, \
        f"π should decrease during recovery: was {pi_at_p6}, now {sched.last_pi}"
    print(f"  π recovery: {pi_at_p6:+.4f} → {sched.last_pi:+.4f}")

    # Eventually after enough fast windows, priority should drop
    print("\n--- 200 more windows @ 80ms (full recovery) ---")
    for i in range(200):
        sched.update(0.080, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"Result: π={sched.last_pi:+.4f}, P{sched.current_priority}")
    assert sched.current_priority in [1, 2, 4], \
        f"Expected P1/P2/P4 after long recovery, got P{sched.current_priority}"

    print("\n✅ Priority transitions work correctly (cumulative deficit behavior verified)")


def test_comm_ratio_emergency():
    """测试窗口通信紧急信号：comm_ratio > 1.3 时即使 π 领先也强制 P6"""
    print("\n" + "=" * 60)
    print("Test 4: Window comm_ratio emergency signal (计算主导负载)")
    print("=" * 60)
    sched = SLOScheduler(slo_threshold=1.2, ema_alpha=0.3, solo_warmup_windows=5)

    # 模拟计算主导负载：wall=1.0s，其中 comm 仅 5ms（真实模型 tiny：计算占 ~99%）
    # 窗口 1 故意加入 NCCL 初始化型污染（comm=8s，iter0 现象）——必须被跳过
    print("\n--- Window 1 (含 NCCL 初始化污染 comm=8s) ---")
    sched.update(9.0, data_size=2048*1024*1024, window_comm_time=8.0)
    print(f"  -> window_count={sched.window_count}, baseline 未被初始化: "
          f"{sched.baseline_comm_time_s}")

    # 后续 warmup 窗口: comm=5ms → min 聚合 → baseline=5ms
    print("\n--- Solo warmup (windows 2-6 @ wall=1.0s, comm=5ms) ---")
    for i in range(5):
        sched.update(1.0, data_size=2048*1024*1024, window_comm_time=0.005)
    print(f"\nAfter solo: baseline={sched.baseline_comm_time_s*1000:.2f}ms, "
          f"π={sched.last_pi:+.4f}, P{sched.current_priority}")
    assert abs(sched.baseline_comm_time_s - 0.005) < 1e-9, \
        f"baseline 应等于 solo 稳态 5ms（跳过污染窗口+min），实际 {sched.baseline_comm_time_s*1000}ms"

    # 竞争: comm 5ms → 6.8ms（1.36×，匹配测试床实测 1.3~1.4× 膨胀水平）
    # wall 1.0s → 1.008s（+0.8%），纯 π 视角几乎不变 → 必须靠 comm_ratio 触发
    print("\n--- Contested phase (5 windows @ wall=1.008s, comm=6.8ms) ---")
    triggered_emergency = False
    for i in range(5):
        sched.update(1.008, data_size=2048*1024*1024, window_comm_time=0.0068)
        print(f"  -> After window {sched.window_count}: "
              f"comm_ratio={sched.last_comm_ratio:.2f}, "
              f"π={sched.last_pi:+.4f}, P{sched.current_priority}")
        if sched.current_priority == 6:
            triggered_emergency = True

    # 关键断言：1.36× 膨胀（测试床实测水平）必须触发 P6（旧 1.5 阈值不可达）
    assert triggered_emergency, \
        "comm_ratio emergency signal FAILED: P6 not triggered although comm inflated 1.36×"
    print("\n✅ comm_ratio emergency signal triggers P6 on compute-dominated load")

    # 恢复：comm 回到 5ms → ratio 回落 → 优先级应下降
    print("\n--- Recovery (5 windows @ wall=1.0s, comm=5ms) ---")
    for i in range(5):
        sched.update(1.0, data_size=2048*1024*1024, window_comm_time=0.005)
        print(f"  -> After window {sched.window_count}: "
              f"comm_ratio={sched.last_comm_ratio:.2f}, P{sched.current_priority}")
    assert sched.current_priority != 6, "Priority should drop after comm recovers"
    print("\n✅ Priority drops after comm recovers")


if __name__ == '__main__':
    test_ema_sliding_baseline()
    test_paper_formula()
    test_priority_transitions()
    test_comm_ratio_emergency()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
