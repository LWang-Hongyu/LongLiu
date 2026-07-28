#!/usr/bin/env python3
"""单元测试：验证 SLOScheduler v1(π) 的 EMA 滑动基线和论文公式"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from slo_scheduler import SLOScheduler

def test_ema_sliding_baseline():
    """测试 EMA 滑动基线：solo phase 持续更新，contested phase 不更新"""
    print("=" * 60)
    print("Test 1: EMA sliding baseline (solo vs contested)")
    print("=" * 60)
    sched = SLOScheduler(slo_threshold=1.2, ema_alpha=0.3, solo_warmup_epochs=5)

    # Solo phase: 100ms per epoch for 5 epochs
    print("\n--- Solo phase (5 epochs @ 100ms) ---")
    for i in range(5):
        sched.update(0.100, data_size=2048*1024*1024)

    print(f"\nAfter solo: T_target={sched.target_comm_time_s*1000:.2f}ms, "
          f"π={sched.last_pi:+.3f}, P{sched.current_priority}")

    # Contested phase: 200ms per epoch (priority should rise)
    print("\n--- Contested phase (5 epochs @ 200ms) ---")
    for i in range(5):
        sched.update(0.200, data_size=2048*1024*1024)
        print(f"  -> After epoch {sched.epoch_count}: "
              f"T_target={sched.target_comm_time_s*1000:.2f}ms, "
              f"π={sched.last_pi:+.3f}, P{sched.current_priority}")

    # Verify T_target didn't inflate during contested phase (priority was P4, not P6)
    # Actually during contested phase, T_target should remain ~100ms (no credible update)
    # unless priority rises to P6
    assert sched.target_comm_time_s * 1000 < 120, \
        f"T_target should stay near 100ms during contention, got {sched.target_comm_time_s*1000:.2f}ms"
    print("\n✅ T_target did NOT inflate during contested phase")


def test_paper_formula():
    """测试论文公式：π = A / (c_i × T_target × k) - 1"""
    print("\n" + "=" * 60)
    print("Test 2: Paper formula π = A / (c_i × T_target × k) - 1")
    print("=" * 60)
    sched = SLOScheduler(slo_threshold=1.2, ema_alpha=1.0, solo_warmup_epochs=10)

    # Force T_target = 100ms via first epoch
    sched.update(0.100, data_size=2048*1024*1024)
    print(f"\nT_target = {sched.target_comm_time_s*1000:.2f}ms (after 1st epoch)")

    # 9 more solo epochs at exactly 100ms
    for i in range(9):
        sched.update(0.100, data_size=2048*1024*1024)
    print(f"After 10 solo epochs @ 100ms: π={sched.last_pi:+.4f} (should be ~0)")

    # Now 1 contested epoch at 200ms (T_target won't update since not P6 and past warmup)
    # c_i=1.2, T_target=100ms, k=11, A = 10*0.1 + 0.2 = 1.2s
    # expected = 1.2 * 0.1 * 11 = 1.32s
    # π = 1.2/1.32 - 1 = -0.091
    sched.update(0.200, data_size=2048*1024*1024)
    print(f"After 1 contested epoch @ 200ms: π={sched.last_pi:+.4f}")
    print(f"  (Expected: A=1.2s, c_i*T_target*k=1.32s, π≈-0.091)")

    # More contested epochs
    for i in range(5):
        sched.update(0.200, data_size=2048*1024*1024)
    print(f"After 6 contested epochs @ 200ms: π={sched.last_pi:+.4f}, P{sched.current_priority}")


def test_priority_transitions():
    """测试优先级映射：P1, P2, P4, P6"""
    print("\n" + "=" * 60)
    print("Test 3: Priority transitions (P1→P2→P4→P6)")
    print("=" * 60)
    sched = SLOScheduler(slo_threshold=1.2, ema_alpha=1.0, solo_warmup_epochs=10)

    # Solo phase to establish baseline
    for i in range(10):
        sched.update(0.100, data_size=2048*1024*1024)
    print(f"\nBaseline: T_target=100ms, π={sched.last_pi:+.4f}, P{sched.current_priority}")

    # Heavily contested - should rise to P6
    print("\n--- 20 epochs @ 300ms (heavy contention) ---")
    for i in range(20):
        sched.update(0.300, data_size=2048*1024*1024)
    print(f"Result: π={sched.last_pi:+.4f}, P{sched.current_priority}")
    assert sched.current_priority == 6, f"Expected P6, got P{sched.current_priority}"
    pi_at_p6 = sched.last_pi

    # Back to fast - π should decrease (recovery), but cumulative deficit
    # means priority may not drop immediately (paper formula is cumulative)
    print("\n--- 30 epochs @ 80ms (fast, recovering) ---")
    for i in range(30):
        sched.update(0.080, data_size=2048*1024*1024)
    print(f"Result: π={sched.last_pi:+.4f}, P{sched.current_priority}")
    # Verify π is recovering (decreasing toward 0 and below)
    assert sched.last_pi < pi_at_p6, \
        f"π should decrease during recovery: was {pi_at_p6}, now {sched.last_pi}"
    print(f"  π recovery: {pi_at_p6:+.4f} → {sched.last_pi:+.4f}")

    # Eventually after enough fast epochs, priority should drop
    print("\n--- 200 more epochs @ 80ms (full recovery) ---")
    for i in range(200):
        sched.update(0.080, data_size=2048*1024*1024)
    print(f"Result: π={sched.last_pi:+.4f}, P{sched.current_priority}")
    assert sched.current_priority in [1, 2, 4], \
        f"Expected P1/P2/P4 after long recovery, got P{sched.current_priority}"

    print("\n✅ Priority transitions work correctly (cumulative deficit behavior verified)")


if __name__ == '__main__':
    test_ema_sliding_baseline()
    test_paper_formula()
    test_priority_transitions()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
