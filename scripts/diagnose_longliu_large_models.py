#!/usr/bin/env python3
"""
诊断 LongLiu 大模型违约的原因。

分析：
1. 大模型的 SAS 分布
2. 大模型的 avg_iter_ms vs target_iter_ms
3. 大模型的通信量 vs 带宽需求
4. 与 CRUX 的对比
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_per_job_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_large_models(data: dict):
    """分析大模型（ci=1.5）的违约情况。"""
    print("=" * 80)
    print("LongLiu 大模型（ci=1.5）违约诊断")
    print("=" * 80)

    ll_jobs = data['LongLiu']
    crux_jobs = data['CRUX']

    # 筛选大模型
    ll_large = [j for j in ll_jobs if j['ci'] == 1.5]
    crux_large = [j for j in crux_jobs if j['ci'] == 1.5]

    print(f"\n大模型总数: {len(ll_large)} 个")
    print()

    # 按 SAS 排序
    ll_large_sorted = sorted(ll_large, key=lambda x: x['sas'], reverse=True)

    print("LongLiu 大模型 SAS 分布（从高到低）:")
    print(f"{'JID':<6} {'Model':<16} {'DP':<4} {'iter_solo':<10} {'avg_iter':<12} {'target':<10} {'SAS':<8} {'Meets?'}")
    print("-" * 80)
    for j in ll_large_sorted:
        meets = '✓' if j['meets_slo'] else '✗'
        print(f"{j['jid']:<6} {j['model']:<16} {j['dp']:<4} "
              f"{j['iter_solo_ms']:<10.1f} {j['avg_iter_ms']:<12.1f} "
              f"{j['target_iter_ms']:<10.1f} {j['sas']:<8.3f} {meets}")

    print()
    print("CRUX 大模型 SAS 分布:")
    print(f"{'JID':<6} {'Model':<16} {'DP':<4} {'iter_solo':<10} {'avg_iter':<12} {'target':<10} {'SAS':<8} {'Meets?'}")
    print("-" * 80)
    for j in sorted(crux_large, key=lambda x: x['sas'], reverse=True):
        meets = '✓' if j['meets_slo'] else '✗'
        print(f"{j['jid']:<6} {j['model']:<16} {j['dp']:<4} "
              f"{j['iter_solo_ms']:<10.1f} {j['avg_iter_ms']:<12.1f} "
              f"{j['target_iter_ms']:<10.1f} {j['sas']:<8.3f} {meets}")

    # 统计分析
    print()
    print("=" * 80)
    print("统计分析")
    print("=" * 80)

    ll_sas = [j['sas'] for j in ll_large]
    crux_sas = [j['sas'] for j in crux_large]

    print(f"\nLongLiu 大模型 SAS:")
    print(f"  Mean: {sum(ll_sas)/len(ll_sas):.3f}")
    print(f"  Median: {sorted(ll_sas)[len(ll_sas)//2]:.3f}")
    print(f"  Min: {min(ll_sas):.3f}")
    print(f"  Max: {max(ll_sas):.3f}")
    print(f"  满足 SLO: {sum(1 for j in ll_large if j['meets_slo'])}/{len(ll_large)}")

    print(f"\nCRUX 大模型 SAS:")
    print(f"  Mean: {sum(crux_sas)/len(crux_sas):.3f}")
    print(f"  Median: {sorted(crux_sas)[len(crux_sas)//2]:.3f}")
    print(f"  Min: {min(crux_sas):.3f}")
    print(f"  Max: {max(crux_sas):.3f}")
    print(f"  满足 SLO: {sum(1 for j in crux_large if j['meets_slo'])}/{len(crux_large)}")

    # 违约程度分析
    print()
    print("=" * 80)
    print("违约程度分析")
    print("=" * 80)

    # LongLiu 严重违约的大模型（SAS < 0.15）
    ll_severe = [j for j in ll_large if j['sas'] < 0.15]
    crux_severe = [j for j in crux_large if j['sas'] < 0.15]

    print(f"\nLongLiu 严重违约（SAS < 0.15）: {len(ll_severe)} 个")
    for j in ll_severe:
        violation_ratio = j['avg_iter_ms'] / j['target_iter_ms']
        print(f"  {j['jid']}: {j['model']}, avg_iter={j['avg_iter_ms']:.1f}ms, "
              f"target={j['target_iter_ms']:.1f}ms, 违约倍数={violation_ratio:.1f}x")

    print(f"\nCRUX 严重违约（SAS < 0.15）: {len(crux_severe)} 个")
    for j in crux_severe:
        violation_ratio = j['avg_iter_ms'] / j['target_iter_ms']
        print(f"  {j['jid']}: {j['model']}, avg_iter={j['avg_iter_ms']:.1f}ms, "
              f"target={j['target_iter_ms']:.1f}ms, 违约倍数={violation_ratio:.1f}x")

    # 通信量分析
    print()
    print("=" * 80)
    print("通信量分析")
    print("=" * 80)

    print("\n大模型通信量（comm_solo_ms）:")
    for j in ll_large_sorted[:5]:  # 前 5 个
        print(f"  {j['jid']}: {j['model']}, comm_solo={j['comm_solo_ms']:.1f}ms, "
              f"dp={j['dp']}, iter_solo={j['iter_solo_ms']:.1f}ms")

    # 关键问题诊断
    print()
    print("=" * 80)
    print("关键问题诊断")
    print("=" * 80)

    # 检查是否有大模型超额满足（SAS > 1.0）
    ll_overachieve = [j for j in ll_large if j['sas'] > 1.0]
    print(f"\nLongLiu 大模型超额满足（SAS > 1.0）: {len(ll_overachieve)} 个")
    for j in ll_overachieve:
        print(f"  {j['jid']}: {j['model']}, SAS={j['sas']:.3f}, "
              f"avg_iter={j['avg_iter_ms']:.1f}ms < target={j['target_iter_ms']:.1f}ms")

    # 检查 DP 分布
    print()
    print("大模型 DP 分布:")
    dp_counts = {}
    for j in ll_large:
        dp_counts[j['dp']] = dp_counts.get(j['dp'], 0) + 1
    for dp, count in sorted(dp_counts.items()):
        dp_jobs = [j for j in ll_large if j['dp'] == dp]
        dp_sas = [j['sas'] for j in dp_jobs]
        print(f"  DP={dp}: {count} 个, Mean SAS={sum(dp_sas)/len(dp_sas):.3f}")

    # 结论
    print()
    print("=" * 80)
    print("结论")
    print("=" * 80)

    if len(ll_severe) > len(crux_severe):
        print(f"\n⚠️  LongLiu 有更多严重违约的大模型（{len(ll_severe)} vs {len(crux_severe)}）")
        print("   可能原因：")
        print("   1. Beta 权重抑制了大模型优先级（demand 越大，pi_eff 越小）")
        print("   2. 动态 T_target 校准被大模型主导，导致中模型饿死")
        print("   3. EMA 冻结机制（ui < 1）阻止了大模型优先级上升")
    else:
        print(f"\n✓ LongLiu 严重违约数量与 CRUX 相近或更少")

    if len(ll_overachieve) > 0:
        print(f"\n✓ LongLiu 有 {len(ll_overachieve)} 个大模型超额满足")
        print("   说明 LongLiu 的动态调整在某些情况下有效")


def main():
    base_dir = Path(__file__).parent.parent
    per_job_path = base_dir / 'outputs' / 'table3_perjob_test' / 'per_job.json'

    data = load_per_job_data(str(per_job_path))
    analyze_large_models(data)


if __name__ == "__main__":
    main()