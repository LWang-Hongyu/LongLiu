#!/usr/bin/env python3
"""
诊断 LongLiu 的 pi 计算过程。

分析每个 job 的：
- avg_iter_ms
- default_T_target
- pi = avg_iter_ms / default_T_target - 1
- DSCP 映射
- 是否能更新 EMA
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_per_job_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_pi_calculation(data: dict):
    """分析每个 job 的 pi 计算过程。"""
    print("=" * 80)
    print("LongLiu pi 计算诊断")
    print("=" * 80)

    ll_jobs = data['LongLiu']

    # DSCP 映射（论文 Table 5）
    DSCP_MAP = [
        (0.2, 46, "EF (P6) 最高"),
        (-0.1, 34, "AF41 (P4)"),
        (-0.5, 18, "AF21 (P2)"),
        (-1.0, 0, "BE (P0) 最低"),
    ]

    def get_dscp(pi):
        for threshold, dscp, name in DSCP_MAP:
            if pi > threshold:
                return dscp, name
        return 0, "BE (P0) 最低"

    print(f"\n{'JID':<6} {'Model':<16} {'DP':<4} {'ci':<4} {'avg_iter':<10} "
          f"{'T_target':<10} {'pi':<8} {'DSCP':<16} {'能更新EMA?'}")
    print("-" * 100)

    for j in sorted(ll_jobs, key=lambda x: x['sas'], reverse=True):
        avg_iter = j['avg_iter_ms']
        T_target = j['target_iter_ms']  # 这是 SLO 允许时间，不是 default_T_target
        default_T_target = j['iter_solo_ms'] * j['ci']  # default_T_target = ci × iter_solo_ms

        # pi = avg_iter_ms / default_T_target - 1
        pi = avg_iter / default_T_target - 1.0
        dscp, dscp_name = get_dscp(pi)
        can_update_ema = "✓ 能" if dscp == 46 else "✗ 不能"

        print(f"{j['jid']:<6} {j['model']:<16} {j['dp']:<4} {j['ci']:<4.1f} "
              f"{avg_iter:<10.1f} {default_T_target:<10.1f} {pi:<8.3f} "
              f"{dscp_name:<16} {can_update_ema}")

    print()
    print("=" * 80)
    print("关键发现")
    print("=" * 80)

    # 统计能更新 EMA 的 job
    can_update = []
    cannot_update = []
    for j in ll_jobs:
        avg_iter = j['avg_iter_ms']
        default_T_target = j['iter_solo_ms'] * j['ci']
        pi = avg_iter / default_T_target - 1.0
        if pi > 0.2:
            can_update.append(j)
        else:
            cannot_update.append(j)

    print(f"\n能更新 EMA（pi > 0.2，DSCP 46）: {len(can_update)} 个")
    for j in can_update:
        print(f"  {j['jid']}: {j['model']}, SAS={j['sas']:.3f}")

    print(f"\n不能更新 EMA（pi <= 0.2）: {len(cannot_update)} 个")
    print("  其中严重违约（SAS < 0.15）:")
    severe = [j for j in cannot_update if j['sas'] < 0.15]
    for j in severe:
        avg_iter = j['avg_iter_ms']
        default_T_target = j['iter_solo_ms'] * j['ci']
        pi = avg_iter / default_T_target - 1.0
        print(f"    {j['jid']}: {j['model']}, pi={pi:.3f}, SAS={j['sas']:.3f}")

    print()
    print("=" * 80)
    print("结论")
    print("=" * 80)
    print("\n1. EMA 更新机制要求 pi > 0.2（DSCP 46）才能更新")
    print("2. 大模型的 default_T_target 很大（ci=1.5 × iter_solo_ms）")
    print("3. 当 avg_iter_ms 还没膨胀时，pi 可能是负值，拿不到 DSCP 46")
    print("4. 拿不到 DSCP 46 → 无法更新 EMA → 恶性循环")
    print("\n建议：放宽 EMA 更新条件，允许 DSCP 34（pi > -0.1）也能更新")


def main():
    base_dir = Path(__file__).parent.parent
    per_job_path = base_dir / 'outputs' / 'table3_perjob_test' / 'per_job.json'

    data = load_per_job_data(str(per_job_path))
    analyze_pi_calculation(data)


if __name__ == "__main__":
    main()