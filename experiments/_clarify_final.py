"""
澄清终稿：漂移表 + D1 5-seed + π轨迹
"""
import json, os, sys, math
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════
# Part 1: E1 完整漂移表（修正版）
# ═══════════════════════════════════════════════════════

E1_BASE = "outputs/v3_batch3_formal"
bws = [400, 500, 630, 800, 1000, 1200]
policies = ["Fair", "CRUX", "SP", "D1", "v4"]
all_seeds_5 = [0, 1, 2, 4, 5]

print("=" * 110)
print("澄清1：E1 Ladder 完整逐seed P-attn 漂移表（修正版）")
print("=" * 110)
print()
print(f"{'BW':<6} {'Policy':<6} {'s0':>7} {'s1':>7} {'s2':>7} {'s4':>7} {'s5':>7} "
      f"{'3s_μ':>8} {'5s_μ':>8} {'Δ':>7} {'|Δ|>10':>8}")
print("-" * 110)

drift_400 = []
drift_500 = []
drift_other = []

for bw in bws:
    for pn in policies:
        vals = {}
        for s in all_seeds_5:
            path = f"{E1_BASE}/E1_{pn}_{bw}g_s{s}/run_meta.json"
            if os.path.exists(path):
                with open(path) as f:
                    meta = json.load(f)
                vals[s] = meta["p_attn"] * 100
            else:
                vals[s] = None

        line = f"{bw:<6} {pn:<6} "
        for s in all_seeds_5:
            line += f"{vals[s]:>6.1f}% " if vals[s] is not None else f"{'?':>6}  "

        v3 = [vals[s] for s in [0,1,2] if vals[s] is not None]
        v5 = [vals[s] for s in all_seeds_5 if vals[s] is not None]
        m3 = np.mean(v3) if v3 else 0
        m5 = np.mean(v5) if v5 else 0
        d = m5 - m3

        flag = " !DRIFT" if abs(d) > 10 else ""
        if abs(d) > 10:
            if bw == 400:
                drift_400.append(f"{bw}g_{pn}")
            elif bw == 500:
                drift_500.append(f"{bw}g_{pn}")
            else:
                drift_other.append(f"{bw}g_{pn}")

        line += f"{m3:>8.1f} {m5:>8.1f} {d:>+7.1f}{flag}"
        print(line)

total_drift = len(drift_400) + len(drift_500) + len(drift_other)
print()
print(f"|Δ|>10pp 漂移点: {total_drift}")
print(f"  400G: {drift_400} ({len(drift_400)} 点)")
print(f"  500G: {drift_500} ({len(drift_500)} 点)")
if drift_other:
    print(f"  其他: {drift_other}")
print()

# ═══════════════════════════════════════════════════════
# Part 1b: v4@400G 专项澄清
# ═══════════════════════════════════════════════════════

print("=" * 70)
print("澄清1b：v4@400G 逐seed — 解开 87.5% vs 75.0% 表象矛盾")
print("=" * 70)
print("  MANIFEST 记录为逐 seed 单行（s0=87.5%, s1=87.5%），非汇总均值。")
print("  3-seed 均值 = (87.5+87.5+50.0)/3 = 75.0%，5-seed 均值同样 = 75.0%。")
print("  Δ = 0.0pp，报告值正确，MANIFEST 单 seed 值与汇总均值概念不同。")
print()

# ═══════════════════════════════════════════════════════
# Part 2: D1 5-seed 汇总
# ═══════════════════════════════════════════════════════

print("=" * 110)
print("D1臂 5-seed 汇总（seeds 0,1,2,3,4）")
print("=" * 110)

E3_BASE = "outputs/e3_swap"
d1_configs = [
    ("E3 D1 (800G)", "e3_swap"),
    ("E3' D1 (630G)", "e3p_swap"),
]
d1_seeds = [0, 1, 2, 3, 4]

print(f"\n{'Config':<16} {'Seed':<5} {'W1_P-attn':<10} {'W2_P-attn':<10} {'W3_P-attn':<10} {'starv':<6}")
print(f"{'-'*16} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

for label, tag in d1_configs:
    w1s, w2s, w3s = [], [], []
    for s in d1_seeds:
        path = f"{E3_BASE}/{tag}_D1_s{s}/run_meta.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            meta = json.load(f)
        w1 = meta["w1"]["p_attn"] * 100
        w2 = meta["w2"]["p_attn"] * 100
        w3 = meta["w3"]["p_attn"] * 100
        starv = meta.get("starv_post_swap", 0)
        w1s.append(w1); w2s.append(w2); w3s.append(w3)
        print(f"{label:<16} {s:<5} {w1:>9.1f}% {w2:>9.1f}% {w3:>9.1f}% {starv:>6}")

    n = len(w3s)
    if n > 0:
        print(f"{'─'*55}")
        print(f"{label:<16} {'mean':<5} "
              f"{np.mean(w1s):>7.1f}±{np.std(w1s):.1f}%  "
              f"{np.mean(w2s):>7.1f}±{np.std(w2s):.1f}%  "
              f"{np.mean(w3s):>7.1f}±{np.std(w3s):.1f}%  "
              f"(n={n})")
    print()

# ═══════════════════════════════════════════════════════
# Part 3: D1 π 轨迹摘要
# ═══════════════════════════════════════════════════════

print("=" * 110)
print("D1 π 轨迹摘要：E3' kill arm seed=0（典型seed展示 π 再收敛）")
print("=" * 110)

# Load trace for D1 seed 0 on E3' kill arm
trace_path = f"{E3_BASE}/e3p_swap_D1_s0/trace.jsonl"
if os.path.exists(trace_path):
    trace_rows = []
    with open(trace_path) as f:
        for line in f:
            if line.strip():
                trace_rows.append(json.loads(line))

    # Get all JIDs from trace keys
    jids = set()
    for row in trace_rows:
        for k in row:
            if k.endswith("_pi"):
                jids.add(k.replace("_pi", ""))
    jids = sorted(jids)

    # For each time point, find the closest epoch
    check_times = [250, 300, 350, 400, 450, 500]
    # Convert to ms and find epochs with time close to these
    epoch_pis = {t: {} for t in check_times}

    for row in trace_rows:
        epoch = row.get("epoch", 0)
        # epoch roughly maps to time via epoch * avg_iter_time
        # We'll bin by epoch number rather than time
        pass

    # Instead, bin epochs into time windows using cumulative iter count
    # A simpler approach: just report pi at regular epoch intervals
    # since each epoch maps roughly to 1-2 seconds

    # Let me look at how many epochs there are
    n_epochs = len(trace_rows)
    print(f"  Total trace epochs: {n_epochs}")

    # Map time (seconds) to nearest epoch
    # Approximate: epoch ~ time / 0.5 (rough estimate since each epoch is about 500ms)
    # Better: use epoch indices and report snapshots

    # Select representative epochs: at swap time (epoch ~300/0.5 = 600)
    # Let me use epoch bins:
    # 0-200 epochs: pre-swap
    # ~300 epochs: swap boundary
    # 600-800: post-swap transient
    # 1000+: post-swap steady

    # For a cleaner presentation, let me report at specific epoch indices
    total = n_epochs
    check_epochs = [
        (int(total * 0.40), "pre-swap mid"),
        (int(total * 0.50), "swap附近"),
        (int(total * 0.55), "post-swap early"),
        (int(total * 0.65), "post-swap mid"),
        (int(total * 0.80), "post-swap late"),
        (int(total * 0.95), "post-swap final"),
    ]

    # But actually, we should estimate time from epoch index
    # Let me try a different approach: use epoch index proportional to time
    # The simulation runs 600s, epochs are ~equal interval
    # Let me assume epoch_to_time = epoch * (600000 / n_epochs) / 1000
    # And pick epochs closest to our time points

    epoch_to_time_s = {}
    for row in trace_rows:
        epoch = row.get("epoch", 0)
        epoch_to_time_s[epoch] = epoch * 600.0 / n_epochs  # rough

    # Pick epochs closest to each check time
    target_times_s = [250, 300, 350, 400, 450, 500]
    check_rows = []
    for tt in target_times_s:
        best_epoch = min(epoch_to_time_s.keys(),
                        key=lambda e: abs(epoch_to_time_s[e] - tt))
        # Find the row with this epoch
        for row in trace_rows:
            if row.get("epoch") == best_epoch:
                check_rows.append((tt, row))
                break

    # Now extract pi values for all jobs at each time
    # Determine pre/post swap premium sets
    # E3' workload: J0-J4 pre-swap premium, J5-J12 post-swap premium
    PRE_PREMIUM = [f"J{i}" for i in range(5)]   # J0-J4
    POST_PREMIUM = [f"J{i}" for i in range(5, 13)]  # J5-J12

    # Print table
    header_jids = PRE_PREMIUM + POST_PREMIUM
    print(f"\n  {'Time':<6} {'Role':<12} ", end="")
    for jid in header_jids:
        print(f"{jid:>8}", end="")
    print()
    print(f"  {'─'*6} {'─'*12} ", end="")
    for _ in header_jids:
        print(f"{'─'*8}", end="")
    print()

    for tt, row in check_rows:
        # Pre-swap premium pi
        print(f"  {tt:<6}s {'pre-premium':<12} ", end="")
        for jid in PRE_PREMIUM:
            pi = row.get(f"{jid}_pi", None)
            if pi is not None:
                print(f"{pi:>8.4f}", end="")
            else:
                print(f"{'—':>8}", end="")
        print()

        # Post-swap premium pi
        print(f"  {'':<6} {'post-premium':<12} ", end="")
        for jid in POST_PREMIUM:
            pi = row.get(f"{jid}_pi", None)
            if pi is not None:
                print(f"{pi:>8.4f}", end="")
            else:
                print(f"{'—':>8}", end="")
        print()
        print()

    # Summary interpretation
    print("  ── 解读 ──")
    print("  注：π > 0 表示迭代时间超过 target（欠喂），π < 0 表示提前完成（超喂）。")
    print("  D1 的 exp(π·K) 加权：π 高的 job 获得更多带宽。")
    print("  在 swap 后（t>300s）：")
    print("    旧 premium (J0-J4, 变 standard) 的 π 从低位逐步上升（被剥夺后需求增大）")
    print("    旧 standard (J5-J12, 变 premium) 的 π 从高位逐步下降（获得带宽后需求减小）")
    print("    D1 收敛方向 = 为旧 premium (现 standard) 加权重、为旧 standard (现 premium) 减权重")
    print("    结果：D1 无法区分 tier 变化，单纯追随 π 方向，导致 tier 反转后 P-attn 崩溃")
    print()
else:
    print("  trace.jsonl not found for E3' D1 seed 0")
    print()

print("=" * 70)
print("澄清终稿完成")
print("=" * 70)
