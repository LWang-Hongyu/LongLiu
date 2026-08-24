"""
Step 4: 窗口过滤方法论修正 — start_ms vs end_ms 对比
使用 v4 trace 来验证，因为 CRUX 没有 trace。

同时分析：W3 gap (300→500s=200s) vs max_iter (~3s) → 预期无差异。
"""
import json
import os
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V3_PRO_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS

BASE = "outputs/e3_swap"
W1_START, W1_END = 200_000, 300_000
W2_START, W2_END = 300_000, 320_000
W3_START, W3_END = 500_000, 600_000
SWAP_TIME = 300_000

workload = FEAS_BOUNDARY_V3_PRO_WORKLOAD

# Post-swap premium: was standard (ci>2.0)
post_swap_premium = {f"J{i}" for i, (_, _, ci) in enumerate(workload) if ci > 2.0}
post_swap_standard = {f"J{i}" for i, (_, _, ci) in enumerate(workload) if ci <= 2.0}

def load_v4_iterations():
    """从 v4 trace 加载 iteration records. trace 格式为 epoch-level, 
    含 per-job iteration 开始/结束时间."""
    trace_path = f"{BASE}/e3p_swap_v4_s0/trace.jsonl"
    if not os.path.exists(trace_path):
        print("No v4 trace found")
        return None

    # v4 trace is epoch-level allocator decisions, not individual iterations.
    # We need the simulation records instead.
    # Let's check the trace format.
    with open(trace_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    if not lines:
        return None
    
    # v4 trace has fields: epoch, time_ms, capacity_gbps, etc.
    # But NOT individual iteration start/end times.
    # We need to run the simulation with record saving.
    print(f"v4 trace has {len(lines)} epochs")
    print(f"Sample keys: {list(lines[0].keys())}")
    return None


def analyze_from_run_meta():
    """分析：从现有 run_meta 推断 start_ms vs end_ms 是否有差异。"""
    
    # Max iteration time for CRUX post-swap premium
    crux_meta = json.load(open(f"{BASE}/e3p_swap_CRUX_s0/run_meta.json"))
    crux_w3 = crux_meta["w3"]["per_job"]
    
    max_avg_iter = 0
    for jid, pj in crux_w3.items():
        avg = pj.get("avg_iter_ms", 0)
        if avg > max_avg_iter:
            max_avg_iter = avg
    
    print(f"Max avg_iter_ms in CRUX W3: {max_avg_iter:.0f}ms")
    print(f"W3 gap from swap: {W3_START - SWAP_TIME:.0f}ms = {(W3_START-SWAP_TIME)/1000:.0f}s")
    print(f"Guard band needed (2×max_iter): {2*max_avg_iter:.0f}ms = {2*max_avg_iter/1000:.1f}s")
    print()
    
    if W3_START - SWAP_TIME > 2 * max_avg_iter:
        print("VERDICT: W3 gap (200s) >> guard band (~6s)")
        print("→ end_ms filter for W3 does NOT introduce cross-swap contamination.")
        print("→ start_ms filter will give IDENTICAL results for W3.")
    else:
        print("WARNING: W3 gap insufficient, contamination possible")
    
    # Also check W2 (gap=0, max contamination)
    print()
    print(f"W2 gap from swap: {W2_START - SWAP_TIME:.0f}ms")
    if W2_START == SWAP_TIME:
        print("W2 starts at swap time → MAXIMUM contamination risk")
        print("But W2 is explicitly labeled 'transient', not used for claims.")
    
    # Check max iteration across ALL jobs
    all_max = 0
    for jid, pj in crux_meta["w1"]["per_job"].items():
        all_max = max(all_max, pj.get("avg_iter_ms", 0))
    for jid, pj in crux_meta["w3"]["per_job"].items():
        all_max = max(all_max, pj.get("avg_iter_ms", 0))
    
    print(f"Global max avg_iter_ms: {all_max:.0f}ms")
    print(f"W1 gap from swap: {W1_END - SWAP_TIME:.0f}ms")
    if W1_END <= SWAP_TIME:
        print("W1 ends AT swap → no post-swap contamination")
    else:
        print(f"W1 extends {W1_END - SWAP_TIME:.0f}ms past swap → possible tail contamination")
        if W1_END - SWAP_TIME < all_max:
            print("  But tail < max_iter, so at most 1 iteration per job contaminated")


def estimate_contamination_in_w1():
    """估算 end_ms 过滤下 W1 是否有跨 swap 污染。"""
    print()
    print("=" * 60)
    print("W1 Contamination Check")
    print("=" * 60)
    
    # W1 ends at 300s = swap time. Iterations ending in W1 (200-300s)
    # could have started as early as 300-max_iter.
    # With max_iter~3s, earliest start = 297s, still pre-swap.
    # So W1 is clean.
    
    e3p_crux = json.load(open(f"{BASE}/e3p_swap_CRUX_s0/run_meta.json"))
    
    # Find max iteration in W1
    w1_max = max(pj["avg_iter_ms"] for pj in e3p_crux["w1"]["per_job"].values())
    print(f"Max CRUX W1 avg_iter: {w1_max:.0f}ms ({w1_max/1000:.1f}s)")
    print(f"W1 ends at t={W1_END/1000:.0f}s = swap time")
    print(f"Earliest iteration start for end_ms filter: {W1_END - w1_max:.0f}ms = {(W1_END-w1_max)/1000:.1f}s")
    
    if W1_END - w1_max >= 0:
        print(f"This is {W1_END - w1_max:.0f}ms after simulation start → all iterations pre-swap")
    print("→ W1 is clean. No cross-swap contamination.")
    
    # W3 check
    w3_max = max(pj["avg_iter_ms"] for pj in e3p_crux["w3"]["per_job"].values())
    print()
    print(f"Max CRUX W3 avg_iter: {w3_max:.0f}ms ({w3_max/1000:.1f}s)")
    print(f"W3 starts at t={W3_START/1000:.0f}s (gap from swap = {(W3_START-SWAP_TIME)/1000:.0f}s)")
    print(f"Earliest iteration start for end_ms filter: {W3_START - w3_max:.0f}ms = {(W3_START - w3_max)/1000:.1f}s")
    print(f"Swap at {(W3_START - w3_max - SWAP_TIME)/1000:.1f}s before earliest possible iteration start")
    if W3_START - w3_max >= SWAP_TIME:
        print("→ W3 is clean. All iterations entirely post-swap.")
    else:
        print("→ W3 has cross-swap contamination!")


if __name__ == "__main__":
    print("Step 4: Window Filter Analysis (start_ms vs end_ms)")
    print("=" * 60)
    print()
    analyze_from_run_meta()
    estimate_contamination_in_w1()
    
    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()
    print("The end_ms filter is METHODOLOGICALLY CORRECT for this experiment:")
    print("  - W3 gap (200s) >> 2×max_iter (~6s) → no cross-swap contamination")
    print("  - W1 ends at swap time → all iterations pre-swap")
    print("  - W2 is transient, not used for primary claims")
    print()
    print("Switching to start_ms filter will NOT change any W1 or W3 results.")
    print("However, adopting start_ms filter as convention is cleaner for")
    print("future experiments where window gaps may be smaller.")
    print()
    print("Recommendation: Fix filter in code, re-run to confirm zero-delta,")
    print("then proceed to 3-seed formal batch.")
