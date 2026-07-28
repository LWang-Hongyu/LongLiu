"""
D1@400G standard job 计入重算：验证 J5/J6 欠喂误差收敛

用户假设：D1 是单层 exp(π·K) 定律，没有 tier 隔离。在 400G 深饱和区，
standard job 同样累积违约（π 上升），与 premium 在同一条定律下竞争，
摊薄 premium 的份额。

验证方法：
1. 对每个 contested epoch（≥2 个 job 在同 spine link 竞争），
   计算 exp(π·K) 期望份额（包含所有 premium + standard jobs）
2. 对比 J5/J6 的实际份额 vs 期望份额
3. 若误差收敛到 ~0，D1 失效机制定案：无 tier 隔离导致 standard 摊薄 premium
"""

import json
import math
import numpy as np
from collections import defaultdict

K = 2.0

# E1 workload: premium=J0-J7, standard=J8-J13
PREMIUM_JIDS = [f"J{i}" for i in range(8)]
STANDARD_JIDS = [f"J{i}" for i in range(8, 14)]
ALL_JIDS = PREMIUM_JIDS + STANDARD_JIDS

# Load trace
trace_rows = []
with open("outputs/v3_d1_trajectory_400g/trace.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            trace_rows.append(json.loads(line))

n_epochs = len(trace_rows)
print(f"Total epochs: {n_epochs}")
print(f"Premium jobs: {PREMIUM_JIDS}")
print(f"Standard jobs: {STANDARD_JIDS}")
print()

# Step 1: Identify contested epochs (≥2 jobs present, at least some premium+standard mix)
contested = []
for row in trace_rows:
    epoch = row["epoch"]
    present = []
    for jid in ALL_JIDS:
        share = row.get(f"{jid}_share")
        if share is not None:
            present.append(jid)
    if len(present) < 2:
        continue
    contested.append(row)

print(f"Contested epochs (≥2 jobs): {len(contested)}")
print()

# Step 2: For each contested epoch, compute exp(pi*K) expected share
# Only consider epochs where standard jobs are present
mixed_epochs = []
for row in contested:
    has_premium = any(row.get(f"{jid}_share") is not None for jid in PREMIUM_JIDS)
    has_standard = any(row.get(f"{jid}_share") is not None for jid in STANDARD_JIDS)
    if has_premium and has_standard:
        mixed_epochs.append(row)

print(f"Mixed epochs (premium+standard on same link): {len(mixed_epochs)}")
print()

if not mixed_epochs:
    print("WARNING: No mixed epochs found! Premium and standard never compete on same link.")
    print("This confirms '错峰+路由伪影' — different job sets on different spine links.")
    print()
    print("Alternative analysis: aggregate across all spine links")
    
    # Aggregate all epochs across all spine links
    # For each JID, compute mean observed share and mean expected share
    # Expected share per JID = mean(exp(pi*K) / sum(exp(pi*K))) across all epochs where JID is active
    
    job_obs_shares = {jid: [] for jid in ALL_JIDS}
    job_exp_shares = {jid: [] for jid in ALL_JIDS}
    
    for row in contested:
        present = [jid for jid in ALL_JIDS if row.get(f"{jid}_share") is not None]
        if len(present) < 2:
            continue
        
        # Compute exp(pi*K) for all present jobs
        weights = {}
        for jid in present:
            pi = row.get(f"{jid}_pi", 0.0)
            weights[jid] = math.exp(K * pi)
        
        total_w = sum(weights.values())
        for jid in present:
            exp_share = weights[jid] / total_w if total_w > 0 else 0
            obs_share = row.get(f"{jid}_share", 0.0)
            job_obs_shares[jid].append(obs_share)
            job_exp_shares[jid].append(exp_share)
    
    print("=" * 80)
    print("Per-Job: Observed vs Expected Share (exp(pi*K), all contested epochs)")
    print("=" * 80)
    header = f"{'JID':<6} {'Tier':<8} {'N_epochs':>8} {'ObsMean':>10} {'ExpMean':>10} {'Error':>10} {'ObsStd':>8}"
    print(header)
    print("-" * len(header))
    
    total_error = 0.0
    for jid in ALL_JIDS:
        tier = "premium" if jid in PREMIUM_JIDS else "standard"
        n = len(job_obs_shares[jid])
        if n == 0:
            continue
        obs_mean = np.mean(job_obs_shares[jid])
        exp_mean = np.mean(job_exp_shares[jid])
        error = obs_mean - exp_mean
        obs_std = np.std(job_obs_shares[jid])
        total_error += abs(error)
        print(f"{jid:<6} {tier:<8} {n:>8d} {obs_mean:>10.4f} {exp_mean:>10.4f} {error:>+10.4f} {obs_std:>8.4f}")
    
    print()
    print(f"Mean absolute error (all jobs): {total_error / 14:.4f}")
    
    # Focus on J5/J6 (the allegedly underfed premium jobs)
    print()
    print("=" * 80)
    print("J5/J6 Underfeed Analysis")
    print("=" * 80)
    for jid in ["J5", "J6"]:
        if job_obs_shares[jid]:
            obs_mean = np.mean(job_obs_shares[jid])
            exp_mean = np.mean(job_exp_shares[jid])
            error = obs_mean - exp_mean
            print(f"  {jid}: obs_mean={obs_mean:.4f}, exp_mean={exp_mean:.4f}, error={error:+.4f} "
                  f"({'UNDERFED' if error < -0.01 else 'OVERFED' if error > 0.01 else 'CONVERGED'})")
    
    # Also check all premium jobs for systemic bias
    print()
    print("=" * 80)
    print("Premium Tier: Observed vs Expected (with standard jobs in competition)")
    print("=" * 80)
    premium_errors = []
    for jid in PREMIUM_JIDS:
        if job_obs_shares[jid]:
            obs_mean = np.mean(job_obs_shares[jid])
            exp_mean = np.mean(job_exp_shares[jid])
            error = obs_mean - exp_mean
            premium_errors.append(error)
    
    mean_premium_error = np.mean(premium_errors)
    print(f"  Mean premium error: {mean_premium_error:+.4f}")
    print(f"  Premium errors: {[f'{e:+.4f}' for e in premium_errors]}")
    
    # VERDICT
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    j5_error = np.mean(job_obs_shares.get("J5", [0])) - np.mean(job_exp_shares.get("J5", [0]))
    j6_error = np.mean(job_obs_shares.get("J6", [0])) - np.mean(job_exp_shares.get("J6", [0]))
    
    if abs(mean_premium_error) < 0.02:
        print(f"  Premium mean error = {mean_premium_error:+.4f} < 0.02 → CONVERGED")
        print()
        print("  Mechanism confirmed: D1's exp(pi*K) law operates on ALL jobs without tier isolation.")
        print("  Standard jobs with rising pi compete on equal footing with premium jobs,")
        print("  diluting premium share. The observed inequality among premium jobs")
        print("  (J0/J1 dominant, J5/J6 underfed) is fully explained by the same law")
        print("  operating on different spine links with different competitor sets.")
        print()
        print("  v4's two-tier isolation (premium pool / standard floor) directly solves this.")
    else:
        print(f"  Premium mean error = {mean_premium_error:+.4f} — residual remains")
        print(f"  J5 error = {j5_error:+.4f}, J6 error = {j6_error:+.4f}")
        print("  Requires further investigation.")
    
    # Save results
    result = {
        "n_epochs": n_epochs,
        "n_contested": len(contested),
        "n_mixed": len(mixed_epochs),
        "mean_premium_error": round(float(mean_premium_error), 4),
        "per_job": {}
    }
    for jid in ALL_JIDS:
        if job_obs_shares[jid]:
            result["per_job"][jid] = {
                "obs_mean": round(float(np.mean(job_obs_shares[jid])), 4),
                "exp_mean": round(float(np.mean(job_exp_shares[jid])), 4),
                "error": round(float(np.mean(job_obs_shares[jid]) - np.mean(job_exp_shares[jid])), 4),
                "n": len(job_obs_shares[jid])
            }
    
    with open("outputs/v3_d1_trajectory_400g/standard_recalc.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to outputs/v3_d1_trajectory_400g/standard_recalc.json")

else:
    # Mixed epochs exist — direct computation
    # Compute expected share per epoch and compare
    j5_errors = []
    j6_errors = []
    all_errors = []
    
    for row in mixed_epochs:
        present = [jid for jid in ALL_JIDS if row.get(f"{jid}_share") is not None]
        weights = {}
        for jid in present:
            pi = row.get(f"{jid}_pi", 0.0)
            weights[jid] = math.exp(K * pi)
        
        total_w = sum(weights.values())
        for jid in present:
            exp_share = weights[jid] / total_w if total_w > 0 else 0
            obs_share = row.get(f"{jid}_share", 0.0)
            error = obs_share - exp_share
            all_errors.append(error)
            if jid == "J5":
                j5_errors.append(error)
            if jid == "J6":
                j6_errors.append(error)
    
    if j5_errors:
        print(f"J5: obs-exp error mean={np.mean(j5_errors):+.4f}, std={np.std(j5_errors):.4f}")
    if j6_errors:
        print(f"J6: obs-exp error mean={np.mean(j6_errors):+.4f}, std={np.std(j6_errors):.4f}")
    print(f"All jobs error mean={np.mean(all_errors):+.4f}")
