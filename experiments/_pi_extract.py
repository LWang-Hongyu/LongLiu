"""
Extract per-job π fine-grained time series from E3' D1 seed=0 trace.
t ∈ [250, 550]s, every 25s.
"""
import json

trace_path = "outputs/e3_swap/e3p_swap_D1_s0/trace.jsonl"

targets_ms = list(range(250000, 550001, 25000))

all_epochs = []
print("Loading trace...")
with open(trace_path) as f:
    for line in f:
        d = json.loads(line)
        epoch = d["epoch"]
        t = d["time_ms"]
        pi_dict = {}
        for k, v in d.items():
            if k.endswith("_pi"):
                jid = k.replace("_pi", "")
                pi_dict[jid] = v
        if pi_dict:
            all_epochs.append((epoch, t, pi_dict))

print(f"Loaded {len(all_epochs)} epochs with pi data")
print(f"Time range: {all_epochs[0][1]:.0f} - {all_epochs[-1][1]:.0f} ms")

# Collect latest pi per job at each target
results = {}
latest_pi = {}
epoch_idx = 0

for target_ms in targets_ms:
    while epoch_idx < len(all_epochs) and all_epochs[epoch_idx][1] <= target_ms:
        _, _, pi_dict = all_epochs[epoch_idx]
        latest_pi.update(pi_dict)
        epoch_idx += 1
    t_key = f"{target_ms/1000:.0f}s"
    results[t_key] = dict(latest_pi)

# All JIDs
all_jids = set()
for pi_dict in results.values():
    all_jids.update(pi_dict.keys())
all_jids = sorted(all_jids, key=lambda x: int(x[1:]))

# Classification
pre_premium = {"J0", "J1", "J2", "J3", "J4"}
pre_standard = {"J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12"}

job_models = {
    "J0": ("BERT-Large-fp16", 2), "J1": ("BERT-Large-fp16", 2),
    "J2": ("BERT-Large-fp16", 4), "J3": ("ViT-Large", 2),
    "J4": ("ViT-Large", 2), "J5": ("LLaMA-2-13B", 8),
    "J6": ("LLaMA-2-13B", 8), "J7": ("LLaMA-2-13B", 8),
    "J8": ("LLaMA-2-7B", 8), "J9": ("LLaMA-2-7B", 8),
    "J10": ("LLaMA-2-7B", 8), "J11": ("T5-11B-fp16", 8),
    "J12": ("T5-11B-fp16", 8),
}

# === TABLE 1: π values ===
print()
print("=" * 160)
print("E3' D1 seed=0: Per-Job π Time Series (t ∈ [250, 550]s, Δ=25s)")
print("=" * 160)
print(f"{'Job':>4s} {'Model':<18s} {'dp':>3s} {'pre':>3s} {'post':>3s}", end="")
for t_label in results:
    print(f" {t_label:>7s}", end="")
print()
print("-" * 160)

for jid in all_jids:
    model, dp = job_models.get(jid, ("?", 0))
    pre_tier = "P" if jid in pre_premium else "S"
    post_tier = "S" if jid in pre_premium else "P"
    print(f"{jid:>4s} {model:<18s} {dp:>3d} {pre_tier:>3s} {post_tier:>3s}", end="")
    for t_label in results:
        pi_val = results[t_label].get(jid, None)
        if pi_val is not None:
            print(f" {pi_val:>7.3f}", end="")
        else:
            print(f" {'N/A':>7s}", end="")
    print()

# === TABLE 2: exp(π·K) weights at key times ===
K = 2.0
print()
print("=" * 120)
print("exp(π·K) weights at key time points (K=2.0)")
print("=" * 120)
key_times = ["250s", "275s", "300s", "325s", "350s", "375s", "400s",
             "425s", "450s", "475s", "500s", "525s", "550s"]
print(f"{'Job':>4s} {'pre':>3s} {'post':>3s}", end="")
for t in key_times:
    print(f" {t:>8s}", end="")
print()
print("-" * 120)

for jid in all_jids:
    pre_tier = "P" if jid in pre_premium else "S"
    post_tier = "S" if jid in pre_premium else "P"
    print(f"{jid:>4s} {pre_tier:>3s} {post_tier:>3s}", end="")
    for t in key_times:
        pi = results[t].get(jid, None)
        if pi is not None:
            w = round(2.71828 ** (pi * K), 3)
            print(f" {w:>8.3f}", end="")
        else:
            print(f" {'N/A':>8s}", end="")
    print()

# === Analysis: π trends per group ===
print()
print("=" * 80)
print("Group-Level π Averages (mean over all jobs in tier)")
print("=" * 80)
print(f"{'Tier Group':<30s}", end="")
for t_label in results:
    print(f" {t_label:>8s}", end="")
print()

groups = {
    "Old premium (J0-J4) → new std": pre_premium,
    "Old standard (J5-J12) → new prem": pre_standard,
}

for gname, gset in groups.items():
    print(f"{gname:<30s}", end="")
    for t_label in results:
        pis = [results[t_label][j] for j in gset if j in results[t_label]]
        if pis:
            print(f" {sum(pis)/len(pis):>8.3f}", end="")
        else:
            print(f" {'N/A':>8s}", end="")
    print()

# === Mechanism analysis: W3 (500-550s) ===
print()
print("=" * 80)
print("Mechanism Analysis: W3 Window (500-600s)")
print("=" * 80)

# New premium = J5-J12 (pre-swap standard, now premium)
new_premium = sorted(pre_standard)
old_premium_now_std = sorted(pre_premium)

print("\n--- New Premium jobs (J5-J12) π in W3 ---")
for jid in new_premium:
    pis_w3 = [results[t][jid] for t in results if t in ["500s", "525s", "550s"] and jid in results[t]]
    if pis_w3:
        print(f"  {jid} ({job_models[jid][0]}): π = {pis_w3}")

print("\n--- Old Premium now Standard (J0-J4) π in W3 ---")
for jid in old_premium_now_std:
    pis_w3 = [results[t][jid] for t in results if t in ["500s", "525s", "550s"] and jid in results[t]]
    if pis_w3:
        print(f"  {jid} ({job_models[jid][0]}): π = {pis_w3}")

# Check: did any old-premium-now-standard π cross 0 (go positive)?
print("\n--- Key question: Did any old premium (now standard) π cross 0? ---")
for jid in old_premium_now_std:
    for t_label in results:
        if jid in results[t_label]:
            pi = results[t_label][jid]
            if pi > -0.01:  # near zero or positive
                print(f"  {jid} @{t_label}: π = {pi:.4f} (near/above zero!)")

# Check: did any new premium π become very negative?
print("\n--- Did new premium π become very negative (indicating overfeeding)? ---")
for jid in new_premium:
    for t_label in ["400s", "450s", "500s", "550s"]:
        if jid in results[t_label]:
            pi = results[t_label][jid]
            if pi < -0.5:
                print(f"  {jid} @{t_label}: π = {pi:.4f} (deeply negative = OVERFED)")

# Check: weight clip_ratio=10 analysis
print("\n--- Weight clipping analysis (clip_ratio=10) ---")
# With K=2.0, exp(pi*K): pi range determines weight range
# If pi varies from -1.0 to 1.0, exp range is 0.135 to 7.389
# clip_ratio=10 means weight ratios capped at 10x
# But exp(pi*K) differences within same class are the issue
for jid in new_premium:
    if jid in results["500s"]:
        pi = results["500s"][jid]
        w = 2.71828 ** (pi * K)
        print(f"  {jid}: π={pi:.4f}, exp(π·K)={w:.4f}")
