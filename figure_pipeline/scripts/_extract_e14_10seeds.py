"""提取 E14 10-seed per-seed 数据到 figure_pipeline/data/e14_probe/。
baseline:  probe0_passive0_800g_s{0-9}  (probe=False, passive=False)
passive_low: probe0_passive1_800g_s{0-9} (probe=False, passive=True)
naive probe: 由 _quick_scan_e14_probe.py 提供（2 seeds）
"""
import json
import os

BASE = "/home/why/LongLiu_rebuild/sim-nextgen/outputs/e14_probe"
OUT = "/home/why/LongLiu_rebuild/sim-nextgen/figure_pipeline/data/e14_probe"
os.makedirs(OUT, exist_ok=True)

CONFIGS = {
    "baseline":    [f"probe0_passive0_800g_s{i}" for i in range(10)],
    "passive_low": [f"probe0_passive1_800g_s{i}" for i in range(10)],
}

rows = []
for cfg, dirs in CONFIGS.items():
    for d in dirs:
        with open(os.path.join(BASE, d, "run_meta.json")) as f:
            m = json.load(f)
        rows.append({
            "config": cfg,
            "seed": m["seed"],
            "p_attn": m["p_attn"],
            "n_frozen_jobs": m["n_frozen_jobs"],
            "starv": m["starv"],
            "total_iters": m["total_iters"],
        })

# 汇总 CSV
with open(os.path.join(OUT, "summary_10seeds_per_seed.csv"), "w") as f:
    f.write("config,seed,p_attn,n_frozen_jobs,starv,total_iters\n")
    for r in sorted(rows, key=lambda x: (x["config"], x["seed"])):
        f.write(f"{r['config']},{r['seed']},{r['p_attn']:.4f},"
                f"{r['n_frozen_jobs']},{r['starv']},{r['total_iters']}\n")

# 打印每 config 均值/配对差值（passive_low - baseline）
import statistics
for cfg in CONFIGS:
    ps = [r["p_attn"] for r in rows if r["config"] == cfg]
    fr = [r["n_frozen_jobs"] for r in rows if r["config"] == cfg]
    print(f"{cfg:12s} p_attn={statistics.mean(ps)*100:5.1f}% ±{statistics.stdev(ps)*100:4.1f} "
          f"frozen={statistics.mean(fr):5.1f}")

b = [r["p_attn"] for r in rows if r["config"] == "baseline"]
p = [r["p_attn"] for r in rows if r["config"] == "passive_low"]
diffs = [x - y for x, y in zip(p, b)]
print(f"paired diff (passive-baseline): mean={statistics.mean(diffs)*100:+.1f}pp")
try:
    from scipy import stats
    t, pv = stats.ttest_rel(p, b)
    print(f"paired t-test: t={t:.3f} p={pv:.3f}")
except ImportError:
    print("scipy not available, skip t-test")
print(f"saved: {OUT}/summary_10seeds_per_seed.csv")
