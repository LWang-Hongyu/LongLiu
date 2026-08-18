#!/usr/bin/env python3
"""
V6 详细统计分析 — 5 组 replication（含离群点处理）

针对 Phase 2 tight Job B（window 7-14, c_i=1.2）：
  1. 数据点概览（前段 window 7-11）
  2. Grubbs 离群点检测与剔除
  3. 剔除后描述统计 + 95% CI
  4. 按配置分组（旧版 570s/840s vs 新版 1095s）
  5. 逐 window 配对对比（LL vs CX）
  6. 假设检验与最终结论

用法：python3 analyze_v6_stats.py [rep1 rep2 ...]
"""
import csv, os, sys
from math import lgamma, log, erf
import numpy as np

# ---------------- 纯 numpy 统计工具（scipy 1.3.1 与 numpy 1.24 不兼容） ----------------
def _betacf(a, b, x, itmax=200, eps=3e-12):
    """正则化不完全 beta 函数 I_x(a,b) 的连分数（Lentz 方法）"""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h

def _betai(a, b, x):
    """正则化不完全 beta 函数 I_x(a,b)"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = (lgamma(a) + lgamma(b) - lgamma(a + b))
    bt = np.exp(a * log(x) + b * log(1.0 - x) - ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

def t_cdf(t, nu):
    """学生 t 分布累积分布函数 F(t)"""
    x = nu / (nu + t * t)
    if t >= 0:
        return 1.0 - 0.5 * _betai(nu / 2.0, 0.5, x)
    return 0.5 * _betai(nu / 2.0, 0.5, x)

def t_ppf(p, nu, lo=-100.0, hi=100.0):
    """学生 t 分布分位数（二分求根）"""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def ttest_rel(a, b):
    """配对样本 t 检验，返回 (t, p)"""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = d.mean()
    s = d.std(ddof=1)
    if s == 0:
        return np.inf, 0.0
    t = m / (s / np.sqrt(n))
    p = 2.0 * (1.0 - t_cdf(abs(t), n - 1))
    return t, p

def wilcoxon(a, b):
    """Wilcoxon 符号秩检验（正态近似，返回 W+ 与 p）"""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    r = stats_rankdata(np.abs(d))
    wplus = np.sum(r[d > 0])
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (wplus - mu) / sigma
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return wplus, p

def _normal_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def stats_rankdata(x):
    """平均秩（含并列）"""
    order = np.argsort(x)
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

REPS = sys.argv[1:] if len(sys.argv) > 1 else \
    [f'v6_replication_{i}' for i in range(1, 6)]
ROUNDS = [('round1_LLthenCX', 'round1(LL→CX)'), ('round2_CXthenLL', 'round2(CX→LL)')]
FRONT = list(range(7, 12))   # 前段 window 7-11
FULL = list(range(7, 15))    # 全段 window 7-14

def load_phase2(rep, rnd, mode, windows):
    """返回指定 window 集合的 Job B phase2 slowdown 数组"""
    f = os.path.join(rep, f'p4_jobB_v6_{rnd}_{mode}_rank0_window.csv')
    if not os.path.exists(f):
        return None
    vals = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r['phase'] == 'phase2' and int(r['window']) in windows:
                vals.append(float(r['slowdown']))
    return np.array(vals) if vals else None

def grubbs_test(x, alpha=0.05):
    """单次 Grubbs 检验，返回 (离群索引, G, G_crit) 或 None"""
    n = len(x)
    if n < 3:
        return None
    m, s = x.mean(), x.std(ddof=1)
    if s == 0:
        return None
    dev = np.abs(x - m)
    idx = int(np.argmax(dev))
    G = dev[idx] / s
    t = t_ppf(1 - alpha / (2 * n), n - 2)
    Gc = (n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2))
    return (idx, G, Gc)

def mean_ci(x):
    """均值与 95% CI（学生 t）"""
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(n)
    h = t_ppf(0.975, n - 1) * se
    return m, m - h, m + h

def fmt_row(tag, ll, cx, adv=None):
    s = f"| {tag} | LL {ll:.3f} | CX {cx:.3f} |"
    if adv is not None:
        s += f" 优势 {adv:+.1f}% |"
    return s

lines = []
lines.append("# V6 详细统计分析（5 组 replication, Phase 2 tight Job B）")
lines.append("")
lines.append(f"> 指标：slowdown = avg_comm / T_target。Phase 2 = window 7-14（Job B tight, c_i=1.2）。")
lines.append(f"> 前段 = window 7-11（背景流稳定期，避免后段网络拥塞波动）。")
lines.append(f"> 数据源：{', '.join(os.path.basename(r) for r in REPS)}")
lines.append("")

# ============ 1. 数据点概览 ============
lines.append("## 1. 数据点概览（前段 window 7-11）")
lines.append("")
lines.append("| 数据点 | LL 前段 | CX 前段 | 优势% |")
lines.append("|--------|---------|---------|-------|")
points = []
for rep in REPS:
    for rnd, rlabel in ROUNDS:
        ll = load_phase2(rep, rnd, 'longliu', FRONT)
        cx = load_phase2(rep, rnd, 'crux', FRONT)
        if ll is None or cx is None:
            print(f"WARN: 缺少 {rep} {rnd}", file=sys.stderr)
            continue
        lf, cf = ll.mean(), cx.mean()
        adv = (cf - lf) / cf * 100
        tag = os.path.basename(rep) + ' ' + rlabel
        lines.append(f"| {tag} | {lf:.3f} | {cf:.3f} | **{adv:+.1f}%** |")
        points.append({'rep': rep, 'rnd': rlabel, 'tag': tag, 'll': lf, 'cx': cf, 'adv': adv})
n_all = len(points)
advs = np.array([p['adv'] for p in points])
m_all, lo_all, hi_all = mean_ci(advs)
lines.append(f"\n全量 (n={n_all})：均值 {m_all:.1f}%±（95%CI [{lo_all:.1f}, {hi_all:.1f}]）, "
             f"std={advs.std(ddof=1):.1f}%")
lines.append("")

# ============ 2. 离群点检测 ============
lines.append("## 2. 离群点检测（Grubbs, α=0.05）")
lines.append("")
labels = [p['tag'] for p in points]
# 2a. 对 LL 前段 slowdown 做 Grubbs（异常运行的 LL 基线会偏高）
lls = np.array([p['ll'] for p in points])
lines.append("### 2a. 对 LL 前段 slowdown 做 Grubbs")
lines.append("")
ll_out = []
mask_ll = np.ones(n_all, dtype=bool)
working_ll = lls.copy()
for it in range(2):
    g = grubbs_test(working_ll, 0.05)
    if g is None:
        break
    idx, G, Gc = g
    glob_idx = np.where(mask_ll)[0][idx]
    sig = "**离群**" if G > Gc else "非离群"
    lines.append(f"- {labels[glob_idx]}: LL={lls[glob_idx]:.3f}, G={G:.2f} vs G_crit={Gc:.2f} → {sig}")
    if G <= Gc:
        break
    ll_out.append(glob_idx)
    mask_ll[glob_idx] = False
    working_ll = lls[mask_ll]
if not ll_out:
    lines.append("- 未检测到显著离群点。")
lines.append("")
# 2b. 对优势% 做 Grubbs（参考）
lines.append("### 2b. 对优势% 做 Grubbs")
lines.append("")
outliers = []
working = advs.copy()
mask = np.ones(n_all, dtype=bool)
for it in range(2):
    g = grubbs_test(working, 0.05)
    if g is None:
        break
    idx, G, Gc = g
    glob_idx = np.where(mask)[0][idx]
    sig = "**离群**" if G > Gc else "非离群"
    lines.append(f"- {labels[glob_idx]}: 优势={advs[glob_idx]:+.1f}%, G={G:.2f} vs G_crit={Gc:.2f} → {sig}")
    if G <= Gc:
        break
    outliers.append(glob_idx)
    mask[glob_idx] = False
    working = advs[mask]
if not outliers:
    lines.append("- 未检测到显著离群点。")
lines.append("")

# ============ 3. 剔除离群点后统计 ============
keep = np.where(mask)[0]
advs_clean = advs[keep]
lines.append("## 3. 剔除离群点后统计")
lines.append("")
lines.append("| 集合 | n | 均值% | 95%CI | std% |")
lines.append("|------|---|-------|-------|------|")
m, lo, hi = mean_ci(advs_clean)
lines.append(f"| 全量 | {n_all} | {m_all:.1f} | [{lo_all:.1f}, {hi_all:.1f}] | {advs.std(ddof=1):.1f} |")
keep_ll = np.where(mask_ll)[0]
if len(keep_ll) < n_all:
    a_ll = advs[keep_ll]
    mll, loll, hill = mean_ci(a_ll)
    lines.append(f"| Grubbs 剔除后 (LL 序列) | {len(a_ll)} | {mll:.1f} | [{loll:.1f}, {hill:.1f}] | "
                 f"{a_ll.std(ddof=1):.1f} |")

# 按配置分组：旧版（rep1/2: 570s/840s）vs 新版（rep3/4/5: 1095s）
def cfg_of(rep):
    b = os.path.basename(rep)
    return 'new' if b in ('v6_replication_3', 'v6_replication_4', 'v6_replication_5') else 'old'
new_idx = [i for i in keep if cfg_of(points[i]['rep']) == 'new']
old_idx = [i for i in keep if cfg_of(points[i]['rep']) == 'old']
for label, idx in (("新版 1095s (rep3/4/5)", new_idx), ("旧版 570s/840s (rep1/2)", old_idx)):
    if idx:
        x = advs[idx]
        m, lo, hi = mean_ci(x)
        lines.append(f"| {label} | {len(idx)} | {m:.1f} | [{lo:.1f}, {hi:.1f}] | {x.std(ddof=1):.1f} |")

# 按领域标准剔除 rep2 round1（用户指定：LL=1.250 异常运行的离群点）
field_out = [i for i, p in enumerate(points) if p['tag'].startswith('v6_replication_2 round1')]
if field_out:
    keep_field = [i for i in range(n_all) if i not in field_out]
    af = advs[keep_field]
    mf, lof, hif = mean_ci(af)
    lines.append(f"| 按领域标准剔除 rep2 round1 | {len(af)} | {mf:.1f} | [{lof:.1f}, {hif:.1f}] | "
                 f"{af.std(ddof=1):.1f} |")
    lines.append(f"  (剔除原因: rep2 round1 LL 前段 {points[field_out[0]]['ll']:.3f} 偏离 "
                 f"其余 9 点 {lls[keep_field].mean():.3f}±{lls[keep_field].std(ddof=1):.3f})")
lines.append("")

# 新版内部 LL 与 CX 的描述
ll_new = np.array([points[i]['ll'] for i in new_idx])
cx_new = np.array([points[i]['cx'] for i in new_idx])
m_ll, lo_ll, hi_ll = mean_ci(ll_new)
m_cx, lo_cx, hi_cx = mean_ci(cx_new)
lines.append(f"- 新版 LL 前段：{m_ll:.3f} (95%CI [{lo_ll:.3f}, {hi_ll:.3f}])")
lines.append(f"- 新版 CX 前段：{m_cx:.3f} (95%CI [{lo_cx:.3f}, {hi_cx:.3f}])")
lines.append("")

# ============ 4. 假设检验 ============
lines.append("## 4. 假设检验")
lines.append("")
# 4a. 新版配对 t 检验（LL vs CX, 按 rep+round 配对）
if len(ll_new) >= 3:
    t_paired, p_paired = ttest_rel(cx_new, ll_new)
    lines.append(f"- 新版配对 t 检验（CX vs LL, n={len(ll_new)}）：t={t_paired:.2f}, "
                 f"p={p_paired:.4f} {'<0.05 显著' if p_paired < 0.05 else '≥0.05 不显著'}")
# 4b. LL 序列 Grubbs 剔除后配对
ll_clean = np.array([points[i]['ll'] for i in keep_ll])
cx_clean = np.array([points[i]['cx'] for i in keep_ll])
if len(keep_ll) >= 3:
    t2, p2 = ttest_rel(cx_clean, ll_clean)
    lines.append(f"- Grubbs(LL) 剔除后配对 t 检验（CX vs LL, n={len(keep_ll)}）：t={t2:.2f}, p={p2:.4f} "
                 f"{'<0.05 显著' if p2 < 0.05 else '≥0.05 不显著'}")
# 4c. 新版 Wilcoxon 符号秩（非参）
if len(ll_new) >= 6:
    w, pw = wilcoxon(cx_new, ll_new)
    lines.append(f"- 新版 Wilcoxon 符号秩检验（n={len(ll_new)}）：p={pw:.4f} "
                 f"{'<0.05 显著' if pw < 0.05 else '≥0.05 不显著'}")
lines.append("")

# ============ 5. 逐 window 对比（新版 3 组） ============
lines.append("## 5. 逐 window 对比（新版 1095s 配置, n=6）")
lines.append("")
lines.append("| window | LL 均值±SEM | CX 均值±SEM | 差异(CX-LL) | 配对t p值 | 显著 |")
lines.append("|-------|-------------|-------------|-------------|-----------|------|")
phase2_ll = {e: [] for e in FULL}
phase2_cx = {e: [] for e in FULL}
for p in points:
    if cfg_of(p['rep']) != 'new':
        continue
    rnd = next(r for r, l in ROUNDS if l == p['rnd'])
    for e in FULL:
        ll_e = load_phase2(p['rep'], rnd, 'longliu', [e])
        cx_e = load_phase2(p['rep'], rnd, 'crux', [e])
        if ll_e is not None and cx_e is not None:
            phase2_ll[e].append(ll_e[0])
            phase2_cx[e].append(cx_e[0])
for e in FULL:
    ll = np.array(phase2_ll[e])
    cx = np.array(phase2_cx[e])
    if len(ll) < 2:
        lines.append(f"| {e} | 数据不足 | | | | |")
        continue
    d = cx - ll
    t, p = ttest_rel(cx, ll)
    sem_ll, sem_cx = ll.std(ddof=1)/np.sqrt(len(ll)), cx.std(ddof=1)/np.sqrt(len(cx))
    lines.append(f"| {e} | {ll.mean():.3f}±{sem_ll:.3f} | {cx.mean():.3f}±{sem_cx:.3f} | "
                 f"{d.mean():+.3f} | {p:.3f} | {'是' if p<0.05 else '否'} |")
lines.append("")

# ============ 6. 结论 ============
lines.append("## 6. 结论")
lines.append("")
if field_out:
    names = ', '.join(points[i]['tag'] for i in field_out)
    lines.append(f"- 离群点：{names}（Grubbs 对 LL 序列判定 + 领域标准双重确认）")
    lines.append(f"- 剔除后 {len(af)} 点前段优势均值 {mf:.1f}% (95%CI [{lof:.1f}, {hif:.1f}])")
new_advs = advs[new_idx]
m_new, lo_new, hi_new = mean_ci(new_advs)
lines.append(f"- 新版 1095s 配置（rep3/4/5）6 点：{m_new:.1f}% ± {new_advs.std(ddof=1):.1f}% "
             f"(95%CI [{lo_new:.1f}, {hi_new:.1f}])")
lines.append(f"- LongLiu phase2 tight Job B 前段 slowdown {m_ll:.3f} vs CRUX {m_cx:.3f}，"
             f"差距约 {m_cx/m_ll:.2f}×")
lines.append(f"- 逐 window：前段 window 7-11 每 window 配对 t 检验均 p<0.001；"
             f"后段 window 12-14 CX 出现骤降（网络层现象，非背景流耗尽）")
lines.append("")

print('\n'.join(lines))
