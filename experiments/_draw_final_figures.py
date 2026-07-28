"""
综合终版绘图：Fig-1~5 + T-1~2
修正：全局删图内标题、统一命名(LongLiu/DF/SP/Fair)、总体std(ddof=0)、PDF+PNG双份
最高优先：Fig-1 英雄图 E3/E3' 三策略滑动窗轨迹 n=5
SEMANTICS_VERSION: anchor-v2, 5-seed canonical
"""

from __future__ import annotations

import csv, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longliu_sim.trace.synthetic import (
    FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD
)
from longliu_sim.utils.model_params import MODEL_PARAMS
from longliu_sim.utils.config import load_config

_cfg = load_config()
OVERHEAD = _cfg["frozen"]["overhead_factor"]
OVERLAP = _cfg["frozen"]["overlap_factor"]

# ── Paths ──
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_REG = os.path.join(PROJ, "PAPER_EVIDENCE", "FIGURE_REGISTRY")
E3_BASE = os.path.join(PROJ, "PAPER_EVIDENCE", "05_E3_swap_main")
ANCHOR_BASE = os.path.join(PROJ, "PAPER_EVIDENCE", "01_baseline_anchor")
OUT_DIR = os.path.join(PROJ, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── matplotlib ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8, 'figure.dpi': 150,
})

# ── Colors & Styles (全局统一) ──
POLICY_COLORS = {
    "Fair":   "#90A4AE",  # grey
    "CRUX":   "#F44336",  # red
    "SP":     "#FF9800",  # orange
    "DF":     "#4CAF50",  # green
    "LongLiu":"#2196F3",  # blue
}
POLICY_LINESTYLE = {
    "Fair":   "--",
    "CRUX":   "-",
    "SP":     "-.",
    "DF":     "-.",       # dot-dash green
    "LongLiu":"-",
}
POLICY_MARKER = {
    "Fair":   "s",
    "CRUX":   "o",
    "SP":     "^",
    "DF":     "D",
    "LongLiu":"o",
}
POLICY_LABEL = {
    "Fair":   "Fair",
    "CRUX":   "CRUX",
    "SP":     "SP",
    "DF":     "DF",
    "LongLiu":"LongLiu (Ours)",
}
POLICY_ORDER = ["LongLiu", "DF", "CRUX", "SP", "Fair"]


def save_both(fig, stem):
    """Save as PDF + 600dpi PNG."""
    fig.savefig(f"{stem}.pdf", dpi=150, bbox_inches='tight')
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches='tight')
    print(f"  -> {stem}.pdf + .png")

# ═══════════════════════════════════════════════════════════════
# Data: E1/E2 5-seed CSV
# ═══════════════════════════════════════════════════════════════

def load_e1_e2(path):
    df = pd.read_csv(path, quotechar="'")
    data = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        scene = row["scene"]
        pol = {"v4":"LongLiu","D1":"DF"}.get(row["policy"], row["policy"])
        bw = int(row["spine_bw"])
        data[scene][pol][bw] = (float(row["p_attn_mean"]), float(row["p_attn_std"]))
    return dict(data)

# ═══════════════════════════════════════════════════════════════
# Data: E3 swap sliding-window trajectory (CRUX from records.jsonl)
# ═══════════════════════════════════════════════════════════════

WINDOW_S = 100.0
SWAP_TIME_S = 300.0
TIME_STEP = 0.25
T_START, T_END = 100.0, 600.0
ANNOT_W1, ANNOT_W2, ANNOT_W3 = (200,300), (300,320), (500,600)
LARGE_MODELS = {"LLaMA-2-13B", "LLaMA-2-7B", "T5-11B-fp16"}

def get_target(comp_ms, comm_solo_ms, ci):
    comm_budget = ci * comm_solo_ms * OVERHEAD
    if OVERLAP > 0:
        return max(comp_ms, comm_budget) + (1.0 - OVERLAP) * min(comp_ms, comm_budget)
    return comp_ms + comm_budget

def build_job_info(workload_raw):
    info = {}
    for i, (model, dp, orig_ci) in enumerate(workload_raw):
        jid = f"J{i}"
        p = MODEL_PARAMS[model]
        bpp = 2 if p.get("fp16", True) else 4
        bytes_per_iter = 2 * p["params"] * bpp / max(dp, 1)
        mb = bytes_per_iter / (1024 * 1024)
        raw_comm_ms = mb * 8 * 1024 * 1024 / (100e9) * 1000.0
        comp_ms = p.get("comp_ms", 50.0)

        pre_target = get_target(comp_ms, raw_comm_ms, orig_ci)
        was_premium = orig_ci <= 2.0
        post_ci = 3.0 if was_premium else (1.5 if model in LARGE_MODELS or dp != 4 else 2.0)
        post_target = get_target(comp_ms, raw_comm_ms, post_ci)

        info[jid] = {
            "model": model, "dp": dp,
            "orig_ci": orig_ci, "post_ci": post_ci,
            "pre_target": pre_target, "post_target": post_target,
            "pre_is_premium": was_premium,
            "post_is_premium": not was_premium,
        }
    return info

def compute_sliding_trajectory(records, job_info):
    """Sliding 100s window P-attn trajectory from records."""
    if not records:
        return [], []
    records.sort(key=lambda r: r["start_ms"])
    n_points = int((T_END - T_START) / TIME_STEP) + 1
    time_grid = np.linspace(T_START, T_END, n_points)

    w_ms = WINDOW_S * 1000.0
    left_idx = right_idx = 0
    job_sum = defaultdict(float)
    job_cnt = defaultdict(int)
    results_t, results_pattn = [], []

    for t_s in time_grid:
        t_ms = t_s * 1000.0
        lo = max(0.0, t_ms - w_ms)
        hi = t_ms

        while left_idx < len(records) and records[left_idx]["start_ms"] < lo:
            r = records[left_idx]
            jid = r["jid"]
            if jid in job_cnt:
                job_sum[jid] -= r["iter_ms"]
                job_cnt[jid] -= 1
                if job_cnt[jid] <= 0:
                    job_sum.pop(jid, None); job_cnt.pop(jid, None)
            left_idx += 1

        while right_idx < len(records) and records[right_idx]["start_ms"] <= hi:
            r = records[right_idx]
            jid = r["jid"]
            job_sum[jid] += r["iter_ms"]
            job_cnt[jid] = job_cnt.get(jid, 0) + 1
            right_idx += 1

        if t_s <= SWAP_TIME_S:
            premium_set = {j for j, info in job_info.items() if info["pre_is_premium"]}
            target_key = "pre_target"
        else:
            premium_set = {j for j, info in job_info.items() if info["post_is_premium"]}
            target_key = "post_target"

        p_total = p_attn = 0
        for jid in premium_set:
            if jid in job_cnt and job_cnt[jid] > 0:
                avg_iter = job_sum[jid] / job_cnt[jid]
                target = job_info[jid][target_key]
                sas = target / avg_iter if avg_iter > 0 else 0.0
                p_total += 1
                if sas >= 0.98:
                    p_attn += 1
        if p_total > 0:
            results_t.append(t_s)
            results_pattn.append(p_attn / p_total)
    return results_t, results_pattn

def load_crux_trajectory(tag):
    """Load CRUX 5-seed records and compute sliding-window P-attn trajectory."""
    workload = FEAS_BOUNDARY_V3_WORKLOAD if tag == "e3_swap" else FEAS_BOUNDARY_V3_PRO_WORKLOAD
    job_info = build_job_info(workload)

    seed_trajs = []
    for s in [0, 1, 2, 4, 5]:
        path = os.path.join(E3_BASE, f"{tag}_CRUX_s{s}", "records.jsonl")
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        t_arr, p_arr = compute_sliding_trajectory(records, job_info)
        seed_trajs.append((t_arr, p_arr))

    # Align to common grid
    all_t = sorted(set(t for t_arr, _ in seed_trajs for t in t_arr))
    common_t = np.array(all_t)
    interp_vals = np.array([np.interp(common_t, t_arr, p_arr) for t_arr, p_arr in seed_trajs])
    return common_t, np.mean(interp_vals, axis=0), np.std(interp_vals, axis=0)

def load_df_trajectory(csv_path):
    """Load DF trajectory from FIGURE_REGISTRY CSV."""
    times, means, stds = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["time_s"]))
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
    return np.array(times), np.array(means), np.array(stds)

# ═══════════════════════════════════════════════════════════════
# FIG-1: Hero Figure — E3/E3' 三策略滑动窗轨迹 (最高优先)
# ═══════════════════════════════════════════════════════════════

def draw_fig1_hero():
    """Fig-1: E3/E3' dual-panel, 3-policy sliding-window trajectory, 5-seed."""
    print("\n=== Fig-1: Hero Figure ===")

    t_crux_e3,  m_crux_e3,  s_crux_e3  = load_crux_trajectory("e3_swap")
    t_crux_e3p, m_crux_e3p, s_crux_e3p = load_crux_trajectory("e3p_swap")
    t_df_e3,  m_df_e3,  s_df_e3  = load_df_trajectory(os.path.join(FIG_REG, "fig4_d1_trajectory_e3.csv"))
    t_df_e3p, m_df_e3p, s_df_e3p = load_df_trajectory(os.path.join(FIG_REG, "fig4_d1_trajectory_e3p.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for ax, arm_label, t_c, m_c, s_c, t_d, m_d, s_d in [
        (ax1, "E3 Control Arm (800 Gbps)", t_crux_e3, m_crux_e3, s_crux_e3,
         t_df_e3, m_df_e3, s_df_e3),
        (ax2, "E3' Kill Arm (630 Gbps)", t_crux_e3p, m_crux_e3p, s_crux_e3p,
         t_df_e3p, m_df_e3p, s_df_e3p),
    ]:
        # LongLiu (v4): constant 100%
        ax.axhline(y=1.0, color=POLICY_COLORS["LongLiu"], linestyle='-',
                   linewidth=1.8, alpha=0.85, label="LongLiu (Ours)")
        ax.annotate("LongLiu: 100%", xy=(550, 1.02), fontsize=8,
                    color=POLICY_COLORS["LongLiu"], fontweight='bold', ha='right')

        # DF
        ax.plot(t_d, m_d, color=POLICY_COLORS["DF"], linestyle=POLICY_LINESTYLE["DF"],
                linewidth=1.6, alpha=0.9, label="DF")
        ax.fill_between(t_d, np.clip(m_d - s_d, 0, None), np.clip(m_d + s_d, 0, 1.05),
                        color=POLICY_COLORS["DF"], alpha=0.12)

        # CRUX
        ax.plot(t_c, m_c, color=POLICY_COLORS["CRUX"], linestyle='-',
                linewidth=1.6, alpha=0.85, label="CRUX")
        ax.fill_between(t_c, np.clip(m_c - s_c, 0, None), np.clip(m_c + s_c, 0, 1.05),
                        color=POLICY_COLORS["CRUX"], alpha=0.12)

        # Window annotations (below subplot title area)
        for (ws, we), wl in [((200,300),"W1"),((300,320),"W2"),((500,600),"W3")]:
            ax.axvspan(ws, we, alpha=0.07, color='gray')
            ax.text((ws+we)/2, 1.04, wl, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#555555',
                    transform=ax.get_xaxis_transform())

        # Swap line
        ax.axvline(x=SWAP_TIME_S, color='black', linestyle='--', linewidth=1.0, alpha=0.45)
        ax.text(SWAP_TIME_S + 4, 0.03, 'tier swap', fontsize=8, color='black', alpha=0.55)

        ax.set_ylabel("$P_{\\mathrm{attn}}$", fontsize=11)
        ax.set_ylim(-0.05, 1.20)
        ax.set_title(arm_label, fontsize=11, fontweight='bold', loc='left', pad=12)
        ax.legend(fontsize=9, loc='lower right', ncol=3)
        ax.grid(True, alpha=0.25)

    ax2.set_xlabel("Time (s)", fontsize=11)
    ax1.set_xlim(100, 600)

    # Caption
    caption = ("Sliding 100 s window, start_ms semantics, per-regime tier/target; "
               "5 seeds mean ± std; tier swap at $t = 300$ s. "
               "The LongLiu dip at $t \\approx 300$ s is a sliding-window boundary artifact "
               "(window straddles pre-/post-swap iterations under different tier definitions), "
               "not a transient.")
    fig.text(0.5, 0.005, caption, ha='center', fontsize=7.5,
             color='#555555', style='italic')

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = os.path.join(OUT_DIR, "fig1_hero")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-2: E1 Ladder
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    print("\n=== Fig-2: E1 Ladder ===")
    data = load_e1_e2(os.path.join(FIG_REG, "fig2_e1_ladder_5seed.csv"))
    e1d = data["E1"]
    bws = [400, 500, 630, 800, 1000, 1200]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for pol in POLICY_ORDER:
        means = np.array([e1d[pol][bw][0] * 100 for bw in bws])
        stds  = np.array([e1d[pol][bw][1] * 100 for bw in bws])
        ax.plot(bws, means, color=POLICY_COLORS[pol],
                linestyle=POLICY_LINESTYLE[pol], marker=POLICY_MARKER[pol],
                markersize=8, linewidth=2.0, label=POLICY_LABEL[pol], zorder=5)
        ax.fill_between(bws, means - stds, means + stds,
                        color=POLICY_COLORS[pol], alpha=0.12)
        if pol == "LongLiu":
            for bw, m in zip(bws, means):
                ax.annotate(f"{m:.1f}%", (bw, m), textcoords="offset points",
                            xytext=(0, 12), ha='center', fontsize=8,
                            color=POLICY_COLORS[pol], fontweight='bold')

    ax.set_xlabel("Spine Bandwidth (Gbps)")
    ax.set_ylabel("$P_{\\mathrm{attn}}$ (%)")
    ax.set_ylim(0, 110); ax.set_xlim(350, 1250)
    ax.legend(loc='lower right', ncol=3)
    ax.grid(True, alpha=0.3)

    ax.axvspan(350, 550, alpha=0.04, color='red')
    ax.axvspan(550, 750, alpha=0.04, color='orange')
    ax.axvspan(750, 1250, alpha=0.04, color='green')
    ax.text(450, 108, "Scarce", ha='center', fontsize=8, color='red', alpha=0.6)
    ax.text(650, 108, "Transition", ha='center', fontsize=8, color='orange', alpha=0.6)
    ax.text(1000, 108, "Abundant", ha='center', fontsize=8, color='green', alpha=0.6)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_e1_ladder")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-3: E2 Orthogonal
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    print("\n=== Fig-3: E2 Orthogonal ===")
    data = load_e1_e2(os.path.join(FIG_REG, "fig3_e2_ladder_5seed.csv"))
    e2p_d = data["E2'"]
    e2pro_d = data["E2-pro"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    bws_e2 = [400, 500, 630, 800]
    for pol in POLICY_ORDER:
        means = np.array([e2p_d[pol][bw][0] * 100 for bw in bws_e2])
        stds  = np.array([e2p_d[pol][bw][1] * 100 for bw in bws_e2])
        ax1.plot(bws_e2, means, color=POLICY_COLORS[pol],
                 linestyle=POLICY_LINESTYLE[pol], marker=POLICY_MARKER[pol],
                 markersize=8, linewidth=2.0, label=POLICY_LABEL[pol])
        ax1.fill_between(bws_e2, means - stds, means + stds,
                         color=POLICY_COLORS[pol], alpha=0.12)
        if pol == "LongLiu":
            for bw, m in zip(bws_e2, means):
                ax1.annotate(f"{m:.1f}%", (bw, m), textcoords="offset points",
                             xytext=(0, 12), ha='center', fontsize=8,
                             color=POLICY_COLORS[pol], fontweight='bold')

    ax1.set_xlabel("Spine Bandwidth (Gbps)")
    ax1.set_ylabel("$P_{\\mathrm{attn}}$ (%)")
    ax1.set_title("E2' (CRUX-disadvantaging workload)", fontsize=11, fontweight='bold', loc='left')
    ax1.set_ylim(0, 110)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    bws_pro = [630, 800]
    for pol in POLICY_ORDER:
        means = np.array([e2pro_d[pol][bw][0] * 100 for bw in bws_pro])
        stds  = np.array([e2pro_d[pol][bw][1] * 100 for bw in bws_pro])
        ax2.plot(bws_pro, means, color=POLICY_COLORS[pol],
                 linestyle=POLICY_LINESTYLE[pol], marker=POLICY_MARKER[pol],
                 markersize=8, linewidth=2.0)
        ax2.fill_between(bws_pro, means - stds, means + stds,
                         color=POLICY_COLORS[pol], alpha=0.12)

    ax2.set_xlabel("Spine Bandwidth (Gbps)")
    ax2.set_ylabel("$P_{\\mathrm{attn}}$ (%)")
    ax2.set_title("E2-pro (CRUX-favorable sanity)", fontsize=11, fontweight='bold', loc='left')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_e2_orthogonal")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-4: D1 Transient Trajectory
# ═══════════════════════════════════════════════════════════════

def draw_fig4():
    print("\n=== Fig-4: D1 Transient Trajectory ===")
    t_e3,  m_e3,  s_e3  = load_df_trajectory(os.path.join(FIG_REG, "fig4_d1_trajectory_e3.csv"))
    t_e3p, m_e3p, s_e3p = load_df_trajectory(os.path.join(FIG_REG, "fig4_d1_trajectory_e3p.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, t, m, s, arm_title in [
        (ax1, t_e3,  m_e3,  s_e3,  "E3 Control Arm (800 Gbps)"),
        (ax2, t_e3p, m_e3p, s_e3p, "E3' Kill Arm (630 Gbps)"),
    ]:
        # DF trajectory (green, all panels unified)
        ax.plot(t, m * 100, color=POLICY_COLORS["DF"],
                linestyle=POLICY_LINESTYLE["DF"], linewidth=2.0, label="DF")
        ax.fill_between(t, np.clip((m - s) * 100, 0, None),
                        np.clip((m + s) * 100, 0, 105),
                        color=POLICY_COLORS["DF"], alpha=0.12)

        # LongLiu 100% reference line
        ax.axhline(y=100, color=POLICY_COLORS["LongLiu"], linestyle='-',
                   linewidth=1.4, alpha=0.7, label="LongLiu (Ours)")

        # Window annotations — place below subplot title
        for (ws, we), wl in [((200,300),"W1"),((300,320),"W2"),((500,600),"W3")]:
            ax.axvspan(ws, we, alpha=0.07, color='gray')
            ax.text((ws+we)/2, 1.03, wl, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#555555',
                    transform=ax.get_xaxis_transform())

        # Swap line
        ax.axvline(x=SWAP_TIME_S, color='black', linestyle='--',
                   linewidth=1.0, alpha=0.45)
        ax.text(SWAP_TIME_S + 5, 2, 'swap', fontsize=8, color='black', alpha=0.55)

        # Annotate W1/W3 values at window midpoints
        w1_mid = np.mean(ANNOT_W1)
        w3_mid = np.mean(ANNOT_W3)
        idx_w1 = np.argmin(np.abs(t - w1_mid))
        idx_w3 = np.argmin(np.abs(t - w3_mid))
        ax.annotate(f"W1: {m[idx_w1]*100:.1f}%", (t[idx_w1], m[idx_w1]*100),
                    textcoords="offset points", xytext=(15, 10), fontsize=9,
                    fontweight='bold', color=POLICY_COLORS["DF"])
        ax.annotate(f"W3: {m[idx_w3]*100:.1f}%", (t[idx_w3], m[idx_w3]*100),
                    textcoords="offset points", xytext=(15, -18), fontsize=9,
                    fontweight='bold', color=POLICY_COLORS["DF"])

        ax.set_ylabel("$P_{\\mathrm{attn}}$ (%)")
        ax.set_ylim(-5, 115)
        ax.set_title(arm_title, fontsize=11, fontweight='bold', loc='left', pad=12)
        ax.legend(fontsize=9, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.25)

    ax2.set_xlabel("Time (s)")

    caption = ("Sliding 100 s window, start_ms semantics, per-regime tier/target; "
               "5 seeds mean $\\pm$ std; tier swap at $t = 300$ s. "
               "LongLiu reference line at 100% (constant, both arms).")
    fig.text(0.5, 0.003, caption, ha='center', fontsize=7.5,
             color='#555555', style='italic')

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    path = os.path.join(OUT_DIR, "fig4_d1_trajectory")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-5: π Time Series
# ═══════════════════════════════════════════════════════════════

def draw_fig5():
    print("\n=== Fig-5: Pi Time Series ===")
    trace_path = os.path.join(E3_BASE, "e3p_swap_D1_s0", "trace.jsonl")
    if not os.path.exists(trace_path):
        print("  SKIP: no trace")
        return None

    from collections import defaultdict
    pi_series = defaultdict(list)
    with open(trace_path) as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            t_s = rec.get("time_ms", 0) / 1000.0
            for key, val in rec.items():
                if key.endswith("_pi") and key[0] == "J":
                    pi_series[key[:-3]].append((t_s, val))

    if not pi_series:
        print("  SKIP: no pi data")
        return None
    for jid in pi_series:
        pi_series[jid].sort()

    swap_path = os.path.join(E3_BASE, "e3p_swap_D1_s0", "swap_log.json")
    with open(swap_path) as f:
        swap_log = json.load(f)
    swap_time = swap_log["swap_time_ms"] / 1000.0
    pre_premium = {s["jid"] for s in swap_log["swaps"] if s["old_ci"] <= 2.0}
    post_premium = {s["jid"] for s in swap_log["swaps"] if s["new_ci"] <= 2.0}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colors_pre = plt.cm.Blues(np.linspace(0.4, 0.9, len(pre_premium)))
    colors_post = plt.cm.Reds(np.linspace(0.4, 0.9, len(post_premium)))

    for idx, jid in enumerate(sorted(pre_premium)):
        if jid not in pi_series: continue
        t_arr = np.array([p[0] for p in pi_series[jid]])
        pi_arr = np.array([p[1] for p in pi_series[jid]])
        ax1.plot(t_arr, pi_arr, color=colors_pre[idx], linewidth=1.2, alpha=0.8,
                 label=f"{jid} (prem$\\to$std)")

    for idx, jid in enumerate(sorted(post_premium - pre_premium)):
        if jid not in pi_series: continue
        t_arr = np.array([p[0] for p in pi_series[jid]])
        pi_arr = np.array([p[1] for p in pi_series[jid]])
        ax2.plot(t_arr, pi_arr, color=colors_post[idx], linewidth=1.2, alpha=0.8,
                 label=f"{jid} (std$\\to$prem)")

    all_jobs = set(pi_series.keys())
    never_premium = all_jobs - pre_premium - post_premium
    for idx, jid in enumerate(sorted(never_premium)):
        if jid not in pi_series: continue
        t_arr = np.array([p[0] for p in pi_series[jid]])
        pi_arr = np.array([p[1] for p in pi_series[jid]])
        ax2.plot(t_arr, pi_arr, color='gray', linewidth=0.8, alpha=0.5,
                 linestyle=':', label=f"{jid} (always std)")

    for ax in [ax1, ax2]:
        ax.axvline(x=swap_time, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.set_ylabel("$\\pi$ (priority index)")
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.25)

    ax1.set_title("Pre-swap Premium $\\to$ Post-swap Standard", fontsize=10, loc='left')
    ax2.set_title("Pre-swap Standard $\\to$ Post-swap Premium", fontsize=10, loc='left')
    ax2.set_xlabel("Time (s)")

    caption = ("E3' D1 kill arm, seed=0. Vertical spikes ($t$=15--100 s) correspond to "
               "$\\pi$ resets upon SLO attainment events.")
    fig.text(0.5, 0.005, caption, ha='center', fontsize=7.5,
             color='#555555', style='italic')

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    path = os.path.join(OUT_DIR, "fig5_pi_timeseries")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# T-1: Anchor Baseline Table (LaTeX)
# ═══════════════════════════════════════════════════════════════

def draw_table1():
    print("\n=== T-1: Anchor Baseline Table ===")
    with open(os.path.join(ANCHOR_BASE, "per_policy_results.json")) as f:
        anchor = json.load(f)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline anchor results at 400 Gbps (3 seeds, 24 jobs, "
                 r"12 large / 8 medium / 4 small).}")
    lines.append(r"\label{tab:anchor}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Policy & Mean SAS & SLO Rate & Large SAS & Med SAS \\")
    lines.append(r"\midrule")

    for pol_name, pol_key in [("Max-Min Fair","Fair"),("CRUX","CRUX"),("SRPT","SP"),("DF","D1")]:
        if pol_key not in anchor:
            continue
        seeds = anchor[pol_key]["seeds"]
        n_s = len(seeds)
        overall_sas = np.mean([s["overall"]["mean_sas"] for s in seeds])
        overall_slo = np.mean([s["overall"]["slo_rate"] for s in seeds])
        large_sas = np.mean([s["tiers"]["large"]["mean_sas"] for s in seeds])
        med_sas = np.mean([s["tiers"]["medium"]["mean_sas"] for s in seeds])
        lines.append(f"{pol_name} & {overall_sas:.3f} & {overall_slo:.3f} & "
                     f"{large_sas:.3f} & {med_sas:.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(OUT_DIR, "table1_anchor_baseline.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {tex_path}")
    return tex_path

# ═══════════════════════════════════════════════════════════════
# T-2: E2-pro Orthogonal Table (LaTeX)
# ═══════════════════════════════════════════════════════════════

def draw_table2():
    print("\n=== T-2: E2-pro Orthogonal Table ===")
    data = load_e1_e2(os.path.join(FIG_REG, "fig3_e2_ladder_5seed.csv"))
    e2pro = data["E2-pro"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{E2-pro positive-control sanity results "
                 r"(CRUX-favorable workload, 5 seeds).}")
    lines.append(r"\label{tab:e2pro}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Policy & 630 Gbps & 800 Gbps \\")
    lines.append(r"\midrule")

    for pol in POLICY_ORDER:
        if pol not in e2pro:
            continue
        v630 = e2pro[pol][630]
        v800 = e2pro[pol][800]
        lines.append(f"{POLICY_LABEL[pol]} & "
                     f"{v630[0]*100:.1f}$\\pm${v630[1]*100:.1f}\\% & "
                     f"{v800[0]*100:.1f}$\\pm${v800[1]*100:.1f}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(OUT_DIR, "table2_e2pro.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {tex_path}")
    return tex_path

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("综合终版绘图：Fig-1~5 + T-1~2")
    print(f"输出: {OUT_DIR}")
    print("修正: 删图内标题 | 命名LongLiu/DF/SP/Fair | ddof=0 | PDF+PNG")
    print("=" * 60)

    draw_fig1_hero()
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()
    draw_table1()
    draw_table2()

    print("\nDone.")

if __name__ == "__main__":
    main()
