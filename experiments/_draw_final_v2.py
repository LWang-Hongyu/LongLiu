"""
LongLiu 论文终版绘图 v2 — INFOCOM 双栏正式发表级
规范：Okabe-Ito色 / Times+STIX / 截断窗v4轨迹 / 无图内标题 / IEEE栏宽
SEMANTICS_VERSION: anchor-v2, 5-seed canonical, ddof=0
"""

from __future__ import annotations
import csv, json, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS
from longliu_sim.utils.config import load_config

_cfg = load_config()
OVERHEAD = _cfg["frozen"]["overhead_factor"]
OVERLAP  = _cfg["frozen"]["overlap_factor"]

PROJ       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_REG    = os.path.join(PROJ, "PAPER_EVIDENCE", "FIGURE_REGISTRY")
E3_BASE    = os.path.join(PROJ, "PAPER_EVIDENCE", "05_E3_swap_main")
ANCHOR_D   = os.path.join(PROJ, "PAPER_EVIDENCE", "01_baseline_anchor")
OUT_DIR    = os.path.join(PROJ, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Matplotlib rcParams — IEEE 双栏发表级
# ═══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':    'serif',
    'font.serif':     ['Nimbus Roman', 'TeX Gyre Termes', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'stix',
    'font.size':       9,
    'axes.titlesize':  9,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth':  0.8,
    'grid.linewidth':  0.4,
    'grid.alpha':      0.3,
    'legend.frameon':  True,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '#CCCCCC',
    'savefig.dpi':      600,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'figure.dpi':       150,
})

# ═══════════════════════════════════════════════════════════════
# Okabe-Ito 色彩 + 线型 + 标记 (全局唯一映射)
# ═══════════════════════════════════════════════════════════════
POLICY_COLOR    = {"LongLiu":"#0072B2","CRUX":"#D55E00","DF":"#009E73",
                   "SP":"#E69F00","Fair":"#999999"}
POLICY_LS       = {"LongLiu":"-","CRUX":"--","DF":"-.","SP":(0,(3,1.5)),"Fair":":"}
POLICY_MARKER   = {"LongLiu":"s","CRUX":"o","DF":"D","SP":"^","Fair":"v"}
POLICY_LABEL    = {"LongLiu":"LongLiu","CRUX":"CRUX","DF":"DF","SP":"SP","Fair":"Fair"}
POLICY_ORDER    = ["LongLiu","DF","CRUX","SP","Fair"]

# Bandwidth labels
BW_LABELS   = {400:"400 Gbps",500:"500 Gbps",630:"630 Gbps",
               800:"800 Gbps",1000:"1000 Gbps",1200:"1200 Gbps"}

# IEEE 栏宽
FULL_W  = 7.16   # inches
FULL_H  = 3.0    # max height
SINGLE_W = 3.5

WINDOW_S    = 100.0
SWAP_TIME_S = 300.0
TIME_STEP   = 0.25
T_START, T_END = 100.0, 600.0
LARGE_MODELS = {"LLaMA-2-13B","LLaMA-2-7B","T5-11B-fp16"}

# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════

def save_both(fig, stem):
    fig.savefig(f"{stem}.pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f"{stem}.png", bbox_inches='tight', pad_inches=0.02)
    print(f"  -> {os.path.basename(stem)}.pdf + .png")

def get_target(comp_ms, comm_solo_ms, ci):
    comm_budget = ci * comm_solo_ms * OVERHEAD
    if OVERLAP > 0:
        return max(comp_ms, comm_budget) + (1.0-OVERLAP)*min(comp_ms, comm_budget)
    return comp_ms + comm_budget

def build_job_info(workload_raw):
    info = {}
    for i, (model, dp, orig_ci) in enumerate(workload_raw):
        jid = f"J{i}"
        p = MODEL_PARAMS[model]
        bpp = 2 if p.get("fp16",True) else 4
        mb = 2 * p["params"] * bpp / max(dp,1) / (1024*1024)
        raw_comm = mb * 8 * 1024 * 1024 / 100e9 * 1000.0
        comp = p.get("comp_ms",50.0)
        was_p = orig_ci <= 2.0
        post_ci = 3.0 if was_p else (1.5 if model in LARGE_MODELS or dp!=4 else 2.0)
        info[jid] = {
            "model": model, "dp": dp,
            "pre_target": get_target(comp, raw_comm, orig_ci),
            "post_target": get_target(comp, raw_comm, post_ci),
            "pre_is_premium": was_p,
            "post_is_premium": not was_p,
        }
    return info

# ═══════════════════════════════════════════════════════════════
# Data: E1/E2 CSV
# ═══════════════════════════════════════════════════════════════

def load_e1_e2(path):
    df = pd.read_csv(path, quotechar="'")
    data = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        scene = row["scene"]
        pol = {"v4":"LongLiu","D1":"DF"}.get(row["policy"], row["policy"])
        data[scene][pol][int(row["spine_bw"])] = (float(row["p_attn_mean"]),
                                                   float(row["p_attn_std"]))
    return dict(data)

# ═══════════════════════════════════════════════════════════════
# Sliding window trajectory — regime-boundary truncation
# ═══════════════════════════════════════════════════════════════

def compute_trajectory(records, job_info):
    """Sliding window P-attn with regime truncation at swap.
    t ≤ 300: window = [t-100, t]
    t > 300: window = [max(300, t-100), t] (truncated at regime boundary)
    """
    if not records:
        return np.array([]), np.array([])
    records.sort(key=lambda r: r["start_ms"])
    n_pts = int((T_END - T_START) / TIME_STEP) + 1
    time_grid = np.linspace(T_START, T_END, n_pts)
    w_ms = WINDOW_S * 1000.0

    results_t, results_p = [], []
    left = right = 0
    jsum = defaultdict(float);  jcnt = defaultdict(int)

    for t_s in time_grid:
        t_ms = t_s * 1000.0
        # regime-truncated window lower bound
        lo = max(0.0, (SWAP_TIME_S if t_s > SWAP_TIME_S else t_s - WINDOW_S) * 1000.0)
        hi = t_ms

        while left < len(records) and records[left]["start_ms"] < lo:
            r = records[left]; jid = r["jid"]
            if jid in jcnt:
                jsum[jid] -= r["iter_ms"]; jcnt[jid] -= 1
                if jcnt[jid] <= 0: jsum.pop(jid,None); jcnt.pop(jid,None)
            left += 1
        while right < len(records) and records[right]["start_ms"] <= hi:
            r = records[right]; jid = r["jid"]
            jsum[jid] += r["iter_ms"]; jcnt[jid] = jcnt.get(jid,0) + 1
            right += 1

        if t_s <= SWAP_TIME_S:
            pset = {j for j,info in job_info.items() if info["pre_is_premium"]}
            tkey = "pre_target"
        else:
            pset = {j for j,info in job_info.items() if info["post_is_premium"]}
            tkey = "post_target"

        ptot = patt = 0
        for jid in pset:
            if jid in jcnt and jcnt[jid] > 0:
                sas = job_info[jid][tkey] / (jsum[jid]/jcnt[jid])
                ptot += 1
                if sas >= 0.98: patt += 1
        if ptot > 0:
            results_t.append(t_s)
            results_p.append(patt/ptot)
    return np.array(results_t), np.array(results_p)

def load_policy_trajectory(tag, pol, n_seeds):
    """Load records.jsonl for a policy, compute 5-seed trajectory."""
    workload = FEAS_BOUNDARY_V3_WORKLOAD if tag=="e3_swap" else FEAS_BOUNDARY_V3_PRO_WORKLOAD
    job_info = build_job_info(workload)
    seeds = list(range(n_seeds)) if pol=="D1" else [0,1,2,4,5]
    seed_trajs = []
    for s in seeds:
        path = os.path.join(E3_BASE, f"{tag}_{pol}_s{s}", "records.jsonl")
        if not os.path.exists(path): continue
        recs = [json.loads(l) for l in open(path) if l.strip()]
        t, p = compute_trajectory(recs, job_info)
        seed_trajs.append((t,p))
    if not seed_trajs: return None,None,None
    all_t = sorted(set(t for t_arr,_ in seed_trajs for t in t_arr))
    common = np.array(all_t)
    interp = np.array([np.interp(common, t_arr, p_arr) for t_arr,p_arr in seed_trajs])
    return common, np.mean(interp,axis=0), np.std(interp,axis=0)

def load_df_csv(path):
    times, means, stds = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["time_s"]))
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
    return np.array(times), np.array(means), np.array(stds)

# ═══════════════════════════════════════════════════════════════
# FIG-1: Hero — E3/E3' 三策略截断窗轨迹
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    print("\n=== Fig-1 Hero ===")
    # v4: real truncated-window trajectory
    tv4_e3,  mv4_e3,  sv4_e3  = load_policy_trajectory("e3_swap",  "v4",  5)
    tv4_e3p, mv4_e3p, sv4_e3p = load_policy_trajectory("e3p_swap", "v4",  5)
    # CRUX
    tc_e3,   mc_e3,   sc_e3   = load_policy_trajectory("e3_swap",  "CRUX",5)
    tc_e3p,  mc_e3p,  sc_e3p  = load_policy_trajectory("e3p_swap", "CRUX",5)
    # DF (from CSV)
    td_e3,   md_e3,   sd_e3   = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3.csv"))
    td_e3p,  md_e3p,  sd_e3p  = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3p.csv"))

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(FULL_W, 3.2), sharex=True)

    for ax, panel, panel_title, (tv,mv,sv),(tc,mc,sc),(td,md,sd) in [
        (ax1, "(a)", "E3, 800 Gbps", (tv4_e3,mv4_e3,sv4_e3),(tc_e3,mc_e3,sc_e3),(td_e3,md_e3,sd_e3)),
        (ax2, "(b)", "E3' kill, 630 Gbps", (tv4_e3p,mv4_e3p,sv4_e3p),(tc_e3p,mc_e3p,sc_e3p),(td_e3p,md_e3p,sd_e3p)),
    ]:
        # Panel label: (a)/(b) + sub-title as left-aligned text
        ax.text(0.02, 0.94, f"{panel}  {panel_title}", transform=ax.transAxes,
                fontsize=9, fontweight='bold')

        # LongLiu — real truncated-window trajectory (no hand-drawn constant)
        if tv is not None:
            ax.plot(tv, mv*100, color=POLICY_COLOR["LongLiu"], ls='-', lw=1.4,
                    label="LongLiu", zorder=5)
            ax.fill_between(tv, np.clip((mv-sv)*100,0,None), np.clip((mv+sv)*100,0,105),
                            color=POLICY_COLOR["LongLiu"], alpha=0.15, lw=0)

        # DF
        ax.plot(td, md*100, color=POLICY_COLOR["DF"], ls=POLICY_LS["DF"], lw=1.4,
                label="DF", zorder=4)
        ax.fill_between(td, np.clip((md-sd)*100,0,None), np.clip((md+sd)*100,0,105),
                        color=POLICY_COLOR["DF"], alpha=0.15, lw=0)

        # CRUX — dashed line per spec, no std band
        if tc is not None:
            ax.plot(tc, mc*100, color=POLICY_COLOR["CRUX"], ls='--', lw=1.4,
                    label="CRUX", zorder=3)

        # Swap line
        ax.axvline(x=SWAP_TIME_S, color='#666666', ls='--', lw=1.0, alpha=0.7)
        ax.text(SWAP_TIME_S+3, 2.5, 'tier swap', fontsize=8, color='#666666')

        # Window region labels
        for (ws,we),wl in [((200,300),"W1"),((300,320),"W2"),((500,600),"W3")]:
            ax.axvspan(ws, we, alpha=0.12, color='gray', lw=0)
            ax.text((ws+we)/2, 104, wl, ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#555555')

        ax.set_ylabel("P-attn (%)", fontsize=9)
        ax.set_ylim(-5, 115)
        ax.legend(fontsize=8, loc='lower right', ncol=3)
        ax.grid(True)

    ax2.set_xlabel("Time (s)", fontsize=9)
    ax1.set_xlim(100, 600)

    fig.tight_layout(rect=[0,0,1,0.98])
    path = os.path.join(OUT_DIR, "fig1_hero")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-2: E1 Ladder
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    print("\n=== Fig-2 E1 Ladder ===")
    data = load_e1_e2(os.path.join(FIG_REG,"fig2_e1_ladder_5seed.csv"))
    e1d = data["E1"]
    bws = [400,500,630,800,1000,1200]

    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    for pol in POLICY_ORDER:
        m = np.array([e1d[pol][b][0]*100 for b in bws])
        s = np.array([e1d[pol][b][1]*100 for b in bws])
        ax.errorbar(bws, m, yerr=s, color=POLICY_COLOR[pol], ls=POLICY_LS[pol],
                    marker=POLICY_MARKER[pol], markersize=7, lw=1.4,
                    capsize=2.5, label=POLICY_LABEL[pol], zorder=5)
        if pol=="LongLiu":
            for bw, mv in zip(bws, m):
                ax.annotate(f"{mv:.1f}", (bw, mv), textcoords="offset points",
                            xytext=(0,-14), ha='center', va='top', fontsize=8,
                            color=POLICY_COLOR[pol], fontweight='bold')

    ax.set_xlabel("Spine bandwidth (Gbps)", fontsize=9)
    ax.set_ylabel("P-attn (%)", fontsize=9)
    ax.set_ylim(0,110); ax.set_xlim(350,1250)
    ax.legend(loc='lower right', ncol=5, fontsize=8)
    ax.grid(True)

    ax.axvspan(350,550, alpha=0.10, color='red',   lw=0)
    ax.axvspan(550,750, alpha=0.10, color='orange',lw=0)
    ax.axvspan(750,1250,alpha=0.10, color='green', lw=0)
    ax.text(450,107,"Scarce",    ha='center',fontsize=8,color='red',   alpha=0.7)
    ax.text(650,107,"Transition",ha='center',fontsize=8,color='orange',alpha=0.7)
    ax.text(1000,107,"Abundant", ha='center',fontsize=8,color='green', alpha=0.7)

    fig.tight_layout()
    path = os.path.join(OUT_DIR,"fig2_e1_ladder")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-3: E2 Orthogonal
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    print("\n=== Fig-3 E2 Orthogonal ===")
    data = load_e1_e2(os.path.join(FIG_REG,"fig3_e2_ladder_5seed.csv"))
    d1 = data["E2'"]; d2 = data["E2-pro"]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(FULL_W, 2.6))

    for ax, panel, policy_data, bws in [
        (ax1,"(a)",d1,[400,500,630,800]),
        (ax2,"(b)",d2,[630,800]),
    ]:
        ax.text(0.04,0.94, panel, transform=ax.transAxes, fontsize=9, fontweight='bold')
        for pol in POLICY_ORDER:
            m = np.array([policy_data[pol][b][0]*100 for b in bws])
            s = np.array([policy_data[pol][b][1]*100 for b in bws])
            ax.errorbar(bws, m, yerr=s, color=POLICY_COLOR[pol],
                        ls=POLICY_LS[pol], marker=POLICY_MARKER[pol],
                        markersize=7, lw=1.4, capsize=2.5, label=POLICY_LABEL[pol])
            if pol=="LongLiu":
                for bw,mv in zip(bws,m):
                    ax.annotate(f"{mv:.1f}",(bw,mv),textcoords="offset points",
                                xytext=(0,-14),ha='center',va='top',fontsize=8,
                                color=POLICY_COLOR[pol],fontweight='bold')
        ax.set_xlabel("Spine bandwidth (Gbps)", fontsize=9)
        ax.set_ylabel("P-attn (%)", fontsize=9)
        ax.set_ylim(0,110)
        ax.grid(True)

    ax1.set_title("E2' (disadvantaging CRUX)", fontsize=8, fontweight='bold', loc='left', pad=2)
    ax2.set_title("E2-pro (favorable CRUX)",  fontsize=8, fontweight='bold', loc='left', pad=2)
    # Single shared legend (remove per-panel duplicates)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=8, ncol=5,
               frameon=True, framealpha=0.9, edgecolor='#CCCCCC')

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR,"fig3_e2_orthogonal")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-4: D1 Transient Trajectory
# ═══════════════════════════════════════════════════════════════

def draw_fig4():
    print("\n=== Fig-4 D1 Trajectory ===")
    te3,me3,se3   = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3.csv"))
    te3p,me3p,se3p = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3p.csv"))

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(FULL_W,3.0),sharex=True)

    for ax,panel,title,t,m,s in [
        (ax1,"(a)","E3, 800 Gbps",te3,me3,se3),
        (ax2,"(b)","E3' kill, 630 Gbps",te3p,me3p,se3p),
    ]:
        ax.text(0.02,0.94,f"{panel}  {title}",transform=ax.transAxes,
                fontsize=9,fontweight='bold')

        # DF
        ax.plot(t,m*100,color=POLICY_COLOR["DF"],ls=POLICY_LS["DF"],lw=1.4,label="DF")
        ax.fill_between(t,np.clip((m-s)*100,0,None),np.clip((m+s)*100,0,105),
                        color=POLICY_COLOR["DF"],alpha=0.15,lw=0)

        # LongLiu reference line (no annotation — added to legend)
        ax.axhline(y=100,color=POLICY_COLOR["LongLiu"],ls='-',lw=1.0,alpha=0.7)

        # Windows
        for (ws,we),wl in [((200,300),"W1"),((300,320),"W2"),((500,600),"W3")]:
            ax.axvspan(ws,we,alpha=0.12,color='gray',lw=0)
            ax.text((ws+we)/2,104,wl,ha='center',va='bottom',
                    fontsize=8,fontweight='bold',color='#555555')

        # Swap
        ax.axvline(x=300,color='#666666',ls='--',lw=1.0,alpha=0.7)
        ax.text(303,2,'tier swap',fontsize=8,color='#666666')

        # Annotate W1/W3
        i1=np.argmin(np.abs(t-250)); i3=np.argmin(np.abs(t-550))
        ax.annotate(f"{m[i1]*100:.1f}%",(t[i1],m[i1]*100),
                    textcoords="offset points",xytext=(12,8),
                    fontsize=8,fontweight='bold',color=POLICY_COLOR["DF"])
        ax.annotate(f"{m[i3]*100:.1f}%",(t[i3],m[i3]*100),
                    textcoords="offset points",xytext=(12,-14),
                    fontsize=8,fontweight='bold',color=POLICY_COLOR["DF"])

        ax.set_ylabel("P-attn (%)",fontsize=9)
        ax.set_ylim(-5,115)
        # Manual legend: DF + LongLiu reference
        leg_handles = [
            Line2D([0],[0],color=POLICY_COLOR["DF"],ls=POLICY_LS["DF"],lw=1.4,label='DF'),
            Line2D([0],[0],color=POLICY_COLOR["LongLiu"],ls='-',lw=1.0,label='LongLiu'),
        ]
        ax.legend(handles=leg_handles,fontsize=8,loc='lower right')
        ax.grid(True)

    ax2.set_xlabel("Time (s)",fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.98])
    path = os.path.join(OUT_DIR,"fig4_d1_trajectory")
    save_both(fig,path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-5: π Time Series
# ═══════════════════════════════════════════════════════════════

def draw_fig5():
    print("\n=== Fig-5 Pi Timeseries ===")
    tp = os.path.join(E3_BASE,"e3p_swap_D1_s0","trace.jsonl")
    sp = os.path.join(E3_BASE,"e3p_swap_D1_s0","swap_log.json")
    if not os.path.exists(tp):
        print("  SKIP"); return None

    # Build model shorthand for combined labels
    workload = FEAS_BOUNDARY_V3_PRO_WORKLOAD
    ji = build_job_info(workload)
    def short_model(m):
        abbrev = {"LLaMA-2-13B":"13B","LLaMA-2-7B":"7B","T5-11B-fp16":"11B",
                  "LLaMA-2-3B":"3B","Vicuna-13B":"V13B","Vicuna-7B":"V7B",
                  "GPT-2-xl":"XL","GPT-2-l":"L","T5-small":"TS","BERT-base":"BERT",
                  "ViT-base":"ViT"}
        return abbrev.get(m, m[:8])
    jlab = {jid: f"{jid} ({short_model(ji[jid]['model'])})" for jid in ji}

    pi_series = defaultdict(list)
    with open(tp) as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            ts = rec.get("time_ms",0)/1000.0
            for k,v in rec.items():
                if k.endswith("_pi") and k[0]=="J":
                    pi_series[k[:-3]].append((ts,v))
    if not pi_series: print("  SKIP"); return None
    for jid in pi_series: pi_series[jid].sort()

    with open(sp) as f: slog = json.load(f)
    swap_t = slog["swap_time_ms"]/1000.0
    pre_p  = {s["jid"] for s in slog["swaps"] if s["old_ci"]<=2.0}
    post_p = {s["jid"] for s in slog["swaps"] if s["new_ci"]<=2.0}

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(FULL_W,3.0),sharex=True)
    cpre  = plt.cm.Blues(np.linspace(0.4,0.9,len(pre_p)))
    cpost = plt.cm.Reds(np.linspace(0.4,0.9,len(post_p)))

    for idx,jid in enumerate(sorted(pre_p)):
        if jid not in pi_series: continue
        ta=np.array([p[0] for p in pi_series[jid]])
        pa=np.array([p[1] for p in pi_series[jid]])
        ax1.plot(ta,pa,color=cpre[idx],lw=1.0,alpha=0.8,label=jlab[jid])
    for idx,jid in enumerate(sorted(post_p-pre_p)):
        if jid not in pi_series: continue
        ta=np.array([p[0] for p in pi_series[jid]])
        pa=np.array([p[1] for p in pi_series[jid]])
        ax2.plot(ta,pa,color=cpost[idx],lw=1.0,alpha=0.8,label=jlab[jid])

    allj = set(pi_series.keys())
    for idx,jid in enumerate(sorted(allj-pre_p-post_p)):
        if jid not in pi_series: continue
        ta=np.array([p[0] for p in pi_series[jid]])
        pa=np.array([p[1] for p in pi_series[jid]])
        ax2.plot(ta,pa,color='gray',lw=0.6,alpha=0.4,ls=':',label=jlab[jid])

    for ax in [ax1,ax2]:
        ax.axvline(x=swap_t,color='#666666',ls='--',lw=1.0,alpha=0.7)
        ax.axhline(y=0,color='black',lw=0.5,alpha=0.25)
        ax.set_ylabel("π",fontsize=9)
        ax.legend(fontsize=8,ncol=2,loc='upper left')
        ax.grid(True)

    # Panel labels + compressed titles in one line (no overlap)
    ax1.text(0.02,0.94,"(a) Pre-swap premium",transform=ax1.transAxes,
             fontsize=9,fontweight='bold')
    ax2.text(0.02,0.94,"(b) Pre-swap standard",transform=ax2.transAxes,
             fontsize=9,fontweight='bold')
    ax2.set_xlabel("Time (s)",fontsize=9)

    fig.tight_layout(rect=[0,0,1,0.98])
    path = os.path.join(OUT_DIR,"fig5_pi_timeseries")
    save_both(fig,path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# T-1: Anchor baseline (LaTeX)
# ═══════════════════════════════════════════════════════════════

def draw_table1():
    print("\n=== T-1 Anchor ===")
    with open(os.path.join(ANCHOR_D,"per_policy_results.json")) as f:
        a = json.load(f)
    lines=[]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline anchor at 400 Gbps (3 seeds, 24 jobs).}")
    lines.append(r"\label{tab:anchor}")
    lines.append(r"\small\begin{tabular}{lcccc}\toprule")
    lines.append(r"Policy & Mean SAS & SLO rate & Large SAS & Med SAS \\\midrule")
    for pn,pk in [("Max-Min Fair","Fair"),("CRUX","CRUX"),("SRPT","SP"),("DF","D1")]:
        if pk not in a: continue
        ss=a[pk]["seeds"]
        ov=np.mean([s["overall"]["mean_sas"] for s in ss])
        sr=np.mean([s["overall"]["slo_rate"] for s in ss])
        ls=np.mean([s["tiers"]["large"]["mean_sas"] for s in ss])
        ms=np.mean([s["tiers"]["medium"]["mean_sas"] for s in ss])
        lines.append(f"{pn} & {ov:.3f} & {sr:.3f} & {ls:.3f} & {ms:.3f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    tp=os.path.join(OUT_DIR,"table1_anchor.tex")
    with open(tp,"w") as f: f.write("\n".join(lines)+"\n")
    print(f"  -> {tp}")
    return tp

def draw_table2():
    print("\n=== T-2 E2-pro ===")
    d=load_e1_e2(os.path.join(FIG_REG,"fig3_e2_ladder_5seed.csv"))["E2-pro"]
    lines=[r"\begin{table}[t]",r"\centering",
           r"\caption{E2-pro positive control (CRUX-favorable, 5 seeds).}",
           r"\label{tab:e2pro}",r"\small\begin{tabular}{lcc}\toprule",
           r"Policy & 630 Gbps & 800 Gbps \\\midrule"]
    for pol in POLICY_ORDER:
        if pol not in d: continue
        v6=d[pol][630]; v8=d[pol][800]
        lines.append(f"{POLICY_LABEL[pol]} & {v6[0]*100:.1f}$\\pm${v6[1]*100:.1f}\\% & "
                     f"{v8[0]*100:.1f}$\\pm${v8[1]*100:.1f}\\% \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    tp=os.path.join(OUT_DIR,"table2_e2pro.tex")
    with open(tp,"w") as f: f.write("\n".join(lines)+"\n")
    print(f"  -> {tp}")
    return tp

# ═══════════════════════════════════════════════════════════════
# 自校验
# ═══════════════════════════════════════════════════════════════

def self_check():
    print("\n"+"="*60)
    print("自校验：轨迹关键点 vs FIGURE_REGISTRY CSV")
    print("="*60)

    # Fig-4 DF trajectory
    for tag,csvn,label in [("e3","fig4_d1_trajectory_e3.csv","E3"),
                            ("e3p","fig4_d1_trajectory_e3p.csv","E3'")]:
        t,m,s=load_df_csv(os.path.join(FIG_REG,csvn))
        for ts,wn in [(250,"W1"),(310,"W2_early"),(365,"W2_peak"),(550,"W3")]:
            idx=np.argmin(np.abs(t-ts))
            print(f"  {label} DF {wn}: t={t[idx]:.1f} P-attn={m[idx]:.4f} (CSV)")

    # v4 trajectory (real, from records)
    print("\n  v4 truncated-window (from records.jsonl):")
    for tag,label in [("e3_swap","E3"),("e3p_swap","E3'")]:
        tv,mv,sv=load_policy_trajectory(tag,"v4",5)
        for ts in [250,300,310,350,400,550]:
            idx=np.argmin(np.abs(tv-ts))
            print(f"    {label} v4 t={tv[idx]:.1f}: P-attn={mv[idx]:.4f}±{sv[idx]:.4f}")

    # CRUX W1/W3 vs run_meta
    print("\n  CRUX trajectory vs run_meta:")
    for tag,label in [("e3_swap","E3"),("e3p_swap","E3'")]:
        tc,mc,sc=load_policy_trajectory(tag,"CRUX",5)
        i300=np.argmin(np.abs(tc-300))
        i550=np.argmin(np.abs(tc-550))
        print(f"    {label} CRUX t=300: {mc[i300]:.4f}  t=550: {mc[i550]:.4f}")
        # verify vs run_meta
        w1s=[]; w3s=[]
        for s in [0,1,2,4,5]:
            p=os.path.join(E3_BASE,f"{tag}_CRUX_s{s}","run_meta.json")
            with open(p) as f: m=json.load(f)
            w1s.append(m["w1"]["p_attn"]); w3s.append(m["w3"]["p_attn"])
        print(f"    {label} run_meta W1={np.mean(w1s):.4f} W3={np.mean(w3s):.4f}")

# ═══════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("LongLiu 终版绘图 v2 — INFOCOM 双栏发表级")
    print("="*60)
    draw_fig1()
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()
    draw_table1()
    draw_table2()
    self_check()
    print("\nDone.")

if __name__=="__main__":
    main()
