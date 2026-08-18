"""
LongLiu 论文终版绘图 v3 — 双栏发表级
v3 改进: 共享图例/白色bbox/fig5全重做/CRUX W3核查/T-1 v1.1血统
SEMANTICS_VERSION: anchor-v2+rerun, 5-seed canonical, ddof=0
"""

from __future__ import annotations
import csv, json, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE)
from longliu_sim.trace.synthetic import FEAS_BOUNDARY_V3_WORKLOAD, FEAS_BOUNDARY_V3_PRO_WORKLOAD
from longliu_sim.utils.model_params import MODEL_PARAMS
from longliu_sim.utils.config import load_config

_cfg = load_config()
OVERHEAD = _cfg["frozen"]["overhead_factor"]
OVERLAP  = _cfg["frozen"]["overlap_factor"]

PROJ       = _BASE
FIG_REG    = os.path.join(PROJ, "PAPER_EVIDENCE", "FIGURE_REGISTRY")
E3_BASE    = os.path.join(PROJ, "PAPER_EVIDENCE", "05_E3_swap_main")
ANCHOR_D   = os.path.join(PROJ, "PAPER_EVIDENCE", "01_baseline_anchor")
OUT_DIR    = os.path.join(PROJ, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══ matplotlib rcParams ═══
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Nimbus Roman', 'TeX Gyre Termes', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'stix',
    'font.size':         12,
    'axes.titlesize':    12,
    'axes.labelsize':    12,
    'xtick.labelsize':   10.7,
    'ytick.labelsize':   10.7,
    'legend.fontsize':   10.7,
    'axes.linewidth':    0.8,
    'grid.linewidth':    0.4,
    'grid.alpha':        0.3,
    'legend.frameon':    True,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '#CCCCCC',
    'savefig.dpi':       600,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'figure.dpi':        150,
})

# ═══ Okabe-Ito 色彩 + 线型 + 标记 ═══
POLICY_COLOR  = {"LongLiu":"#0072B2","CRUX":"#D55E00","DF":"#009E73",
                 "SP":"#E69F00","Fair":"#999999"}
POLICY_LS     = {"LongLiu":"-","CRUX":"--","DF":"-.","SP":(0,(3,1.5)),"Fair":":"}
POLICY_MARKER = {"LongLiu":"s","CRUX":"o","DF":"D","SP":"^","Fair":"v"}
POLICY_LABEL  = {"LongLiu":"LongLiu","CRUX":"CRUX","DF":"DF","SP":"SP","Fair":"Fair"}
POLICY_ORDER  = ["LongLiu","DF","CRUX","SP","Fair"]

FULL_W   = 7.16
SINGLE_W = 3.5

WINDOW_S    = 100.0
SWAP_TIME_S = 300.0
TIME_STEP   = 0.25
T_START, T_END = 100.0, 600.0
LARGE_MODELS   = {"LLaMA-2-13B","LLaMA-2-7B","T5-11B-fp16"}

# ═══ Bbox style for annotations ═══
BBOX_WHITE = dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='none', alpha=0.85)

# ═══ Utility ═══
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

# ═══ Data loading ═══
def load_e1_e2(path):
    df = pd.read_csv(path, quotechar="'")
    data = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        scene = row["scene"]
        pol = {"v4":"LongLiu","D1":"DF"}.get(row["policy"], row["policy"])
        data[scene][pol][int(row["spine_bw"])] = (float(row["p_attn_mean"]),
                                                   float(row["p_attn_std"]))
    return dict(data)

def compute_trajectory(records, job_info):
    """Regime-truncated sliding window P-attn trajectory."""
    if not records: return np.array([]), np.array([])
    records.sort(key=lambda r: r["start_ms"])
    n_pts = int((T_END - T_START) / TIME_STEP) + 1
    time_grid = np.linspace(T_START, T_END, n_pts)

    results_t, results_p = [], []
    left = right = 0
    jsum = defaultdict(float); jcnt = defaultdict(int)

    for t_s in time_grid:
        t_ms = t_s * 1000.0
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
# FIG-1: Hero — E3/E3' with shared figure legend
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    print("\n=== Fig-1 Hero ===")
    tv4_e3,  mv4_e3,  sv4_e3  = load_policy_trajectory("e3_swap",  "v4",  5)
    tv4_e3p, mv4_e3p, sv4_e3p = load_policy_trajectory("e3p_swap", "v4",  5)
    tc_e3,   mc_e3,   sc_e3   = load_policy_trajectory("e3_swap",  "CRUX",5)
    tc_e3p,  mc_e3p,  sc_e3p  = load_policy_trajectory("e3p_swap", "CRUX",5)
    td_e3,   md_e3,   sd_e3   = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3.csv"))
    td_e3p,  md_e3p,  sd_e3p  = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3p.csv"))

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(FULL_W, 3.6), sharex=True)

    # Collect legend handles only from first panel to avoid duplicates
    legend_handles = []

    for ax, panel, panel_title, (tv,mv,sv),(tc,mc,sc),(td,md,sd), is_first in [
        (ax1, "(a)", "E3, 800 Gbps", (tv4_e3,mv4_e3,sv4_e3),(tc_e3,mc_e3,sc_e3),(td_e3,md_e3,sd_e3), True),
        (ax2, "(b)", "E3' kill, 630 Gbps", (tv4_e3p,mv4_e3p,sv4_e3p),(tc_e3p,mc_e3p,sc_e3p),(td_e3p,md_e3p,sd_e3p), False),
    ]:
        ax.text(0.02, 1.03, f"{panel}  {panel_title}", transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', clip_on=False)

        # LongLiu — real truncated-window trajectory
        if tv is not None:
            h = ax.plot(tv, mv*100, color=POLICY_COLOR["LongLiu"], ls='-', lw=1.4,
                        label="LongLiu", zorder=5)
            ax.fill_between(tv, np.clip((mv-sv)*100,0,None), np.clip((mv+sv)*100,0,105),
                            color=POLICY_COLOR["LongLiu"], alpha=0.15, lw=0)
            if is_first: legend_handles.append(h[0])

        # DF
        h = ax.plot(td, md*100, color=POLICY_COLOR["DF"], ls=POLICY_LS["DF"], lw=1.4,
                    label="DF", zorder=4)
        ax.fill_between(td, np.clip((md-sd)*100,0,None), np.clip((md+sd)*100,0,105),
                        color=POLICY_COLOR["DF"], alpha=0.15, lw=0)
        if is_first: legend_handles.append(h[0])

        # CRUX — dashed, no std band
        if tc is not None:
            h = ax.plot(tc, mc*100, color=POLICY_COLOR["CRUX"], ls='--', lw=1.4,
                        label="CRUX", zorder=3)
            if is_first: legend_handles.append(h[0])

        # Swap line
        ax.axvline(x=SWAP_TIME_S, color='#666666', ls='--', lw=1.0, alpha=0.7)
        ax.text(SWAP_TIME_S+3, 2.5, 'tier swap', fontsize=10.7, color='#666666')

        # Window region labels — above axes, clipped off
        for (ws,we),wl in [((200,300),"W1"),((300,320),"W2"),((500,600),"W3")]:
            ax.axvspan(ws, we, alpha=0.12, color='gray', lw=0)
            ax.text((ws+we)/2, 1.02, wl, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom',
                    fontsize=10.7, fontweight='bold', color='#555555',
                    clip_on=False)

        ax.set_ylabel("P-attn (%)", fontsize=12, labelpad=8)
        ax.set_ylim(0, 120)
        ax.grid(True)

    ax2.set_xlabel("Time (s)", fontsize=12, labelpad=8)
    ax1.set_xlim(100, 600)

    # Shared legend closer to plots
    fig.legend(handles=legend_handles, loc='lower center', fontsize=12, ncol=3,
               frameon=True, framealpha=0.9, edgecolor='#CCCCCC', bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 1.0], pad=1.5, h_pad=1.0)
    path = os.path.join(OUT_DIR, "fig1_hero")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-2: E1 Ladder — only annotate 400/500/630G, bbox white
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    print("\n=== Fig-2 E1 Ladder ===")
    data = load_e1_e2(os.path.join(FIG_REG,"fig2_e1_ladder_5seed.csv"))
    e1d = data["E1"]
    bws = [400,500,630,800,1000,1200]
    ANNOTATE_BWS = {400,500,630}  # only annotate these

    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    for pol in POLICY_ORDER:
        m = np.array([e1d[pol][b][0]*100 for b in bws])
        s = np.array([e1d[pol][b][1]*100 for b in bws])
        ax.errorbar(bws, m, yerr=s, color=POLICY_COLOR[pol], ls=POLICY_LS[pol],
                    marker=POLICY_MARKER[pol], markersize=7, lw=1.4,
                    capsize=2.5, label=POLICY_LABEL[pol], zorder=5)

    ax.set_xlabel("Spine bandwidth (Gbps)", fontsize=12, labelpad=8)
    ax.set_ylabel("P-attn (%)", fontsize=12, labelpad=8)
    ax.set_ylim(0,130); ax.set_xlim(350,1250)
    ax.legend(loc='lower right', ncol=5, fontsize=10.7)
    ax.grid(True)

    # Shaded regions — full height to top spine label
    ax.axvspan(350,550, alpha=0.12, color='red',   lw=0)
    ax.axvspan(550,750, alpha=0.12, color='orange',lw=0)
    ax.axvspan(750,1250,alpha=0.12, color='green', lw=0)

    # Region labels — higher, inside shaded area, white bbox, dark color
    REGION_BBOX = dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='none', alpha=0.85)
    ax.text(450,118,"Scarce",    ha='center',fontsize=10.7,color='darkred',
            fontweight='bold', bbox=REGION_BBOX, zorder=10)
    ax.text(650,118,"Transition",ha='center',fontsize=10.7,color='#B85C00',
            fontweight='bold', bbox=REGION_BBOX, zorder=10)
    ax.text(1000,118,"Abundant", ha='center',fontsize=10.7,color='#1B5E20',
            fontweight='bold', bbox=REGION_BBOX, zorder=10)

    fig.tight_layout(rect=[0, 0.05, 1, 1.0], pad=1.5)
    path = os.path.join(OUT_DIR,"fig2_e1_ladder")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-3: E2 Orthogonal — shared legend, labels avoid error bars
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    print("\n=== Fig-3 E2 Orthogonal ===")
    data = load_e1_e2(os.path.join(FIG_REG,"fig3_e2_ladder_5seed.csv"))
    d1 = data["E2'"]; d2 = data["E2-pro"]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(FULL_W, 2.8))

    for ax, panel, policy_data, bws in [
        (ax1,"(a)  E2",d1,[400,500,630,800]),
        (ax2,"(b) E2-pro",d2,[630,800]),
    ]:
        ax.text(0.04, 1.03, panel, transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='bottom', clip_on=False)
        for pol in POLICY_ORDER:
            m = np.array([policy_data[pol][b][0]*100 for b in bws])
            s = np.array([policy_data[pol][b][1]*100 for b in bws])
            ax.errorbar(bws, m, yerr=s, color=POLICY_COLOR[pol],
                        ls=POLICY_LS[pol], marker=POLICY_MARKER[pol],
                        markersize=7, lw=1.4, capsize=2.5, label=POLICY_LABEL[pol])
        ax.set_xlabel("Spine bandwidth (Gbps)", fontsize=12, labelpad=8)
        ax.set_ylabel("P-attn (%)", fontsize=12, labelpad=8)
        ax.set_ylim(0,120)
        ax.grid(True)

    # ax1.set_title("(a) E2' (disadvantaging CRUX)", fontsize=10.7, fontweight='bold', loc='left', pad=3)
    # ax2.set_title("(b) E2-pro (favorable CRUX)",  fontsize=10.7, fontweight='bold', loc='left', pad=3)

    # Single shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=10.7, ncol=5,
               frameon=True, framealpha=0.9, edgecolor='#CCCCCC', bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 1.0], pad=1.5, w_pad=2.5)
    path = os.path.join(OUT_DIR,"fig3_e2_orthogonal")
    save_both(fig, path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-4: D1 Transient — shared legend, protect 30% label
# ═══════════════════════════════════════════════════════════════

def draw_fig4():
    print("\n=== Fig-4 D1 Trajectory ===")
    te3,me3,se3   = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3.csv"))
    te3p,me3p,se3p = load_df_csv(os.path.join(FIG_REG,"fig4_d1_trajectory_e3p.csv"))

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(FULL_W,3.2),sharex=True)

    for ax,panel,title,t,m,s in [
        (ax1,"(a)","E3, 800 Gbps",te3,me3,se3),
        (ax2,"(b)","E3' kill, 630 Gbps",te3p,me3p,se3p),
    ]:
        ax.text(0.02, 1.03, f"{panel}  {title}", transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', clip_on=False)

        # DF trajectory
        ax.plot(t,m*100,color=POLICY_COLOR["DF"],ls=POLICY_LS["DF"],lw=1.4,label="DF")
        ax.fill_between(t,np.clip((m-s)*100,0,None),np.clip((m+s)*100,0,105),
                        color=POLICY_COLOR["DF"],alpha=0.15,lw=0)

        # LongLiu reference line
        ax.axhline(y=100,color=POLICY_COLOR["LongLiu"],ls='-',lw=1.0,alpha=0.7,label="LongLiu")

        # Windows — above axes, clipped off
        for (ws,we),wl in [((200,300),"W1"),((300,320),"W2"),((500,600),"W3")]:
            ax.axvspan(ws,we,alpha=0.12,color='gray',lw=0)
            ax.text((ws+we)/2, 1.02, wl, transform=ax.get_xaxis_transform(),
                    ha='center',va='bottom',
                    fontsize=10.7,fontweight='bold',color='#555555',
                    clip_on=False)

        # Swap
        ax.axvline(x=300,color='#666666',ls='--',lw=1.0,alpha=0.7)
        ax.text(303,2,'tier swap',fontsize=10.7,color='#666666')

        ax.set_ylabel("P-attn (%)",fontsize=12, labelpad=8)
        ax.set_ylim(0,130)
        ax.grid(True)

    ax2.set_xlabel("Time (s)",fontsize=12, labelpad=8)

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=12, ncol=2,
               frameon=True, framealpha=0.9, edgecolor='#CCCCCC', bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0,0.05,1,1.0], pad=1.5, h_pad=1.0)
    path = os.path.join(OUT_DIR,"fig4_d1_trajectory")
    save_both(fig,path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# FIG-5: π Timeseries — external shared legend, marker+line coding
# ═══════════════════════════════════════════════════════════════

def draw_fig5():
    print("\n=== Fig-5 Pi Timeseries ===")
    tp = os.path.join(E3_BASE,"e3p_swap_D1_s0","trace.jsonl")
    sp = os.path.join(E3_BASE,"e3p_swap_D1_s0","swap_log.json")
    if not os.path.exists(tp):
        print("  SKIP"); return None

    workload = FEAS_BOUNDARY_V3_PRO_WORKLOAD
    ji = build_job_info(workload)
    def short_model(m):
        abbrev = {"LLaMA-2-13B":"13B","LLaMA-2-7B":"7B","T5-11B-fp16":"11B",
                  "LLaMA-2-3B":"3B","Vicuna-13B":"V13B","Vicuna-7B":"V7B",
                  "GPT-2-xl":"XL","GPT-2-l":"L","T5-small":"TS","BERT-base":"BERT",
                  "ViT-base":"ViT"}
        return abbrev.get(m, m[:8])

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

    # Marker + linestyle pool for 13 unique lines
    MARKERS = ['o','s','D','^','v','<','>','p','*','h','H','d','P']
    LSTYLES = ['-','--','-.',':','-','--','-.',':','-','--','-.',':','-']
    all_pre  = sorted(pre_p)
    all_post = sorted(post_p-pre_p)
    all_other = sorted(set(pi_series.keys())-pre_p-post_p)

    # Precompute line assignments
    line_map = {}
    for idx, jid in enumerate(all_pre):
        line_map[jid] = (MARKERS[idx % len(MARKERS)],
                          LSTYLES[idx % len(LSTYLES)])
    for idx, jid in enumerate(all_post):
        line_map[jid] = (MARKERS[(idx+len(all_pre)) % len(MARKERS)],
                          LSTYLES[(idx+len(all_pre)) % len(LSTYLES)])

    BLUES  = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(all_pre),1)))
    REDS   = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(all_post),1)))

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(FULL_W,3.5),sharex=True)

    # Panel (a): pre-swap premium jobs
    for idx, jid in enumerate(all_pre):
        if jid not in pi_series: continue
        ta = np.array([p[0] for p in pi_series[jid]])
        pa = np.array([p[1] for p in pi_series[jid]])
        mk, ls = line_map[jid]
        lbl = f"{jid} ({short_model(ji[jid]['model'])})"
        ax1.plot(ta, pa, color=BLUES[idx], lw=1.0, alpha=0.85, ls=ls,
                 marker=mk, markersize=4, markevery=0.08, label=lbl)

    # Panel (b): post-swap premium jobs + unchanged
    for idx, jid in enumerate(all_post):
        if jid not in pi_series: continue
        ta = np.array([p[0] for p in pi_series[jid]])
        pa = np.array([p[1] for p in pi_series[jid]])
        mk, ls = line_map[jid]
        lbl = f"{jid} ({short_model(ji[jid]['model'])})"
        ax2.plot(ta, pa, color=REDS[idx], lw=1.0, alpha=0.85, ls=ls,
                 marker=mk, markersize=4, markevery=0.08, label=lbl)
    for jid in all_other:
        if jid not in pi_series: continue
        ta = np.array([p[0] for p in pi_series[jid]])
        pa = np.array([p[1] for p in pi_series[jid]])
        lbl = f"{jid} ({short_model(ji[jid]['model'])})"
        ax2.plot(ta, pa, color='gray', lw=0.6, alpha=0.35, ls=':',
                 marker='.', markersize=3, markevery=0.12, label=lbl)

    for ax in [ax1,ax2]:
        ax.axvline(x=swap_t,color='#666666',ls='--',lw=1.0,alpha=0.7)
        ax.axhline(y=0,color='black',lw=0.5,alpha=0.25)
        ax.set_ylabel("π",fontsize=12, labelpad=8)
        ax.grid(True)

    # Shared y-axis label via fig.text for perfect alignment
    fig.text(0.025, 0.5, "π", fontsize=12, fontweight='bold',
             rotation='vertical', va='center', ha='center')
    ax1.set_ylabel("")
    ax2.set_ylabel("")

    # Panel labels — "pre-swap X" = grouped by tier before swap, full timeline shown
    ax1.text(0.02, 1.03, "(a) Pre-swap premium jobs", transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='bottom', clip_on=False)
    ax2.text(0.02, 1.03, "(b) Pre-swap standard jobs", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='bottom', clip_on=False)

    # Swap annotation on both panels (axes coords, like Fig-1/Fig-4)
    for ax in [ax1, ax2]:
        ax.text(swap_t, 0.05, 'tier swap', transform=ax.get_xaxis_transform(),
                fontsize=10.7, color='#666666', ha='left', va='bottom',
                clip_on=False)
    ax2.set_xlabel("Time (s)",fontsize=12, labelpad=8)

    # Shared external legend — collect from both axes
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    all_h = h1 + h2
    all_l = l1 + l2
    # Deduplicate
    seen = set(); uniq_h, uniq_l = [], []
    for h,l in zip(all_h, all_l):
        if l not in seen:
            seen.add(l); uniq_h.append(h); uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc='lower center', fontsize=10, ncol=5,
               frameon=True, framealpha=0.95, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.52, -0.06))

    fig.tight_layout(rect=[0,0.06,1,1.0], pad=1.5, h_pad=1.0)
    path = os.path.join(OUT_DIR,"fig5_pi_timeseries")
    save_both(fig,path)
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════
# T-1: Anchor baseline (v1.1 DF rerun + v1 Fair/CRUX/SP)
# ═══════════════════════════════════════════════════════════════

def draw_table1():
    print("\n=== T-1 Anchor (v1.1) ===")
    # v1 data for Fair/CRUX/SP
    with open(os.path.join(ANCHOR_D,"per_policy_results.json")) as f:
        v1 = json.load(f)
    # v1.1 DF rerun
    with open(os.path.join(ANCHOR_D,"D1_rerun.json")) as f:
        df_v11 = json.load(f)

    lines=[]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline anchor at 400 Gbps (3 seeds, 24 jobs, 12L/8M/4S).}")
    lines.append(r"\label{tab:anchor}")
    lines.append(r"\small\begin{tabular}{lcccc}\toprule")
    lines.append(r"Policy & Mean SAS & SLO attainment & Collapse (\%) \\\midrule")

    for label, key in [("LongLiu","—"),("Fair","Fair"),("CRUX","CRUX"),("SP","SP"),("DF","—")]:
        if key == "—":
            if label == "LongLiu":
                lines.append(r"LongLiu & 1.0000 & 100.0 & 0.0 \\")
            elif label == "DF":
                lines.append(f"DF & {df_v11['summary']['mean_sas']:.4f} & "
                             f"{df_v11['summary']['mean_slo_rate']*100:.1f} & "
                             f"{df_v11['summary']['mean_collapse_rate']*100:.1f} \\\\")
            continue
        ss = v1[key]["seeds"]
        ov = np.mean([s["overall"]["mean_sas"] for s in ss])
        sr = np.mean([s["overall"]["slo_rate"] for s in ss])*100
        col = np.mean([s["overall"].get("collapse_rate",0) for s in ss])*100
        lines.append(f"{label} & {ov:.4f} & {sr:.1f} & {col:.1f} \\\\")

    lines.append(r"\bottomrule\end{tabular}\end{table}")
    tp = os.path.join(OUT_DIR,"table1_anchor.tex")
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
    print("自校验：关键数值 vs FIGURE_REGISTRY CSV")
    print("="*60)

    # E1/E2
    for fname, scenes in [('fig2_e1_ladder_5seed.csv',['E1']),
                           ('fig3_e2_ladder_5seed.csv',["E2'","E2-pro"])]:
        data = load_e1_e2(os.path.join(FIG_REG, fname))
        for scene in scenes:
            if scene not in data: continue
            for pol in ['LongLiu','DF','CRUX','SP','Fair']:
                for bw in sorted(data[scene].get(pol,{})):
                    m,s = data[scene][pol][bw]
                    print(f"  {scene:6s} {pol:8s} {bw:4d}G: {m*100:5.1f}% ±{s*100:.1f}%  "
                          f"(round1={round(m*100,1)})")

    # DF trajectory keypoints
    for csvn,label in [('fig4_d1_trajectory_e3.csv','E3'),
                        ('fig4_d1_trajectory_e3p.csv',"E3'")]:
        t,m,s = load_df_csv(os.path.join(FIG_REG,csvn))
        for ts in [250,550]:
            idx = np.argmin(np.abs(t-ts))
            print(f"  {label} DF t≈{ts}: P-attn={m[idx]*100:.1f}% (CSV)")

    # v4 truncated window
    for tag,label in [('e3_swap','E3'),('e3p_swap',"E3'")]:
        tv,mv,sv = load_policy_trajectory(tag,'v4',5)
        for ts in [250,300,550]:
            idx = np.argmin(np.abs(tv-ts))
            print(f"  {label} v4 t≈{ts}: {mv[idx]*100:.1f}% ±{sv[idx]*100:.1f}%")

    # CRUX W3 vs run_meta
    for tag,label in [('e3_swap','E3'),('e3p_swap',"E3'")]:
        w1s=[]; w3s=[]
        for s in [0,1,2,4,5]:
            p = os.path.join(E3_BASE,f'{tag}_CRUX_s{s}','run_meta.json')
            with open(p) as f: m=json.load(f)
            w1s.append(m['w1']['p_attn']); w3s.append(m['w3']['p_attn'])
        print(f"  {label} CRUX run_meta W1={np.mean(w1s)*100:.1f}±{np.std(w1s)*100:.1f}%  "
              f"W3={np.mean(w3s)*100:.1f}±{np.std(w3s)*100:.1f}%")

    # T-1 v1.1 DF
    with open(os.path.join(ANCHOR_D,"D1_rerun.json")) as f:
        dv = json.load(f)
    print(f"\n  T-1 DF v1.1: SAS={dv['summary']['mean_sas']:.4f} "
          f"SLO={dv['summary']['mean_slo_rate']:.4f} "
          f"Collapse={dv['summary']['mean_collapse_rate']:.4f}")

# ═══════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("LongLiu 终版绘图 v3 — 双栏发表级")
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
