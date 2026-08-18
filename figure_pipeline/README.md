# Figure Pipeline（图表/证据管线）

统一的图表与证据归档目录：**运行脚本 → 数据 → 图片 → LaTeX** 全链路索引，
保证论文中每个数字可溯源、每个实验可复现。

```
figure_pipeline/
├── README.md            # 本文件：全链索引 + 复现指南
├── scripts/             # 绘图脚本（生成论文图）
│   ├── _draw_e10_wfs.py ... _draw_e16_beta.py   # E 系列图
│   ├── _draw_final_v3.py         # fig1-fig5 + table1/2
│   ├── _draw_trace_compare.py    # fig6（Lingjun trace）
│   └── experiments/              # 正式实验运行脚本（软链到 ../experiments/）
├── data/                # 数据源（实验 summary + 权威 CSV）
│   ├── e10_wfs/ e11_overlap/ e11b_overlap_waste/ e12_dscp/ e13_window/
│   ├── e14_probe/ e15_straggler/ e16_beta/       # E 系列 summary.csv
│   ├── figure_registry/          # fig2/3/4/6 权威数据 CSV
│   ├── e3_swap/                  # E3 Swap 轨迹（软链，原始 2.5G）
│   ├── anchor/                   # 锚点基线 per_policy_results（Fair/CRUX/SP/DF）
│   └── evidence/                 # 论文补充证据（per-seed 明细，见 §3）
└── figs/                # 图片输出（fig*.pdf/png, table*.tex）
```

---

## 1. 实验 → 脚本 → 数据 → 图片 → LaTeX 全链索引

> 运行命令统一在仓库根目录执行：`cd /home/why/LongLiu_rebuild/sim-nextgen`

### E 系列正式实验（5-seed，E14 含 10-seed）

| 实验 | 运行脚本 | 数据目录（`outputs/`） | 图片 | LaTeX 章节 |
|---|---|---|---|---|
| E10 WFS 基线 | `exp_e10_wfs.py --seeds 5` | `e10_wfs/` | `fig_e10_wfs` | `paper/evaluation_e10_e15.tex` §E10 |
| E11 Overlap | `exp_e11_overlap.py --seeds 5` | `e11_overlap/` | `fig_e11_overlap` | 同上 §E11 |
| E11b 分配精度 | `exp_e11b_overlap_waste.py --seeds 5` | `e11b_overlap_waste/` | `fig_e11b_overlap_waste` | `paper/evaluation_e11b_e16.tex` §E11b |
| E12 DSCP 量化 | `exp_e12_dscp.py --seeds 5` | `e12_dscp/` | `fig_e12_dscp` | `paper/evaluation_e10_e15.tex` §E12 |
| E13 窗口大小 | `exp_e13_window.py --seeds 5` | `e13_window/` | `fig_e13_window` | 同上 §E13 |
| E14 锚冻结 | `exp_e14_probe.py --seeds 10` | `e14_probe/` | `fig_e14_anchor` | `paper/evaluation_e14_anchor.tex` |
| E14 v4 对照 | `exp_e14_probe_v4.py --seeds 10` | `e14_probe/v4_closure_800g_s*/` | — | `paper/evaluation_e14_anchor.tex` |
| E15 Straggler | `exp_e15_straggler.py --seeds 5` | `e15_straggler/` | `fig_e15_straggler` | `paper/evaluation_e10_e15.tex` §E15 |
| E16 β 敏感性 | `exp_e16_beta.py --seeds 5` | `e16_beta/` | `fig_e16_beta` | `paper/evaluation_e11b_e16.tex` §E16 |
| Anchor v4 补跑 | `python scripts/run_anchor_v4.py` | `anchor_v4/` | — | tab:anchor（Table 1） |

> 软链版本位于 `scripts/experiments/`，仅作统一浏览入口；运行请使用上表
> `experiments/` 下的原始路径（脚本依赖自身路径定位项目根目录）。

### 论文主图（fig1–fig6）

| 图片 | 绘制脚本 | 数据源 |
|---|---|---|
| `fig1_hero` | `_draw_final_v3.py` | `data/e3_swap/` + `data/figure_registry/fig4_d1_*.csv` |
| `fig2_e1_ladder` | 同上 | `data/figure_registry/fig2_e1_ladder_5seed.csv` |
| `fig3_e2_orthogonal` | 同上 | `data/figure_registry/fig3_e2_ladder_5seed.csv` |
| `fig4_d1_trajectory` | 同上 | `data/figure_registry/fig4_d1_trajectory_e3*.csv` |
| `fig5_pi_timeseries` | 同上 | `data/e3_swap/`（π 轨迹） |
| `fig6_trace_compare` | `_draw_trace_compare.py` | `data/figure_registry/fig6_trace_compare.csv` |

---

## 2. 复现步骤

```bash
cd /home/why/LongLiu_rebuild/sim-nextgen

# 1) 正式实验（E 系列）
python3 experiments/exp_e10_wfs.py --seeds 5
python3 experiments/exp_e11_overlap.py --seeds 5
python3 experiments/exp_e11b_overlap_waste.py --seeds 5
python3 experiments/exp_e12_dscp.py --seeds 5
python3 experiments/exp_e13_window.py --seeds 5
python3 experiments/exp_e14_probe.py --seeds 10
python3 experiments/exp_e14_probe_v4.py --seeds 10   # E14 v4 闭式解对照
python3 experiments/exp_e15_straggler.py --seeds 5
python3 experiments/exp_e16_beta.py --seeds 5
python3 scripts/run_anchor_v4.py                     # tab:anchor v4 行

# 2) E14 验证链（naive probe 48.3 溯源）
python3 experiments/_quick_scan_e14_probe.py
python3 experiments/_quick_scan_e14_passive.py       # probe_pass = 48.3（2-seed 扫描）
python3 experiments/_validate_e14_passive.py
python3 experiments/_validate_e14_passive_10seeds.py # 10-seed + 配对 t-test
python3 experiments/_verify_e14_hardfrozen.py

# 3) 绘图
python3 figure_pipeline/scripts/_draw_final_v3.py     # fig1-fig5 + table1/2
python3 figure_pipeline/scripts/_draw_trace_compare.py # fig6
python3 figure_pipeline/scripts/_draw_e10_wfs.py ... _draw_e16_beta.py
```

---

## 3. 论文证据明细（`data/evidence/`）

论文补充数字的每 seed 原始数据，供核查/统计检验：

```
data/evidence/
├── e14_anchor_frozen/          # E14：锚冻结 + 被动校准 + v4 对照
│   ├── summary.csv                    (旧版 5-seed，已废弃，仅存档)
│   ├── summary_passive_10seeds.csv    (10-seed 定稿：baseline 53.3±11.5 / passive 55.3±8.6)
│   ├── naive_probe/summary_2seeds.csv (naive probe 48.3，2-seed 扫描)
│   ├── baseline/    s0..s9.json       (控制环 baseline 每 seed run_meta)
│   ├── passive_low/ s0..s9.json       (passive 每 seed run_meta)
│   ├── v4_closure/  s0..s9.json + summary_v4.csv  (v4 闭式解 83.7±6.9，10-seed)
│   └── scripts/     (exp_e14_probe*.py + 扫描/验证脚本)
├── trace_replay/               # Lingjun trace 重放（10-seed）
│   ├── summary.csv                   (SAS mean/min、p-values、总迭代——权威)
│   ├── fig6_trace_compare.csv        (P-attn)
│   ├── exp_trace_replay.py           (生成脚本)
│   └── run_meta/  run_meta_{Fair,CRUX,SP,D1,v4}_s0..s9.json  (50 个)
└── anchor/                     # tab:anchor（24-job / 400G / 3 seeds）
    ├── run_meta.json                (场景元数据)
    ├── per_policy_results.json      (Fair/CRUX/SP)
    ├── per_policy_results_regen_v1.json
    ├── D1_rerun.json                (DF)
    ├── per_policy_results_v4.json + run_meta_v4.json  (v4 补跑)
    └── baseline_regen.py + run_anchor_v4.py           (生成脚本)
```

---

## 4. 关键结论速查（论文叙事）

| 实验 | 结论 |
|---|---|
| E10 | 带宽稀缺区 v4 优于 WFS（400G：72.5% vs 52.5%）；优势随带宽充裕收窄 |
| E11 | v4 全 ρ∈{0,0.3,0.5,0.85,1.0} 稳定（500G 87.5%，630G 92.5-97.5%） |
| E11b | v4 分配精度 ≈1.0（按需求精确分配）；WFS 权重分配超配 0.70-0.88 |
| E12 | 7 级量化下 v4 优于 D1；优势随规模扩大（n100：36.8% vs 6.0%） |
| E13 | W 无统计显著影响；W=20 均值最高、方差最低（工程默认） |
| E14 | 冻结是真实问题：baseline 53.3±11.5、9-10/30 硬冻结；naive 探测 48.3 有害；passive 消除冻结但均值不显著；**v4 闭式解 83.7±6.9%、零冻结** |
| E15 | 400G 稀缺带宽下 v4 全 straggler 因子领先（67.5-72.5%） |
| E16 | P-attn 对 β∈[0.3,1.0] 平坦（85-97.5%）；S-cont 单调上升——运营自由度大 |
| Anchor | v4 mean_sas 0.9685 远超基线（0.72-0.84），collapse=0，slo_rate 31.9% 最高 |

---

## 5. 风格规范（E 系列图）

- serif（Times New Roman）、无数据标签、无网格、黑色粗边框（linewidth 2.0）
- rcParams：font.size=26 / labelsize=28.6 / titlesize=31.2 / ticks=23.4 / legend=20.8
- 扁平宽图 figsize=(13, 4.2)，图例在框外上方 `bbox_to_anchor=(0.5, 1.05)`，`subplots_adjust(top=0.80, bottom=0.12)`
- 颜色：v4=`#1f77b4`（蓝）、WFS=`#A0522D`（棕）、Fair=`#808080`、CRUX=`#D2691E`、SP=`#DAA520`、D1=`#2E8B57`
