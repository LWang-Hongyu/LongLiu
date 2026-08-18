# 实验归档索引：E10-E16（Robustness & Sensitivity）

> 本文件为 E10-E16 全部实验的权威归档索引：脚本 → 数据 → 图片 → LaTeX 章节。
> 生成日期：2026-08-10。数据均来自 5-seed 正式实验（E14 含 10-seed；E11b/E16 为
> 审稿回应新增实验）。完整复现指南见 [figure_pipeline/README.md](figure_pipeline/README.md)。

---

## E10 — WFS Baseline（加权公平共享基线）

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e10_wfs.py` |
| 数据目录 | `outputs/e10_wfs/`（`summary.csv` + 120× `run_meta.json`/`trace.jsonl`） |
| 绘图脚本 | `outputs/figures/_draw_e10_wfs.py` |
| 图片 | `outputs/figures/fig_e10_wfs.png` / `.pdf`（扁平 13×4.2，字体 ×1.3） |
| LaTeX | `paper/evaluation_e10_e15.tex` §E10 |

**结论**：带宽稀缺区 v4 优于 WFS——400G：v4 72.5% vs WFS 52.5%；630G：97.5% vs 75.0%。优势随带宽充裕而收窄，印证 deficit-aware 调度的价值集中在拥塞区。

---

## E11 — Overlap Factor Sensitivity（重叠因子敏感性）

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e11_overlap.py` |
| 数据目录 | `outputs/e11_overlap/` |
| 绘图脚本 | `outputs/figures/_draw_e11_overlap.py` |
| 图片 | `outputs/figures/fig_e11_overlap.png` / `.pdf`（双面板 500/630G，字体 ×1.3） |
| LaTeX | `paper/evaluation_e10_e15.tex` §E11 |

**结论**：v4 在全部 $\rho\in\{0,0.3,0.5,0.85,1.0\}$ 下稳定——500G 达 87.5%，630G 92.5-97.5%；CRUX 等基线随 $\rho$ 增大明显退化。

---

## E12 — DSCP Quantization Error（量化误差）

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e12_dscp.py` |
| 数据目录 | `outputs/e12_dscp/`（n14/n50/n100 × v4/D1 × 5 seeds） |
| 绘图脚本 | `outputs/figures/_draw_e12_dscp.py` |
| 图片 | `outputs/figures/fig_e12_dscp.png` / `.pdf`（4:3，figsize 8×6） |
| LaTeX | `paper/evaluation_e10_e15.tex` §E12 |

**结论**：n14 两者均 97.5%；n50 v4 60.0% vs D1 35.7%；n100 36.8% vs 6.0%。闭式解（连续 $\pi$）规避 7 级量化碰撞，优势随规模扩大。

---

## E13 — Window Size Sensitivity（窗口大小，修复后定稿）

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e13_window.py` |
| 数据目录 | `outputs/e13_window/`（`summary.csv`：w5/w10/w20/w50 × 500G × 5 seeds） |
| 绘图脚本 | `outputs/figures/_draw_e13_window.py` |
| 图片 | `outputs/figures/fig_e13_window.png` / `.pdf`（4:3，figsize 8×6） |
| LaTeX | `paper/evaluation_e10_e15.tex` §E13 |

**5-seed 结果**（`summary.csv`，window_size 同步修复后）：

| W | P-attn | std |
|---|---|---|
| 5 | 70.9% | ±18.5 |
| 10 | 69.1% | ±22.0 |
| **20** | **76.4%** | **±17.8** |
| 50 | 72.7% | ±24.4 |

**结论**：W 无统计显著影响（均值差 ≤7.3pp，误差棒大幅重叠）；W=20 均值最高、方差最低，定为工程默认。

**代码变更**：`longliu_sim/job/job.py` 新增 `sliding_window_len` 属性；`longliu_sim/policy/longliu.py` `allocate()` 惰性同步 `window_size` 到注入新 job（修复前新 job 恒用 W=8）。

---

## E14 — Anchor Freezing（设计案例，v2 定位）

> 叙事变更：由"主动探测有效"改为"锚点冻结设计洞察"——冻结是真实问题；naive 探测在 SP 下有害；被动校准消除冻结但均值不显著；反衬 v4 闭式解天然免疫。

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e14_probe.py`（正式，含 `ema_passive`/`ema_weights` 参数） |
| 扫描/验证脚本 | `experiments/_quick_scan_e14_passive.py`（2-seed 基/被动/探测） |
| | `experiments/_quick_scan_e14_weights.py`（P4/P5 权重网格，2-seed） |
| | `experiments/_validate_e14_passive.py`（5-seed） |
| | `experiments/_validate_e14_passive_10seeds.py`（10-seed + 配对 t/Wilcoxon/Cohen's d） |
| | `experiments/_verify_e14_hardfrozen.py`（硬冻结插桩验证） |
| 数据目录 | `outputs/e14_probe/` |
| 定稿数据 | `outputs/e14_probe/summary_passive_10seeds.csv` |
| 中间数据 | `outputs/e14_probe/summary_passive.csv`（5-seed）、`summary.csv`（旧代码 probe 5-seed，仅存档） |
| 绘图脚本 | `outputs/figures/_draw_e14_anchor.py` |
| 图片 | `outputs/figures/fig_e14_anchor.png` / `.pdf`（4:3，figsize 8×6） |
| LaTeX | `paper/evaluation_e14_anchor.tex`（独立 E14 文件，含架构描述） |

**10-seed 定稿**（`summary_passive_10seeds.csv` + `v4_closure/summary_v4.csv`）：

| 配置 | P-attn | 硬冻结 job | seeds |
|---|---|---|---|
| baseline（控制环） | 53.3±11.5% | 9-10 / 30 | 10 |
| naive 探测（best） | 48.3% | — | 2（扫描） |
| passive 校准（P4=0.2/P5=0.4） | 55.3±8.6% | 0 / 30 | 10 |
| **v4 闭式解** | **83.7±6.9%** | **0 / 30** | **10** |

**统计**：被动 vs 基线均值差 +2.0pp，配对 t p=0.66（不显著）；Cohen's d=0.14；方差 -25%。
**v4 闭式解对照**（`exp_e14_probe_v4.py --seeds 10`）：无 EMA 锚、每窗口按链路容量重解最优
分配，天然免疫冻结——P-attn 83.7±6.9%（+30.4pp vs baseline）、零硬冻结、零饥饿，实证支撑
"闭式解免疫锚冻结"叙事。per-seed 数据：`outputs/e14_probe/v4_closure_800g_s{0..9}/run_meta.json`，
归档：`figure_pipeline/data/evidence/e14_anchor_frozen/v4_closure/`。

**代码变更**：`job.py` `update_ema_from_comm_time(weight)` 支持信任加权 + 单边更新（快观测全额采纳、慢观测按 α×weight 折扣）+ `ema_update_count` 插桩；`longliu.py` 新增 `ema_passive`/`ema_weights` 参数与 `EMA_PRIORITY_WEIGHTS` 表（P6=1.0 → P0=0.05）。

**弃用产物**：`outputs/figures/fig_e14_probe.png`/`.pdf` 与 `_draw_e14_probe.py`（旧代码"探测有效"叙事，保留仅供存档）。

---

## E15 — Straggler Injection（400G 拥塞定稿）

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e15_straggler.py` |
| 数据目录 | `outputs/e15_straggler/`（`summary.csv`：factor {1,2,3,5} × 5 policies × 5 seeds） |
| 绘图脚本 | `outputs/figures/_draw_e15_straggler.py` |
| 图片 | `outputs/figures/fig_e15_straggler.png` / `.pdf`（扁平 13×4.2，字体 ×1.3） |
| LaTeX | `paper/evaluation_e10_e15.tex` §E15 |

**结论**：400G 稀缺带宽下 v4 全 straggler 因子领先——1×:72.5%、2×:72.5%、3×:67.5%、5×:67.5%；D1 从 52.5% 降至 45.0%，CRUX 65.0%→60.0%。窗口平均 + 闭式重分配吸收计算膨胀。

---

## E11b — Allocation Precision（分配精度，审稿回应）

> 回应审稿人对"串行模型是否过度分配"的关切。原 Waste Ratio 指标在 flow-level +
> work-conserving 下退化为 1−util（Σalloc≈Σused），按用户决策换为 Allocation Precision：
> Prec = Σ min(a_j, b_j^att) / Σ a_j——衡量分配的"按需精确度"（v4 精确分配 → ≈1.0；
> WFS 权重分配给部分 job 超需求带宽 → <1.0）。

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e11b_overlap_waste.py`（`--seeds 5`，含 `BwProbeSimulator` per-spine cap） |
| 数据目录 | `outputs/e11b_overlap_waste/summary.csv`（ρ∈{0,0.3,0.5,0.85,1.0} × 500/630G × v4/WFS × 5 seeds） |
| 绘图脚本 | `figure_pipeline/scripts/_draw_e11b_overlap_waste.py` |
| 图片 | `figure_pipeline/figs/fig_e11b_overlap_waste.png` / `.pdf`（扁平 13×4.2，双 panel） |
| LaTeX | `paper/evaluation_e11b_e16.tex` §E11b |

**5-seed 结果**（Allocation Precision）：

| ρ | v4 @500G | WFS @500G | v4 @630G | WFS @630G |
|---|---|---|---|---|
| 0.0 | 1.000 | 0.784 | 1.000 | 0.703 |
| 0.3 | 0.996 | 0.820 | 0.996 | 0.731 |
| 0.5 | 0.998 | 0.835 | 0.998 | 0.760 |
| 0.85 | 1.000 | 0.871 | 1.000 | 0.801 |
| 1.0 | 1.000 | 0.880 | 1.000 | 0.829 |

**结论**：v4 分配精度 ≈1.0 恒成立（精确按各 job 可达带宽分配，不超配）；WFS 权重分配
在稀缺带宽下超配 12-30%（500G: 0.78-0.88、630G: 0.70-0.83），即把需求外的带宽
"浪费"给无需者。P-attn 同步验证：v4 82.5-87.5%（500G）、92.5-97.5%（630G）领先 WFS 70-80%。

---

## E16 — β Sensitivity（β 敏感性，审稿回应）

> 回应"β 是运营者最敏感的 knob"。扫描 β∈{0.3,0.5,0.7,1.0}，报告 P-attn / S-cont /
> utilization / starved。叙事：P-attn 平坦、S-cont 单调——运营自由度大。

| 项 | 路径 |
|---|---|
| 运行脚本 | `experiments/exp_e16_beta.py`（`--seeds 5`） |
| 数据目录 | `outputs/e16_beta/summary.csv`（β×500/630G×5 seeds） |
| 绘图脚本 | `figure_pipeline/scripts/_draw_e16_beta.py` |
| 图片 | `figure_pipeline/figs/fig_e16_beta.png` / `.pdf`（扁平 13×4.2，单 panel 双轴：P-attn 左/S-cont 右） |
| LaTeX | `paper/evaluation_e11b_e16.tex` §E16 |

**5-seed 结果**：

| β | P-attn @500G | S-cont @500G | P-attn @630G | S-cont @630G |
|---|---|---|---|---|
| 0.3 | 87.5% | 0.978 | 92.5% | 0.998 |
| 0.5 | 87.5% | 0.982 | 97.5% | 0.999 |
| 0.7 | 85.0% | 0.987 | 97.5% | 0.999 |
| 1.0 | 95.0% | 0.999 | 97.5% | 1.000 |

**结论**：P-attn 在 β∈[0.3,1.0] 平坦（500G 85-95%、630G 92.5-97.5%），S-cont 单调上升
0.978→0.999（β 越大标准作业越快，串行时间占比提升）——"β 运营自由度大"叙事成立；
util≈0.32/0.27，starved=0（所有配置）。

---

## 图片总览（E 系列）

| 图片 | 比例 | 说明 |
|---|---|---|
| `fig_e10_wfs.png` | 扁平（字体×1.3） | WFS 基线，图例在框外上方 |
| `fig_e11_overlap.png` | 扁平（字体×1.3） | Overlap 双面板，图例框外上方 |
| `fig_e11b_overlap_waste.png` | 扁平（字体×1.3） | Allocation Precision 双 panel（500/630G） |
| `fig_e12_dscp.png` | 4:3 | v4 vs D1 三规模 |
| `fig_e13_window.png` | 4:3 | W 敏感性 |
| `fig_e14_anchor.png` | 4:3 | 锚点冻结设计案例（新叙事） |
| `fig_e15_straggler.png` | 扁平（字体×1.3） | Straggler 注入，图例框外上方 |
| `fig_e16_beta.png` | 扁平（字体×1.3） | β 敏感性单 panel 双轴（P-attn 左 / S-cont 右） |

**风格规范**：serif（Times New Roman）、无数据标签、无网格、黑色粗边框（linewidth 2.0）、E12-14 为 4:3、E10/11/11b/15/16 扁平且字体 ×1.3。

---

## 运行方式

```bash
# 正式实验（5-seed；E14 为 10-seed）
python3 experiments/exp_e10_wfs.py --seeds 5
python3 experiments/exp_e11_overlap.py --seeds 5
python3 experiments/exp_e11b_overlap_waste.py --seeds 5
python3 experiments/exp_e12_dscp.py --seeds 5
python3 experiments/exp_e13_window.py --seeds 5
python3 experiments/exp_e14_probe.py --seeds 10
python3 experiments/exp_e14_probe_v4.py --seeds 10   # v4 闭式解对照（83.7±6.9%）
python3 experiments/exp_e15_straggler.py --seeds 5
python3 experiments/exp_e16_beta.py --seeds 5

# E14 验证链（2/5/10-seed + 硬冻结）
python3 experiments/_quick_scan_e14_passive.py
python3 experiments/_quick_scan_e14_weights.py
python3 experiments/_validate_e14_passive.py
python3 experiments/_validate_e14_passive_10seeds.py
python3 experiments/_verify_e14_hardfrozen.py

# 绘图（统一在 figure_pipeline/scripts/ 下）
python3 figure_pipeline/scripts/_draw_final_v3.py
python3 figure_pipeline/scripts/_draw_trace_compare.py
python3 figure_pipeline/scripts/_draw_e10_wfs.py
python3 figure_pipeline/scripts/_draw_e11_overlap.py
python3 figure_pipeline/scripts/_draw_e11b_overlap_waste.py
python3 figure_pipeline/scripts/_draw_e12_dscp.py
python3 figure_pipeline/scripts/_draw_e13_window.py
python3 figure_pipeline/scripts/_draw_e14_anchor.py
python3 figure_pipeline/scripts/_draw_e15_straggler.py
python3 figure_pipeline/scripts/_draw_e16_beta.py
```

> 绘图脚本与图片已迁移至 `figure_pipeline/`（scripts/ + figs/），`outputs/figures/` 旧目录仅存档。
