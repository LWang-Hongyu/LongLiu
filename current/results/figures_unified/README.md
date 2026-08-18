# figures_unified — 论文图片统一管理目录

> 创建日期：2026-08-10
> 目的：将分散在各实验目录的论文图片（数据源 + 绘图脚本 + 图输出）统一收纳，便于管理与复现。

## 目录结构

```
figures_unified/
├── README.md                      # 本文件
├── fig6_v6stats/                  # 图：V6 统计重演（Fig-6s）
│   ├── data/                      # 输入数据（v6_replication_3/4/5 的 epoch CSV 副本）
│   ├── scripts/plot_v6_stats_fig.py   # 绘图脚本（输出 → figures/）
│   └── figures/                   # 图输出（PDF + 600dpi PNG）
└── fig_expC/                      # 图：实验 C v2（S1 P-attn / S2 per-job / S1 premium）
    ├── data/expC_v2_per_round.csv # 输入数据
    ├── scripts/plot_expC_v2.py    # 绘图脚本（输出 → figures/）
    └── figures/                   # 图输出（PDF + PNG）
```

## 图 → 数据源 → 脚本 映射

| 图 | 数据源（原始位置） | 本目录数据 | 绘图脚本 | 输出 |
|----|--------------------|-----------|----------|------|
| fig6_v6stats_testbed_600.png | `experiments_evaluation/P4_dumbbell_slo/v6_replication_{3,4,5}/p4_jobB_v6_*_rank0_epoch.csv` | `fig6_v6stats/data/v6_replication_{3,4,5}/` | `fig6_v6stats/scripts/plot_v6_stats_fig.py` | `fig6_v6stats/figures/fig6_v6stats_testbed.{pdf,600.png}` |
| fig_expC_s1_pattn.png | `experiments_supplementary/03_exp_C_scale_ladder/analysis/expC_v2_per_round.csv` | `fig_expC/data/expC_v2_per_round.csv` | `fig_expC/scripts/plot_expC_v2.py` | `fig_expC/figures/fig_expC_s1_pattn.{pdf,png}` |
| fig_expC_s2_perjob.png | 同上 | 同上 | 同上 | `fig_expC/figures/fig_expC_s2_perjob.{pdf,png}` |
| fig_expC_s1_premium_sd.png（同脚本副产） | 同上 | 同上 | 同上 | `fig_expC/figures/fig_expC_s1_premium_sd.{pdf,png}` |

## 复现方法

```bash
# 图1：fig6_v6stats
cd current/results/figures_unified/fig6_v6stats/scripts
python3 plot_v6_stats_fig.py          # 输出到 ../figures/

# 图2/3：实验 C v2
cd current/results/figures_unified/fig_expC/scripts
python3 plot_expC_v2.py               # 输出到 ../figures/
```

## 数据刷新说明

- **fig6_v6stats/data/**：为 `P4_dumbbell_slo/v6_replication_{3,4,5}` 的 epoch CSV 副本；
  重跑 V6 实验后需重新复制。
- **fig_expC/data/**：为 `03_exp_C_scale_ladder/analysis/expC_v2_per_round.csv` 的副本；
  若重跑分析（`analysis/analyze_expC_v2.py`）重新生成该 CSV，请同步覆盖本目录副本。
