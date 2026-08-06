# Fig-6：V6-P4 物理床对照（LongLiu vs CRUX）

> **论文用途**：物理床 outcome 核心证据 — LongLiu 动态优先级 vs CRUX 静态 P3 的 slowdown 对比。
> **对应证据**：PAPER_EVIDENCE/01_V6_main + 02_replications
> **数据就绪**：✅

## 内容

```
fig6_v6physical/
├── README.md                           # 本文件
├── fig6_data.csv                       # 核心数据（28 行，三口径）
├── round1_console.log                  # 原件：V6-P4 Round 1 (LL→CX)
├── round2_console.log                  # 原件：V6-P4 Round 2 (CX→LL)
├── run_meta_round1.txt                 # 实验元数据
├── run_meta_round2.txt                 # 实验元数据
├── p4_job_reverse.py                   # 实验运行器（核心脚本）
├── analyze_reverse.py / _v2.py         # CSV 分析器
├── plot_p4.py                          # 仿真侧 5-baseline 绘图脚本（不适用）
├── plot_fig6.py                        # Fig-6 物理床 2×2 子图绘图脚本
├── fig6_testbed.pdf                    # Fig-6 矢量输出（通栏 7.16in）
├── fig6_testbed_600.png                # Fig-6 600dpi 光栅输出
└── fig6_self_check.csv                 # 自校验：图均值 vs CSV 逐格对照
```

## 数据口径

| phase | epochs | 说明 | 用途 |
|-------|--------|------|------|
| phase1_tight | 0-6 | Job A solo + ramp-up | 基线参照 |
| phase2_tight_stable | 7-11 | Job B 稳定争抢窗 | **论文主口径** |
| phase2_tight_full | 7-14 | Job B 含 CRUX 衰减 | 附录备用 |

## 关键结果（稳定窗 7-11）

| Round | 方向 | LL mean | CRUX mean | 差幅 |
|-------|------|---------|-----------|------|
| orig_r1 | LL→CX | 1.1075 | 1.3022 | **−15.0%** |
| orig_r2 | CX→LL | 1.1069 | 1.2824 | **−13.7%** |
| rep2_r1 | LL→CX | 1.2496 | 1.2541 | −0.4%（重叠） |
| rep2_r2 | CX→LL | 1.1123 | 1.3365 | **−16.8%** |

**裁决**：3/4 决定性无重叠，1/4 重叠不定。论文声明：4 轮独立实验。

## 数据来源

- `fig6_data.csv` 由 `../_build_fig6_csv.py`（在 FIGURE_REGISTRY 外）生成
- 原始 epoch 数据来自 `round{1,2}_console.log` 的 Results Summary 段
- 原始 CSV 因文件名冲突灭失（详见 `PAPER_EVIDENCE/01_V6_main/README.md` 勘误②）

## 复现命令

```bash
# 一次完整 V6 实验（需物理床双机）
bash experiments/P4_dumbbell_slo/run_v6_atomic.sh

# 分析已有 CSV
python3 analyze_reverse.py p4_job*.csv

# 绘图（需统一格式，仿真侧执行）
python3 plot_p4.py --input fig6_data.csv --output fig6.png
```
