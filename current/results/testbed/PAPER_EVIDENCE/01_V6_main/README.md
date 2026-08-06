# 01_V6_main — V6-P4 正式两轮（血缘声明）

## 数据状态

| 文件 | 属主 | 说明 |
|------|------|------|
| `round1_console.log` | **原件** | Header Date=2026-07-24T16:24:14，含完整 results_summary |
| `round2_console.log` | **原件** | Header Date=2026-07-24T16:33:52，含完整 results_summary |
| `run_meta_round1.txt` | **原件** | 实验元数据 |
| `run_meta_round2.txt` | **原件** | 实验元数据 |
| `*.csv` | **复制1（非原件）** | 原始 CSV 因文件名冲突被统计重演覆盖 |
| `results_summary.txt` | **复制1（非原件）** | 同上 |

## 原始数据灭失声明

V6-P4 正式两轮（2026-07-24 16:33-17:08）的 epoch 级 CSV 文件因后续统计重演（复制1）使用相同文件名（如 `p4_jobA_v6_round1_LLthenCX_longliu_rank0_epoch.csv`）而被覆盖。

**权威记录来源**：
1. `round1_console.log` 和 `round2_console.log` 内的 Results Summary 区块（epoch 级数据，见文件末尾 `=== Results Summary ===` 段）。
2. `QUOTA_EXPERIMENT_RESULTS.md` 文档表格值（L1277-L1309，V6 Outcome 节）。

## 原始两轮关键指标（console.log results_summary 提取）

> ⚠️ **勘误说明**：下表 Round 2 数据的原始估算值（来自首版 README）已在无记录情况下被修正为实测值。详见文末【勘误】。

### Round 1: LL→CX（2026-07-24T16:24:14）

| Epoch | LL JobB slowdown | CRUX JobB slowdown | LL 优势 |
|-------|-----------------|-------------------|---------|
| 7 | 1.1441 | 1.3062 | −12.4% |
| 8 | 1.122 | 1.2997 | −13.7% |
| 9 | 1.1036 | 1.2942 | −14.7% |
| 10 | 1.078 | 1.3101 | −17.7% |
| 11 | 1.0897 | 1.3008 | −16.2% |

### Round 2: CX→LL（2026-07-24T16:33:52）

| Epoch | LL JobB slowdown | CRUX JobB slowdown | LL 优势 |
|-------|-----------------|-------------------|---------|
| 7 | 1.1075 | 1.3034 | −15.0% |
| 8 | 1.1346 | 1.2831 | −11.6% |
| 9 | 1.0840 | 1.2784 | −15.2% |
| 10 | 1.1012 | 1.2883 | −14.5% |
| 11 | 1.1071 | 1.2586 | −12.0% |

**结论**：Phase 2 tight B 无区间重叠，LL 决定性优势 −12–17%。（已勘误，见文末）

## 目录文件清单

```
01_V6_main/
├── README.md                                    # 本文件（血缘声明）
├── round1_console.log                           # 原件：round1 完整日志 + results_summary
├── round2_console.log                           # 原件：round2 完整日志 + results_summary
├── run_meta_round1.txt                          # 原件
├── run_meta_round2.txt                          # 原件
├── fig6_data.csv                                # Fig-6 数据源（从 console.log 提取，见勘误③）
```

---

## 勘误

### ① Round 2 数值修正（2026-07-27）

**问题**：首版 README 的 Round 2 表格填入了估算值（非 console.log 实测值），后续在无记录情况下被直接覆盖修正。

**原值（已删除线标记，保留归档）**：

~~| Epoch | 7 | 8 | 9 | 10 | 11 |~~
~~| LL JobB | 1.1388 | 1.1315 | 1.1133 | 1.1074 | 1.1018 |~~
~~| CRUX JobB | 1.3257 | 1.3218 | 1.3146 | 1.3082 | 1.3102 |~~

**修正后（实测值）**：

| Epoch | 7 | 8 | 9 | 10 | 11 |
| LL JobB | 1.1075 | 1.1346 | 1.0840 | 1.1012 | 1.1071 |
| CRUX JobB | 1.3034 | 1.2831 | 1.2784 | 1.2883 | 1.2586 |

**修正原因**：首版 README 的 Round 2 值来源于中间提取脚本的错误转换，未直接核对 console.log 原件。实测值直接取自 `round2_console.log` 的 Results Summary 段。

**修正日期**：2026-07-27
**状态**：经验收批准

### ② rep1 与原件同一性（2026-07-27）

**发现**：`02_replications/v6_replication_1/` 的两个 `console.log` 文件与 `01_V6_main/` 对应原件的 md5 完全相同（`round1: 9438c116`; `round2: 3f8170eb`），确认复制1 因文件名冲突覆盖事故而实际为原件的副本，非独立实验轮次。

**影响**：独立实验轮次从 6 轮下修为 **4 轮**（2 orig + 2 rep2），复现率从 "5/6 决定性" 下修为 "**3/4 决定性 + 1/4 重叠不定**"。

### ③ fig6_data.csv 生成说明（2026-07-27）

`fig6_data.csv` 由 `_build_fig6_csv.py`（存放于 `PAPER_EVIDENCE/../`，即归档外）生成。生成脚本不在归档内。

**口径**：
- `phase1_tight`：epochs 0-6，Job A（solo + ramp-up 基线）
- `phase2_tight_stable`：epochs 7-11，Job B（稳定争抢窗）— **论文主口径**
- `phase2_tight_full`：epochs 7-14，Job B（含 CRUX 衰减）— **附录备用**

**独立轮次列表**（共 4 轮，rep1 非独立）：

| round_id | order | 说明 |
|----------|-------|------|
| orig_r1 | LL→CX | 原件 round1 |
| orig_r2 | CX→LL | 原件 round2 |
| rep2_r1 | LL→CX | 复制2 round1（独立） |
| rep2_r2 | CX→LL | 复制2 round2（独立） |
