# 实验重做指导文档

## 背景
CRUX 实现已修正（比例分配 → 严格优先级，GPU intensity 映射 DSCP 优先级队列）。
旧名 SP/D1/v4/WFS 统一重命名，新增 CASSINI 独立 baseline。

## 2026-08-24 追加修复：CASSINI comm offsets
审查中发现 CASSINI 在主表/E11/E15/trace/anchor 中仅以裸 `CASSINI()` 运行，
其 `allocate()` 与 Fair 完全相同（逐指标一致），comm offsets 从未被应用（仅 S3 消融调用了
`apply_cassini_offsets`）。经用户确认：**修复并重跑** —— 在上述所有脚本的 jobs 创建后、
submit 前调用 `CASSINI.compute_offsets` 并设置 `job.comm_offset_ms`（与 S3 一致）。
受影响的实验（P0+P1+P2 全部 6 个脚本）已修改并重启。

## Baseline 定义（最终 7 策略主表）

| 策略 | 代码标识 | 设计思路 | 分配方式 |
|------|----------|----------|----------|
| Fair | `Fair()` | 无区分均分 | 比例分配 |
| SRPT | `SRPT()` | flow 级最短剩余时间（旧名 SP） | 比例分配 |
| CRUX | `CRUX()` | GPU intensity + 严格优先级 | SP |
| CASSINI | `CASSINI()` | 通信相位错开 | Fair 分配 + offset |
| DF | `LongLiuDWRR()` | 动态反馈有界加权（旧名 D1） | DWRR |
| LL-S | `LongLiu(use_dynamic_T_target=False)` | 动态 pi key + 静态 T_target（旧名 SP 混淆点澄清） | SP |
| LongLiu | `LongLiuAllocatorV4()` | 闭式分配器，动态 pi key + 动态 T_target（旧名 v4） | SP |

注：WFS（`WFS()`）经讨论确定为 LongLiu 变体，移出主表对照，不再进入 E1/E2 图。

## 种子策略（最终验证标准）

| 实验 | 种子数 | 依据 |
|------|--------|------|
| 主表 E1/E2/E2-pro（exp_v3_batch3_formal） | 10 | 最终验证 ≥10 seeds |
| E3/E3' swap（exp_e3_swap） | 5 | 动态实验，5 seeds |
| E11 overlap / E15 straggler | 5 | 附录敏感性 |
| Trace replay | 10 | 真实 trace 对照 |
| S3 组件消融 | 10 | 消融统计（paired t-test） |
| Anchor baseline（baseline_regen） | 10 | 附录 Table anchor |

## 需要重做的实验

### 优先级 P0（主表，必须重做）

| 实验 | 脚本 | 用途 |
|------|------|------|
| E1 + E2' + E2-pro | `experiments/exp_v3_batch3_formal.py` | 主表阶梯/对抗/正对照（12 配置 × 7 策略 × 10 seeds = 840 runs） |
| E3/E3' Swap | `figure_pipeline/data/e3_swap/exp_e3_swap.py` | Hero 图（2 场景 × 7 策略 × 5 seeds） |

### 优先级 P1（附录实验）

| 实验 | 脚本 | 用途 |
|------|------|------|
| Trace Replay | `experiments/exp_trace_replay.py` | 真实 trace 鲁棒性（7 策略 × 10 seeds） |
| Straggler | `experiments/exp_e15_straggler.py` | Straggler 鲁棒性（4 因子 × 7 策略 × 5 seeds） |
| Overlap | `experiments/exp_e11_overlap.py` | 重叠率敏感性（5 overlap × 2 bw × 7 策略 × 5 seeds） |

### 优先级 P2（消融 + 锚点）

| 实验 | 脚本 | 说明 |
|------|------|------|
| S3 Component | `experiments/exp_s3_component_ablation.py` | 3 组消融：priority key / T_target 校准 / phase 错开（10 seeds） |
| Anchor baseline | `figure_pipeline/data/evidence/anchor/baseline_regen.py` | 24-job anchor @400G 附录表（7 策略 × 10 seeds） |

## 不需要重做的实验

| 实验 | 原因 |
|------|------|
| E12 DSCP (exp_e12_dscp.py) | 不涉及 CRUX |
| E13 Window (exp_e13_window.py) | 不涉及 CRUX |
| E14 Anchor Freezing (exp_e14_anchor.py) | 不涉及 CRUX |
| E16 Beta (exp_e16_beta.py) | 不涉及 CRUX |
| Scale (exp_scalability.py) | 只有 LongLiu vs DF |
| E10 WFS (exp_e10_wfs.py) | WFS 移出主表对照 |
| 各轨迹实验 | 单策略行为分析 |

## 判定阈值（exp_v3_batch3_formal）

- LongLiu 保障：三场景 @800G mean P-attn ≥ 0.98
- P1a: E2' @630G CRUX mean P-attn < LongLiu 至少 10pp
- P1b: E2' @500G CRUX P-cap ≤ 0.90（严格优先级后性能提升，旧阈值 0.35 失效）
- P2: E1 @400/500G LongLiu mean P-attn ≥ DF mean

## 执行步骤

1. 修正 CRUX 实现 ✅
2. 统一策略名与策略列表 ✅（7 策略）
3. 按优先级运行实验：P0 → P1 → P2 ✅（10/5 seeds）
4. 汇总数据：输出 CSV 到 figure_pipeline/data/figure_registry/，备份 outputs → figure_pipeline/data/ ✅
5. 更新绘图脚本（_draw_final_v3 等）策略名与种子数，重新生成 figure_pipeline/figs/ ✅
6. 更新 evaluation.tex 和 appendix.tex ✅

## 2026-08-24 完成记录（全部实验重做完毕）

所有实验已完成并汇总：

- **主表 840 runs（10 seeds）完成**：`_make_registry_csvs.py` 生成
  `fig2_e1_ladder_10seed.csv`（42 行）、`fig3_e2_ladder_10seed.csv`（42 行）、`fig6_trace_compare.csv`。
  注意：输出目录混有 2026-07-27 旧残留（D1/v4/SP 共 180 目录），`load_main_metas` 已按
  `timestamp.startswith("2026-08-24")` + policy 白名单过滤。
- **脚本验证断言（Matrix v2.1）有两项 FAIL**：LongLiu E1@800G=97.5%、E2'@800G=96.7% 略低于预设 98%
  阈值 → 触发 `*** BATCH FAILED — stopping ***`，但 840 个 run_meta.json 全部写入，数据完整。
  论文中按真实结果 97.5%/96.7% 表述（研究诚信优先）。
- **E3/E3' swap（5 seeds）**：LongLiu W1/W2/W3 均 100%（两臂），通过预注册验证。
- **E11/E15/trace/S3/anchor** 全部完成。

## 2026-08-25 E17 混合集合通信（新增验证）

用户质疑"方案只验证了 DDP（AllReduce）"，经讨论确认方案理论层（带宽分配抽象，
通信时间 = 数据量/带宽）与集合通信类型无关，但缺实验证据。采取"论证 + 补混合实验"方案：

- **模拟器扩展**：`Job.collective_type`（allreduce/allgather/reduce_scatter/alltoall），
  `simulator._create_collective_flows` 按类型展开：ring 类 = N 条环流；
  alltoall = N×(N-1) 条全连接流（多瓶颈）。
- **E17 场景**（`experiments/exp_e17_mixed_collective.py`）：复用主表 E1 的 model/dp/ci 结构
  （14 jobs，8P/6S），混合 5×AllReduce（DDP）+ 4×AllGather（ZeRO-3）+ 3×All-to-All（MoE）
  + 2×ReduceScatter（ZeRO-3），通信量按类型缩放（单阶段 = 0.5×）。
  4 档 spine × 7 策略 × 10 seeds = 280 runs，84 分钟完成。
- **结果**：LongLiu 400/500/630G 三档最优（67.5/71.2/73.8%）；400G 显著优于
  DF(p=0.006)/CRUX(p=0.003)/CASSINI(p=2e-4)/LL-S(p=0.04)；800G 收敛区无显著差异。
  E1 的定性排序完整保留 → 方案对集合通信类型不敏感。
- **论文**：evaluation.tex 新增 "Generality across collectives" 段 + fig7_e17_mixed；
  appendix.tex 新增 E17 小节 + tab:e17 全表；绘图脚本 `_draw_e17.py`。
- **绘图修复**：`_draw_final_v3.py` 中 `load_e1_e2` 的 `quotechar="'"` 导致 CSV 解析错乱（KeyError 'E1'），
  已改为默认 quotechar；`_make_fig4_csv.py` 修复 `workload_raw` 未传参 bug。
- **figs 重新生成**：fig1_hero ~ fig5 + table1_anchor.tex/table2_e2pro.tex + fig6_trace_compare/
  fig_e11_overlap/fig_e15_straggler（14:25-14:26）。
- **evaluation.tex / appendix.tex 已更新**：
  - 主表数字全部换为 10 seeds（E1 400G LongLiu 76.3%、500G 90.0%、630G 96.3%、800G 97.5%）
  - E2'/E2-pro、E3/E3'（CRUX 新表现：control 33.3%/kill 47.5%）、trace（LL-S 85% vs LongLiu 75%，
    LongLiu 最差 SAS 最高 0.731 且无饥饿）、E15（LongLiu 5× 达 75.0%）、S3 消融段落（新增）均已如实更新。
  - baseline 列表：Fair/SRPT/CRUX/CASSINI/DF/LL-S（7 策略，SP→SRPT，WFS 移除）。

## 2026-09-09 Trace 实验 10→30 seeds（用户质疑 LL-S > LongLiu 是否偶然）

用户要求重跑 trace 实验验证 "LongLiu P-attn 低于 LL-S" 是否偶然现象：

- **执行**：`exp_trace_replay.py --seeds 30`（seeds 0-29，154s）。s0-s9 与旧 run_meta 逐字节
  diff 全部一致（完全确定性），新增 s10-s29 为独立 trace 重放样本。
- **结果（30 seeds 配对）**：LL-S mean P-attn 显著更高 —— 85.0±17.3% vs 75.0±13.1%，
  paired t p=0.0169，Wilcoxon p=0.0194，LL-S 19 胜 / 9 负 / 2 平。**不是偶然**。
- **机制**：LL-S（静态 T-target）系统性过冲（mean SAS 1.819 vs 1.170），在 0.98 二元
  attention 门槛上占优；代价同样显著 —— worst-case SAS 0.538±0.549 vs 0.760±0.235
  (p=0.030)，premium 饿死 11/30 seeds（22 次）vs 2/30（2 次），吞吐低 4.9%
  （2146.7 vs 2252.3 iters，p=0.0064）。mean-vs-tail 双向系统性 trade-off。
- **数据管线**：`_make_registry_csvs.py` 重跑 → fig6_trace_compare.csv（30 seeds 列）+
  data/trace_replay 备份（242 items）；`_draw_trace_compare.py` 重画 fig6 并修正 docstring
  （10→30 seeds）；fig6 PDF/PNG 同步到 paper/figure/。
- **论文更新**（10→30 seeds，叙事从 "near-tie" 改为 "mean-vs-tail 双向显著 trade-off"）：
  - 0Main 摘要：margin 26.4-39.3 → "24.8-54.3 pp over five of six baselines" + LL-S trade-off 一句
  - 1Introduction 验证段：数字更新 + LL-S 过冲机制表述
  - 5Evaluation：trace 主段重写（p=0.017/Wilcoxon 0.019/19-of-30、尾部与吞吐代价）、
    caption n=30、key-finding 段改为 "sharpens the design claim"
  - 7Discussion：scope 段补充 LL-S mean 优势即动态目标刻意避免的过冲
  - 9Appendix tab:trace-replay：全表 30-seed 数字 + p 值更新
- 原始数据备份：`outputs/trace_replay_s0_9_backup/`（10-seed 版本存档）。

## 2026-09-09 追加：CASSINI time-shift simulator bug 修复 + 全实验补跑

用户发现 anchor 表 CASSINI SAS=0.4249 "比 Fair 还低很多，不符合常理"，排查发现模拟器级 bug。

### Bug 与修复

- **现象**：所有给 CASSINI 设置 `comm_offset_ms` 的实验中，CASSINI 的 SAS/P-attn 系统性塌陷
  （如 anchor SAS 0.4249、s3 消融 SLO 10.8%），远低于其 Fair 分配应有的水平。
- **根因**：`simulator.py` 在每轮迭代的 collective flow 发射逻辑中都重复应用
  `job.comm_offset_ms`，导致相位偏移随迭代次数累积放大（应仅在首轮发射时应用一次）。
- **修复**：`job.py` 的 `comm_offset_ms` 改为 property + `_comm_offset_pending` 标志
  （仅在 pending 时允许应用一次）；`simulator.py` 发射 flow 时按 `apply_offset` 判定。
  对 `comm_offset_ms=0` 的所有策略（Fair/SRPT/CRUX/DF/LL-S/LongLiu）是完全 no-op。
- **单元测试**：`tests/test_cassini_offset.py`（同时修复 num_workers=2 无迭代问题），全部通过。

### 补跑清单（复用原脚本，重跑受影响数据）

| 实验 | 脚本 | 规模 | 状态 |
|------|------|------|------|
| Trace replay | `experiments/exp_trace_replay.py` | 30 seeds × 7 策略 | ✅ |
| E11 overlap | `experiments/exp_e11_overlap.py` | CASSINI 臂 × 5 seeds × 10 配置 | ✅ |
| E15 straggler | `experiments/exp_e15_straggler.py` | CASSINI 臂 × 5 seeds × 4 因子 | ✅ |
| E17 mixed | `experiments/exp_e17_mixed_collective.py` | CASSINI 臂 × 10 seeds × 4 档 | ✅ |
| E1/E2'/E2-pro 主表 | `experiments/exp_v3_batch3_formal.py` | CASSINI 臂 × 10 seeds × 12 配置 | ✅ |
| S3 消融 | `experiments/exp_s3_component_ablation.py` | CASSINI + Staggered 变体 × 10 seeds | ✅ |
| Anchor 表 | `figure_pipeline/data/evidence/anchor/baseline_regen.py` | 7 策略 × 10 seeds（全量） | ✅ |
| E3/E3' swap | `figure_pipeline/data/e3_swap/exp_e3_swap.py` | 7 策略 × 5 seeds × 2 场景（全量） | ✅ |

非 CASSINI 策略在同 seed 下逐字节复现（确定性自检通过），数据可复核。

### 结论修正（论文相应更新）

- **trace**：CASSINI P-attn $20.7\% \to 37.4\pm7.4\%$，排序回到 Fair 与 CRUX 之间；
  "CASSINI 最差" 系 bug 伪影，撤回。
- **E11**：CASSINI 500G $45.0\% \to 65.0\%$（400G 起点 $82.5\%$）。
- **E17**：CASSINI p=0.39 不显著，论文删除该显著性表述（DF/CRUX/LL-S 保留）。
- **E1**：630G CASSINI 修复后与 Fair 并列最优 baseline（91.3%），正文改为
  "best baselines (Fair, CASSINI)"。
- **S3 消融**：Staggered 变体（相位错开）SLO 39.2±6.6% 与 LongLiu 无显著差异（p=0.186），
  旧结论 "staggering collapses" 系 bug 伪影；CASSINI 短板源于其 allocation
  （25.8±6.2%，p=0.0067 vs LongLiu；iters 少 ~8%，p=3.4e-4），9Appendix S3 段重写。
- **anchor 表**：CASSINI 行数字更新（见 `figure_pipeline/data/anchor/per_policy_results.json`，
  tab:anchor 由 `_draw_final_v3.py` 自动生成 `table1_anchor.tex`）。
- **E3/E3' swap**：CASSINI 臂更新（fig1_hero 不含 CASSINI，图不变；summary JSON 更新）。
  重跑后论文引用数字交叉验证全部保持一致：LongLiu W1/W3 $100.0\pm0.0\%$（两场景）、
  E3 DF W3 $96.7\pm6.7\%$、LL-S W3 $60.0\pm13.3\%$；E3' DF W3 $25.0\%$、
  CRUX W3 $47.5\%$ 且 S-cont $0.659{\approx}0.66$；脚本内 pre-registered
  verification 全部 PASS。E3 段论文文字无需修改。
  CASSINI 修复后 E3 W3=33.3%、E3' W3=17.5%（论文未直接引用）。

### 数据备份

旧（bug 期）CASSINI 数据归档于 `outputs/cassini_offset_fix_backup/`：
`{v3_batch3_formal, e11_overlap, e15_straggler, e17_mixed_collective, s3_component_ablation,
anchor_regen_v1, e3_swap}`。
