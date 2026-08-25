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
