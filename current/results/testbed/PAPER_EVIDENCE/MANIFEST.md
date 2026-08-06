# PAPER_EVIDENCE — 物理侧证据清单

> 台账编号：a-e 对应物理证据五支柱
> 最后更新：2026-07-27（Fig-6 + T-3 登记）
> 源文档：`QUOTA_EXPERIMENT_RESULTS.md` 最终版
> 统计口径：3/4 决定性 + 1/4 重叠不定（见 02_replications；rep1 md5 确认为原件副本，已排除）

---

## 目录结构

```
PAPER_EVIDENCE/
├── MANIFEST.md                   # 本文件
├── DO_NOT_USE.md                 # 作废/排除数据清单
├── 00_documents/                 # 最终文档 (a-e 汇总)
│   ├── HANDOFF_physical_evidence.md  # 论文支撑材料交接清单
│   ├── t3_data_table.csv             # T-3 拓扑与映射验证数据
│   ├── ttarget_alignment_report.md   # T_target 对齐核查报告
│   ├── sec6_6_eq5.tex                # §6.6 EQ5 LaTeX 正文
│   └── sec6_6_eq5_crossref.tex       # 正文→CSV 逐格对照表（审计件，不进正文）
├── 01_V6_main/                   # V6-P4 正式两轮 — 仅原件5件 (c)
├── 02_replications/              # 统计重演 ×2 + 覆写备份 (c)
├── 03_class_probe/               # 全类探测 DSCP 0-56 (e)
├── 04_226_probe/                 # 226 分类能力探针三轮 (d)
├── 05_tc_map/                    # DSCP→TC 映射修正记录 (b)
├── 06_calibration/               # 6G 校准 + T_target 标定
└── 07_V5_diagnosis/              # V5 漂移归因 (a)
```

---

## 台账 a-e

### a. 机制环路证据（V4/V5/V6 闭环）
**路径**：`07_V5_diagnosis/`, `01_V6_main/`（π 轨迹）
**内容**：
- V5 π→priority 闭环轨迹（jobA: P2→P4→P2, jobB: 全程 P6）
- V6 π 轨迹（Phase 2 tight B P2: +0.22~0.24）
- CRUX 静态 P3 对照
**论文用途**：证明 Progress Deficit → priority 翻转环路在真实 RoCE 上按设计工作

### b. 映射反转发现 + 修正（系统洞见）
**路径**：`05_tc_map/`
**内容**：
- 旧映射（p×8: P6→DSCP=48→tc:6）配置快照
- 新映射（P6→DSCP=8→tc:0）修正配置快照
- mlnx_qos 采集命令记录
- 发现路径：倒挂→全类探测→修正→验证
**论文用途**：独立一小节——软件优先级命名空间与硬件 TC 解耦的陷阱

### c. V6 Outcome：Phase 2 tight 13–17% 无重叠优势
**路径**：`01_V6_main/`（console.log 原件 + 灭失声明）, `02_replications/`（统计重演 CSV 全量）
**内容**：
- V6-P4 正式两轮：console.log 原件（含 epoch 级 results_summary）+ run_meta + 灭失声明
- 原始 CSV 因命名冲突被后续重演覆盖，文档表格值与 console.log 为权威记录
- 统计重演 ×2：复制1（md5 确认为原件副本，排除）、复制2（1/4 重叠不定）
- `v6_replication_1_overlaid/`：从 01_V6_main 迁出的复制1覆写文件（与 `v6_replication_1/` 完全重复，保留仅作血缘追溯）
- **3/4 轮决定性无区间重叠**（平均 LL −13–17%），1/4 轮统计不可分
- 论文 §6.6 正文：`00_documents/sec6_6_eq5.tex`（含 tab:eq5-summary 汇总表）
- 正文数字→CSV 逐格对照：`00_documents/sec6_6_eq5_crossref.tex`（审计件）
- Fig-6 绘图：`FIGURE_REGISTRY/fig6_v6physical/plot_fig6.py`（脚本）+ `fig6_testbed.pdf`（PDF 矢量）+ `fig6_testbed_600.png`（600dpi PNG）+ `fig6_self_check.csv`（自校验对照）
**论文用途**：物理床 outcome 核心证据

### d. 226 不分类的拓扑约束声明
**路径**：`04_226_probe/`
**内容**：
- 反向模式 iperf3 探针三轮
- NIC 计数器确认：全部 8 prio 仅 prio0 有增量
- 结论：226 NIC 硬件/驱动不支持 DSCP→prio 分类
**论文用途**：披露不对称拓扑——226 侧交换机看不到 DSCP 优先级分类

### e. 探测实证 DSCP→TC 映射表
**路径**：`03_class_probe/`, `05_tc_map/`
**内容**：
- DSCP 0-56 全表 vs 6G P3 背景流
- 10.1 NIC tx_prio 计数器（P3=5.3TB, P4=49GB, P6=1.2GB）
- 226 NIC 计数器（全部 0 — 不分类互验）
- RoCE 硬件计数器基线
**论文用途**：映射表实证列——NIC 级验证而非推测
- T-3 三合一 LaTeX 表：`FIGURE_REGISTRY/t3_topo/tab_t3.tex`（\label{tab:testbed-hw}，含 tc_map 修正 + 全类探测 + NIC 不对称三子表）
- 逐格对照：`FIGURE_REGISTRY/t3_topo/tab_t3_crossref.tex`（审计件）

---

## 入档规则（本目录遵守）

1. 只收录有完整 run_meta 的运行（五要素：背景流速、持续时间、钳位值、顺序、NIC 计数器摘要）
2. DO_NOT_USE.md 收录被排除数据及原因
3. 复制不移动，统计口径以文档最终版为准
4. 归档数据须能逐格复现文档表格
