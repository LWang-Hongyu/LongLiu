# LongLiu Simulator (sim-nextgen)

面向多租户 AI 训练网络调度的 **flow-level 离散事件仿真器**，专为 Data Parallel (DDP) 训练场景设计。

论文: *LongLiu: Tier-Aware Dynamic Priority Scheduling for Multi-Tenant DDP Training*

---

## 核心特性

- **Fat-Tree 拓扑感知**：分层带宽建模（host / ToR / spine / rack oversub），ECMP 多路径
- **迭代级 workload**：per-job 按迭代生成 compute + AllReduce 通信事件，barrier 语义（tail flow 判定）
- **6 种调度策略**：LongLiu (v4)、DF (D1)、CRUX、SP (SRPT)、Fair、CASSINI
- **Progress Deficit (π) 反馈**：基于 SLO 松弛系数 ci 的目标迭代时间，动态调整优先级
- **Tiered SLO 分层**：大模型 ci=1.5（紧约束）、中模型 ci=2.0、小模型 ci=3.0（松约束）
- **DSCP 优先级映射**：7 级优先级 + 首轮最高优先级 (DSCP 38) + 同级别 exp(π·K) 加权
- **Poisson 任务到达**：模拟真实云环境任务分布
- **物理原型机校准**：overhead_factor (NCCL/PCIe)、overlap_factor (compute-comm 重叠) 来自实测

---

## 目录结构

```
sim-nextgen/
├── config.yaml                 # 唯一配置源（冻结参数、拓扑、workload）
├── CONVENTION.md               # 语义版本、锚点公式、完整约定链
├── DESIGN.md                   # 完整实现方案文档（算法、参数、权衡）
├── HANDOFF.md                  # 关键决策与约束记录
├── PLAN.md                     # 项目规划与实施路线图
├── requirements.txt            # Python 依赖
├── .gitignore
├── longliu_sim/                # 核心仿真库（受保护源码，见"源码保护"）
│   ├── core/                   # 事件驱动仿真引擎
│   │   ├── simulator.py        # 主事件循环 + 带宽重分配
│   │   └── event.py            # 事件定义
│   ├── network/                # 网络模型
│   │   ├── topology.py         # FatTree 拓扑（冻结，严禁修改）
│   │   ├── link.py             # 链路模型
│   │   └── flow.py             # flow 定义
│   ├── job/                    # 任务模型
│   │   └── job.py              # Job 类（迭代生命周期、barrier）
│   ├── policy/                 # 调度策略实现
│   │   ├── base.py             # Policy 基类
│   │   ├── longliu.py          # LongLiu v4（tier-aware DWRR）
│   │   ├── dwrr.py             # DWRR / DF（deficit-feedback）
│   │   ├── crux.py             # CRUX（GPU-intensity 权重）
│   │   ├── fair.py             # Fair（均分带宽）
│   │   ├── srpt.py             # SRPT / SP（最短剩余处理时间）
│   │   └── cassini.py          # CASSINI（time-shift 交错）
│   ├── trace/                  # Trace 生成与解析
│   │   ├── synthetic.py        # 合成 workload 生成
│   │   ├── lingjun.py          # Alibaba Lingjun trace loader
│   │   └── synthetic_128.py    # 128 节点合成 trace
│   ├── metrics/                # 指标计算与可视化
│   │   ├── stats.py            # SLO attainment、SAS、collision rate
│   │   └── plot.py             # 图表绘制
│   └── utils/                  # 工具
│       ├── config.py           # 配置解析
│       ├── metrics.py          # SAS / target_iter_ms 公式单实现
│       └── model_params.py     # 模型参数量（参数 MB 推导）
├── experiments/                # 实验脚本
│   ├── exp_e10_wfs.py ... exp_e16_beta.py   # E10-E16 正式实验
│   ├── exp_e14_probe_v4.py     # E14 v4 闭式解对照（10-seed）
│   ├── exp_trace_replay.py     # Lingjun trace 时段重放（E09）
│   ├── _quick_scan_e14_*.py    # E14 验证链（扫描/校验）
│   └── ...                     # 早期分析/审计脚本
├── figure_pipeline/            # 图表/证据管线（脚本+数据+图片统一管理）
│   ├── README.md               # 全链索引 + 复现指南（见下）
│   ├── scripts/                # 绘图脚本 + 正式实验软链（experiments/）
│   ├── data/                   # 数据源（E 系列 summary + figure_registry + evidence）
│   └── figs/                   # 图片输出（fig1-fig6, fig_e10-e16, table1/2）
├── configs/                    # 实验配置（按实验分文件）
│   ├── trace_replay.yaml       # E09 trace 重放配置
│   └── fatree_*.yaml           # 拓扑规模配置
├── scripts/                    # 工具脚本（28 个）
│   ├── gatekeeper.py           # 门禁脚本（锚点复现验证）
│   ├── baseline_regen.py       # 基线重生成
│   ├── run_anchor_v4.py        # tab:anchor v4 补跑
│   ├── capacity_accounting.py  # 容量核算
│   └── ...
├── outputs/                    # 实验输出
│   ├── figures/                # 论文图表（fig1-fig6, table1-table3）
│   │   ├── _draw_final_v3.py       # 终版图表绘制脚本
│   │   ├── _draw_trace_compare.py  # fig6 trace 对比绘制脚本
│   │   ├── sections_6_2_to_6_5.tex # 论文 §6.2-6.5 章节（\Name 宏）
│   │   └── section_6_6_trace_replay.tex # 论文 §6.6 trace 章节
│   ├── v3_batch3_formal/       # 正式实验数据
│   ├── trace_replay/           # E09 trace 重放结果（run_meta_*.json, summary.csv）
│   └── ...
├── PAPER_EVIDENCE/             # 论文证据（冻结，只读）
│   ├── FIGURE_REGISTRY/        # 图表数据 CSV（权威数据源，fig2/3/4/6）
│   ├── 01_baseline_anchor/     # 锚点基线数据
│   ├── 03_E1_ladder/           # E1 Ladder 实验
│   ├── 04_E2_orthogonal/       # E2 正交对照实验
│   ├── 05_E3_swap_main/        # E3 Swap 实验
│   ├── 06_D1_mechanism/        # D1 机制分析
│   └── 09_trace_replay/        # E09 trace 重放归档（50 run_meta + summary + config + script）
└── tests/                      # 单元测试
    └── test_simulator.py
```

---

## 快速开始

### 环境要求

- Python >= 3.9
- 依赖：`numpy`, `matplotlib`, `pyyaml`

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
cd /home/why/LongLiu_rebuild/sim-nextgen
python3 tests/test_simulator.py
```

### 运行对比实验

```bash
# 锚点基线实验（3 seeds, 400G, 24 jobs）
python3 experiments/exp_compare.py

# E1 Ladder 实验
python3 experiments/exp_v3_batch1.py

# E3 Swap 实验
python3 experiments/exp_e3_swap.py

# E09 Lingjun trace 时段重放对照实验（5 策略 × 10 seeds）
python3 experiments/exp_trace_replay.py --seeds 10
```

### 生成论文图表

```bash
python3 figure_pipeline/scripts/_draw_final_v3.py
python3 figure_pipeline/scripts/_draw_trace_compare.py   # fig6：合成 E1 vs Lingjun trace 对比
```

图表输出至 `figure_pipeline/figs/`，包括：
- `fig1_hero.pdf` — 全景英雄图（v4 vs DF 截断窗轨迹）
- `fig2_e1_ladder.pdf` — E1 保障阶梯图
- `fig3_e2_orthogonal.pdf` — E2 正交对照图
- `fig4_d1_trajectory.pdf` — D1 瞬态轨迹图
- `fig5_pi_timeseries.pdf` — π 机制证据图
- `fig6_trace_compare.pdf` — 合成 E1 vs Lingjun trace 对比图（§6.6）
- `table1_anchor.tex` — 锚点基线表
- `table2_e2pro.tex` — E2-pro 正相关对照表

E10-E16 系列图（`fig_e10_wfs` ~ `fig_e16_beta`，含 `fig_e11b_overlap_waste`、`fig_e14_anchor`）
与绘图脚本见 [EXPERIMENTS_ARCHIVE.md](EXPERIMENTS_ARCHIVE.md) 与 [figure_pipeline/README.md](figure_pipeline/README.md)。

---

## 配置说明

所有参数集中在 [config.yaml](config.yaml)，分为以下几个部分：

| 配置项 | 说明 | 关键参数 |
|--------|------|----------|
| `frozen` | 冻结参数（锚点语义） | `overhead_factor=1.3`, `overlap_factor=0.85`, `K=2.0` |
| `topology` | 拓扑默认值 | `k=4`, `host_bw=100G`, `spine_bw=400G` |
| `simulation` | 仿真设置 | `duration_ms=600000`, `seeds=[0..9]` |
| `v2_anchor_workload` | 锚点 workload（24 jobs） | 12 大 / 8 中 / 4 小模型 |
| `v4` | LongLiu v4 分配器参数 | `beta=0.5` (standard 降级界限) |

**冻结令**：`frozen.*`、`topology.*`、`v2_anchor_workload` 的变更需显式批准并在 HANDOFF 记录。

---

## 论文实验体系

| 实验编号 | 名称 | 数据源 | 用途 |
|----------|------|--------|------|
| 01 | Baseline Anchor | `PAPER_EVIDENCE/01_baseline_anchor/` | 锚点基线表 (T-1)，验证各策略在 400G/24 jobs 下的基础性能 |
| 03 | E1 Ladder | `PAPER_EVIDENCE/03_E1_ladder/` | 保障阶梯 (Fig-2)：稀缺区 v4 优势、充裕区收敛 |
| 04 | E2 Orthogonal | `PAPER_EVIDENCE/04_E2_orthogonal/` | 正交对照 (Fig-3)：v4 对 CRUX 的 workload 结构优势 |
| 05 | E3 Swap | `PAPER_EVIDENCE/05_E3_swap_main/` | 动态 Swap (Fig-1)：v4 无瞬态 vs DF/CRUX 换档瞬态 |
| 06 | D1 Mechanism | `PAPER_EVIDENCE/06_D1_mechanism/` | π 机制证据 (Fig-4/5)：DF 的表达力墙诊断 |
| 09 | Trace Replay | `PAPER_EVIDENCE/09_trace_replay/` | Lingjun 2023 trace 时段重放 (Fig-6, §6.6)：真实到达模式下 v4=75.0% vs Fair=35.7% (p=1.09e-06) |
| 10 | WFS Baseline | `outputs/e10_wfs/` | 加权公平共享基线对照：证明 LongLiu 闭式解优于线性权重映射 |
| 11 | Overlap Sensitivity | `outputs/e11_overlap/` | 串行模型保守性敏感性分析：验证不同重叠率下的鲁棒性 |
| 12 | DSCP Quantization | `outputs/e12_dscp/` | DSCP 量化误差宏观影响：证明 7 级量化下 SLO 达成率下降 <5% |
| 13 | Window Sensitivity | `outputs/e13_window/` | 窗口大小 W 敏感性（修复后定稿）：W 无统计显著差异，W=20 均值最高、方差最低（见 `EXPERIMENTS_ARCHIVE.md` §E13） |
| 14 | Anchor Freezing | `outputs/e14_probe/` | 锚点冻结设计案例（v2）：基线 9-10/30 job 硬冻结；naive 探测在 SP 下有害；被动校准消除冻结、方差 -25%，但均值不显著；v4 闭式解补跑 83.7±6.9%、零冻结（见 `EXPERIMENTS_ARCHIVE.md` §E14） |
| 15 | Straggler Injection | `outputs/e15_straggler/` | Straggler 注入实验（400G）：v4 全因子领先，窗口平均 + 闭式重分配吸收 2-5× 计算膨胀 |
| 11b | Allocation Precision | `outputs/e11b_overlap_waste/` | 审稿回应：v4 分配精度 ≈1.0（按需精确），WFS 权重分配超配 0.70-0.88 |
| 16 | β Sensitivity | `outputs/e16_beta/` | 审稿回应：P-attn 对 β∈[0.3,1.0] 平坦（85-97.5%），S-cont 单调 0.978→0.999——运营自由度大 |

所有数据数字必须与 `PAPER_EVIDENCE/FIGURE_REGISTRY/` 中的 CSV 逐格一致。实验数据目录已冻结（chmod a-w），任何修改需先解冻并记录。

---

## 源码保护

`longliu_sim/` 是受保护的核心仿真源码，任何修改都会影响所有已归档结果的可复现性。约束如下：

1. **禁止未批准修改**：`longliu_sim/` 下任何文件的变更需先说明动机、评估对既有结果（E01-E09）的影响，经确认后修改，并在 [HANDOFF.md](HANDOFF.md) 记录。
2. **最高冻结等级**：`network/topology.py`、`utils/metrics.py`（公式单实现）、`config.yaml` 的 `frozen.*` 为最高冻结等级，严禁修改（见 [CONVENTION.md](CONVENTION.md)）。
3. **实验脚本隔离**：实验参数/场景变更一律通过 `configs/` 新配置文件与 `experiments/` 新脚本实现，不得改动 `longliu_sim/` 来"适配"某个实验。
4. **归档校验**：任何源码变更后，需重跑 `scripts/gatekeeper.py`（锚点复现验证）确认历史 headline 数字不变。
5. **Git 纪律**：`longliu_sim/` 的变更必须显式提交并注明影响范围；`PAPER_EVIDENCE/` 保持只读（chmod a-w），需修改时先解冻、记录、再冻结。

---

## 约定与纪律

1. **公式单实现**：SAS / target_iter_ms 计算仅在 `utils/metrics.py` 中定义，禁止复制粘贴
2. **配置集中**：所有参数在 `config.yaml` 中定义，代码内禁止字面量
3. **拓扑冻结**：`network/topology.py` 的路由/链路模型严禁修改，变更会使所有历史结果作废
4. **口径统一**：通信量 = 2 × params × 2 bytes (gradient sync, fp16)；SLO attainment = 累计平均迭代时间
5. **引用纪律**：论文写作中：
   - DF 是自实现 baseline，不引用文献
   - 名称统一：LongLiu (v4)、DF (D1)、CRUX、SP (SRPT)、Fair
   - 禁止 Max-Min Fair、D1、v4、SRPT
6. **审计链**：任何偏离预设分支的结果需上报，禁止自行决策

详见 [CONVENTION.md](CONVENTION.md) 和 [HANDOFF.md](HANDOFF.md)。

---

## 关键技术决策

- **Flow-level 而非 Packet-level**：调度策略设计为核心，flow-level 足够表达迭代级 SLO；packet-level 单次数小时，不适合参数扫描
- **仿真器将 fabric 抽象为 spine 争抢 + pod 内无损**：ToR 上行与 host NIC 不参与争抢建模，v2/v3/v4 全部结果在此假设下产生
- **仅支持 DDP**：排除 TP/PP/EP/CP 混合并行，与论文物理原型一致
- **per-NIC 上限不建模**：sim 的带宽分配仅在 spine link 层面，场景设计自行维护物理合理性约束（per_flow_attain ≤ 100G）

---

## Git 仓库

```
https://github.com/LWang-Hongyu/LongLiu.git
```

- **`master`** 分支：实现端代码（物理原型、NCCL 插件等）
- **`simulation`** 分支：仿真端代码（本目录）
- 大文件（trace.jsonl, 100-178MB）已通过 `.gitignore` 排除

---

## 引用

如果使用本仿真器进行研究，请引用：

```bibtex
@inproceedings{longliu2025,
  title     = {LongLiu: Tier-Aware Dynamic Priority Scheduling for Multi-Tenant DDP Training},
  author    = {Hongyu Wang and others},
  booktitle = {IEEE INFOCOM},
  year      = {2025}
}
```
