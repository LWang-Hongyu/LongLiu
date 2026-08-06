# LongLiu 项目汇总

> 最后更新：2026-08-06

---

## 一、目录结构总览

```
LongLiu_rebuild/
├── legacy/                          # 旧实现 + 早期机制验证实验
│   ├── nccl-dscp/                   # C++ NCCL Shim 修改版（已弃用）
│   ├── nccl-master/                 # NCCL 2.18.3 源码（编译 nccl-dscp 用）
│   ├── LongLiu_py/                  # 早期 Python 原型（未使用）
│   ├── experiments_mechanism/       # P1-P3 机制验证实验
│   └── *.md                         # 旧文档（DSCP验证日志等）
│
├── current/                         # 当前主力实现 + 评估实验
│   ├── multi_comm_slo/              # Python + C 多 Communicator（当前主力）
│   ├── experiments_evaluation/      # P4/V6/ExpA/B/C 评估实验
│   │   ├── P4_dumbbell_slo/         # 哑铃 SLO 对比 + V6 物理实验
│   │   └── experiments_supplementary/  # 补充实验 A/B/C
│   └── results/                     # 实验结果汇总 + 论文图表
│       ├── testbed/                 # 物理床结果（QUOTA_EXPERIMENT_RESULTS.md 等）
│       ├── physical_result/         # 物理实验原始数据
│       └── fig_arch.*               # 架构图
│
── README.md                        # 项目总览
── PROJECTS_SUMMARY.md              # 本文件
├── LongLiu_补充实验方案.md           # 补充实验规划
├── LongLiu_INFOCOM_Draft.md         # 论文草稿
├── LONGLIU_PAPER_PLAN.md            # 论文计划
└── LONGLIU_IMPLEMENTATION.md        # 实现说明
```

---

## 二、实现版本

### 1. nccl-dscp（C++ NCCL Shim 修改版）— legacy

**路径**：`legacy/nccl-dscp/`

**状态**：⚠️ 早期实验使用，P4 之后已不再使用

**核心文件**：
```
legacy/nccl-dscp/
├── src/
│   ├── misc/
│   │   ├── dscp_adapter.cc      # 核心：epoch管理、Ui计算、DSCP映射、EMA带宽
│   │   └── comm_stats.cc        # per-iteration统计收集 + JSON导出
│   ├── include/
│   │   ├── dscp_adapter.h       # 数据结构定义
│   │   └── comm_stats.h         # 迭代统计结构
│   ├── transport/
│   │   └── net_ib.cc            # QP创建时TC注入 + 运行时ibv_modify_qp动态修改
│   ├── enqueue.cc               # NCCL op触发epoch检查
│   └── init.cc                  # DSCP adapter初始化
├── build/
│   └── libnccl.so.2             # 编译产物（替换PyTorch自带NCCL）
└── DESIGN_REFERENCE.md          # 代码分析文档
```

**架构**：
- 直接修改 NCCL 2.18.3 源码
- 运行时通过 `ibv_modify_qp(IBV_QP_AV)` 动态修改 QP 的 `traffic_class`
- 所有 QP 共享同一个 communicator
- 需要完整编译 NCCL，替换 PyTorch 的 libnccl.so

**π 公式**：Ui = ai/ei（墙钟时间比，含 compute）

**DSCP 映射**：7 级连续值 {0, 18, 26, 28, 48, 50, 52}

**TC 计算**：`tc = dscp << 2`（标准 IP ToS）

**使用的实验**：
- P1：DSCP 注入验证（证明 TC 能正确映射到线路）
- P2：双峰间隔分布（证明负载是间歇争抢）
- P3：EMA 带宽收敛（证明保护门有效）

---

### 2. multi_comm_slo（Python + C 多 Communicator）— current

**路径**：`current/multi_comm_slo/`

**状态**：✅ **当前主力实现**，P4/V6/ExpB/ExpC 均使用

**核心文件**：
```
current/multi_comm_slo/
├── src/
│   ├── multi_comm.c             # C库：预创建7个communicator（P0-P6），运行时切换索引
│   ├── multi_comm.h             # 头文件
│   ├── slo_scheduler.py         # Python调度器：π计算、优先级映射、DSCP映射
│   └── Makefile                 # 编译脚本
├── build/
│   └── libmulti_comm.so         # 编译产物
└── DESIGN.md                    # 设计文档（含与nccl-dscp对比）
```

**架构**：
- 不修改 NCCL 源码，通过 Python ctypes 调用 C 库
- 每个优先级预创建一个独立的 NCCL communicator（创建时设置 `config.trafficClass`）
- 运行时通过 `multi_comm_set_priority()` 切换 `current_priority` 索引
- 独立 .so，通过 LD_LIBRARY_PATH 加载，不影响 PyTorch 内置 NCCL

**π 公式**：π = A_i/(c_i × T_target × k_i) - 1（纯通信时间比）

**DSCP 映射**：6 级离散值：P6→8, P4→0, P3→16, P2→24, P1→32, P0→40

**TC 计算**：硬编码映射表 `prio_dscp[]`

**使用的实验**：
- P4：哑铃 SLO 对比（5 策略：Fair/LongLiu/CRUX/SRPT/MLTCP）
- V6：LongLiu vs CRUX 物理对比（4 轮，13-15% 优势）
- Experiment B：tier swap 动态锚点（4 轮 AB×2 + BA×2）
- Experiment C v1：规模阶梯（27 轮，3 regimes × 3 arms × 3 rounds）

---

### 3. epoch_emulator + alloc_daemon（CPU 模拟器 + 守护进程）— current

**路径**：`current/experiments_evaluation/experiments_supplementary/03_exp_C_scale_ladder/`

**状态**：✅ **ExpC v2 专用**

**核心文件**：
```
03_exp_C_scale_ladder/
├── emulator/
│   ├── epoch_emulator.c         # RDMA write模拟器，支持Multi-QP（4个QP对应P6/P4/P2/P1）
│   └── build/epoch_emulator     # 编译产物
├── daemon/
│   ├── alloc_daemon.py          # v1守护进程
│   └── alloc_daemon_v2.py       # v2守护进程（c_policy/c_eval拆分）
├── scenarios/
│   └── scenarios_v2.json        # 场景配置（S1/S2双场景，5 regimes）
├── scripts/
│   ├── run_expC.sh              # v1运行脚本
│   └── run_expC_v2.sh           # v2运行脚本
├── data_v2/                     # v2实验数据（75轮）
├── analysis/
│   ├── plot_expC_v2.py          # 绘图脚本（生成3张IEEE风格图）
│   ├── expC_v2_results.md       # 结果文档
│   └── expC_v2_analysis.md      # 分析文档
└── README.md                    # 实验说明
```

**架构**：
- **emulator**：C 程序，模拟 RDMA 通信 + 计算，每 job 一对进程（client/server）
- **daemon**：Python 守护进程，读取 emulator 统计，运行 SLOScheduler（import 自 multi_comm_slo），写 DSCP 控制文件
- **Multi-QP 方案**：mlx5 不支持 `ibv_modify_qp(IBV_QP_AV)`，预创建 4 个 QP（P6/P4/P2/P1），运行时切换 `active_qp_idx`

**调度逻辑**：与 multi_comm_slo 完全一致（同一份 SLOScheduler 代码）

**使用的实验**：
- Experiment C v2：规模/稀缺锚点（75 轮，5 regimes × 3 arms × 5 rounds）

---

## 三、实验目录

### legacy/experiments_mechanism/（早期机制验证 P1-P3）

```
legacy/experiments_mechanism/
├── P1_dscp_injection/           # DSCP注入验证（nccl-dscp）
├── P2_bimodal_interval/         # 双峰间隔分布（nccl-dscp）
└── P3_ema_convergence/          # EMA带宽收敛（nccl-dscp）
```

### current/experiments_evaluation/（评估实验 P4 + 补充实验 A/B/C）

```
current/experiments_evaluation/
├── P4_dumbbell_slo/             # 哑铃SLO对比 + V6物理实验（multi_comm_slo）
│   ├── p4_job1_crux.py          # CRUX job
│   ├── p4_job2.py               # LongLiu job
│   ├── bench_payload.py         # payload基准测试
│   ├── run_v6_calib_atomic.sh   # V6校准运行脚本
│   ── FIGURE_REGISTRY/         # 图表注册（含fig6_v6physical）
│
└── experiments_supplementary/   # 补充实验 A/B/C
    ├── 00_prerequisites/        # 前置事项（DSCP探针、solo基线）
    ├── 01_exp_A_static_anchor/  # 实验A：静态锚点（HW vs Sim保真度）
    ├── 02_exp_B_tier_swap/      # 实验B：动态锚点tier swap
    ├── 03_exp_C_scale_ladder/   # 实验C：规模/稀缺锚点CPU模拟器
    │   ├── data/                # v1数据（27轮）
    │   └── data_v2/             # v2数据（75轮）
    ├── 04_bonus/                # 加分项（shim开销、锚点收敛等）
    ├── figure/                  # 论文图表
    └── Evaluation.tex           # 论文评估章节LaTeX
```

### current/results/（实验结果汇总）

```
current/results/
├── testbed/                     # 物理床结果
│   ├── QUOTA_EXPERIMENT_RESULTS.md  # 实验结果汇总表
│   ├── HANDOFF_physical_evidence.md # 物理证据包
│   ├── DWRR_RUNBOOK.md          # DWRR实验手册
│   ├── FIGURE_REGISTRY/         # 图表注册
│   └── PAPER_EVIDENCE/          # 论文证据归档
├── physical_result/             # 物理实验原始数据
│   ├── probe_dscp_priority.sh   # DSCP优先级探针
│   ├── probe_226_classify.sh    # 226 NIC分类能力探针
│   └── v6_replication_{1,2}/    # V6复制实验
├── fig_arch.py                  # 架构图绘制脚本
├── fig_arch_longliu.png/pdf     # 架构图输出
├── P4_DSCP_实验完整记录.md       # P4实验记录
└── 实现进度与结果汇总.md         # 进度汇总
```

---

## 四、依赖关系

```
legacy/nccl-dscp (7/16)
    ↓ 机制验证（P1-P3）
    ↓ 编译复杂、兼容性差
    ↓
current/multi_comm_slo (8/6) ← 当前主力
    ↓ P4/V6/ExpB/ExpC v1
    ↓ SLOScheduler被复用
    ↓
current/.../03_exp_C_scale_ladder/daemon/alloc_daemon_v2.py (8/6) ← ExpC v2专用
    ↓ import SLOScheduler from multi_comm_slo
    ↓ 调度逻辑完全一致
    ↓
实验结果 → plot_expC_v2.py → 论文图表 → Evaluation.tex
```

---

## 五、关键结论

1. **nccl-dscp 没有完全过期**，P1-P3 机制验证实验仍依赖它（在 `legacy/` 中归档）
2. **multi_comm_slo 是当前主力**，P4 之后的所有实验均使用它（在 `current/` 中）
3. **alloc_daemon_v2 复用了 multi_comm_slo 的 SLOScheduler**，调度逻辑完全一致
4. **三套实现的 π 公式不同**：
   - nccl-dscp：Ui = ai/ei（墙钟时间，含 compute）
   - multi_comm_slo：π = A_i/(c_i×T_target×k_i) - 1（纯通信时间）
   - alloc_daemon：同 multi_comm_slo（同一份代码）
5. **DSCP 注入方式不同**：
   - nccl-dscp：运行时 `ibv_modify_qp` 动态修改
   - multi_comm_slo：预创建 7 个 communicator，切换索引
   - alloc_daemon：Multi-QP 预创建 4 个 QP，切换 `active_qp_idx`
6. **所有物理实验跑的是纯通信 benchmark**（AllReduce 随机 tensor + GPU sleep），不是真实模型训练
