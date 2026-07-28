以下是整合后的 **完整设计与协作规范文档**，可直接交付给所有 AI 助手。

---

# LongLiu v3.0 设计与 AI 协作统一规范

> **版本**：v3.0-INFOCOM  
> **日期**：2026-07-04  
> **适用范围**：所有负责 LongLiu 代码实现、实验仿真、论文写作的 AI 助手  
> **前置约束**：本文档为最高优先级指令，任何子任务规范不得与之冲突。

---

## 第一部分：架构红线（Architecture Red Line）

### 1.1 v3.0 核心架构声明
LongLiu v3.0 是 **NCCL-centric 的纯发送端方案**。所有实现必须围绕 **NCCL communicator** 作为状态隔离与调度单元。

### 1.2 明确废弃组件（绝对禁止出现）
以下组件在 v3.0 中已彻底废弃，任何代码、文档、图表、注释中均不得引用、复用或退回：

| 废弃组件 | 旧版本 | 替换方案 | 禁止形式 |
|---------|--------|---------|---------|
| Count-Min Sketch (NIC) | ToN/Conext v2.0 | NCCL Proxy 状态机 | 代码、注释、文档、PPT |
| 交换机优先级衰减（PFC/RunID/RunFlag） | ToN/Conext v2.0 | 无需交换机修改 | 任何提及 PFC/RunID 的内容 |
| 包间隔阈值推断（100ms 启发式） | ToN/Conext v2.0 | NCCL collective 完成事件 / PyTorch Hook | 固定阈值检测迭代边界 |
| JCT_SOLO 离线测量 | ToN v2.0 | 两阶段 $T_{\text{target}}$ 自动测量 | 需要 solo 运行或隔离 profiling |
| ibv_modify_qp 动态 TC | 实验探索 | NCCL proxy quota 控制 | RDMA QP 动态修改 |
| DSCP/TC 标记 | 实验探索 | 无 | 内核流量控制标记 |

**AI 自查命令**：提交前执行 `grep -ri "Count-Min\|Sketch\|PFC\|RunID\|RunFlag\|JCT_SOLO\|100ms.*threshold\|ibv_modify_qp\|DSCP" .` 若返回非空，立即删除并重构。

### 1.3 统一术语表
所有输出文件必须使用以下术语，禁用旧术语：

| 禁用旧术语 | 强制新术语 |
|-----------|-----------|
| Urgency Factor $U_i$ | Progress Deficit $\pi_i$ |
| Urgency Coefficient | Slack Coefficient $c_i$ |
| Count-Min Sketch | 迭代边界识别（Iteration Boundary Detection） |
| Priority Decay（交换机） | 动态优先级映射（Dynamic Priority Mapping） |
| Size/Time/Count Sketch | 不适用（已删除） |
| 交换机队列 | NCCL Proxy Quota |

---

## 第二部分：核心设计规格（Core Design Specification）

### 2.1 全局状态结构（Global State）
每个 NCCL communicator 维护独立的 `LongLiuState`：

```c
struct LongLiuState {
    // Progress Deficit 计算
    double A_i;              // 累积实际通信时间 (μs)
    uint64_t k_i;            // 已完成迭代数
    double T_target;         // 目标迭代时间 (μs)
    double c_i;              // 松弛系数 (环境变量 LONGLIU_C_I)
    
    // T_target 测量状态
    int stage;               // 0=RTT阶段, 1=修正阶段, 2=稳定阶段
    double T_ema;            // EMA 平滑后的 T_target
    double ema_alpha;        // EMA 系数 (默认 0.3)
    bool has_highest_priority; // 当前是否获得最高优先级
    
    // 迭代边界识别（自动推断路径）
    uint64_t last_collective_end;  // 上次 collective 完成时间戳 (ns)
    uint64_t intra_iter_gap;        // 迭代内间隔阈值 (ns)
    uint64_t inter_iter_gap;        // 迭代间间隔阈值 (ns)
    int gap_learning_count;         // 已学习样本数
    bool boundary_learned;          // 是否已完成学习
    
    // 优先级与配额
    int priority_level;      // 当前优先级 P0-P6
    int quota;               // 本轮 operation 配额
    int base_quota;          // 默认配额 (环境变量 LONGLIU_BASE_W)
    
    // 控制开关
    bool enabled;            // LONGLIU_ENABLED
};
```

**原则**：状态按 communicator 隔离，禁止按 flow/connection 共享。

### 2.2 Progress Deficit 计算
**公式**（必须严格实现）：
$$\pi_i = \frac{A_i / k_i}{c_i \cdot T_{\text{target}}} - 1$$

**离散优先级映射表**（必须严格遵循）：

| 优先级 | $\pi$ 范围 | 配额系数 | 语义 |
|--------|-----------|---------|------|
| P6 | $\pi > 0.3$ | $4 \times$ base | 严重违约，抢占带宽 |
| P5 | $0.15 < \pi \leq 0.3$ | $3 \times$ base | 中度违约 |
| P4 | $0.05 < \pi \leq 0.15$ | $2 \times$ base | 轻度违约 |
| P3 | $-0.05 < \pi \leq 0.05$ | $1 \times$ base | SLO 边界 |
| P2 | $-0.2 < \pi \leq -0.05$ | $0.8 \times$ base | 正常，轻微让出 |
| P1 | $-0.5 < \pi \leq -0.2$ | $0.5 \times$ base | 超前，让出带宽 |
| P0 | $\pi \leq -0.5$ | $0.2 \times$ base | 严重超前，最小带宽 |
| P7 | 保留 | — | 控制包/ACK（不用于数据） |

**关键原则**：
- 纯 SLO 驱动，当前版本 **不引入 GPU Intensity 作为二级排序键**（双层级映射移至 §VII Discussion）
- 每个 job 独立计算 $\pi_i$，无需全局协调
- `quota` 必须为整数且 $\geq 1$

### 2.3 两阶段 $T_{\text{target}}$ 测量

**阶段一：RTT 探测（启动后 0-3 个迭代）**
- 发送 RDMA read 或 small send 探测包
- 估计：$T_{\text{target}} \approx \text{RTT} \times \text{steps\_per\_iter} / 0.3$（保守估计，假设计算占 70%）
- 输出：初始 $T_{\text{ema}}$，进入阶段二

**阶段二：最高优先级修正（运行时）**
- 触发条件：连续 3 个 iteration 获得 P6（最高数据优先级）
- 更新：$T_{\text{ema}} = \alpha \cdot t_{\text{iter}} + (1-\alpha) \cdot T_{\text{ema}}$
- 输出：精修后的 $T_{\text{target}} = T_{\text{ema}}$，进入阶段三（稳定）

**状态转换**：
```
[INIT] --RTT探测--> [STAGE1] --等待连续3轮P6--> [STAGE2] --EMA更新--> [STABLE]
```

### 2.4 迭代边界识别（双路径）

**路径 A：NCCL 自动推断（默认）**
- 收集前 5 个 iteration 的 collective 完成间隔
- 双模态聚类：排序后找最大间隔，区分 `intra_iter_gap`（小）与 `inter_iter_gap`（大）
- 判定：间隔 > `inter_iter_gap × 0.5` 则触发新迭代

**路径 B：PyTorch Hook 显式标记（增强）**
- 自动注入 `DistributedDataParallel.forward` 循环
- 调用 `ncclLongLiuIterStart(comm)` / `ncclLongLiuIterEnd(comm)`
- 100% 精确，无视并行策略复杂度

**原则**：自动推断为默认；显式 Hook 仅用于 PP 或极短迭代（<20ms）场景。

### 2.5 NCCL Proxy 修改
- **文件**：`src/proxy.h`（结构体）、`src/proxy.cc`（quota 控制）、`src/libnccl.map`（符号导出）
- **核心逻辑**：`ncclProxyProgress` 每轮循环读取 `priority_level`，映射为 `quota`，限制本轮处理 ops 数量
- **禁止**：修改交换机、内核模块、DSCP/TC 标记

---

## 第三部分：目录与文件组织

```
LongLiu_v3.0/
├── DESIGN_DOC/          # 设计文档（只读模板，AI 不得修改）
│   ├── paper_plan.md
│   └── ai_spec.md       # 本文件
├── src/                 # NCCL 修改源码
│   ├── proxy.h
│   ├── proxy.cc
│   └── libnccl.map
├── hook/                # PyTorch Hook
│   └── longliu_hook.py
├── testbed/             # 物理原型实验
│   ├── scripts/
│   ├── logs/            # 原始日志（按 YYYYMMDD 子目录）
│   └── data/            # 清洗后的 CSV
├── sim/                 # 仿真实验
│   ├── ns3/
│   ├── traces/
│   └── results/
├── docs/                # 过程文档与 diff
│   ├── YYYYMMDD_AI_NAME_task.md
│   └── diffs/
└── archive/             # 旧版本归档（只进不出）
    └── v2.0_ton/
```

**强制规则**：
- 代码文件必须放在 `src/` 或 `hook/`，不得放在根目录
- 实验数据必须按 `YYYYMMDD_实验名_版本号` 命名子目录
- 临时文件（`*.tmp`, `*.bak`, `*~`）必须在任务完成后删除

---

## 第四部分：输出格式规范

### 4.1 代码文件
- **语言**：C/C++ 使用 `.cc` / `.h`；Python 使用 `.py`
- **编码**：UTF-8，LF 行尾
- **注释**：所有函数必须包含 Doxygen 注释，标注对应论文章节（如 `// §III-B: Two-stage T_target`）

### 4.2 实验数据（CSV）
文件名格式：`YYYYMMDD_实验名_节点数_拓扑_版本.csv`

**必填字段**：
```csv
timestamp,job_id,iteration_idx,iter_time_us,pi_value,priority_level,quota,T_target_us,stage,c_i
```

**元数据**：每个 CSV 必须伴随同名 `.meta.json`：
```json
{
  "version": "3.0",
  "experiment": "slo_attainment",
  "topology": "fat-tree-k8",
  "nodes": 128,
  "baseline": ["Fair", "SRPT", "CRUX"],
  "ai_author": "AI_NAME",
  "date": "2026-07-04",
  "commit_hash": "abc123",
  "notes": "异常或手动干预说明"
}
```

### 4.3 图表
- 格式：PDF（矢量图）+ PNG（预览）
- 命名：`fig<章节号>_<描述>_v3.0.pdf`
- 颜色：LongLiu **#2E7D32**（深绿），Baseline **#757575**（灰），提升 **#C62828**（红）

---

## 第五部分：版本控制与防回退

### 5.1 禁止回退（No-Regression Rule）
- **绝对禁止**：将 v3.0 文件替换为 v2.0 文件
- **绝对禁止**：通过 `#ifdef OLD_VERSION` 保留旧逻辑
- **允许**：从 v2.0 提取纯数据（如 trace 格式说明），但必须标注 `_from_v2.0`

### 5.2 修改追踪
任何修改必须生成 diff 并存放在 `docs/diffs/`：
```bash
git diff > docs/diffs/YYYYMMDD_AI_NAME_module.diff
```
无 git 时提供前后对比摘要。

### 5.3 备份原则
旧文件不得直接删除，移动到 `archive/YYYYMMDD_原文件名`，并添加注释：
```c
// ARCHIVED on YYYY-MM-DD by AI_NAME. Reason: <原因>
```

---

## 第六部分：最大努力原则（Anti-Laziness）

### 6.1 禁止行为（视为偷懒）
| 禁止行为 | 正确做法 |
|---------|---------|
| "交换机方案更简单，改回交换机" | 交换机方案已废弃，必须坚持 NCCL |
| "Count-Min Sketch 可以复用旧代码" | 必须重写 NCCL 层状态机 |
| "JCT_SOLO 测量更直接" | 必须使用两阶段 $T_{\text{target}}$ |
| "物理实验太难，只做仿真" | 物理原型是必要组件，必须完成 2 节点验证 |
| "理论证明简化，只给直观解释" | 必须提供可调度性条件 + Lyapunov 推导 |

### 6.2 困难上报机制
遇到确实无法克服的障碍时：
1. 创建 `docs/YYYYMMDD_blocker_AI_NAME.md`
2. 描述：障碍是什么、尝试了哪些方案、为什么失败
3. 提出**替代方案**（而非退回旧版本）
4. 等待人工确认后方可继续

### 6.3 完整性自评
完成模块后回答：
- [ ] 本模块是否完全遵循 v3.0 架构？（无旧组件混入）
- [ ] 本模块是否与其他模块接口一致？
- [ ] 本模块是否有明确的输入/输出规范？
- [ ] 本模块是否包含自检/断言机制？
- [ ] 实验数据是否按 CSV + JSON 元数据格式输出？

---

## 第七部分：实验规范

### 7.1 必须包含的 Baseline
Fair、SRPT、CRUX、LongLiu。

### 7.2 必须报告的核心指标
| 指标 | 必须报告 |
|------|---------|
| SLO 达成率（$\pi_i \leq 0$ 占比） | ✅ |
| 平均/P95/P99 迭代时间 | ✅ |
| SLO 违约严重程度分布 | ✅ |
| T_target 收敛速度 | ✅ |
| GPU 利用率 | ⚠️ 可选 |

### 7.3 可复现性
- 随机种子固定（默认 `random_seed=42`）
- 仿真参数完整记录于 `.meta.json`
- 物理实验记录：硬件型号、固件版本、NCCL/CUDA 版本、环境变量

---

## 第八部分：交付检查清单（Pre-Submission）

**代码**：
- [ ] 无 v2.0 组件混入（grep 检查通过）
- [ ] 所有函数有 Doxygen 注释，标注论文章节
- [ ] 编译通过（`-Wall -Wextra` 无 warning）
- [ ] 单元测试通过（Progress Deficit、优先级映射、T_target 更新）

**数据**：
- [ ] CSV 包含必填字段，无缺失值
- [ ] 每个 CSV 有对应的 `.meta.json`
- [ ] 图表使用统一颜色方案

**文档**：
- [ ] 修改 diff 已存档于 `docs/diffs/`
- [ ] 旧文件已备份到 `archive/`
- [ ] 术语与 §1.3 一致

---

**本规范为所有 AI 助手的最高优先级约束。任何输出必须首先符合本文档，其次符合子任务说明。**