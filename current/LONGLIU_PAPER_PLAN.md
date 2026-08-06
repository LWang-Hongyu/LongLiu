# LongLiu Paper Plan — INFOCOM (v3.0)

> 最后更新: 2026-06-18
> 核心 claim: Progress Deficit — 第一个在线、per-iteration、仅需松弛系数 c_i 的 DNN 训练 SLO 调度信号

---

## 一、Motivation 故事线

### 1.1 商业驱动：SLA 的"性能真空"

当前云 GPU 集群的 SLA（AWS SageMaker、Azure ML、阿里云 PAIS）**仅保证基础设施可用性**（"机器活着"），明确排除网络性能保证。这意味着：

- 租户为高端 GPU（如 A100/H100）按小时付费，但训练任务的实际完成时间无法被承诺
- 网络拥塞导致的迭代时间膨胀被定义为"Best-Effort"，租户没有追索权
- 尤其高风险：训练 LLM 等长周期任务，一次拥塞可能导致数万美元的计算资源浪费

这个 "performance vacuum" 是 LongLiu 的商业 motivation——**将不可控的 Best-Effort 网络转化为可承诺的确定性服务**。

### 1.2 技术问题：多租户带宽竞争

多租户 GPU 集群中，多个 DDP 训练 job 共享 NIC 带宽（RDMA）。
每轮迭代的 AllReduce 同步需要等最慢的 flow 完成。
当多个 job 并发训练时，网络拥塞导致迭代时间不可预测地膨胀，job 系统性偏离其 SLO。

### 1.3 已有方案的粒度鸿沟

| 类别 | 代表工作 | 调度信号 | 信号性质 | 更新频率 |
|------|---------|----------|---------|---------|
| Job级调度器 | Tiresias [NSDI 2019], Themis [SOSP 2020] | 无调度信号 | - | 一次性 |
| GPU利用率优化 | CRUX [SIGCOMM 2024] | GPU Intensity | **静态**（前几轮测一次） | 一次性 |
| 流量模式预测 | CASSINI [NSDI 2024] | Off-ON pattern | **静态**（离线预测） | 一次性 |
| 拥塞控制 | DCTCP, HPCC, TIMELY | RTT/ECN | 动态（每RTT） | 每RTT |
| **LongLiu（本文）** | - | **Progress Deficit** | **动态（每轮迭代）** | **per-iteration** |

### 1.4 关键洞察

**现有调度器都用静态或半静态信号**：

- CRUX 的 GPU Intensity I_j = W_j / t_j，是 job 的**固有属性**（计算/通信比），在 job 前几轮测一次后不变。它优化的是**集群 GPU 利用率**，不关心单个 job 的进度
- CASSINI 的 traffic pattern 是离线预测的，在实际运行中可能偏移
- 它们都无法响应 job 运行过程中的动态变化（其他 job 加入/退出、数据加载波动、拥塞状态变化）

**DNN 训练的迭代结构天然提供了动态信号源**：
- 每完成一轮迭代，job 就离 SLO 近了一步
- 比较"应该完成多少"和"实际完成多少"就是进度信号
- 这个信号每轮迭代都在更新，天然动态
- 每个 job 独立计算，天然分布式（类似 TCP 的 backward-facing 控制）

### 1.5 额外挑战

原始论文识别了两个实际部署中的关键问题：

- **短作业陷阱**：短 job（几十轮迭代）没有足够的迭代次数来适应。如果前几轮刚好遇到网络拥塞，T_target 的初始测量值被污染，整条 job 的 SLO 都无法恢复
- **级联故障**：一个 job 的网络延迟可以通过共享链路级联——它占着带宽不放，其他 job 等它的 AllReduce 完成，然后一起变慢

LongLiu 的两阶段 T_target 测量方案（见 §3.2）专门用于解决这两个问题。

### 1.6 Profiling 需求的取舍

一个关键的系统级问题是：**LongLiu 要求用户提供一个参数 c_i，这是负担吗？**

对比来看：

| 方案 | 用户需要提供什么 | 信号性质 |
|------|----------------|---------|
| CRUX | 无（自动测量 GPU Intensity） | 静态（一次测量终身有效） |
| CASSINI | 无（离线预测 traffic pattern） | 静态（部署前确定） |
| **LongLiu** | **c_i（松弛系数，>1 的浮点数）** | **动态（每轮迭代更新）** |

CRUX 和 CASSINI 的"用户零输入"是以**信号静态性**为代价的：
- CRUX 无法响应运行时的拥塞变化，因为它不知道 job 当前的进度
- CASSINI 的 traffic pattern 预测可能在运行中偏差
- 它们能优化集群吞吐，但不能为单个 job 提供 SLO 保证

LongLiu 的 c_i 承担了完全不同的角色：
- 不是技术参数（如学习率、batch size），而是**商业 SLA 等级**（"你愿意接受比理想慢多少？"）
- c_i = 1.5 是一个直观的值——"接受 50% 的性能折扣"——任何租户都能理解
- 这个**单一标量参数**换来了：动态 per-iteration 调度 + 形式化 SLO 保证 + 分布式部署

**一句话**：c_i 不是 profiling 的负担，而是用户表达其 SLA 意愿的商业接口。用最小代价换取了 CRUX/CASSINI 无法提供的动态 SLO 保证。

### 1.7 LongLiu 的核心创新

**Progress Deficit**——每轮迭代在线计算，只需要松弛系数 c_i：
- 不需要离线 profiling（CRUX 需要测 GPU Intensity、CASSINI 需要预测 traffic pattern）
- 不需要全局信息（CRUX 需要全局路径选择 + 优先级分配）
- 每个 job 独立计算，天然分布式
- 用户唯一需要提供的 c_i 是商业 SLA 参数，不是技术参数

### 1.8 贡献总结

1. **Progress Deficit**：第一个在线、per-iteration、仅需松弛系数 c_i 的 DNN 训练 SLO 调度信号
2. **T_target 两阶段混合测量**：RTT 预测量 + 最高优先级修正，零初始化开销，避免短作业陷阱
3. **迭代级调度机制**：第一个在 DNN 训练中以迭代为粒度进行运行时网络调度的工作
4. **Lyapunov SLO 稳定性定理**：形式化 SLO 保证 P(max_i π_i > B) ≤ D·exp(-θ·B)，第一个提供 DNN 训练形式化 SLO 保证的工作
5. **轻量级实现**：~50 行 C 代码修改 NCCL proxy thread，无需硬件支持、无需内核模块、无需中心调度器

---

## 二、SLO 定义

### 2.1 为什么不是作业级 SLO

传统的"作业完成时间 SLO"在多租户环境下不可行：
- 调度器不知道 job 的总迭代轮数（用户不暴露）
- 不知道模型结构（无法预估收敛行为）
- 不知道用户期望的 deadline（用户可能自己也不知道）
- 即使知道，不同 job 的 deadline 无法公平比较

作业级 SLO 需要的信息在多租户环境下均不可获得。

### 2.2 通信周期级 SLO 定义

LongLiu 将 SLO 定义在**迭代（通信周期）级别**：

| 参数 | 定义 | 来源 |
|------|------|------|
| A_i(t) | 任务 i 的累积实际通信时间 | 运行时测量（AllReduce hook） |
| k_i(t) | 已完成的迭代轮数 | 运行时计数 |
| T_i_target | 任务 i 的目标迭代时间 | 两阶段混合测量（自动） |
| c_i | 松弛系数，c_i > 1 | **用户唯一需要提供的参数**（SLA 等级） |

**核心公式**：

```
π_i(t) = A_i(t) / (c_i × T_i_target × k_i(t)) - 1
```

- π_i(t) < 0 → 进度超前 → 低优先级（让出带宽）
- π_i(t) = 0 → 恰好处于 SLO 边界
- π_i(t) > 0 → **SLO 违约** → 需提升优先级

**物理含义**：π_i(t) = 0.5 表示"job 的实际平均迭代时间比 SLO 允许的慢了 50%"。

### 2.3 松弛系数 c_i 的商业含义

c_i 是用户与云厂商签订的 SLA 等级参数：
- c_i = 1.2 → 严格 SLO（允许比 solo 慢 20%），成本高
- c_i = 1.5 → 中等 SLO（允许慢 50%），默认值
- c_i = 2.0 → 宽松 SLO（允许慢 100%），成本低
- c_i → 1.0 → 要求 100% 理想性能（技术上不可行）

这跟商业 SLA 直接对齐——AWS 的 GPU 实例按"性能等级"定价，LongLiu 提供了实现这种分级定价的技术手段。

---

## 三、T_target 测量

### 3.1 核心矛盾

T_target 应该是"job 在无竞争条件下的 solo 迭代时间"。
但问题：**在拥挤的多租户网络中，你永远无法获得真正的 solo 时间**。
第一轮测量可能被其他 job 的流量污染。

### 3.2 两阶段混合测量方案

**阶段 1：RTT 预测量（初始估计）**
- Job 启动时，发送一个小的 RTT 探测包（RDMA read 或 ibv_send）
- 通过 RTT 估计当前的网络延迟
- T_target ≈ RTT × 每轮迭代的通信步数
- 这是一个**粗略估计**，误差可能较大，但提供了初始值，避免了"短作业陷阱"

**阶段 2：最高优先级修正（运行时校准）**
- 运行期间，每当 job 获得**当前可用的最高优先级**时（π_i < 0 且没有其他 job 有更高的 deficit），记录当前迭代时间
- 核心洞察：**最高优先级 ≈ 独占带宽**——此时测得的迭代时间最接近 solo 时间
- 只在此时更新 T_target（使用 EMA 平滑更新）

**为什么这可行**：
- "高优先级 ≈ 独占带宽"是一个物理事实——严格优先级调度下，最高优先级的 job 不受其他 job 干扰
- 不需要单独的执行环境，在正常训练过程中即可完成校准
- T_target 随着网络条件变化自动适应（比如其他 job 加入/退出）

### 3.3 对短作业陷阱的免疫

短 job 的关键脆弱性：如果前几轮测量被污染，整个 job 的 SLO 都会失效。

LongLiu 的处理：
- RTT 预测量提供了安全的初始 T_target（不需要等实际迭代）
- 即使第一轮迭代被严重污染，系统不会立即大幅调整 T_target
- T_target 只在获得最高优先级时才更新，保证了测量质量

---

## 四、设计

### 4.1 核心信号：Progress Deficit

```
π_i(t) = A_i(t) / (c_i × T_i_target × k_i(t)) - 1
```

其中：
- A_i(t) / k_i(t) = 实际平均迭代时间（EMA 平滑后）
- c_i × T_i_target = SLO 允许的单轮时间上限
- π_i > 0 → 落后于 SLO → quota↑ → 占更多带宽
- π_i < 0 → 超前于 SLO → quota↓ → 让出带宽

### 4.2 优先级映射函数 φ(π) — 离散队列映射

真实 RDMA NIC 支持有限数量的硬件优先级队列（通常 P0-P7，8 级）。
LongLiu 使用离散映射，而非连续函数：

| 优先级 | π 范围 | 调度行为 |
|--------|--------|----------|
| P7 | 保留 | ACK 等关键控制包 |
| P6 | π > 0.3 | 严重违约，最高优先级 |
| P4-P5 | -0.1 < π ≤ 0.3 | 轻度违约，高优先级 |
| P2-P3 | -0.5 < π ≤ -0.1 | 正常，中优先级 |
| P0-P1 | π ≤ -0.5 | 超前，低优先级 |

**为什么用离散映射而不是连续函数**：
- 硬件只支持 8 级优先级（严格优先级队列）
- 连续函数的细微变化在硬件层面无法表达
- 离散映射更易于分析和证明（Theorem 2）
- P6 专门用于 T_target 的"最高优先级修正"测量

### 4.3 在 NCCL 中的实现

```
PyTorch DDP 训练进程
  |
  ├── [PyTorch Hook] 测量每轮迭代时间
  |   → A_i(t), k_i(t) → π_i(t)
  |   → 写入 NCCL comm 的 priority 字段（原子变量）
  |
  └── [NCCL 修改] ncclProxyProgress()
      → 每轮循环从 comm 读取 priority
      → quota = map_priority_to_quota(priority)
      → 控制本轮处理的 ops 数量
```

### 4.4 为什么不需要跨进程协调

| Job A (π=+0.3, 落后) | Job B (π=-0.2, 超前) |
|------------------------|------------------------|
| priority = P6 (最高) | priority = P2-P3 (正常) |
| quota = 最大 → 满速发 | quota = 正常 → 保持 |
| 占更多带宽 | 自动退让 |
| 追赶 SLO | 不受影响（原本就超前） |

各自独立 → 效果等价于全局协调。这是 backward-facing 控制的优势：
- 落后 job 自然变 aggressive（因为它的 π 更大）
- 超前 job 自然变 passive（因为它的 π 更小）
- 交换机按严格优先级调度 → 自动收敛

### 4.5 与原设计的差异

| 方面 | 原设计（v2） | 当前设计（v3） |
|------|-------------|---------------|
| T_target 来源 | 用户提供 T_deadline | **系统自动测量** + 用户仅提供 c_i |
| Deficit 公式 | deficit = expected - actual | π = actual_avg / (c × T_target) - 1 |
| 映射函数 | 连续线性 φ(deficit) | **离散 8 级队列映射** |
| 短作业陷阱 | 未处理 | RTT 预测量 + 最高优先级修正 |
| 商业动机 | 性能优化 | **SLA 性能真空 → 可承诺服务** |

---

## 五、理论

### 5.1 Lemma 1: EMA 收敛性（支撑性引理）

EMA 速率估计器以指数速度收敛到真实迭代速率。这是标准 EMA 收敛结果，作为 Theorem 2 的支撑引理。

### 5.2 Theorem（唯一核心定理）: Deficit 稳定性 (Lyapunov)

在离散优先级映射 φ(π) 下，deficit 向量收敛到平稳分布，
且对任意 B > 0：

```
P(max_i π_i > B) ≤ D · exp(-θ · B)
```

**这是本文的核心理论贡献**——DNN 训练调度中第一个形式化 SLO 保证。

证明策略：
- 构建 Lyapunov 函数 V = Σ π_i²（deficit 平方和）
- 证明严格优先级调度下 V 的漂移为负（负反馈循环的数学表达）
- 用 Foster-Lyapunov 准则得到几何遍历性
- 尾部界的推导基于马尔可夫链的漂移条件

### 5.3 性能下界分析

即使调度最优，存在理论上的性能下界：
- 瓶颈链路带宽 B 和竞争 job 数量 N 决定了不可压缩的延迟下界
- 这个下界是 SLO 可行性的必要条件——如果 c_i × T_target 低于这个下界，SLO 不可满足
- 有助于云厂商在 SLA 签订时评估可行性

---

## 六、实现

### 6.1 为什么不是其他方案

| 方案 | 状态 | 原因 |
|------|------|------|
| eBPF/TC 标记 DSCP | ❌ 废弃 | RDMA 绕过内核，eBPF 看不到 RDMA 包 |
| LD_PRELOAD 拦截 ibv_modify_qp() | ❌ 实验失败 | BlueField-3 DPU 返回 EINVAL |
| LD_PRELOAD 拦截 ibv_post_send() | ❌ 不可行 | static inline，无法拦截 |
| **NCCL proxy 修改（当前）** | ✅ **可行** | **已验证编译，deficit 计算正确** |

### 6.2 NCCL 修改

**文件**: nccl-master/src/proxy.cc → ncclProxyProgress()

关键逻辑：每轮循环读 comm->priority → 映射到 quota → 限制 ops 处理量。

代码量：~50 行 C。

### 6.3 PyTorch Hook

**文件**: longliu_hook.py

关键逻辑：time all_reduce → 更新 A_i(t), k_i(t) → 计算 π_i(t) → 写回 comm。

代码量：~30 行 Python。

---

## 七、评估

### 7.1 仿真（核心评估）

| 维度 | 内容 |
|------|------|
| 规模 | 6-128 节点（当前 6-18 完成，需扩展） |
| 拓扑 | Fat-Tree (k=4 / k=8) |
| 负载 | 合成负载 + Philly / Alibaba trace (TODO) |
| 策略 | Fair, SRPT, CRUX-like, CASSINI-like, LongLiu |

**CRUX baseline 实现**：
每个 job 分配固定的 GPU intensity I_j（按模型类型：GPT=high, BERT=medium, ResNet=low），
优先级按 I_j 降序分配，job 启动时设定，不再变更。

**当前结果（6-18 节点）**：
- avg 迭代时间: LongLiu 比 SRPT 低 10-33%
- p95 迭代时间: LongLiu 比 SRPT 低 14-51%
- 吞吐量: LongLiu 最优或持平

### 7.2 物理实验（机制验证）

| 维度 | 内容 |
|------|------|
| 硬件 | 2 节点, 3×GPU (2×RTX5000 + 1×RTX4000), RoCEv2 @40Gbps |
| 基线 | 单 DDP job, 50MB AllReduce, ~20ms 迭代时间 ✅ |
| TODO | 双 communicator 竞争实验 (P0) |
| TODO | T_target 两阶段测量验证 (P1) |

### 7.3 指标

- **SLO 达标率**：π_i(t) ≤ 0 的时间占比（核心指标）
- **迭代时间**：avg 和 p95（与传统指标对齐）
- **GPU 利用率**：与 CRUX 对比（secondary）
- **T_target 收敛速度**：两阶段混合测量的有效性

### 7.4 论文定位

> Due to hardware constraints (3 GPUs across 2 nodes), the physical prototype serves as mechanism verification: confirming that (a) the NCCL proxy modification works, (b) deficit is computed correctly, and (c) the control loop is end-to-end functional. The core claims are validated through calibrated simulation at 6-128 nodes, demonstrating consistent SLO attainment improvement over all baselines.

---

## 八、与 CRUX 的完整对比

### CRUX 做了什么

1. **Fact**: 多 job 共存时，低 GPU Intensity job 的通信阻塞高 GPU Intensity job → 降低整体 GPU 利用率
2. **Metric**: GPU Intensity I_j = W_j / t_j（**静态**，前几轮测一次）
3. **Theory**: Max GPU utilization = max sum of GPU intensities on bottleneck link
4. **System**: 路径选择（给高 intensity job 选最少拥塞的路）+ 优先级分配（按 intensity 排序）+ 优先级压缩（适配有限硬件队列）

### CRUX 没做什么

- ❌ 不做 per-iteration 调整——优先级一次赋值终身有效
- ❌ 不关心 SLO——目标是集群 GPU 利用率最大化，不是单个 job 的完成时间
- ❌ 不处理运行时动态——GPU Intensity 是静态的
- ❌ 不可分布式——需要中心调度器掌握全局信息（路径 + 优先级分配）

### LongLiu vs CRUX 的本质区别

| 维度 | CRUX | LongLiu |
|------|------|---------|
| 信号 | GPU Intensity（计算/通信比，静态） | Progress Deficit（SLO 进度，动态） |
| 决策时机 | job 启动时 | **每轮迭代** |
| 优化目标 | GPU 利用率（集群视角） | SLO 达标率（每个 job） |
| 用户输入 | 无（自动测量） | **仅需 c_i**（SLA 等级） |
| 全局视图 | ✅ 需要（路径选择+优先级分配） | ❌ **不需要（端主机独立）** |
| 理论保证 | NP-Complete 问题转化 | **Lyapunov 稳定性 + SLO 概率保证** |
| 形式化 SLO | ❌ 无 | ✅ **概率界 P(π_i > B) ≤ D·exp(-θ·B)** |

---

## 九、原始的论文可取之处

原始 ToN 论文（SLO_Driven_Network_Scheduling_Integrated.docx）中包含以下可取之处，
已整合到本设计：

| 可取之处 | 整合位置 | 说明 |
|----------|---------|------|
| c_i 松弛系数 | §2.2-2.3 | 用户只需提供 SLA 等级，系统自动测 T_target |
| 两阶段 T_target 测量 | §3 | RTT 预测量 + 最高优先级修正，避免短作业陷阱 |
| 离散优先级映射 | §4.2 | 8 级硬件队列映射，更实际 |
| SLA 性能真空 | §1.1 | 商业 motivation，填补云 SLA 的市场空白 |
| 短作业陷阱 | §1.5 | 短 job 的脆弱性及解决方案 |
| 级联故障 | §1.5 | 网络延迟的级联效应 |
| 性能下界分析 | §5.3 | SLO 可行的必要条件 |
| 负反馈循环 | §4.4 | 独立计算 → 自动收敛的直觉 |

---

## 十、TODO

### P0（论文必须）
- [ ] 64-128 节点仿真结果
- [ ] Theorem（Lyapunov）完整证明 ← **唯一核心定理**
- [ ] 双 communicator 物理实验
- [ ] CRUX baseline 仿真实现（GPU intensity 分配）
- [ ] Introduction 重写（基于本文 v3 的故事线）

### P1（加分项）
- [ ] Philly / Alibaba trace 驱动的仿真
- [ ] T_target 两阶段测量验证实验
- [ ] 不同 c_i 值的 SLO 达标率敏感性分析
- [ ] 8 级离散映射 vs 连续映射的性能对比

### P2（未来工作）
- [ ] DPU/IPU 卸载方案
- [ ] GPU Intensity + Deficit 联合信号
- [ ] 多级 QoS 定价模型（基于 c_i 的商业化）

