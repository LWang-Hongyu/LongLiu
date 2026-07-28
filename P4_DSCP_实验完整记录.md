# P4 DSCP 优先级队列实验完整记录

> 更新日期: 2026-07-16
> 实验目标: 验证 LongLiu DSCP 方案在多 Job 竞争场景下的带宽优先级保护能力

---

## 一、项目概述

### 1.1 研究背景

LongLiu 是一个**纯发送端（sender-side）**的优先级调度方案，通过动态设置 RDMA QP 的 DSCP（Differentiated Services Code Point）值，利用 RoCEv2 网络的 QoS 机制实现带宽优先级控制。

### 1.2 核心思想

- **Epoch-Level 调度**: 以训练 epoch 为调度单元，每个 epoch 内所有 collective 操作使用同一个 DSCP 优先级
- **EMA 带宽自估计**: 通过指数滑动平均（EMA）自适应估计无竞争下的理想带宽
- **Urgency Index**: 计算 `Ui = ai / ei`，其中 ai 是实际累积时间，ei 是期望累积时间
- **DSCP 映射**: 将 Urgency Index 映射到 7 级 DSCP 码点，对应不同优先级队列

---

## 二、实验环境

### 2.1 硬件配置

| 节点 | 管理 IP | GPU | 显存 | RDMA设备 | 网络 |
|------|---------|-----|------|----------|------|
| guolab-226 | 10.157.197.107 | 2x Quadro RTX 5000 | 2x16GB | BlueField-3 B3220 (mlx5_0/mlx5_1) | 100GbE RoCEv2 |
| guolab-10 | 10.157.197.26 | 1x Quadro RTX 4000 | 8GB | BlueField-3 B3220 (mlx5_0/mlx5_1) | 100GbE RoCEv2 |

### 2.2 网络拓扑

```
┌──────────┐  管理网络 (192.10.10.x)   ┌──────────┐
│  10.1    │◄────── TCP socket ──────►│   226    │
│          │   (ID 交换, ctrl)         │          │
│ GPU0     │                           │ GPU0     │  ← Job1
│ GPU0 ────┤   RDMA (10.10.10.x)      ├─ GPU0    │  ← Job1
│ (共享)   │◄──── mlx5_0 ── 100G ───►│ GPU1     │  ← Job2
│          │                           │          │
│ Job1 r0  │                           │ Job1 r1  │
│ Job2 r0  │                           │ Job2 r1  │ (GPU1)
└──────────┘                           └──────────┘
```

### 2.3 软件版本

| 组件 | 226 | 10.1 |
|------|-----|------|
| OS | Ubuntu 22.04 / Python 3.10 | Ubuntu 20.04 / Python 3.8 |
| PyTorch | 2.3.0+cu121 | 2.3.0+cu121 |
| CUDA | 12.6 (driver) / 12.1 (runtime) | 12.1 |
| NCCL | 2.30.7 (trafficClass API) | 2.30.7 (trafficClass API) |

### 2.4 关键环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| NCCL_IB_HCA | mlx5_0 | 强制使用 mlx5_0（mlx5_1 不可用） |
| NCCL_IB_GID_INDEX | 3 | RoCEv2 GID 索引（IPv4） |
| NCCL_SOCKET_IFNAME | enp130s0f0np0 / enp59s0f0np0 | 网络接口 |
| NCCL_ALGO | RING | 使用 Ring AllReduce |
| NCCL_PROTO | SIMPLE | 简单协议 |

---

## 三、实现架构

### 3.1 方案演进

| 方案 | 状态 | 核心思想 |
|------|------|----------|
| Proxy Quota (nccl-master) | 已废弃 | 修改 NCCL proxy thread 的 ops quota |
| DSCP Adapter (nccl-dscp) | 已废弃 | 修改 NCCL 源码，通过 ibv_modify_qp 动态修改 QP traffic_class |
| Multi-Comm SLO (multi_comm_slo) | **当前方案** | 预创建 7 个不同 DSCP 的 NCCL communicator，运行时切换 |

### 3.2 当前方案：Multi-Comm SLO

#### 架构图

```
┌─────────────────────────────────────────────────────┐
│                    p4_train_gpt.py                   │
│  ┌─────────────────────────────────────────────────┐ │
│  │  SLOScheduler                                    │ │
│  │   - EMA 带宽估计 (ei)                            │ │
│  │   - Ui = ai / ei 计算                             │ │
│  │   - PRIORITY_THRESHOLDS 查表                     │ │
│  └──────────────┬──────────────────────────────────┘ │
│                 │ priority ∈ {0..6}                   │
│  ┌──────────────▼──────────────────────────────────┐ │
│  │  MultiCommWrapper (ctypes)                      │ │
│  │   - epoch_start(epoch)   → set_priority         │ │
│  │   - allreduce(...)       → 当前 priority 的 comm│ │
│  │   - epoch_end(epoch)     → update + set_priority│ │
│  └──────────────┬──────────────────────────────────┘ │
└─────────────────┼────────────────────────────────────┘
                  │ ctypes FFI
┌─────────────────▼────────────────────────────────────┐
│  libmulti_comm.so (C)                                │
│  ┌──────────────────────────────────────────────────┐│
│  │  multi_comm_init()                               ││
│  │    ├─ exchange_ids_via_tcp()  ← TCP ID 交换      ││
│  │    └─ ncclCommInitRankConfig()                   ││
│  │         └─ trafficClass = priority * 8           ││
│  │                                                   ││
│  │  multi_comm_set_priority(p)                       ││
│  │    → g_handle.current_priority = p                ││
│  │                                                   ││
│  │  multi_comm_allreduce(...)                        ││
│  │    → ncclAllReduce(..., comms[priority][dev])     ││
│  └──────────────────────────────────────────────────┘│
└─────────────────┼────────────────────────────────────┘
                  │ RDMA verbs
┌─────────────────▼────────────────────────────────────┐
│  NCCL 2.30.7 (trafficClass API)                      │
│  → QP 创建时自动设置 TClass = DSCP                   │
│  → RDMA 流量携带对应 DSCP 标记                       │
└─────────────────┼────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────┐
│  P4 可编程交换机 (Strict Priority QoS)               │
│  Queue 0 (DSCP 0)  → 最高优先级队列                  │
│  Queue 1 (DSCP 8)  →                                  │
│  ...                                                  │
│  Queue 6 (DSCP 48) → 最低优先级队列                  │
└──────────────────────────────────────────────────────┘
```

#### 核心组件

1. **C 库 — `libmulti_comm.so`**
   - 预创建 7 个不同 DSCP 的 NCCL communicator
   - 通过 TCP socket 交换 NCCL Unique ID
   - O(1) 优先级切换（指针切换）

2. **SLO 调度器 — `SLOScheduler`**
   - EMA 带宽估计：`ei = α × actual_bw + (1-α) × ei`
   - Urgency Index：`Ui = ai / ei`
   - 7 级优先级映射

3. **Python 封装 — `MultiCommWrapper`**
   - epoch_start/epoch_end 管理
   - allreduce 接口

---

## 四、核心算法

### 4.1 EMA 带宽自估计

```python
if not ema_initialized:
    ema_bandwidth = actual_bw          # 首次直接赋值
else:
    ema_bandwidth = (                  # 指数滑动平均
        alpha × actual_bw + 
        (1 - alpha) × ema_bandwidth
    )
```

- `alpha = 0.3`：对拥塞响应较快（~3-4 epoch 稳定）

### 4.2 Priority 阈值表

| Ui 区间 | Priority | DSCP | 含义 |
|---------|----------|------|------|
| Ui < 0.6 | P0 | 0 | 最紧急（严重拥塞） |
| 0.6 ≤ Ui < 0.8 | P1 | 8 | 紧急 |
| 0.8 ≤ Ui < 1.0 | P2 | 16 | 轻度拥塞 |
| 1.0 ≤ Ui < 1.2 | P3 | 24 | 正常 |
| 1.2 ≤ Ui < 1.4 | P4 | 32 | 轻微空闲 |
| 1.4 ≤ Ui < 1.6 | P5 | 40 | 空闲 |
| Ui ≥ 1.6 | P6 | 48 | 最空闲（让出带宽） |

### 4.3 调度时序

```
Epoch N:
  epoch_start(N)
    ├─ multi_comm_set_priority(Px)   ← 设置当前 priority
    ├─ iter 0: allreduce()           ← 使用 Px 的 communicator
    ├─ iter 1: allreduce()           ← 使用 Px 的 communicator
    ├─ ...
    └─ iter 19: allreduce()          ← 使用 Px 的 communicator
  epoch_end(N, data_size)
    ├─ 计算 epoch wall time
    ├─ ai = data_size / wall_time
    ├─ ei = EMA(ai, ei)
    ├─ Ui = ai / ei
    └─ 查表决定新 priority Py → 设置给 Epoch N+1
```

---

## 五、P4 Dumbbell SLO 实验

### 5.1 实验设置

- **模型**: TinyGPT（~44M 参数，d=512, 6 layers, seq=256, batch=2）
- **链路**: 10.1 ←50G RDMA→ 226，经过 P4 交换机 DSCP 队列
- **LongLiu 调度**: Job1 = 严格 SLO（c_i=1.5），Job2 = 宽松 SLO（c_i=2.5）

### 5.2 Job 配置

| 参数 | Job1 | Job2 |
|------|------|------|
| SLO c_i | 1.5 | 2.5 |
| Epochs | 15 | 10 |
| Iters/epoch | 20 | 20 |
| Total iters | 300 | 200 |
| 优先级 | 高 | 低 |

### 5.3 实验模式

| 模式 | 说明 |
|------|------|
| solo | 仅 Job1 运行，作为基线 |
| fair | 两 Job 运行，无优先级（标准 NCCL） |
| longliu | 两 Job 运行，MultiCommWrapper + DSCP 优先级 |

### 5.4 运行命令

```bash
# 在 10.1 节点上运行
cd /home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo

# Solo 基线
bash run_p4.sh solo

# Fair 竞争
bash run_p4.sh fair

# LongLiu 调度
bash run_p4.sh longliu

# GPT 训练模式
bash run_p4.sh train_gpt longliu
```

---

## 六、实验结果

### 6.1 核心结果

| 指标 | Solo | Job1(LongLiu) | Job2(LongLiu) |
|------|------|---------------|---------------|
| Bus Bandwidth | 23.7 Gbps | 21.5 Gbps (竞争期) | ~18.4 Gbps |
| 带宽保持率 | 100% | **90.5%** | N/A |

### 6.2 关键发现

1. **DSCP 优先级生效**: Job1（严格 SLO）在竞争期维持了 90.5% 的 solo 通信带宽，说明高优先级 DSCP 队列有效保护了紧急 Job 的通信

2. **自适应调度**: Job2 的带宽从最初 1.8 Gbps 逐渐上升到 25.3 Gbps，说明 LongLiu 调度器根据 epoch 进度动态调整优先级

3. **链路利用率**: 两 Job 竞争时总 bus 带宽 ~40 Gbps，接近 50G 物理链路极限

### 6.3 问题与挑战

- 10.1 节点只有 1 个 GPU，两 Job 必须共享 GPU，导致训练速度极慢（~10x slowdown）
- Job2 未能完整跑完，但已获取的 105 个 iter 数据足以说明 DSCP 优先级行为

---

## 七、项目文件结构

```
LongLiu_rebuild/
├── DESIGN_REFERENCE.md              # 设计参考文档（论文 Design 章节参考）
├── DSCP_VERIFICATION_LOG.md         # DSCP 编译、部署、验证日志
├── LONGLIU_IMPLEMENTATION.md        # 实现文档（已废弃的 proxy quota 方案）
├── LONGLIU_PAPER_PLAN.md            # 论文规划
├── LongLiu_INFOCOM_Draft.md         # 论文初稿
├── P4_DSCP_实验完整记录.md          # 本文档
├── talking.txt                      # 上次对话记录
├── build_u20.sh                     # Ubuntu 20.04 编译脚本
├── 实验环境与实现方案.md             # 实验环境配置
├── 实现进度与结果汇总.md            # 进度汇总
├── DSCP_端到端测试记录.md           # DSCP 测试记录
│
├── nccl-dscp/                       # NCCL DSCP 修改版（已废弃）
│   ├── src/
│   │   ├── misc/dscp_adapter.cc     # DSCP 适配器核心
│   │   ├── include/dscp_adapter.h   # 数据结构定义
│   │   ├── transport/net_ib.cc      # QP traffic_class 修改
│   │   └── ...
│   └── build/
│
├── nccl-master/                     # NCCL 2.30.7（当前使用）
│   └── build/lib/
│
├── multi_comm_slo/                  # 当前方案：Multi-Comm SLO
│   ├── DESIGN.md                    # 设计文档
│   ├── src/
│   │   ├── multi_comm.c             # C 核心实现
│   │   ├── multi_comm.h             # C 头文件
│   │   ├── slo_scheduler.py         # Python 调度器
│   │   └── Makefile
│   ├── build/
│   │   └── libmulti_comm.so         # 编译产物
│   ├── build.sh
│   └── test_cross_node.sh
│
├── experiments/
│   ├── P4_dumbbell_slo/             # P4 实验脚本
│   │   ├── p4_train_gpt.py          # GPT 训练脚本
│   │   ├── p4_job1.py               # Job1 脚本
│   │   ├── p4_job2.py               # Job2 脚本
│   │   ├── run_p4.sh                # 一键启动脚本
│   │   └── ib_prio_strict.sh        # 交换机配置
│   ├── P1_dscp_injection/           # P1: DSCP 注入实验
│   ├── P2_bimodal_interval/         # P2: 双峰间隔实验
│   ├── P3_ema_convergence/          # P3: EMA 收敛实验
│   └── traces/                      # 仿真 trace 数据
│
├── LongLiu_py/                      # Python 工具
└── testbed/                         # 测试环境
```

---

## 八、关键约束与注意事项

### 8.1 硬件约束

1. **NCCL 必须使用 mlx5_0 接口**: mlx5_1 的 RDMA 连接不可用
2. **10.1 机器只能通过 MPS 初始化一个 NCCL communicator**: 两个 Job 需要共享 GPU
3. **GPU 内存限制**: 10.1 只有 8GB 显存，需要使用 tiny config

### 8.2 软件约束

1. **NCCL 版本兼容**: NCCL 2.18.3 与宿主机 rdma-core 58mlnx43 不兼容，需在 Ubuntu 20.04 容器中编译
2. **NCCL 2.30.7 支持 trafficClass API**: 当前方案依赖此特性
3. **P4 交换机配置**: 需配置 strict priority QoS（8 队列对应 DSCP 0~56）

### 8.3 实验注意事项

1. **端口冲突**: 每次运行使用随机端口避免 TIME_WAIT 冲突
2. **清理时间**: 实验间需等待 70s 让 TIME_WAIT 过期
3. **GPU 共享**: 10.1 节点两 Job 共享 GPU 会导致 ~10x slowdown

---

## 九、实验结果数据位置

- **Solo 基线**: `/tmp/p4_train_JOB1_solo_rank0.csv`
- **Fair 竞争**: `/tmp/p4_train_JOB1_fair_rank0.csv`, `/tmp/p4_train_JOB2_fair_rank0.csv`
- **LongLiu 调度**: `/tmp/p4_train_JOB1_longliu_rank0.csv`, `/tmp/p4_train_JOB2_longliu_rank0.csv`
- **NCCL 日志**: `/tmp/nccl_j1_101_%h_%p.log`, `/tmp/nccl_j1_226_%h_%p.log`
- **训练日志**: `/tmp/p4_job1_node101.log`, `/tmp/p4_job1_node226.log`

---

## 十、后续工作

### 10.1 短期

- [ ] 修复 10.1 GPU 共享导致的性能问题
- [ ] 完成 Job2 的完整运行
- [ ] 收集更多数据点（不同 SLO 阈值、不同模型大小）

### 10.2 中期

- [ ] 扩展到 4+ 节点实验
- [ ] 与仿真结果对比验证
- [ ] 论文实验章节完善

### 10.3 长期

- [ ] 支持更多调度策略
- [ ] 集成到生产环境
- [ ] 开源发布

---

## 十一、参考文档

| 文档 | 说明 |
|------|------|
| [DESIGN_REFERENCE.md](file:///home/why/LongLiu_rebuild/DESIGN_REFERENCE.md) | 设计参考（论文 Design 章节） |
| [multi_comm_slo/DESIGN.md](file:///home/why/LongLiu_rebuild/multi_comm_slo/DESIGN.md) | Multi-Comm SLO 设计文档 |
| [实验环境与实现方案.md](file:///home/why/LongLiu_rebuild/实验环境与实现方案.md) | 实验环境配置 |
| [实现进度与结果汇总.md](file:///home/why/LongLiu_rebuild/实现进度与结果汇总.md) | 进度汇总 |
| [DSCP_VERIFICATION_LOG.md](file:///home/why/LongLiu_rebuild/DSCP_VERIFICATION_LOG.md) | DSCP 验证日志 |
| [LongLiu_INFOCOM_Draft.md](file:///home/why/LongLiu_rebuild/LongLiu_INFOCOM_Draft.md) | 论文初稿 |
