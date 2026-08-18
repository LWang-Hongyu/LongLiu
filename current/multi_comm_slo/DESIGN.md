# Multi-Comm SLO 调度器 — 设计文档

## 概述

本方案基于 **NCCL 原生 `trafficClass` API**，用多 Communicator 架构替代之前修改 NCCL 源码的方法，实现动态 DSCP 优先级调度。核心思路是：**预先创建 7 个不同 DSCP 标记的 NCCL communicator，运行时只切换指针，无需 `ibv_modify_qp`**。

## 架构图

```
┌─────────────────────────────────────────────────────┐
│                    p4_job*.py                        │
│  ┌─────────────────────────────────────────────────┐ │
│  │  SLOScheduler                                    │ │
│  │   - EMA 带宽估计 (ei)                            │ │
│  │   - Ui = ai / ei 计算                             │ │
│  │   - PRIORITY_THRESHOLDS 查表                     │ │
│  └──────────────┬──────────────────────────────────┘ │
│                 │ priority ∈ {0..6}                   │
│  ┌──────────────▼──────────────────────────────────┐ │
│  │  MultiCommWrapper (ctypes)                      │ │
│  │   - window_start(w)      → set_priority         │ │
│  │   - allreduce(...)       → 当前 priority 的 comm│ │
│  │   - window_end(w,size)   → update + set_priority│ │
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

## 核心组件

### 1. C 库 — `libmulti_comm.so`

**路径**: `src/multi_comm.c`、`src/multi_comm.h`

#### 数据结构

```c
typedef struct {
    ncclComm_t comms[NUM_PRIORITIES][MAX_DEVICES];  // 7 × N 个 communicator
    int num_devices;
    int current_priority;  // 当前活跃 priority (0-6)
    int rank;
    int size;
} MultiCommHandle;

static MultiCommHandle g_handle;  // 全局单例
```

每个 (priority, device) 组合有一个独立的 `ncclComm_t`。

#### TCP ID 交换 (`exchange_ids_via_tcp`)

跨节点时需要交换 7 个 NCCL Unique ID。方案：

- **Rank 0 (server)**：bind 到指定端口 → 生成 7 个 `ncclUniqueId` → accept client 连接 → 发送全部 ID
- **Rank 1+ (client)**：gethostbyname 解析 master_addr → connect (30s 超时+重试) → 接收全部 ID

ID 交换在**管理网络**（192.10.10.x）上进行，与 RDMA 数据面分离。

#### Communicator 创建

```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.trafficClass = p * 8;        // DSCP = priority × 8
ncclCommInitRankConfig(&comm, world_size, ids[p], rank, &config);
```

`trafficClass` 是 NCCL 2.30.7 引入的原生 API 字段，NCCL 在建立 RDMA QP 时会自动将其设置为 QP 的 `traffic_class` 属性。**无需修改 NCCL 源码**。

#### Priority 切换

`multi_comm_set_priority(p)` 只修改 `g_handle.current_priority`，后续 `multi_comm_allreduce` 从 `comms[p][device]` 中选择对应的 communicator。**无运行时开销**（O(1) 指针切换）。

### 2. SLO 调度器 — `SLOScheduler`

**路径**: `src/slo_scheduler.py`，类 `SLOScheduler`

调度粒度：**WINDOW**（论文 Section IV-F：W=20 个迭代组成一个窗口，调度器在每个 window boundary 重新计算分配并切换 DSCP）。

#### 算法: Progress Deficit + 窗口通信紧急信号（双信号）

```
π_i(t) = A_i(t) / (c_i × T_target × k_i(t)) - 1

其中:
  A_i(t)        = 累计实际墙钟时间
  T_target      = solo 窗口墙钟时间基线（EMA 滑动，仅 warmup 窗口更新）
  k_i(t)        = 已完成窗口数
  c_i           = SLO 松弛系数（租户提供）
```

**窗口通信紧急信号**（新增，解决计算主导负载下 π 不敏感的问题）：

```
comm_ratio = 当前窗口纯通信时间 / 基线窗口纯通信时间
urgency    = max(π, comm_ratio × 1.5)     # comm_ratio > 1.3 时触发
```

纯通信时间由 `MultiCommWrapper.allreduce()` 内部计时累计（NCCL 调用阻塞返回即完成）。**comm 基线校准**：跳过第一个窗口（含 NCCL 首次 allreduce 初始化开销，iter0 可达数秒），从窗口 2 起在 solo warmup 期间取 min 聚合（对 iter0 型突发与 warmup 期竞争污染鲁棒），warmup 后冻结。**阈值 1.3 依据**：10.1 测试床实测两作业共享 100G 链路时自然竞争膨胀为 1.3~1.4×（solo ~21 Gbps 算法带宽 / 竞争 ~16 Gbps，链路远未饱和，1.5× 物理不可达），1.3× 高于 solo 稳态噪声（±15%）。计算主导负载（如真实模型训练）中通信膨胀即使只有 1.36×，纯 π 也几乎不变（wall 时间仅涨几个百分点），comm_ratio 信号保证通信恶化超过阈值时强制进入 P6。

#### EMA 更新公式

```python
if not ema_initialized:
    ema_bandwidth = actual_bw          # 首次直接赋值
else:
    ema_bandwidth = (                  # 指数滑动平均
        alpha × actual_bw + 
        (1 - alpha) × ema_bandwidth
    )
```

- `alpha = 0.3`：对拥塞响应较快（~3-4 window 稳定）
- T_target / comm 基线仅在 solo warmup（前 `SOLO_WARMUP_WINDOWS=5` 个窗口）EMA 更新，之后冻结——避免拥塞自我放大

#### Priority 阈值表（π / urgency → Priority）

| 区间 | Priority | DSCP | 含义 |
|---------|----------|------|------|
| urgency > 0.3 | P6 | 8 | 显著落后（紧急） |
| -0.1 < urgency ≤ 0.3 | P4 | 0 | SLO 边界 |
| -0.5 < urgency ≤ -0.1 | P2 | 24 | 轻度领先 |
| urgency ≤ -0.5 | P1 | 32 | 大幅领先 |

注意：**DSCP→TC 映射以 10.1 测试床实测为准**（`mlnx_qos --trust dscp` 探针实验）：DSCP=8→tc:0（最高）、DSCP=0→tc:1、DSCP=16→tc:2、DSCP=24→tc:3、DSCP=32→tc:4、DSCP=40→tc:5。P6 使用 DSCP=8 直达 tc:0 最高队列。

#### 典型行为

| 场景 | π | comm_ratio | urgency | 结果 |
|------|----|------------|---------|------|
| Solo 独占 | ~0 | 1.0 | ~0 | P4（保持） |
| 计算主导 + 通信膨胀 1.36× | -0.39（误判领先） | 1.36 | 2.04 | → P6（紧急） |
| 通信恢复 | 负值累积 | 1.0 | <0 | → P2/P1（让出带宽） |

### 3. Python 封装 — `MultiCommWrapper`

**路径**: `src/slo_scheduler.py`，类 `MultiCommWrapper`

核心接口：

| 方法 | 功能 |
|------|------|
| `__init__(scheduler, rank, world_size, device_list, master_addr, port)` | 加载 .so → 初始化 7 个 communicator |
| `window_start(window)` | 记录窗口起始时间 + 清零通信计时 + 应用当前 priority |
| `allreduce(sendbuf, recvbuf, count, datatype, op, device_idx)` | 用当前 priority 的 comm 执行 AllReduce，内部累计窗口纯通信时间 |
| `window_end(window, data_size)` | 计算窗口墙钟时间 → scheduler.update(wall, size, comm) → 设置新 priority |
| `destroy()` | 销毁所有 communicator |

### 4. 实验脚本集成

**路径**: `../experiments/P4_dumbbell_slo/`

#### 双代码路径架构

```
p4_job*.py
├── longliu 模式
│   ├── 不调用 dist.init_process_group
│   ├── 使用 MultiCommWrapper 创建 7 个 communicator
│   ├── 使用 mc.allreduce() 替代 dist.all_reduce()
│   └── epoch_start/end 由 MultiCommWrapper 管理
│
├── fair 模式
│   ├── 标准 dist.init_process_group('nccl')
│   └── 标准 dist.all_reduce()
│
└── solo 模式
    └── 同 fair，但只跑 Job1
```

#### run_p4.sh 模式差异

| 配置项 | longliu | fair/solo |
|--------|---------|-----------|
| NCCL 来源 | NCCL 2.30.7 (`nccl-master/build/lib`) | PyTorch 内置 NCCL |
| LD_LIBRARY_PATH | NCCL 2.30.7 | PyTorch 的 nvidia/nccl/lib |
| PYTHONPATH | `multi_comm_slo/src` | 不设置 |
| MULTI_COMM_PORT | 设为 PORT_J1/J2 | 不设置 |

## 与旧方案（修改 NCCL 源码）的对比

| 方面 | 旧方案 (nccl-dscp) | 新方案 (multi_comm_slo) |
|------|-------------------|------------------------|
| NCCL 修改 | 修改 nccl_dscp_epoch_start/end 源码 | **无修改**，使用原生 trafficClass API |
| 编译复杂度 | 需完整编译 NCCL | 仅编译一个 .so 文件 |
| 兼容性 | 必须替换 PyTorch 的 NCCL，socket bootstrap 异常 | 独立 .so，通过 LD_LIBRARY_PATH 加载 |
| fair/solo 模式 | 需卸载自定义 NCCL | 自动使用 PyTorch 内置 NCCL |
| 优先级切换 | ibv_modify_qp（运行时改 QP） | **预创建 7 个 comm**，O(1) 切换 |
| 跨节点 ID 交换 | 自动由 NCCL bootstrap 处理 | TCP socket 手动交换 |
| RDMA IP 配置 | 管理 IP 和 RDMA IP 分离 | 管理 IP 仅用于 TCP ID 交换 |
| 调度粒度 | per window | per window |

## 调度时序

```
Window N:
  window_start(N)
    ├─ multi_comm_set_priority(Px)   ← 设置当前 priority
    ├─ iter 0: allreduce()           ← 使用 Px 的 communicator，累计纯通信时间
    ├─ iter 1: allreduce()           ← 使用 Px 的 communicator，累计纯通信时间
    ├─ ...
    └─ iter 19: allreduce()          ← 使用 Px 的 communicator，累计纯通信时间
  window_end(N, data_size)
    ├─ 计算 window wall time
    ├─ π = A / (c_i × T_target × k) - 1
    ├─ comm_ratio = 窗口纯通信时间 / 基线通信时间
    ├─ urgency = max(π, comm_ratio × 1.5)
    └─ 查表决定新 priority Py → 设置给 Window N+1

Window N+1: (使用 Py 调度)
  window_start(N+1)  ← 用新 priority
  ...
```

## 实验拓扑

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

## 关键文件

```
multi_comm_slo/
├── DESIGN.md                    ← 本文档
├── src/
│   ├── multi_comm.c             ← C 核心实现
│   ├── multi_comm.h             ← C 头文件
│   ├── slo_scheduler.py         ← Python 调度器 + ctypes 封装
│   └── Makefile                 ← 编译配置
├── build/
│   └── libmulti_comm.so         ← 编译产物
├── build.sh                     ← 一键编译
└── test_cross_node.sh           ← 跨节点验证脚本

experiments/P4_dumbbell_slo/
├── p4_job1.py                   ← Job 1（tight SLO, c_i=1.5）
├── p4_job2.py                   ← Job 2（loose SLO, c_i=2.5）
├── run_p4.sh                    ← 一键实验启动脚本
└── ib_prio_strict.sh            ← 交换机 strict priority 配置
```

## 依赖

- NCCL 2.30.7+（必须支持 `ncclConfig_t.trafficClass`）
- CUDA（与编译时一致）
- Python 3.8+（ctypes, PyTorch）
- 交换机需配置 strict priority QoS（8 队列对应 DSCP 0~56）
