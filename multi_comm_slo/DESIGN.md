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

#### 算法: EMA 带宽自估计

```
ei  = EMA(历史实测带宽)      ← 自适应 solo 基线
ai  = epoch_data_size /    ← 实测带宽（含拥塞影响）
      epoch_wall_time
Ui  = ai / ei               ← 利用率比
```

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

- `alpha = 0.3`：对拥塞响应较快（~3-4 epoch 稳定）

#### Priority 阈值表

| Ui 区间 | Priority | DSCP | 含义 |
|---------|----------|------|------|
| Ui < 0.6 | P0 | 0 | 最紧急（严重拥塞） |
| 0.6 ≤ Ui < 0.8 | P1 | 8 | 紧急 |
| 0.8 ≤ Ui < 1.0 | P2 | 16 | 轻度拥塞 |
| 1.0 ≤ Ui < 1.2 | P3 | 24 | 正常 |
| 1.2 ≤ Ui < 1.4 | P4 | 32 | 轻微空闲 |
| 1.4 ≤ Ui < 1.6 | P5 | 40 | 空闲 |
| Ui ≥ 1.6 | P6 | 48 | 最空闲（让出带宽） |

注意：**P0 (DSCP=0) 在交换机上优先级最高，P6 (DSCP=48) 最低**（取决于 P4 交换机队列映射配置）。

#### 典型行为

| 场景 | ai | ei | Ui | 结果 |
|------|----|----|-----|------|
| Solo 独占 100Gbps | ~12 GB/s | 12 GB/s | ~1.0 | P3（不变） |
| 竞争出现，带宽被分 | ~5 GB/s | 12 GB/s | ~0.42 | → P0（最高优先级） |
| 持续竞争 | ~7 GB/s | EMA 缓慢下降 | ~0.67 | → P1（保持高优先级） |
| 竞争结束 | ~11 GB/s | EMA 仍在低位 | ~1.57 | → P5（降低优先级） |

### 3. Python 封装 — `MultiCommWrapper`

**路径**: `src/slo_scheduler.py`，类 `MultiCommWrapper`

核心接口：

| 方法 | 功能 |
|------|------|
| `__init__(scheduler, rank, world_size, device_list, master_addr, port)` | 加载 .so → 初始化 7 个 communicator |
| `epoch_start(epoch)` | 记录起止时间 + 应用当前 priority |
| `allreduce(sendbuf, recvbuf, count, datatype, op, device_idx)` | 用当前 priority 的 comm 执行 AllReduce |
| `epoch_end(epoch, data_size)` | 计算耗时 → scheduler.update() → 设置新 priority |
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
| 调度粒度 | per epoch | per epoch |

## 调度时序

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

Epoch N+1: (使用 Py 调度)
  epoch_start(N+1)  ← 用新 priority
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
