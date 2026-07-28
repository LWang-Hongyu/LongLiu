# LongLiu DSCP Design Reference

> 基于当前 nccl-dscp 实现的代码分析，对照文件与行号，供论文 Design 章节参考。

---

## 1. 系统架构

LongLiu 是一个 **纯发送端（sender-side）** 的优先级调度方案。所有逻辑实现在 NCCL 通信库内部，交换机仅需标准 DSCP-to-CoS 优先级队列配置，无需状态跟踪。

### 修改范围

| 文件 | 说明 |
|------|------|
| `src/misc/dscp_adapter.cc` | 核心：epoch 管理、priority 计算、DSCP 映射 |
| `src/include/dscp_adapter.h` | 数据结构定义 |
| `src/transport/net_ib.cc` | QP creation: TC 注入；QP update: 动态修改 traffic_class |
| `src/misc/comm_stats.cc` | per-iteration 统计收集 + JSON 导出 |
| `src/include/comm_stats.h` | 迭代统计结构定义 |
| `src/enqueue.cc` | 在每次 NCCL op 时触发 epoch 检查与统计记录 |
| `src/init.cc` | DSCP adapter 初始化 + 销毁时导出 |

---

## 2. Epoch-Level 调度范例

### 2.1 调度粒度

LongLiu 以 **training epoch**（而非单个 iteration）为调度单元。一个 epoch 包含 N 个 training iteration（N = batches per epoch），epoch 内所有 collective 操作使用 **同一个 DSCP 优先级**。

```
Epoch K:        [iter i, iter i+1, ..., iter i+N-1]  → all use DSCP = D_K
Epoch K+1:      [iter i+N, ..., iter i+2N-1]          → all use DSCP = D_{K+1}
```

**代码路径**: `ncclDscpAdapterUpdateDscpForNextEpoch` → `ncclIbSetTc` + `ncclDscpAdapterUpdateIbQpPriority` 仅在 epoch 边界调用一次（[dscp_adapter.cc:425-486](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L425-L486)）。

### 2.2 Epoch 生命周期

```
Epoch N-1 结束 → EndEpoch(N-1) → 聚合统计 → 计算 priority → 更新 DSCP → 所有 QP 同步
Epoch N 开始   → StartEpoch(N)
Epoch N 运行   → 所有 iteration 使用 DSCP_D_{N-1} 计算结果
Epoch N 结束   → EndEpoch(N)   → ...
```

---

## 3. 核心算法

### 3.1 Epoch 统计聚合

当 `EndEpoch` 触发时（[dscp_adapter.cc:179-270](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L179-L270)），从 `comm_stats` 中聚合 epoch 内所有 iteration：

- **totalBytes**: epoch 总通信字节量
- **commDuration**: epoch 总通信时间（所有 collective 操作的 `endTime - startTime` 之和）
- **computeDuration**: epoch 总计算时间 = `epochWallTime - commDuration`
- **numIterations**: epoch 内的 iteration 数量

### 3.2 在线带宽校准（EMA Bandwidth）

每 epoch 结束时计算实测带宽并 EMA 更新（[dscp_adapter.cc:248-265](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L248-L265)）：

```
newBw = totalBytes / commDuration           (Gbps)
emaBw = α · newBw + (1-α) · emaBw          (α = 0.3, EMA)
```

- 冷启动：第一个 epoch 完成后用 `newBw` 初始化 EMA
- 运行时：每 epoch 用 EMA 平滑带宽估计，追踪无竞争下的理想带宽
- 不需要 solo profiling

### 3.3 Urgency Index 计算

使用 epoch 1 的统计数据作为 reference，计算理想每 epoch 时间（[dscp_adapter.cc:273-348](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L273-L348)）：

```
idealCommTimePerEpoch = epoch1_totalBytes / emaBw       (理想通信时间)
idealTimePerEpoch = epoch1_computeDuration + idealCommTimePerEpoch   (理想每 epoch 时间)

ai = epoch_K.endTime - firstEpochStartTime               (实际累积时间)
ei = SLO_threshold × (epoch0_actualDuration + K × idealTimePerEpoch)  (期望累积时间)

Urgency Index: U = ai / ei
```

- `U > 1`: job 落后 SLO，需要更高优先级
- `U = 1`: 正好匹配 SLO
- `U < 1`: job 超前 SLO，可以让出带宽

**Epoch 0 特殊处理**: 使用实际时间而非理想模型，因为 epoch 0 可能受 NCCL 初始化/warming 影响。

**SLO threshold** 默认为 1.2，可通过 `NCCL_DSCP_SLO_THRESHOLD` 环境变量调整。

### 3.4 Priority → DSCP 两级映射

#### 阶段一：固定阈值映射（第 1-2 个 epoch）

[dscp_adapter.cc:397-413](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L397-L413)

| Urgency Index | DSCP 值 | AF Class | 含义 |
|:------------:|:------:|:--------:|------|
| ≥ 1.6 | 38 | AF43 | 严重落后，最高优先级 |
| [1.4, 1.6) | 34 | AF41 | 中度落后 |
| [1.2, 1.4) | 36 | AF42 | 轻度落后 |
| [1.0, 1.2) | 26 | AF31 | SLO 临界 |
| [0.8, 1.0) | 28 | AF32 | 正常偏快 |
| [0.6, 0.8) | 18 | AF21 | 明显超前 |
| < 0.6 | 0 | BE | 严重超前，最低优先级 |

共 7 级（`{0, 18, 28, 26, 36, 34, 38}`），对应 DSCP BE 到 AF43。

#### 阶段二：动态自适应映射（第 3 个 epoch 起）

[dscp_adapter.cc:378-396](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L378-L396)

收集前两轮 priority 的 min/max，后续归一化到 [0,1] 区间映射到 7 级：

```
normalized = (U - minPriority) / (maxPriority - minPriority)  (带 10% buffer)
levelIndex = floor(normalized × 7)
DSCP = dscpMapping[levelIndex]
```

优势：自动适应不同 SLO 阈值、不同带宽环境，无需手动校准阈值。

---

## 4. DSCP → RoCEv2 网络层注入

### 4.1 两条注入路径

**路径 A — QP 创建时注入**（[net_ib.cc:522](file:///home/why/LongLiu_rebuild/nccl-dscp/src/transport/net_ib.cc#L522)）：

```c
qpAttr.ah_attr.grh.traffic_class = ncclIbGetTc();   // 读取当前全局 TC
// 在 ibv_modify_qp (RTR state) 时设置
```

**路径 B — QP 运行时动态更新**（[net_ib.cc:565-614](file:///home/why/LongLiu_rebuild/nccl-dscp/src/transport/net_ib.cc#L565-L614)）：

```
1. ibv_query_qp(IBV_QP_AV)          // 查询当前 QP 属性
2. 跳过非 RTS (Ready To Send) 状态  // 避免修改未就绪 QP
3. 修改 ah_attr.sl 和 ah_attr.grh.traffic_class
4. ibv_modify_qp(IBV_QP_AV)         // 注入内核态
5. ibv_query_qp(IBV_QP_AV)          // 验证确认
```

仅适用于 RoCEv2。IB link layer 的 traffic_class 与 RoCEv2 的 DSCP 映射方式不同。

### 4.2 DSCP → SL/TC 映射

```c
ibSl = (dscp × 15) / 63       // Service Level: 0..15
ibTc = (dscp × 7)  / 63       // Traffic Class: 0..7
```

（[dscp_adapter.cc:462-466](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L462-L466)）

### 4.3 全局 TC 状态

`g_ncclIbTc`（[net_ib.cc:88](file:///home/why/LongLiu_rebuild/nccl-dscp/src/transport/net_ib.cc#L88)）是全局变量，存储当前生效的 TC 值：
- QP 创建时调用 `ncclIbGetTc()` 读取 → 新 QP 继承当前优先级
- QP 动态更新时调用 `ncclIbSetTc(tc)` 写入 → 后续新建 QP 使用新 TC
- DSCP=0（BE）时回退到 `NCCL_IB_TC` 环境变量（[net_ib.cc:99-104](file:///home/why/LongLiu_rebuild/nccl-dscp/src/transport/net_ib.cc#L99-L104)）

### 4.4 全 QP 遍历

`ncclDscpAdapterUpdateIbQpPriority` 遍历 **所有 channel × 所有 peer × 所有 connector × 所有 QP**[dscp_adapter.cc:564-616](file:///home/why/LongLiu_rebuild/nccl-dscp/src/misc/dscp_adapter.cc#L564-L616)）：

- 遍历 `comm->nChannels` 个 channel
- 每个 channel 遍历 `comm->nRanks` 个 peer
- 每个 peer 遍历 `NCCL_MAX_CONNS` 个 connector（send + recv）
- 每个 connector 遍历 `ibSendComm->nqps` 或 `ibRecvComm->nqps` 个 QP

**非 RTS QP 跳过**，无后续补更新机制。QP 重建时从 `g_ncclIbTc` 读取当前值。

---

## 5. PyTorch 集成

### 5.1 ctypes API

两个导出 C 符号（`__attribute__((visibility("default")))`）：

```c
ncclResult_t ncclDscpEpochStart(int epoch);   // PyTorch 调用，设 pending 标志
ncclResult_t ncclDscpEpochEnd(int epoch);     // PyTorch 调用，设 pending 标志
```

Python 端通过 `ctypes.CDLL` 直接调用，仅设置 pending 标志（一次 `pthread_mutex_lock` + 一次整数赋值），开销 < 1μs。

### 5.2 异步处理模型

```
PyTorch (user thread)                  NCCL (NCCL op path)
─────────────────────                 ─────────────────────
ncclDscpEpochEnd(K)                    (pending 标志)
  └→ pendingEndEpoch = K               ↓
                                   ncclDscpAdapterCheckEpochTriggers()
                                     └→ Process End(K):
                                         聚合迭代统计
                                         计算 Urgency Index
                                         映射 DSCP
                                         ibv_modify_qp × 所有 QP
                                     └→ Process Start(K+1):
                                         记录 epoch 起止时间

ncclDscpEpochStart(K+1)
  └→ pendingStartEpoch = K+1
```

**Epoch 边界触发点**: 每次 NCCL op (e.g. `ncclAllReduce`) 的 `ncclEnqueueCheck` 中检查 pending 标志（[enqueue.cc:1610-1612](file:///home/why/LongLiu_rebuild/nccl-dscp/src/enqueue.cc#L1610-L1612)），先处理 End 再处理 Start。

---

## 6. 统计收集与导出

### 6.1 Per-Iteration 统计

`comm_stats`（[comm_stats.h](file:///home/why/LongLiu_rebuild/nccl-dscp/src/include/comm_stats.h)）按 iteration 记录每次 collective 操作：

- **容量**: 10000 iterations × 1000 ops/iter
- **粒度**: 每次 NCCL op（AllReduce, Broadcast 等）的 func、bytes、start/end time
- **时钟**: `CLOCK_MONOTONIC` 高精度时间戳（ns 级）

### 6.2 JSON 导出

销毁 communicator 时自动导出到 `/longliu8/nccl/staticsJson/comm_stats_rankN.json`（[init.cc:248-264](file:///home/why/LongLiu_rebuild/nccl-dscp/src/init.cc#L248-L264)）：

```json
{
  "version": "1.0",
  "epochs": [{
    "epoch": 0,
    "ui": 0.0,
    "dscp": 26,
    "iterations": [{
      "iteration": 0,
      "total_bytes": 1048576,
      "duration": 0.001234,
      "operations": [{ "func": "AllReduce", "bytes": 1048576, ... }]
    }]
  }]
}
```

可选 per-epoch 单独导出（`NCCL_STATS_EXPORT_EPOCH_FILES=1`）。

---

## 7. 环境变量接口

| 变量 | 作用 | 默认值 |
|------|------|:------:|
| `NCCL_DSCP_ADAPTER_ENABLED` | 启用/禁用适配器（`0`=禁用） | `1` |
| `NCCL_DSCP_SLO_THRESHOLD` | SLO 阈值（ei 的缩放因子） | `1.2` |
| `NCCL_STATS_BATCHES_PER_EPOCH` | 每 epoch 的 iteration 数 | 自动检测 |
| `NCCL_STATS_EXPORT_EPOCH_FILES` | 是否导出 per-epoch JSON 文件 | `0` |

---

## 8. 部署方式

当前唯一部署方式：**LD_PRELOAD**。

```bash
LD_PRELOAD=~/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2  python train.py
```

替换 PyTorch 自带 NCCL，训练脚本零修改。

---

## 9. 数据流总览

```
Training Loop
     │
     ├─ [PyTorch] ncclDscpEpochStart(N) ────→ pending flag
     │
     ├─ for iter in range(batches_per_epoch):
     │     ├─ ncclAllReduce(...)
     │     │      ├─ ncclCommStatsStartOp()   ← 记录 op 起始
     │     │      ├─ ncclDscpAdapterCheckEpochTriggers()  ← 处理 pending epoch events
     │     │      ├─ taskAppend()             ← NCCL 内部调度
     │     │      └─ ncclCommStatsEndOp()     ← 记录 op 结束
     │     └─ ...
     │
     ├─ [PyTorch] ncclDscpEpochEnd(N) ──────→ pending flag
     │
     [下一轮 NCCL op 时触发:]
     │  EndEpoch(N): 聚合迭代统计 → U = ai/ei → DSCP = f(U)
     │       ├─ ncclIbSetTc()
     │       └─ ibv_modify_qp(IBV_QP_AV) × all QPs
     │  StartEpoch(N+1): 记录起始时间
     │
     [训练结束, ncclCommDestroy]
     └─ ncclCommStatsExportToFile()  → /longliu8/nccl/staticsJson/comm_stats_rank*.json
```
