# Ui（优先级计算）模块说明

## 概述

Ui模块负责计算每个epoch的优先级值（ui），用于评估训练进度是否滞后。**ui值越大，表示训练进度越慢，紧急程度越高**。

---

## 1. Ui计算公式

### 1.1 基本公式

```
ui = ai / ei
```

其中：
- **ai**：实际累积时间（Actual accumulated time）
- **ei**：期望累积时间（Expected accumulated time）

### 1.2 详细计算

**ai（实际累积时间）**：
```c
ai = targetEpoch->endTime - firstEpochStartTime
```
- 从第一个epoch开始到当前epoch结束的总时间

**ei（期望累积时间）**：
```c
ei = sloThreshold * (epoch + 1) * idealTimePerEpoch
```
- `sloThreshold`：SLO阈值（默认1.2，可通过环境变量`NCCL_DSCP_SLO_THRESHOLD`配置）
- `(epoch + 1)`：已完成的epoch数量（从0开始计数）
- `idealTimePerEpoch`：理想情况下每个epoch的耗时

**idealTimePerEpoch（理想epoch耗时）**：
```c
idealTimePerEpoch = computeDuration + idealCommTimePerEpoch
```
- `computeDuration`：计算时间（从epoch1获取）
- `idealCommTimePerEpoch`：理想通信时间

**idealCommTimePerEpoch（理想通信时间）**：
```c
idealCommTimePerEpoch = (epoch1->totalBytes * 8.0) / (idealBandwidth * 1e9)
```

---

## 2. 核心函数

### 2.1 优先级计算函数

**位置**：`src/misc/dscp_adapter.cc` (第184-240行)

```184:240:src/misc/dscp_adapter.cc
ncclResult_t ncclDscpAdapterCalculatePriority(struct ncclDscpAdapter* adapter,
                                               int epoch,
                                               double* priority) {
  if (adapter == NULL || priority == NULL) return ncclInvalidArgument;
  if (!adapter->enabled) return ncclSuccess;
  
  pthread_mutex_lock(&adapter->mutex);
  
  // Need at least 2 epochs to calculate priority
  if (adapter->numEpochs < 2 || epoch < 0 || epoch >= adapter->numEpochs) {
    pthread_mutex_unlock(&adapter->mutex);
    return ncclInvalidArgument;
  }
  
  struct ncclEpochStats* targetEpoch = &adapter->epochs[epoch];
  struct ncclEpochStats* epoch1 = &adapter->epochs[1]; // Use epoch 1 as reference
  
  // Calculate ideal bandwidth if not yet calculated
  if (adapter->idealBandwidth == 0.0) {
    if (epoch1->commDuration > 0) {
      adapter->idealBandwidth = (epoch1->totalBytes * 8.0 / 1e9) / epoch1->commDuration;
    } else {
      pthread_mutex_unlock(&adapter->mutex);
      return ncclInvalidArgument;
    }
  }
  
  // Calculate ideal communication time per epoch (using ideal bandwidth)
  double idealCommTimePerEpoch = (epoch1->totalBytes * 8.0) / (adapter->idealBandwidth * 1e9);
  
  // Ideal time per epoch = compute time + ideal communication time
  double idealTimePerEpoch = epoch1->computeDuration + idealCommTimePerEpoch;
  
  // Calculate ai (actual accumulated time) and ei (expected accumulated time)
  if (adapter->firstEpochStartTime == 0.0) {
    adapter->firstEpochStartTime = adapter->epochs[0].startTime;
  }
  
  if (targetEpoch->endTime == 0.0 || adapter->firstEpochStartTime == 0.0) {
    pthread_mutex_unlock(&adapter->mutex);
    return ncclInvalidArgument;
  }
  
  double ai = targetEpoch->endTime - adapter->firstEpochStartTime;
  double ei = adapter->sloThreshold * (epoch + 1) * idealTimePerEpoch;
  
  if (ei > 0) {
    *priority = ai / ei;
  } else {
    pthread_mutex_unlock(&adapter->mutex);
    return ncclInvalidArgument;
  }
  
  pthread_mutex_unlock(&adapter->mutex);
  
  return ncclSuccess;
}
```

### 2.2 函数参数

- **`adapter`**：DSCP适配器结构体指针
- **`epoch`**：要计算优先级的epoch编号（从0开始）
- **`priority`**：输出参数，返回计算得到的ui值

### 2.3 返回值

- **`ncclSuccess`**：计算成功
- **`ncclInvalidArgument`**：参数无效或epoch数量不足（需要至少2个epoch）

---

## 3. 关键参数说明

### 3.1 idealBandwidth（理想带宽）

**计算时机**：在`ncclDscpAdapterEndEpoch()`中，当完成前两个epoch后计算

**位置**：`src/misc/dscp_adapter.cc` (第159-177行)

```159:177:src/misc/dscp_adapter.cc
// Calculate ideal bandwidth from first two epochs if not yet calculated
if (adapter->idealBandwidth == 0.0 && adapter->numEpochs >= 2) {
  // Calculate ideal bandwidth from epochs 0 and 1
  struct ncclEpochStats* epoch0 = &adapter->epochs[0];
  struct ncclEpochStats* epoch1 = &adapter->epochs[1];
  
  if (epoch0->commDuration > 0 && epoch1->commDuration > 0) {
    // Average bandwidth during communication
    double avgBw0 = (epoch0->totalBytes * 8.0 / 1e9) / epoch0->commDuration;
    double avgBw1 = (epoch1->totalBytes * 8.0 / 1e9) / epoch1->commDuration;
    double avgBw = (avgBw0 + avgBw1) / 2.0;
    
    // Calculate 95th percentile bandwidth from individual operations
    // (simplified: use max of average bandwidths for now)
    adapter->idealBandwidth = avgBw;
    
    // TODO: Calculate 95th percentile from individual operations if needed
  }
}
```

**计算公式**：
```
idealBandwidth = (avgBw0 + avgBw1) / 2.0
```
其中：
- `avgBw0 = (epoch0->totalBytes * 8.0 / 1e9) / epoch0->commDuration`
- `avgBw1 = (epoch1->totalBytes * 8.0 / 1e9) / epoch1->commDuration`

**单位**：Gbps（千兆比特每秒）

### 3.2 sloThreshold（SLO阈值）

**默认值**：1.2

**配置方式**：通过环境变量`NCCL_DSCP_SLO_THRESHOLD`设置

**作用**：在计算期望时间时作为缓冲系数，允许一定的时间裕度

**位置**：`src/init.cc`（初始化时读取环境变量）

### 3.3 computeDuration（计算时长）

**计算方式**：在`ncclDscpAdapterEndEpoch()`中计算

**位置**：`src/misc/dscp_adapter.cc` (第154-157行)

```154:157:src/misc/dscp_adapter.cc
// Calculate compute duration = total wall-clock time - communication time
double totalDuration = epochStats->endTime - epochStats->startTime;
epochStats->computeDuration = (totalDuration > totalCommDuration) ? 
                             (totalDuration - totalCommDuration) : 0.0;
```

**公式**：
```
computeDuration = totalDuration - totalCommDuration
```

---

## 4. 计算流程

### 4.1 前置条件

1. **至少需要2个epoch**：使用epoch1作为参考基准
2. **epoch必须已完成**：`targetEpoch->endTime`必须已设置
3. **适配器必须启用**：`adapter->enabled == 1`

### 4.2 计算步骤

```
步骤1: 检查前置条件
    ↓
步骤2: 计算idealBandwidth（如果尚未计算）
    - 使用epoch0和epoch1的平均带宽
    ↓
步骤3: 计算idealCommTimePerEpoch
    - 基于epoch1的字节数和idealBandwidth
    ↓
步骤4: 计算idealTimePerEpoch
    - idealTimePerEpoch = computeDuration + idealCommTimePerEpoch
    ↓
步骤5: 计算ai（实际累积时间）
    - ai = targetEpoch->endTime - firstEpochStartTime
    ↓
步骤6: 计算ei（期望累积时间）
    - ei = sloThreshold * (epoch + 1) * idealTimePerEpoch
    ↓
步骤7: 计算ui
    - ui = ai / ei
```

### 4.3 调用时机

**位置**：`src/misc/dscp_adapter.cc` (第313-339行)

在`ncclDscpAdapterUpdateDscpForNextEpoch()`函数中调用：

```313:339:src/misc/dscp_adapter.cc
ncclResult_t ncclDscpAdapterUpdateDscpForNextEpoch(struct ncclDscpAdapter* adapter,
                                                     int currentEpoch,
                                                     double* priority,
                                                     int* dscp) {
  if (adapter == NULL || priority == NULL || dscp == NULL) return ncclInvalidArgument;
  if (!adapter->enabled) return ncclSuccess;

  // Calculate priority for current epoch
  double calculatedPriority = 0.0;
  ncclResult_t ret = ncclDscpAdapterCalculatePriority(adapter, currentEpoch, &calculatedPriority);
  if (ret != ncclSuccess) {
    return ret;
  }

  *priority = calculatedPriority;

  // Map priority to DSCP
  *dscp = ncclDscpAdapterPriorityToDscp(adapter, calculatedPriority);

  // Update current DSCP
  pthread_mutex_lock(&adapter->mutex);
  adapter->currentDscp = *dscp;
  pthread_mutex_unlock(&adapter->mutex);

  // Set DSCP via environment variable
  return ncclDscpAdapterSetDscpViaEnv(adapter, *dscp);
}
```

---

## 5. Ui值的含义

### 5.1 值域解释

- **ui < 1.0**：训练进度**超前**，实际时间少于期望时间
- **ui = 1.0**：训练进度**正常**，实际时间等于期望时间
- **ui > 1.0**：训练进度**滞后**，实际时间超过期望时间
  - ui越大，滞后越严重，紧急程度越高

### 5.2 示例

假设：
- epoch0开始时间：0秒
- epoch1结束时间：100秒
- idealTimePerEpoch：50秒
- sloThreshold：1.2

计算epoch1的ui：
- ai = 100 - 0 = 100秒
- ei = 1.2 × 2 × 50 = 120秒
- ui = 100 / 120 = 0.83

**结果**：ui = 0.83 < 1.0，表示训练进度超前，性能良好。

---

## 6. 数据结构

### 6.1 ncclDscpAdapter结构

**位置**：`src/include/dscp_adapter.h` (第34-60行)

关键字段：
```c
struct ncclDscpAdapter {
  double sloThreshold;            // SLO阈值（默认1.2）
  double idealBandwidth;          // 理想带宽（Gbps）
  double firstEpochStartTime;     // 第一个epoch开始时间（用于ai计算）
  struct ncclEpochStats epochs[NCCL_DSCP_MAX_EPOCHS];  // Epoch统计信息
  // ...
};
```

### 6.2 ncclEpochStats结构

**位置**：`src/include/dscp_adapter.h` (第21-31行)

关键字段：
```c
struct ncclEpochStats {
  double startTime;               // Epoch开始时间
  double endTime;                 // Epoch结束时间
  size_t totalBytes;              // 总通信字节数
  double commDuration;            // 通信时长
  double computeDuration;         // 计算时长
  // ...
};
```

---

## 7. 线程安全

所有计算操作都通过互斥锁保护：

```c
pthread_mutex_lock(&adapter->mutex);
// ... 计算逻辑 ...
pthread_mutex_unlock(&adapter->mutex);
```

确保多线程环境下数据的一致性和计算的准确性。

---

## 8. 总结

Ui模块的核心功能：

1. ✅ **动态优先级计算**：基于实际训练进度计算ui值
2. ✅ **自适应基准**：使用前两个epoch确定理想带宽和理想耗时
3. ✅ **SLO感知**：通过sloThreshold考虑服务等级目标
4. ✅ **实时评估**：每个epoch结束后立即计算优先级
5. ✅ **线程安全**：通过互斥锁保护共享数据

计算得到的ui值将用于后续的DSCP映射，实现动态QoS调整。
