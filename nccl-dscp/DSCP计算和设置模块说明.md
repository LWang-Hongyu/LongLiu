# DSCP计算和设置模块说明

## 概述

DSCP计算和设置模块负责将计算得到的ui优先级值映射到具体的DSCP（Differentiated Services Code Point）值，并将该值应用到网络层的IP数据包中，实现动态QoS调整。

---

## 1. DSCP值映射

### 1.1 7级DSCP映射表

**位置**：`src/include/dscp_adapter.h` (第59行)

DSCP适配器支持7个优先级级别，对应不同的DSCP值：

```c
int dscpMapping[7];  // DSCP values: [0, 18, 28, 26, 36, 34, 46]
```

**映射表**：

| 级别 | DSCP值 | 名称 | 说明 |
|------|--------|------|------|
| 0 | 0 | BE (Best Effort) | 尽力而为，最低优先级 |
| 1 | 18 | AF21 | 保证转发，低优先级 |
| 2 | 28 | AF32 | 保证转发，中低优先级 |
| 3 | 26 | AF31 | 保证转发，中优先级（默认） |
| 4 | 36 | AF42 | 保证转发，中高优先级 |
| 5 | 34 | AF41 | 保证转发，高优先级 |
| 6 | 46 | EF (Expedited Forwarding) | 加速转发，最高优先级 |

**初始化**：`src/misc/dscp_adapter.cc` (第42-43行)

```c
adapter->dscpMapping[0] = 0;   // BE
adapter->dscpMapping[1] = 18; // AF21
adapter->dscpMapping[2] = 28; // AF32
adapter->dscpMapping[3] = 26; // AF31 (default)
adapter->dscpMapping[4] = 36; // AF42
adapter->dscpMapping[5] = 34; // AF41
adapter->dscpMapping[6] = 46; // EF
```

### 1.2 优先级到DSCP映射函数

**位置**：`src/misc/dscp_adapter.cc` (第242-311行)

```242:311:src/misc/dscp_adapter.cc
int ncclDscpAdapterPriorityToDscp(struct ncclDscpAdapter* adapter, 
                                    double priority) {
  if (adapter == NULL) return 26; // Default: AF31
  
  pthread_mutex_lock(&adapter->mutex);
  
  // Update priority history
  if (adapter->numPriorities < NCCL_DSCP_MAX_EPOCHS) {
    adapter->priorityHistory[adapter->numPriorities++] = priority;
    
    // Update min/max
    if (adapter->numPriorities == 1) {
      adapter->minPriority = priority;
      adapter->maxPriority = priority;
    } else {
      if (priority < adapter->minPriority) adapter->minPriority = priority;
      if (priority > adapter->maxPriority) adapter->maxPriority = priority;
    }
    
    // Enable dynamic mapping after 2 priorities
    if (adapter->numPriorities >= 2) {
      adapter->useDynamicMapping = 1;
    }
  }
  
  int dscp;
  
  // Use dynamic mapping if available
  if (adapter->useDynamicMapping && 
      (adapter->maxPriority - adapter->minPriority) > 0.1) {
    // Dynamic mapping: normalize priority to [0, 1] based on historical range
    double range = adapter->maxPriority - adapter->minPriority;
    double buffer = range * 0.1; // 10% buffer
    if (buffer < 0.1) buffer = 0.1;
    
    double minNorm = adapter->minPriority - buffer;
    double maxNorm = adapter->maxPriority + buffer;
    
    double normalized = (priority - minNorm) / (maxNorm - minNorm);
    if (normalized < 0.0) normalized = 0.0;
    if (normalized > 1.0) normalized = 1.0;
    
    // Map to 7 levels
    int levelIndex = (int)(normalized * 7);
    if (levelIndex > 6) levelIndex = 6;
    
    dscp = adapter->dscpMapping[levelIndex];
  } else {
    // Default fixed-threshold mapping
    if (priority >= 1.6) {
      dscp = adapter->dscpMapping[6]; // EF (46)
    } else if (priority >= 1.4) {
      dscp = adapter->dscpMapping[5]; // AF41 (34)
    } else if (priority >= 1.2) {
      dscp = adapter->dscpMapping[4]; // AF42 (36)
    } else if (priority >= 1.0) {
      dscp = adapter->dscpMapping[3]; // AF31 (26)
    } else if (priority >= 0.8) {
      dscp = adapter->dscpMapping[2]; // AF32 (28)
    } else if (priority >= 0.6) {
      dscp = adapter->dscpMapping[1]; // AF21 (18)
    } else {
      dscp = adapter->dscpMapping[0]; // BE (0)
    }
  }
  
  pthread_mutex_unlock(&adapter->mutex);
  
  return dscp;
}
```

---

## 2. 映射策略

### 2.1 动态映射（Dynamic Mapping）

**启用条件**：
1. 至少收集了2个优先级值
2. 优先级范围（maxPriority - minPriority）> 0.1

**映射算法**：

```
步骤1: 计算优先级范围
    range = maxPriority - minPriority

步骤2: 添加10%缓冲
    buffer = range * 0.1
    if (buffer < 0.1) buffer = 0.1  // 最小缓冲0.1

步骤3: 计算归一化范围
    minNorm = minPriority - buffer
    maxNorm = maxPriority + buffer

步骤4: 归一化当前优先级
    normalized = (priority - minNorm) / (maxNorm - minNorm)
    // 限制在 [0.0, 1.0] 范围内

步骤5: 映射到7个级别
    levelIndex = (int)(normalized * 7)
    if (levelIndex > 6) levelIndex = 6

步骤6: 获取DSCP值
    dscp = dscpMapping[levelIndex]
```

**优势**：
- 自适应：根据历史优先级范围动态调整阈值
- 敏感：能够更准确地反映训练进度的变化
- 灵活：适应不同训练场景的优先级分布

**示例**：

假设历史优先级范围：[0.5, 1.8]
- buffer = (1.8 - 0.5) * 0.1 = 0.13
- minNorm = 0.5 - 0.13 = 0.37
- maxNorm = 1.8 + 0.13 = 1.93

当前优先级 = 1.5：
- normalized = (1.5 - 0.37) / (1.93 - 0.37) = 0.72
- levelIndex = (int)(0.72 * 7) = 5
- dscp = dscpMapping[5] = 34 (AF41)

### 2.2 固定阈值映射（Fixed Threshold Mapping）

**使用场景**：
- 优先级数量不足（< 2个）
- 优先级范围过小（≤ 0.1）

**阈值表**：

| 优先级范围 | DSCP值 | 级别 |
|-----------|--------|------|
| ui ≥ 1.6 | 46 (EF) | 6 |
| 1.4 ≤ ui < 1.6 | 34 (AF41) | 5 |
| 1.2 ≤ ui < 1.4 | 36 (AF42) | 4 |
| 1.0 ≤ ui < 1.2 | 26 (AF31) | 3 |
| 0.8 ≤ ui < 1.0 | 28 (AF32) | 2 |
| 0.6 ≤ ui < 0.8 | 18 (AF21) | 1 |
| ui < 0.6 | 0 (BE) | 0 |

**特点**：
- 简单直接，易于理解
- 适用于优先级分布稳定的场景
- 作为动态映射的备用方案

---

## 3. DSCP设置流程

### 3.1 环境变量设置

**位置**：`src/misc/dscp_adapter.cc` (第341-358行)

```341:358:src/misc/dscp_adapter.cc
ncclResult_t ncclDscpAdapterSetDscpViaEnv(struct ncclDscpAdapter* adapter, 
                                           int dscp) {
  if (adapter == NULL) return ncclInvalidArgument;
  
  // Set environment variables for NCCL to pick up
  char dscpStr[16];
  snprintf(dscpStr, sizeof(dscpStr), "%d", dscp);
  
  // Set both NCCL_SOCKET_DSCP and NCCL_NET_DSCP
  setenv("NCCL_SOCKET_DSCP", dscpStr, 1);
  setenv("NCCL_NET_DSCP", dscpStr, 1);
  
  if (adapter->rank == 0) {
    INFO(NCCL_ENV, "DSCP adapter: Set DSCP to %d (priority-based)", dscp);
  }
  
  return ncclSuccess;
}
```

**功能**：
- 将DSCP值转换为字符串
- 设置环境变量 `NCCL_SOCKET_DSCP` 和 `NCCL_NET_DSCP`
- 在rank 0打印日志信息

**环境变量**：
- `NCCL_SOCKET_DSCP`：用于socket通信的DSCP值
- `NCCL_NET_DSCP`：用于网络通信的DSCP值

### 3.2 环境变量读取

**位置**：`src/misc/socket.cc` (第121-136行)

```121:136:src/misc/socket.cc
// Get DSCP value from environment variable
static int envSocketDscp(void) {
  int dscp = -1; // DSCP not set by default
  char* env = getenv("NCCL_SOCKET_DSCP");
  if (env == NULL)
    return dscp;

  dscp = atoi(env);
  if (dscp < 0 || dscp > 63) {
    WARN("NCCL_SOCKET_DSCP: invalid value %s, must be between 0-63", env);
    return -1;
  }

  INFO(NCCL_ENV, "NCCL_SOCKET_DSCP set by environment to %d", dscp);
  return dscp;
}
```

**功能**：
- 从环境变量 `NCCL_SOCKET_DSCP` 读取DSCP值
- 验证DSCP值范围（0-63）
- 返回DSCP值，如果未设置或无效则返回-1

### 3.3 Socket层DSCP设置

**位置**：`src/misc/socket.cc` (第138-172行)

```138:172:src/misc/socket.cc
// Set DSCP (Differentiated Services Code Point) on socket
// DSCP is in the high 6 bits of IP_TOS field, so we need to shift left by 2 bits
static ncclResult_t socketSetDscp(struct ncclSocket* sock) {
  if (sock == NULL || sock->fd == -1) {
    return ncclInvalidArgument;
  }

  int dscp = envSocketDscp();
  if (dscp < 0) {
    // DSCP not set, skip
    return ncclSuccess;
  }

  // Convert DSCP to IP_TOS value (DSCP is in high 6 bits, shift left by 2)
  int tos = dscp << 2;
  
  // Set IP_TOS for IPv4
  if (sock->addr.sa.sa_family == AF_INET) {
    if (setsockopt(sock->fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos)) != 0) {
      WARN("socketSetDscp: Failed to set IP_TOS for IPv4 socket: %s", strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_NET, "Set DSCP=%d (IP_TOS=0x%02x) on IPv4 socket", dscp, tos);
  }
  // Set IPV6_TCLASS for IPv6 (equivalent to IP_TOS for IPv6)
  else if (sock->addr.sa.sa_family == AF_INET6) {
    if (setsockopt(sock->fd, IPPROTO_IPV6, IPV6_TCLASS, &tos, sizeof(tos)) != 0) {
      WARN("socketSetDscp: Failed to set IPV6_TCLASS for IPv6 socket: %s", strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_NET, "Set DSCP=%d (IPV6_TCLASS=0x%02x) on IPv6 socket", dscp, tos);
  }

  return ncclSuccess;
}
```

**关键技术点**：

1. **DSCP到IP_TOS的转换**：
   - DSCP值占用IP_TOS字段的高6位
   - 需要左移2位：`tos = dscp << 2`
   - 例如：DSCP=26 (0x1A) → IP_TOS=104 (0x68)

2. **IPv4支持**：
   - 使用 `setsockopt(sock->fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos))`
   - 设置IP数据包的Type of Service字段

3. **IPv6支持**：
   - 使用 `setsockopt(sock->fd, IPPROTO_IPV6, IPV6_TCLASS, &tos, sizeof(tos))`
   - IPv6使用Traffic Class字段，功能等同于IPv4的IP_TOS

### 3.4 Socket设置调用点

DSCP设置在以下三个位置被调用：

1. **接受连接时**（第504行）：
   ```c
   // Set DSCP on accepted socket if configured
   NCCLCHECK(socketSetDscp(sock));
   ```

2. **建立连接时**（第698行）：
   ```c
   // Set DSCP on connecting socket if configured (must be set before connect)
   NCCLCHECK(socketSetDscp(sock));
   ```

3. **创建socket时**（第826行）：
   ```c
   // Set DSCP on socket if configured
   if (sock->fd >= 0 && addr != NULL) {
     NCCLCHECKGOTO(socketSetDscp(sock), ret, fail);
   }
   ```

**重要**：对于主动连接，必须在调用 `connect()` 之前设置DSCP，否则设置可能无效。

---

## 4. 完整调用流程

### 4.1 从ui到DSCP设置的完整流程

```
Epoch结束
    ↓
ncclDscpAdapterEndEpoch()  // 聚合统计信息
    ↓
ncclDscpAdapterUpdateDscpForNextEpoch()  // 触发DSCP更新
    ↓
ncclDscpAdapterCalculatePriority()  // 计算ui值
    ↓
ncclDscpAdapterPriorityToDscp()  // 将ui映射到DSCP值
    ├─ 更新优先级历史
    ├─ 选择映射策略（动态/固定）
    └─ 返回DSCP值
    ↓
ncclDscpAdapterSetDscpViaEnv()  // 设置环境变量
    ├─ setenv("NCCL_SOCKET_DSCP", dscpStr, 1)
    └─ setenv("NCCL_NET_DSCP", dscpStr, 1)
    ↓
下次创建/连接socket时
    ↓
socketSetDscp()  // 从环境变量读取并设置
    ├─ envSocketDscp()  // 读取环境变量
    ├─ tos = dscp << 2  // 转换为IP_TOS
    └─ setsockopt()  // 应用到socket
```

### 4.2 关键函数调用

**位置**：`src/misc/dscp_adapter.cc` (第313-339行)

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

## 5. 优先级历史管理

### 5.1 历史记录

**数据结构**：`src/include/dscp_adapter.h` (第48-53行)

```c
double priorityHistory[NCCL_DSCP_MAX_EPOCHS];  // 优先级历史数组
int numPriorities;                              // 已记录的优先级数量
double minPriority;                             // 历史最小优先级
double maxPriority;                             // 历史最大优先级
int useDynamicMapping;                          // 是否启用动态映射
```

### 5.2 更新逻辑

在 `ncclDscpAdapterPriorityToDscp()` 中：

1. **记录优先级**：将当前优先级添加到历史数组
2. **更新最值**：
   - 第一个优先级：同时作为min和max
   - 后续优先级：更新min和max
3. **启用动态映射**：当优先级数量 ≥ 2时启用

---

## 6. 线程安全

所有DSCP计算和设置操作都通过互斥锁保护：

```c
pthread_mutex_lock(&adapter->mutex);
// ... 计算和更新逻辑 ...
pthread_mutex_unlock(&adapter->mutex);
```

确保多线程环境下：
- 优先级历史的一致性
- min/max值的准确性
- DSCP映射的正确性

---

## 7. 默认值

### 7.1 默认DSCP值

**位置**：`src/misc/dscp_adapter.cc` (第37行)

```c
adapter->currentDscp = 26; // Default: AF31 (medium priority)
```

如果适配器未初始化或映射失败，返回默认值26（AF31，中优先级）。

### 7.2 默认映射策略

- 初始状态：使用固定阈值映射
- 收集2个优先级后：切换到动态映射（如果范围 > 0.1）

---

## 8. 总结

DSCP计算和设置模块的核心功能：

1. ✅ **7级DSCP映射**：支持从BE(0)到EF(46)的7个优先级级别
2. ✅ **双映射策略**：动态映射（自适应）和固定阈值映射（备用）
3. ✅ **环境变量机制**：通过环境变量在Python和C++之间传递DSCP值
4. ✅ **Socket层应用**：在IPv4/IPv6 socket上设置IP_TOS/IPV6_TCLASS
5. ✅ **优先级历史管理**：记录历史优先级，支持动态阈值调整
6. ✅ **线程安全**：通过互斥锁保护共享数据
7. ✅ **实时调整**：每个epoch结束后立即计算并应用新的DSCP值

该模块实现了从训练性能指标（ui）到网络QoS（DSCP）的完整转换和应用流程。
