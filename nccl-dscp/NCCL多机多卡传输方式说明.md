# NCCL多机多卡传输方式说明

本文档详细说明NCCL在多机多卡分布式训练环境下的各种传输方式。

---

## 一、传输方式总览

NCCL定义了**4种主要传输方式**（Transport Types），按优先级从高到低：

| 传输类型 | 代码常量 | 优先级 | 适用场景 |
|---------|---------|--------|---------|
| **P2P** | `TRANSPORT_P2P` (0) | 最高 | 同一节点内GPU间直接通信 |
| **SHM** | `TRANSPORT_SHM` (1) | 高 | 同一节点内进程间共享内存通信 |
| **NET** | `TRANSPORT_NET` (2) | 中 | 多机间网络通信 |
| **COLLNET** | `TRANSPORT_COLLNET` (3) | 特殊 | 集合网络优化通信 |

**代码位置**：`src/include/transport.h:15-19`

```c
#define NTRANSPORTS 4
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3
```

---

## 二、详细传输方式说明

### 2.1 P2P (Peer-to-Peer) 传输

**文件**：`src/transport/p2p.cc`

**功能**：同一节点内GPU之间的直接通信

**特点**：
- ✅ **最高性能**：GPU间直接内存访问，无需CPU参与
- ✅ **最低延迟**：绕过CPU和系统内存
- ✅ **高带宽**：利用NVLink或PCIe直连

**P2P类型**（`p2p.cc:13`）：
```c
enum p2pType { 
  P2P_DIRECT,      // 直接P2P（NVLink）
  P2P_INTERMEDIATE, // 中间节点转发
  P2P_IPC,         // IPC（Inter-Process Communication）
  P2P_CUMEM        // CUDA统一内存
};
```

**适用场景**：
- 单机多卡训练
- 同一节点内GPU通信

**限制**：
- ❌ 仅适用于同一节点内的GPU
- ❌ 需要GPU支持P2P访问（NVLink或PCIe）

**环境变量控制**：
- `NCCL_P2P_DISABLE=1` - 禁用P2P传输
- `NCCL_P2P_LEVEL` - 设置P2P级别

---

### 2.2 SHM (Shared Memory) 传输

**文件**：`src/transport/shm.cc`

**功能**：同一节点内进程间通过共享内存通信

**特点**：
- ✅ **高效**：使用系统共享内存，避免网络开销
- ✅ **低延迟**：内存拷贝速度快
- ✅ **适合多进程**：同一节点上多个进程通信

**实现方式**：
- 使用`/dev/shm`共享内存
- 支持CUDA内存拷贝（GDR）
- 支持发送端或接收端内存拷贝

**适用场景**：
- 单机多进程训练
- 同一节点内进程间通信
- 作为P2P的备选方案

**环境变量控制**：
- `NCCL_SHM_DISABLE=1` - 禁用SHM传输
- `NCCL_SHM_USE_CUDA_MEMCPY=1` - 使用CUDA内存拷贝
- `NCCL_SHM_MEMCPY_MODE` - 设置内存拷贝模式（发送端/接收端）

**代码位置**：`src/transport/shm.cc:40-46`

```c
NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);
NCCL_PARAM(ShmUseCudaMemcpy, "SHM_USE_CUDA_MEMCPY", 0);
NCCL_PARAM(ShmMemcpyMode, "SHM_MEMCPY_MODE", SHM_SEND_SIDE);
```

---

### 2.3 NET (Network) 传输

**文件**：
- `src/transport/net_socket.cc` - Socket传输（TCP/IP）
- `src/transport/net_ib.cc` - InfiniBand传输

**功能**：多机间通过网络通信

**NET传输包含两种子类型**：

#### 2.3.1 Socket传输（TCP/IP）

**文件**：`src/transport/net_socket.cc`

**特点**：
- ✅ **通用性强**：适用于所有支持TCP/IP的网络
- ✅ **兼容性好**：标准以太网即可使用
- ✅ **易于配置**：无需特殊硬件

**实现方式**：
- 使用TCP Socket进行数据传输
- 支持IPv4和IPv6
- 支持DSCP QoS设置（我们的修改）

**适用场景**：
- 标准以太网环境
- 多机训练（如双机双卡）
- 网络环境简单的情况

**环境变量控制**：
- `NCCL_SOCKET_IFNAME` - 指定网络接口
- `NCCL_SOCKET_FAMILY` - 指定IP协议族（IPv4/IPv6）
- `NCCL_SOCKET_DSCP` - 设置DSCP值（我们的修改）

**DSCP设置代码位置**：`src/misc/socket.cc:122-172`

```c
static int envSocketDscp(void) {
  // 从环境变量读取DSCP值
  char* env = getenv("NCCL_SOCKET_DSCP");
  // ...
}

static ncclResult_t socketSetDscp(struct ncclSocket* sock) {
  // 设置IP_TOS/IPV6_TCLASS
  // ...
}
```

#### 2.3.2 InfiniBand传输

**文件**：`src/transport/net_ib.cc`

**特点**：
- ✅ **高性能**：低延迟、高带宽
- ✅ **RDMA支持**：远程直接内存访问
- ✅ **GPU Direct RDMA**：GPU内存直接访问

**实现方式**：
- 使用InfiniBand Verbs API (libibverbs)
- 支持RDMA操作
- 支持GPU Direct RDMA (GDR)

**适用场景**：
- 高性能计算集群
- 需要低延迟的场景
- 有InfiniBand硬件的环境

**环境变量控制**：
- `NCCL_IB_DISABLE=1` - 禁用InfiniBand
- `NCCL_IB_HCA` - 指定InfiniBand HCA设备
- `NCCL_IB_GID_INDEX` - 指定GID索引

**代码位置**：`src/transport/net_ib.cc:26-50`

---

### 2.4 COLLNET (Collective Network) 传输

**文件**：`src/transport/coll_net.cc`

**功能**：集合网络优化，用于特定的集合通信操作

**特点**：
- ✅ **集合通信优化**：针对AllReduce等操作优化
- ✅ **硬件加速**：利用网络硬件特性
- ✅ **降低延迟**：减少通信步骤

**适用场景**：
- 大规模集合通信
- 支持集合网络的硬件环境
- 需要极致性能的场景

**限制**：
- ❌ 需要硬件支持
- ❌ 主要用于特定的集合操作

---

## 三、传输方式选择机制

### 3.1 自动选择流程

NCCL会根据以下因素自动选择传输方式：

1. **拓扑信息**：GPU和网络拓扑
2. **硬件能力**：GPU是否支持P2P、是否有NVLink等
3. **网络配置**：可用的网络接口和类型
4. **优先级**：按P2P → SHM → NET → COLLNET的顺序尝试

**代码位置**：`src/transport.cc:20-40`

```c
template <int type>
static ncclResult_t selectTransport(...) {
  // 按优先级顺序尝试各种传输方式
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      // 找到可用的传输方式，使用它
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(...));
      return ncclSuccess;
    }
  }
  // 没有找到可用的传输方式
  return ncclSystemError;
}
```

### 3.2 选择逻辑

```
同一节点内GPU通信：
  1. 尝试 P2P（如果GPU支持NVLink/PCIe P2P）
  2. 如果P2P不可用，尝试 SHM
  3. 如果SHM不可用，回退到 NET

不同节点间通信：
  1. 尝试 NET（Socket或InfiniBand）
  2. 如果NET不可用，报错

集合通信优化：
  1. 如果支持COLLNET，优先使用
  2. 否则使用常规传输方式
```

---

## 四、多机多卡场景下的传输组合

### 4.1 双机双卡场景（您的环境）

**配置**：
- 服务器A：1张RTX 4000
- 服务器B：1张RTX 5000

**传输方式**：
- **节点内**：无（单卡节点）
- **节点间**：**NET (Socket/TCP/IP)**
  - 使用标准以太网
  - 通过TCP Socket通信
  - 支持DSCP QoS设置

**代码路径**：
```
训练脚本 → PyTorch DDP → NCCL API
    ↓
NCCL Enqueue层（记录统计、DSCP处理）
    ↓
NCCL Transport层（选择NET传输）
    ↓
Socket传输层（net_socket.cc）
    ↓
Socket层（socket.cc，设置DSCP）
    ↓
TCP/IP网络层
```

### 4.2 多机多卡场景（通用）

**典型配置**：
- 每个节点：4-8张GPU
- 节点数：2-100+

**传输方式组合**：

#### 节点内通信：
- **P2P**：GPU间直接通信（NVLink）
- **SHM**：进程间共享内存（备选）

#### 节点间通信：
- **NET (InfiniBand)**：高性能网络（优先）
- **NET (Socket)**：标准以太网（备选）

**示例**：
```
节点1 (4卡)          节点2 (4卡)
GPU0 ←P2P→ GPU1      GPU4 ←P2P→ GPU5
  ↓                    ↓
  └──NET(IB)──────────┘
  ↓                    ↓
GPU2 ←P2P→ GPU3      GPU6 ←P2P→ GPU7
```

---

## 五、传输方式性能对比

| 传输方式 | 带宽 | 延迟 | 适用场景 | 硬件要求 |
|---------|------|------|---------|---------|
| **P2P (NVLink)** | 极高 (600GB/s) | 极低 (<1μs) | 单机多卡 | NVLink |
| **P2P (PCIe)** | 高 (32GB/s) | 低 (~5μs) | 单机多卡 | PCIe |
| **SHM** | 高 (内存带宽) | 低 (~10μs) | 单机多进程 | 共享内存 |
| **NET (IB)** | 高 (200Gbps) | 低 (~1μs) | 多机高性能 | InfiniBand |
| **NET (Socket)** | 中 (10-100Gbps) | 中 (~50μs) | 多机通用 | 以太网 |
| **COLLNET** | 极高 | 极低 | 集合通信 | 专用硬件 |

---

## 六、环境变量控制

### 6.1 禁用特定传输方式

```bash
# 禁用P2P
export NCCL_P2P_DISABLE=1

# 禁用SHM
export NCCL_SHM_DISABLE=1

# 禁用InfiniBand（强制使用Socket）
export NCCL_IB_DISABLE=1
```

### 6.2 网络配置

```bash
# Socket传输配置
export NCCL_SOCKET_IFNAME=eth0        # 指定网络接口
export NCCL_SOCKET_FAMILY=AF_INET    # IPv4
export NCCL_SOCKET_DSCP=26           # 设置DSCP值（我们的功能）

# InfiniBand配置
export NCCL_IB_HCA=mlx5_0            # 指定HCA设备
export NCCL_IB_GID_INDEX=0           # GID索引
```

### 6.3 调试和日志

```bash
# 启用NCCL调试日志
export NCCL_DEBUG=INFO

# 查看传输方式选择
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
```

---

## 七、代码位置速查

| 功能 | 文件 | 行号/说明 |
|------|------|----------|
| 传输方式定义 | `include/transport.h` | 15-19 |
| 传输方式选择 | `transport.cc` | 20-40 |
| P2P实现 | `transport/p2p.cc` | 完整文件 |
| SHM实现 | `transport/shm.cc` | 完整文件 |
| Socket传输 | `transport/net_socket.cc` | 完整文件 |
| InfiniBand传输 | `transport/net_ib.cc` | 完整文件 |
| COLLNET实现 | `transport/coll_net.cc` | 完整文件 |
| Socket DSCP设置 | `misc/socket.cc` | 122-172 |

---

## 八、总结

### 8.1 传输方式优先级

```
单机多卡：
  P2P (NVLink) > P2P (PCIe) > SHM > NET

多机多卡：
  节点内：P2P > SHM
  节点间：NET (IB) > NET (Socket)
```

### 8.2 您的双机双卡环境

- **传输方式**：NET (Socket/TCP/IP)
- **DSCP支持**：✅ 已实现（我们的修改）
- **性能特点**：中等带宽，中等延迟
- **适用性**：标准以太网环境，通用性强

### 8.3 性能优化建议

1. **单机多卡**：确保P2P和SHM可用
2. **多机通信**：
   - 优先使用InfiniBand（如果可用）
   - 使用DSCP QoS优化（我们的功能）
   - 选择合适的网络接口
3. **大规模训练**：考虑使用COLLNET（如果硬件支持）

---

**文档版本**：v1.0  
**最后更新**：2024年  
**维护者**：longliu8
