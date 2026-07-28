# DSCP NCCL 适配器端到端测试记录

## 1. 问题背景

### 1.1 目标

实现并验证 DSCP (Differentiated Services Code Point) NCCL 适配器的双节点分布式通信能力。DSCP 适配器是 NCCL 的一个扩展模块，通过在通信链路上设置 DSCP 标记，实现对 RDMA 流量的 QoS 控制。

### 1.2 测试环境

| 属性 | 节点 1 (Master) | 节点 2 |
|------|----------------|--------|
| 主机名 | guolab-226 | guolab-10 |
| 简称 | 226 | 10.1 |
| RDMA IP | 192.10.10.226 | 192.10.10.110 |
| 管理网 IP | 192.10.10.107 | 192.10.10.110 |

### 1.3 网络拓扑

两台服务器通过交换机在 `192.10.10.0/24` 网段实现 RDMA 互联。RDMA 网络与管理网络在逻辑上隔离，测试中必须使用 RDMA 网段 IP 进行通信。

---

## 2. 关键问题与修复

### 问题 1：RDMA IP 配置错误

**现象**：NCCL 初始化阶段 `ibv_modify_qp` 失败，通信无法建立。

**根因**：测试时错误地使用了管理网 IP（例如 `192.10.10.107`）而非 RDMA 网段 IP。

**纠正方案**：

- 226 的 RDMA IP 为 `192.10.10.226`
- 10.1 的 RDMA IP 为 `192.10.10.110`

**教训**：必须通过以下命令确认正确的 RDMA 端口和 IP 地址：

```bash
# 查看 RDMA 网卡与网络设备的映射关系
ibdev2netdev -v

# 查看网卡绑定的 IP 地址
ip addr show
```

### 问题 2：226 节点 LD_PRELOAD 路径错误

**现象**：`NCCL_DEBUG=INFO` 日志显示加载的 NCCL 版本为 2.29.7（系统默认版本），而非预期支持 DSCP 的 2.18.3。

**根因**：226 节点上 PyTorch 安装在 Python 3.10 环境中，但 `LD_PRELOAD` 环境变量指向了 `~/.local/lib/python3.8/site-packages/nvidia/nccl/lib/libnccl.so.2`，而该路径在 226 上并不存在。因此 PyTorch 回退加载了系统默认的 NCCL 2.29.7，该版本不包含 `ncclDscpAdapter` 相关符号。

**修复方案**：将 `LD_PRELOAD` 路径修正为正确的 Python 3.10 路径：

```bash
export LD_PRELOAD=/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
```

**验证方法**：通过以下 Python 代码检查实际加载的 NCCL 版本：

```bash
python3 -c "
import ctypes
n = ctypes.CDLL('libnccl.so.2')
v = ctypes.c_int()
n.ncclGetVersion(ctypes.byref(v))
print(f'NCCL version: {v.value}')
"
```

版本号对照：

| 版本号 | 说明 |
|--------|------|
| 21803 | NCCL 2.18.3（DSCP 适配版） |
| 23007 | NCCL 2.30.7（系统默认版，不含 DSCP 支持） |

---

## 3. 正确的运行命令

### 3.1 环境变量说明

```bash
# RDMA 网卡选择：指定使用 mlx5_0 网卡
NCCL_IB_HCA=mlx5_0

# RoCE v2 GID 索引：index 3 对应 192.10.10.x 网段的 RoCE v2 类型 GID
NCCL_IB_GID_INDEX=3

# DSCP 适配器启用开关
NCCL_DSCP_ADAPTER_ENABLED=1

# DSCP SLO (Service Level Objective) 阈值，单位为毫秒
NCCL_DSCP_SLO_THRESHOLD=1.5

# LD_PRELOAD：DSCP 版 NCCL 库路径
# 注意：路径因 Python 版本而异
#
# 226 节点 (Python 3.10):
#   LD_PRELOAD=/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
#
# 10.1 节点 (Python 3.8):
#   LD_PRELOAD=/home/why/.local/lib/python3.8/site-packages/nvidia/nccl/lib/libnccl.so.2
```

### 3.2 步骤 1：启动 rank 0（226，作为 Master）

```bash
ssh -o StrictHostKeyChecking=no why@192.10.10.226 \
  'MASTER_ADDR=192.10.10.226 MASTER_PORT=29501 RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 \
   NCCL_DEBUG=INFO NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
   LD_PRELOAD=/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2 \
   NCCL_DSCP_ADAPTER_ENABLED=1 NCCL_DSCP_SLO_THRESHOLD=1.5 \
   nohup timeout 60 python3 -u /home/why/LongLiu_rebuild/testbed/dscp_debug.py \
   > /tmp/dscp_debug_226.log 2>&1 &'
```

### 3.3 步骤 2：启动 rank 1（10.1）

```bash
MASTER_ADDR=192.10.10.226 MASTER_PORT=29501 RANK=1 LOCAL_RANK=0 WORLD_SIZE=2 \
NCCL_DEBUG=INFO NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
NCCL_DSCP_ADAPTER_ENABLED=1 NCCL_DSCP_SLO_THRESHOLD=1.5 \
timeout 60 python3 -u /home/why/LongLiu_rebuild/testbed/dscp_debug.py
```

### 3.4 步骤 3：运行 DSCP quota_bench 端到端测试

rank 0（226 节点）命令：

```bash
ssh -o StrictHostKeyChecking=no why@192.10.10.226 \
  'MASTER_ADDR=192.10.10.226 MASTER_PORT=29502 RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 \
   NCCL_DEBUG=INFO NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
   LD_PRELOAD=/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2 \
   NCCL_DSCP_ADAPTER_ENABLED=1 NCCL_DSCP_SLO_THRESHOLD=1.5 \
   nohup timeout 120 python3 -u /home/why/LongLiu_rebuild/testbed/quota_bench.py \
   > /tmp/quota_bench_dscp_226.log 2>&1 &'
```

rank 1（10.1 节点）命令：

```bash
MASTER_ADDR=192.10.10.226 MASTER_PORT=29502 RANK=1 LOCAL_RANK=0 WORLD_SIZE=2 \
NCCL_DEBUG=INFO NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 \
NCCL_DSCP_ADAPTER_ENABLED=1 NCCL_DSCP_SLO_THRESHOLD=1.5 \
timeout 120 python3 -u /home/why/LongLiu_rebuild/testbed/quota_bench.py
```

---

## 4. 验证方法

### 4.1 检查 RDMA 网络

```bash
# 查看 RDMA 网卡与系统网络设备的对应关系
ibdev2netdev -v

# 查看 GID 表，确认 index 3 对应 RoCE v2 类型
for i in 0 1 2 3; do
  echo "GID $i: type=$(cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/$i) val=$(cat /sys/class/infiniband/mlx5_0/ports/1/gids/$i)"
done

# ping 测试 RDMA 网络延迟
# 从 10.1 ping 226：
ping -c 2 192.10.10.226

# 从 226 ping 10.1：
ping -c 2 192.10.10.110
```

### 4.2 检查 DSCP 库

```bash
# 验证 DSCP 符号是否存在于 NCCL 库中
nm -D /path/to/libnccl.so.2 | grep ncclDscpAdapter

# 验证 NCCL 版本号
python3 -c "
import ctypes
n = ctypes.CDLL('libnccl.so.2')
v = ctypes.c_int()
n.ncclGetVersion(ctypes.byref(v))
print(f'NCCL version: {v.value}')
"
# 结果对照：
#   21803 -> NCCL 2.18.3 (DSCP 适配版)
#   23007 -> NCCL 2.30.7 (系统默认版)
```

### 4.3 检查实际加载的 NCCL 库

通过 `/proc/self/maps` 查看 PyTorch 运行时实际加载的 NCCL 共享库路径：

```bash
python3 -c "
import os
import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group('nccl', init_method='tcp://127.0.0.1:12345', rank=0, world_size=1)

with open(f'/proc/{os.getpid()}/maps') as f:
    for line in f:
        if 'nccl' in line.lower() and 'libnccl' in line:
            print(line.strip())

dist.destroy_process_group()
"
```

---

## 5. 测试结果

### 5.1 dscp_debug.py -- 基础通信测试

- **测试内容**：双节点 `all_reduce` 操作
- **结果**：两端均输出 `all_reduce ok`，通信链路正常建立
- **状态**：通过

### 5.2 quota_bench.py -- DSCP 配额性能测试

- **测试内容**：在 DSCP 适配器启用状态下运行 200 次 all_reduce 迭代，统计延迟分布
- **结果**：

| 指标 | 数值 |
|------|------|
| 平均延迟 (avg) | 22.55 ms |
| P95 延迟 (p95) | 23.29 ms |
| 采样数 (n) | 200 |

- **状态**：通过（两端结果一致）

---

## 6. 关键技巧总结

### 6.1 网络隔离

RDMA 网段和管理网段是相互隔离的。所有通信测试必须使用 RDMA 网段的 IP 地址（`192.10.10.x`）进行，不能使用管理网 IP。

### 6.2 SSH 连接

使用 RDMA 网段 IP 进行 SSH 连接（确保连通性测试准确）：

```bash
ssh why@192.10.10.226   # 而非管理网 IP
```

### 6.3 LD_PRELOAD 路径

`LD_PRELOAD` 路径必须与目标机器的 Python 版本严格匹配。可通过以下命令确认 Python 的 site-packages 路径：

```bash
python3 -c "import sys; print(sys.path)"
```

### 6.4 GID 索引选择

`NCCL_IB_GID_INDEX=3` 对应 RoCE v2 类型的 GID，用于 RoCE v2 通信。不同环境下索引值可能不同，务必通过 `cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/<index>` 确认。

### 6.5 NCCL 调试日志

`NCCL_DEBUG=INFO` 可查看 NCCL 初始化过程的详细信息，包括：

- 加载的 NCCL 库版本
- 使用的 RDMA 网卡
- GID 索引选择
- 端到端连接建立过程

### 6.6 NCCL 版本一致性

双机 NCCL 版本必须一致。版本不一致会导致 `ibv_modify_qp` 操作失败，进而导致通信无法建立。
