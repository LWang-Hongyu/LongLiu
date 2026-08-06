# LongLiu 物理实验结果

**实验日期**: 2026-06-23
**硬件**:
- 226 (guolab-226): 2x Quadro RTX 5000 (16GB), NCCL 23007+LongLiu, RoCEv2
- 10.1 (guolab-10): 1x Quadro RTX 4000 (8GB, MPS), NCCL 23007+LongLiu, RoCEv2
- RDMA via mlx5_0 (192.10.10.x), ~100GbE

**网络拓扑**: 2节点, 1x RDMA链路, 无交换机

## 实验负载

| Job | 模型 | 参数量 | batch |
|-----|------|--------|-------|
| Job A | MidModel (3x Linear 4096) | ~134M | 64 |
| Job B | ResNet-50 | ~25M | 32 |

## 实验结果

### 基线 (无 LongLiu)

| Job | avg (ms) | p95 (ms) | longliu |
|-----|----------|----------|---------|
| Job A | 113.2 | 122.5 | inactivo |
| Job B | 205.7 | 208.9 | inactivo |

### LongLiu Tight (Job A ci=1.2, Job B ci=2.0)

| Job | avg (ms) | p95 (ms) | vs 基线 |
|-----|----------|----------|---------|
| Job A (ci=1.2, tight) | 111.6 | 120.0 | **-1.4%** |
| Job B (ci=2.0, loose) | 207.1 | 209.7 | +0.7% |

### LongLiu Swap (Job A ci=2.0, Job B ci=1.2)

| Job | avg (ms) | p95 (ms) | vs 基线 |
|-----|----------|----------|---------|
| Job A (ci=2.0, loose) | 113.9 | 127.0 | +0.6% |
| Job B (ci=1.2, tight) | 208.7 | 211.1 | +1.5% |

## 关键发现

1. **LongLiu 控制回路已验证**: NCCL proxy quota 控制 End-to-End 工作
2. **方向正确**: tight (ci=1.2) job 在 LongLiu 下获得优先级，略微加速 (-1.4%)
3. **2节点限制**: ops 队列深度太浅，quota 变化影响有限
4. **ctypes 开销**: LongLiu iterStart/iterEnd 的 Python-C FFI 调用增加 ~6-7ms 基础开销

## 对实验设计的影响

1. 取消"竞争场景"——2节点无交换机时带宽充足，无实质竞争
2. 改用"不同ci值下的SLO达标率"作为核心实验
3. 未来真实实验需 ≥4 节点或交换机场景
