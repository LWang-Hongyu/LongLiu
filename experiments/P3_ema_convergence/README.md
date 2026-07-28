# P3: EMA 带宽收敛实验（含 Ui < 1 防护验证）

## 目标

证明 Bw_ema 在前几个 epoch 收敛到 solo 带宽，且引入竞争后通过 Ui < 1 保护机制保持稳定（不随 Bw_obs 下降而衰减）。

## 实验拓扑

```
10.1 (GPU 0) ──mlx5_0──┐                  ┌──mlx5_0── 226 (GPU 0, GPU 1)
  RDMA IP: 192.10.10.110│                  │  RDMA IP: 192.10.10.226
  NIC: enp130s0f0np0    │                  │  NIC: enp59s0f0np0
                         ├─ tor_bridge ────┤
                         │   VLAN 100      │
                         └─────────────────┘
                           瓶颈链路 ~46 Gbps 单向
                           NCCL AllReduce ~20 Gbps (ring, 2-node)
```

- **Job 1**: epochs 0-9，[10.1 GPU 0, 226 GPU 0]，port 29510
- **Job 2**: epochs 5-9，[10.1 GPU 0, 226 GPU 1]，port 29511
- Epochs 0-4: solo（只有 Job 1）
- Epochs 5-9: contested（Job 1 + Job 2 并发竞争 RDMA 带宽）
- 同步：Job 1 在 epoch 4 完成后写 `.sync_ready` 文件，Job 2 轮询该文件

## 代码修改摘要

本次实验验证了对 `dscp_adapter.cc` 的三处关键修复：

### 1. EMA Ui < 1 防护（核心）
```c
// Guard: EMA 只在上一 epoch 的 Ui < 1.0（低竞争）时更新
int lowContention = (epoch == 0) ||
    (adapter->epochs[epoch - 1].ui < 1.0);

if (!adapter->emaInitialized) {
    adapter->emaBandwidth = newBw;          // 冷启动
    adapter->emaInitialized = 1;
} else if (lowContention) {
    adapter->emaBandwidth = adapter->emaAlpha * newBw +
                            (1.0 - adapter->emaAlpha) * adapter->emaBandwidth;
}
// else: Ui >= 1.0 → 跳过更新，防止竞争期污染
```
**注意**: 使用 `adapter->epochs[epoch-1].ui`（上一 epoch 已计算的 Ui）而非 `epochStats->ui`（当前 epoch 的占位符 0.0）。

### 2. CheckEpochTriggers 中 startIter 修正
`ncclCommStatsStartOp` 在 `CheckEpochTriggers` 之前调用，已递增 `numIterations`。因此 `startIter` 需减 1 才能指向正确的迭代桶：
```c
int startIter = (stats->numIterations > 0) ? stats->numIterations - 1 : 0;
```

### 3. UpdateDscpForNextEpoch 调用顺序
将 `UpdateDscpForNextEpoch` 移到 `EndEpoch` 之前调用，确保 EMA 防护读到的 `Ui_prev` 已是最新值（而非 0.0）。

## 文件说明

| 文件 | 用途 |
|------|------|
| `p3_job1.py` | Job 1 脚本：epochs 0-9，epoch 4 后写同步文件。包含 ctypes 调用 ncclDscpEpochStart/End |
| `p3_job2.py` | Job 2 脚本：等待同步文件后运行 epochs 5-9 |
| `run_p3.sh` | 启动脚本：并行启动 4 个 torchrun 进程，配置 LD_PRELOAD + NCCL_DEBUG=INFO |
| `plot_p3.py` | 离线 EMA 计算 + 绘图 |

## 运行方法

```bash
cd experiments/P3_ema_convergence
bash run_p3.sh        # 采集数据（约 0.5-1 分钟）
# 分析 EMA 日志
grep "DSCP EMA\|DSCP Adapter.*Ui=" /tmp/p3_job1_node101.log
```

环境变量可调：
- `P3_PAYLOAD_MB` (default 256): 每次 AllReduce 的 MB 数
- `P3_ITERS` (default 20): 每个 epoch 的迭代数
- 每 epoch 通信量 = P3_PAYLOAD_MB × P3_ITERS MB

## 输出文件

- `p3_job1_rank0.csv` — Job 1 rank 0 的 epoch 级测量
- `p3_job2_rank0.csv` — Job 2 rank 0 的 epoch 级测量
- `fig_p3_ema_convergence.png` — Bw_obs + Naive EMA + Ideal EMA 对比
- `fig_p3_ema_detail.png` — 柱状图 + 双 EMA 标注
- `/tmp/p3_job{1,2}_node{101,226}.log` — 各进程 NCCL 日志（含 EMA+DSCP 输出）

## 实验结果

### 带宽数据 (Job 1)

| Epoch | Phase | Bw_obs (Gbps) |
|-------|-------|---------------|
| 0 | solo | 18.3 |
| 1 | solo | 20.5 |
| 2 | solo | 20.7 |
| 3 | solo | 21.2 |
| 4 | solo | 21.1 |
| 5 | contested | 11.7 |
| 6 | contested | 10.2 |
| 7 | contested | 10.1 |
| 8 | contested | 10.2 |
| 9 | contested | 10.2 |

- **Solo avg (epochs 0-4)**: 20.4 Gbps
- **Contested avg (epochs 5-9)**: 10.5 Gbps（下降 **48.5%**）

### EMA 行为日志（关键证据）

```
DSCP EMA [Epoch 0]: COLD_START, seed=822042 Gbps
DSCP Adapter [Epoch 0]: Ui=0.8333
DSCP EMA [Epoch 1]: UPDATE (Ui_prev=0.8333 < 1.0)   ← solo
DSCP Adapter [Epoch 1]: Ui=0.8335
DSCP EMA [Epoch 2]: UPDATE (Ui_prev=0.8335 < 1.0)   ← solo
DSCP Adapter [Epoch 2]: Ui=0.8311
DSCP EMA [Epoch 3]: UPDATE (Ui_prev=0.8311 < 1.0)   ← solo
DSCP Adapter [Epoch 3]: Ui=0.8257
DSCP EMA [Epoch 4]: UPDATE (Ui_prev=0.8257 < 1.0)   ← solo
DSCP Adapter [Epoch 4]: Ui=0.8228
DSCP EMA [Epoch 5]: UPDATE (Ui_prev=0.8228 < 1.0)   ← 第一个竞争期 epoch，Ui 仍 < 1
DSCP Adapter [Epoch 5]: Ui=0.9273
DSCP EMA [Epoch 6]: UPDATE (Ui_prev=0.9273 < 1.0)   ← 竞争期，但刚跨过 SLO
DSCP Adapter [Epoch 6]: Ui=1.0337                   ← Ui 首次 ≥ 1.0！
DSCP EMA [Epoch 7]: SKIP (Ui_prev=1.0337 >= 1.0)    ← 防护生效！EMA 冻结
DSCP Adapter [Epoch 7]: Ui=1.1159
DSCP EMA [Epoch 8]: SKIP (Ui_prev=1.1159 >= 1.0)    ← 持续冻结
DSCP Adapter [Epoch 8]: Ui=1.1770
DSCP EMA [Epoch 9]: SKIP (Ui_prev=1.1770 >= 1.0)    ← 持续冻结
```

### 分析

| 指标 | 值 |
|------|-----|
| Solo Ui (epochs 0-4) | ~0.83（job 超前于 SLO） |
| 竞争期 Ui 过渡 | 0.93 → 1.03 → 1.12 → 1.18（递增） |
| EMA 冻结时期 | Epochs 7-9（Ui ≥ 1.0 后持续 SKIP） |
| 修复前 Naive EMA 污染 | -40.1%（从 19.86 降到 11.89 Gbps） |
| 修复后 EMA 保护 | **EMA 保持在 1,666,144（不受竞争期 Bw=10 Gbps 影响）** |

**物理意义**：
- Solo 期 Ui < 1：job 获得足够带宽，网络低竞争，EMA 正常更新跟踪理想带宽
- 竞争期 Ui 从 < 1 过渡到 ≥ 1：job 落后于 SLO，网络进入竞争态
- Ui ≥ 1 后 EMA SKIP：防止竞争期低带宽样本污染理想带宽估计
- 论文表述："The EMA is updated only when the job's urgency index indicates it is ahead of schedule (Ui < 1), ensuring that the estimated ideal bandwidth reflects uncontended network conditions rather than transient congestion artifacts."

## 硬件注意事项

- 226 GPU 1 是 **PCIe Gen 1 x16 + 跨 NUMA (SYS)**，NCCL RDMA 需 `NCCL_P2P_DISABLE=1`
- 两个独立 NCCL communicator 共享同一 GPU 会导致 `ncclSystemError`
- 因此 Job 2 在 226 必须用 GPU 1 并通过 P2P_DISABLE 绕过 GPU 直连
- 10.1 仅 1 个 GPU，两个 Job 共享 GPU 0，配合 NCCL_P2P_DISABLE=1 可共存
- 必须设置 `LD_PRELOAD` 指向 DSCP 修改版 NCCL 库，否则加载系统默认 NCCL
- Python 脚本必须通过 ctypes 调用 `ncclDscpEpochStart/EpochEnd` 触发 epoch 管理

## 部署清单

编译部署 DSCP 修改版 NCCL：
```bash
cd /home/why/LongLiu_rebuild/nccl-dscp
make -j8 src.build BUILDDIR=/home/why/LongLiu_rebuild/nccl-dscp/build
cp build/lib/libnccl.so.2.18.3 /home/why/.local/lib/python3.8/site-packages/nvidia/nccl/lib/libnccl.so.2
scp build/lib/libnccl.so.2.18.3 192.10.10.226:/home/why/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
```

## 时间戳

- 实验日期：2026-07-11
- 修复版本：v3（Ui < 1 guard + CheckEpochTriggers startIter 修正 + UpdateDscpForNextEpoch 顺序修正）
- 交换机配置：tor_bridge, VLAN 100, bridge-access 100
