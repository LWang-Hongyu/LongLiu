# 实验1：主动重校准探针物理验证 — 配置文档

> 目的：在真实物理网卡环境中验证"主动重校准探针"功能——当系统持续拥塞导致锚点（T_target/EMA 带宽）失效时，通过一次 P6 高优先级单次 AllReduce 探测，主动获取无拥塞带宽样本，用于重校准锚点。

## 1. 实验环境

| 项 | 配置 |
|----|------|
| 节点 | guolab-10（master, RTX 4000, ConnectX-6 Dx 50G）+ guolab-226（RTX 5000, BlueField-3 100G） |
| 网络 | RoCEv2, NCCL_IB_HCA=mlx5_0, GID_INDEX=3, 有效链路 50Gbps |
| 交换机 | SN2700, Cumulus Linux 5.1.0（SP 队列，tc:0 > tc:1 > tc:2 …） |
| 调度实现 | multi_comm_slo（7 个预创建 communicator，P6→DSCP=8→tc:0, P3→DSCP=16→tc:2） |

## 2. 实验步骤

1. **校准（solo）**：无背景流，作业以 P3 运行 8 个 epoch，学习 T_target 与 solo 带宽基线（`ttarget.json`）。
2. **背景流打满链路**：双向各 12 路 iperf3 UDP（DSCP=P3/TOS=64，与作业同队列 tc:2），每方向 48Gbps 提供 → 50G 链路打满，作业自身带宽样本被污染（锚点失效场景）。
3. **主阶段**：作业以 **P3 初始 + 锚点冻结**（`preset_target=True`，T_target 不更新；`max_priority=3`，优先级封顶 P3，模拟失效锚点无法自救）。
4. **P6 探测**：每 3 个 epoch，显式切换到 P6 communicator 执行**单次 AllReduce**，测量带宽后切回 P3。
5. **NIC 硬件计数器**：全程 1s 采样 10.1 + 226 的 ethtool prio 队列字节计数、RoCE hw_counters、IB port 计数器、IRQ 计数（`nic_*.csv`）。

## 3. 验证指标与判定

| 指标 | 定义 | 判定标准 |
|------|------|----------|
| 无拥塞样本比例 | 探测带宽 ≥ 0.9×solo 带宽的探测占比 | P6 探测在 SP 队列下不受 P3 背景流影响 → 比例应接近 100% |
| EMA 更新准确性 | 探测样本 EMA 追踪 solo 带宽的偏差 | 分析脚本用 EMA(α=0.3) 模拟更新，稳态偏差 < 5% |
| 探测前后性能变化 | 探测前后 epoch 平均通信时间变化 | 探测为单次操作，对作业扰动 < 探测自身开销 |
| 探测走队列验证 | NIC prio 计数器增量与探测带宽比对 | 探测期间 prio1（P6）计数器增量 ≈ 探测流量 |

## 4. 关键参数（可复现）

| 参数 | 值 |
|------|-----|
| payload | 1024 MB（float32, AllReduce） |
| sleep（模拟计算） | 30 ms/iter |
| iters_per_epoch | 20 |
| num_epochs | 15 |
| probe_every | 3（epoch 5/8/11/14 边界探测，共 4 次） |
| initial_priority / max_priority | 3 / 3（冻结） |
| probe_priority | 6 |
| slo_threshold（π 计算用） | 1.5 |
| EMA α（分析阶段） | 0.3 |
| 背景流 | 12×4G 双向 iperf3 UDP, TOS=64, 300s |
| NCCL | ALGO=RING, PROTO=SIMPLE |

## 5. 运行与数据

```bash
# 1) 同步脚本到 226
bash ../00_common/sync_to_226.sh
# 2) 运行（重复 3 轮）
bash scripts/run_exp1.sh 1
bash scripts/run_exp1.sh 2
bash scripts/run_exp1.sh 3
# 3) 分析
python3 scripts/analyze_exp1.py
```

数据落盘：`data/exp1_r<round>_<ts>/`：
- `exp1_jobA_rank0_probe.csv` — 探测样本（时间戳/带宽/是否无拥塞）
- `exp1_jobA_rank0_iter.csv` / `_epoch.csv` — 探测前后带宽变化曲线
- `nic_10.csv` / `nic_226.csv` — NIC 硬件计数器（原始值，分析脚本算增量）
- `gpu_10.csv` / `gpu_226.csv` — GPU 状态
- `ttarget.json` — 校准锚点与 solo 带宽
- `env/` — 环境核实记录

## 6. 已知约束

- 226 NIC 不做 DSCP→prio 分类（全走 prio0），优先级实施下沉到交换机首跳，不影响 P6 探测结论（瓶颈在 10.1 侧 50G 链路）。
- 探测为同步单次操作，期间作业暂停；扰动计入"系统响应时间"指标。
