# 实验3：图11异常点深度分析（Forensic）— 配置文档

> 目的：审稿人指出 rep2 r1 轮次中 LongLiu 与 CRUX 置信区间重叠仅归咎于"硬件噪声"的解释不足。
> 本实验重跑 rep2 r1，配合 BlueField-3 DPU 硬件计数器与 GPU 监控，对异常点做底层原因定位。

## 1. 异常点背景（图11 / fig6）

| Round | 方向 | LL mean | CRUX mean | 差幅 |
|-------|------|---------|-----------|------|
| orig_r1 | LL→CX | 1.1075 | 1.3022 | −15.0% |
| rep2_r1 | LL→CX | **1.2496** | 1.2541 | −0.4%（**重叠**） |
| rep2_r2 | CX→LL | 1.1123 | 1.3365 | −16.8% |

rep2_r1 中 LL 紧作业 slowdown 从 e7 起升至 1.25~1.29（正常轮次约 1.10~1.11），需定位底层原因。

## 2. 重跑条件（与 run_meta_round1.txt 一致）

| 参数 | 值 |
|------|-----|
| 顺序 | Round 1: LL→CX（LL 先跑） |
| 背景流 | 12×500M iperf3 UDP, DSCP=P3 (TOS=64), 6 Gbps |
| Warmup | 5 min（背景流运行） |
| payload | 1024 MB, sleep 30 ms, 300 iters（20/epoch, 15 epochs） |
| c_i | tight=1.2 / loose=3.0, epoch 7 翻转 |
| T_target | A=4201.087ms（ttarget_v5_jobA）, B=3905.163ms（ttarget_v5_jobB） |
| LongLiu | initial-priority 3，无 max cap（π>0.3 → P6） |
| CRUX | 双 job 静态 P3 |
| 脚本 | p4_job_reverse_ts.py（原脚本 hash ca6c271a + 仅加时间戳，行为不变） |

## 3. Forensic 监控（BlueField-3 DPU + 双端）

| 数据 | 来源 | 用途 |
|------|------|------|
| RoCE 重传率 | 226 `/sys/class/infiniband/mlx5_0/ports/1/hw_counters/roce_adp_retrans` | 网络重传异常 |
| out_of_buffer / RNR NAK | 同上 hw_counters | PFC/队列溢出下游证据 |
| PCIe 吞吐 | 226 IB port_xmit_data+port_rcv_data 速率 | BF-3 PCIe 吞吐变化（宿主流量经 PCIe） |
| 网卡中断率 | 双端 `/proc/interrupts` mlx5 计数 | 中断风暴 |
| GPU 降频日志 | `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_throttle_reasons.*` | thermal/power throttling |
| epoch 墙钟 | 作业日志时间戳（HH:MM:SS）+ run_start.epoch | 精确对齐异常时间戳 |

## 4. 分析方法（analyze_forensic.py）

1. 从 epoch CSV 定位 LL 紧作业 slowdown > 1.20 的异常 epoch
2. 从带时间戳日志建立 epoch → 墙钟映射，对齐硬件计数器
3. 异常窗 vs 基线窗（epoch 0-6）对比：SM 时钟、温度、RoCE 重传、OOB、IRQ
4. 检查 LL 优先级轨迹（是否到达 P6）与硬件异常的相关性
5. 输出时间线图 + 判定结论（thermal throttling / PFC 效应 / 其他）

## 5. 运行与数据

```bash
bash ../00_common/sync_to_226.sh
bash scripts/rerun_rep2_r1.sh          # ~10 分钟（5min warmup + 2 mode）
python3 scripts/analyze_forensic.py
```

数据落盘：`data/exp3_rerun_<ts>/`：
- `exp3_job[AB]_{longliu,crux}_rank0_epoch.csv` — per-epoch slowdown/priority/π
- `exp3_job[AB]_{longliu,crux}_node101.log` — 带时间戳日志
- `run_start.epoch` — 时间对齐锚点
- `nic_10.csv` / `nic_226.csv` — 硬件计数器（原始）
- `gpu_10.csv` / `gpu_226.csv` — GPU 状态与降频原因

## 6. 已知约束

- 交换机 SN2700 PFC 计数器需交换机凭据（本环境无免密），PFC 触发以 NIC 侧
  out_of_buffer/RNR NAK/重传增量作为下游证据，必要时人工登录交换机核实。
- 226 BF-3 为 VPI 模式 PF（enp59s0f0np0），PCIe 吞吐以 IB port 计数器近似。
