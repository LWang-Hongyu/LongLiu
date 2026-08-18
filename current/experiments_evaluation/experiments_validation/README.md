# LongLiu 物理测试床补充验证实验（Experiments Validation）

> 创建日期：2026-08-07
> 对应审稿意见的 4 项补充验证，统一在物理测试床（2 节点）上执行。
> 所有实验遵循统一目录规范：`config/`（配置文档）与 `data/`（原始数据）分开存储，`analysis/`（分析报告与图表）独立于数据。

## 目录结构

```
experiments_validation/
├── README.md                        # 本文件
├── 00_common/                       # 共享工具（环境核实、硬件监控、同步）
│   ├── env_check.sh                 # 环境核实（版本、链路、QoS 配置）
│   ├── monitor_nic.sh               # NIC 硬件计数器监控（ethtool + RoCE hw_counters + IRQ）
│   ├── monitor_gpu.sh               # GPU 监控（nvidia-smi dmon：温度/功耗/时钟/利用率）
│   └── sync_to_226.sh               # 同步实验脚本到 226 节点
├── 01_exp_probe_recalib/            # 实验1：主动重校准探针物理验证
│   ├── config/                      # 实验配置文档
│   ├── scripts/                     # 运行/分析脚本
│   ├── data/                        # 原始数据（NIC 计数器 CSV、探测日志、epoch CSV）
│   └── analysis/                    # 分析报告与图表
├── 02_exp_hw_quantization/          # 实验2：硬件量化误差与 P6 抢占微观验证
│   ├── config/ scripts/ data/ analysis/
├── 03_exp_anomaly_forensic/         # 实验3：图11异常点深度分析（Forensic）
│   ├── config/ scripts/ data/ analysis/
└── 04_exp_comm_primitives/          # 实验4：通信原语多样性验证（AllGather）
    ├── config/ scripts/ data/ analysis/
```

## 测试床环境速查

| 项 | guolab-10 (master) | guolab-226 |
|----|--------------------|------------|
| GPU | Quadro RTX 4000 ×1 | Quadro RTX 5000 ×2 |
| NIC | ConnectX-6 Dx (mlx5_0) | BlueField-3 B3220 (mlx5_0) |
| RDMA IP | 192.10.10.110 (enp130s0f0np0) | 192.10.10.226 (enp59s0f0np0) |
| 链路 | 50GbE | 100GbE（有效带宽受限于 50G 侧） |
| 交换机 | Mellanox Spectrum SN2700, Cumulus Linux 5.1.0 | — |

## 通用执行流程

1. `bash 00_common/env_check.sh`：核实环境一致性（每次实验前运行，输出入 data/env/）
2. `bash 00_common/sync_to_226.sh`：推送最新脚本与 libmulti_comm.so 到 226
3. 按各实验 `scripts/` 下的运行脚本执行（内部会自动启动 NIC/GPU 监控）
4. 结果统一落盘至对应 `data/` 与 `analysis/`

## 关键实现约定

- 调度核心：`current/multi_comm_slo/src/slo_scheduler.py`（SLOScheduler）+ C 库 `multi_comm.c`
- 优先级→DSCP 映射：P6→8, P4→0, P3→16, P2→24, P1→32, P0→40（tc:0 > tc:1 > tc:2 …）
- 226 节点使用旧式扁平路径 `/home/why/LongLiu_rebuild/multi_comm_slo/src`（无 `current/` 前缀）
- 脚本中通过环境变量 `MULTI_COMM_SRC` 指定调度器源码路径，双端按各自布局设置
