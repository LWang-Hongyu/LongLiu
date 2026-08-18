# Exp4 配置文档 — 通信原语多样性验证（AllGather 替换 AllReduce）

> 目标：将 LongLiu 调度所驱动的通信原语从 AllReduce 替换为 AllGather，
> 保持核心参数不变，多轮次验证：
>   1. **DSCP 切换准确性**：调度器随 π 更新切换 priority → DSCP 映射的正确性；
>   2. **锚点测量精度**：solo 校准学到的 T_target / solo_bw 在竞争场景下的可复现性。

## 1. 实验链路与拓扑

```
guolab-10 (rank0, RTX 4000, ConnectX-6 Dx, enp130s0f0np0 = 192.10.10.110, 50G)
      │  RoCEv2 直连
guolab-226 (rank1, RTX 5000, BlueField-3 B3220, enp59s0f0np0 = 192.10.10.226, 100G)
```

- 链路速率：50G（受 10 端 ConnectX-6 限制）
- 背景流：双向 iperf3 UDP 各 12 路，TOS=64（P3），默认每方向 48Gbps 打满
- 通信原语：NCCL Ring **AllGather**（对比实验1~3的 AllReduce）

## 2. 关键参数（与 V6 / LongLiu 核心参数一致）

| 参数 | 值 | 说明 |
|---|---|---|
| 调度算法 | SLOScheduler（π 公式 + EMA 锚点） | 与 AllReduce 版完全一致 |
| c_i | 1.7 | SLO 松弛系数 |
| initial_priority | 3 (P3) | 与 V6 一致 |
| max_priority | 6 | 不封顶，允许升到 P6 |
| payload | 512 MB/rank | float32 tensor |
| sleep（模拟计算） | 30 ms/iter | |
| iters/epoch | 20 | |
| 总 iters（main） | 300（15 epochs） | |
| calib epochs | 5（solo） | 学 T_target + solo_bw |
| DSCP 映射 | P6→8, P4→0, P3→16, P2→24, P1→32, P0→40 | 与硬件 TC 一致 |
| 轮次 | ≥3 轮 | 评估跨轮稳定性 |

## 3. 执行流程（每轮）

1. **Phase 0**：`env_check.sh` 采集双端环境 + solo AllGather 校准
   （写入 `ttarget.json`：`target_comm_time_ms`、`solo_bw_gbps`）。
2. **Phase 1**：`bg_saturate.sh start` 打满背景流 + `monitor_nic.sh` / `monitor_gpu.sh`。
3. **Phase 2**：main — LongLiu 调度 AllGather 作业，每 iter 记录
   `priority/dscp/pi/comm_dur/bw`。
4. **Phase 3**：停止监控与背景流。

## 4. 指标与判定

| 指标 | 判定标准 |
|---|---|
| DSCP 切换准确性 | priority→DSCP 查表无跳变；有背景流时调度器应升优先级的轮次
  能观察到切换（≥1 次）；NIC prio 计数器与 dominant DSCP 队列一致 |
| 锚点测量精度 | main 全段平均带宽 / solo_bw 偏差 ≤ ±10%（带宽损耗来自背景流竞争，
  属预期）；跨轮 solo_bw 与 T_target 变异系数 CV ≤ 5% |
| SLO 守护 | main 期间 slowdown ≤ c_i（1.7）容忍线，π ≤ 0 时优先级应上调 |

## 5. 数据与产物

```
04_exp_comm_primitives/
├── config/EXP4_config.md          本文件
├── scripts/
│   ├── job_allgather.py           AllGather 作业（calib/main）
│   ├── run_exp4.sh                运行脚本（多轮次）
│   └── analyze_exp4.py            分析脚本
├── data/exp4_r<N>_<ts>/           原始数据（每轮）
│   ├── ttarget.json               T_target + solo_bw
│   ├── exp4_jobA_rank0_iter.csv   per-iter priority/dscp/pi/bw
│   ├── exp4_jobA_rank0_epoch.csv  per-epoch 汇总
│   ├── nic_10.csv / nic_226.csv   NIC 硬件计数器
│   ├── gpu_10.csv / gpu_226.csv   GPU 状态
│   └── *.log                       运行日志（含时间戳）
└── analysis/
    ├── exp4_dscp_trajectory.png   DSCP 切换轨迹（多轮分面）
    ├── exp4_anchor_accuracy.png   锚点 vs 观测对比
    └── exp4_report.md             分析报告
```

## 6. 可复现性说明

- NCCL：系统 libnccl.so.2.30.7；头文件 vendored 至
  `multi_comm_slo/src/include/nccl.h`（2.29.7 header，含 trafficClass），
  规避 `/usr/local/include/nccl.h`（v21700 无 trafficClass）的遮蔽问题。
- `libmulti_comm.so` 由 `multi_comm_slo/src/Makefile` 编译，
  导出符号含 `multi_comm_allgather`（`nm -D` 验证）。
- 226 端扁平路径 `/home/why/LongLiu_rebuild/multi_comm_slo/{src,build}`。
- 每次运行记录 `env_check` 输出至 `data/env/`，日期时间戳入 `RUN_ID`。

## 7. 预期结论（论文可用性）

- 若 AllGather 下 DSCP 切换与 AllReduce 一致 → 证明 LongLiu 的
  **优先级调度不依赖特定通信原语**（通用性）。
- 锚点精度跨轮稳定 → 证明校准协议与重校准探针的**可复现性**。
