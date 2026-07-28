============================================================
P4 Dumbbell SLO Verification — Experiment v3
============================================================

实验日期：2026-07-15 (v3 longliu 验证)
硬件：
  - 10.1: Quadro RTX 4000 (8GB), mlx5_0 RoCEv2 HCA
  - 226:  2x Quadro RTX 5000 (16GB each), mlx5_0 RoCEv2 HCA
网络：50Gbps RoCEv2 (10.1 ConnectX-6 Dx @50G ↔ 226 BlueField-3 @100G)
      RDMA实测 ~46 Gbps (单QP ib_write_bw)
  P4-programmable交换机
  DSCP->CoS mapping:
    - CoS 0: DSCP 0-7     → Q0
    - CoS 1: DSCP 8-15    → Q1
    - CoS 2: DSCP 16-23   → Q2
    - CoS 3: DSCP 24-31   → Q3 (高优先级)
    - CoS 4-6: DSCP 32-55 → 更高优先级

============================================================
运行模式
============================================================

  bash run_p4.sh <mode>

  mode ∈ {solo, fair, longliu}

  solo    — 仅 Job1，标准 NCCL (基线)
  fair    — 双 Job，标准 NCCL，无优先级 (公平竞争基线)
  longliu — 双 Job，MultiCommWrapper + NCCL 2.30.7 trafficClass
            (自适应 DSCP 优先级调度)

============================================================
当前配置 (v3 longliu)
============================================================

Job1: 2048MB payload, 50ms compute sleep
      300 iters (15 epochs × 20 iters/epoch)
      SLO c_i=1.5 (严格)

Job2: 2048MB payload, 50ms compute sleep
      200 iters (10 epochs × 20 iters/epoch)
      SLO c_i=2.5 (宽松)

调度: epoch_start() 设置 priority → 20× AllReduce →
      epoch_end() 测量耗时并计算新 priority

NCCL: NCCL 2.30.7 (ncclConfig_t.trafficClass)
      NCCL_ALGO=RING, NCCL_PROTO=SIMPLE
      7 priority communicators (DSCP 0/8/16/24/32/40/48)

============================================================
文件清单
============================================================

脚本:
  run_p4.sh             启动脚本 (solo/fair/longliu 三种模式)
  p4_job1.py            Job1 训练脚本 (GPU0, 300 iters)
  p4_job2.py            Job2 训练脚本 (GPU0, 200 iters)
  plot_p4.py            数据可视化脚本
  sync_to_226.sh        同步脚本到远程节点 226

辅助:
  ib_prio_test.py       RDMA IB优先级测试 (独立验证)
  ib_prio_test.sh       RDMA双QP优先级实验 (shell版)
  ib_prio_strict.sh     RDMA三QP严格优先级实验

CSV 结果文件:
  p4_job1_solo_rank0.csv        solo 模式基线
  p4_job1_fair_rank0.csv        fair 模式 Job1
  p4_job1_longliu_rank0.csv     longliu 模式 Job1
  p4_job2_longliu_rank0.csv     longliu 模式 Job2

============================================================
复现步骤
============================================================

1. 同步脚本到 226:
   bash sync_to_226.sh

2. 运行实验 (三种模式):
   bash run_p4.sh solo     # ～2min
   bash run_p4.sh fair     # ～3min
   bash run_p4.sh longliu  # ～3min

3. 结果在 p4_job*_<mode>_rank0.csv 中

4. 可视化:
   python3 plot_p4.py

============================================================
带宽计算公式说明
============================================================

CSV/日志中的 bw_gbps = BYTES_PER_ITER * 8/1e9 * (n-1)/n / time
即 NCCL 的 "bus bandwidth"（单方向链路带宽）。
对 2-rank Ring AllReduce 来说，链路传输量是 tensor 的 2 倍
(reduce-scatter → allgather)，因此 busbw = size/2/time。

相比之下 ib_write_bw 的 46 Gbps 是单 QP 单方向带宽，
而 NCCL 的 2 channel 并行能更好地利用链路。
二者不应直接比较——NCCL busbw 衡量的是 AllReduce 中的
有效单方向链路利用率，ib_write_bw 是裸 RDMA 吞吐。

============================================================
关键日志位置
============================================================

应用日志 (/tmp/ 可能被后续运行覆盖):
  /tmp/p4_job1_node101.log      10.1 Job1 标准输出
  /tmp/p4_job2_node101.log      10.1 Job2 标准输出
  /tmp/p4_job1_node226.log      226  Job1 标准输出 (通过 SSH)
  /tmp/p4_job2_node226.log      226  Job2 标准输出 (通过 SSH)

NCCL DEBUG 日志 (包含初始化详情):
  /tmp/nccl_j1_101_*.log        10.1 Job1 NCCL
  /tmp/nccl_j2_101_*.log        10.1 Job2 NCCL
  /tmp/nccl_j1_226_*.log        226  Job1 NCCL
  /tmp/nccl_j2_226_*.log        226  Job2 NCCL

============================================================
依赖
============================================================

- NCCL 2.30.7 (编译版): /home/why/LongLiu_rebuild/nccl-master/build/lib/
- MultiComm 库:          /home/why/LongLiu_rebuild/multi_comm_slo/
  - src/multi_comm.c     C 核心 (7 priority communicator)
  - src/slo_scheduler.py Python 调度器包装
  - build/libmulti_comm.so 编译产物
- iptables: 需放行端口 29500:61000 (192.10.10.0/24)
