# LongLiu: Sender-Side RDMA Priority Scheduling for Distributed DNN Training

LongLiu is a **sender-side, NCCL-centric** priority scheduling framework for
distributed deep neural network (DNN) training over RDMA fabrics. It assigns
DSCP-based priority classes to NCCL collectives at **epoch granularity** using
a closed-form shakedown heuristic, without requiring switch state tracking or
in-network modification.

This repository contains the **physical testbed** implementation, including
the modified NCCL v2.18.3 with DSCP injection, a Python-level multi-communicator
SLO scheduler, experimental validation on a two-node RoCEv2 cluster, and all
paper evidence for the INFOCOM 2026 submission.

---

## Key Idea

Existing packet-level priority schedulers (e.g., AuTO, CRUX) require either
in-network switch modification or per-packet ACK timestamping. LongLiu moves
the intelligence entirely to the **sender side**: a lightweight progress
deficit (π) computed from NCCL collective completion times dictates whether a
job needs strict priority, avoiding both switch modification and per-packet
overhead. The closed-form allocation guarantees that under moderate over-
subscription, every job meets its SLO without iterative optimization.

---

## Repository Structure

```
.
├── legacy/                     # 旧实现 + 早期机制验证实验
│   ├── nccl-dscp/              # C++ NCCL Shim 修改版（已弃用）
│   │   ├── src/misc/dscp_adapter.cc    # Epoch管理 & DSCP映射
│   │   ├── src/transport/net_ib.cc     # QP traffic class 注入
│   │   └── DESIGN_REFERENCE.md         # 代码分析文档
│   ├── nccl-master/            # NCCL 2.18.3 源码（编译 nccl-dscp 用）
│   ├── LongLiu_py/             # 早期 Python 原型（未使用）
│   └── experiments_mechanism/  # P1-P3 机制验证实验
│       ├── P1_dscp_injection/  # DSCP 注入验证
│       ├── P2_bimodal_interval/# 双峰间隔分布
│       └── P3_ema_convergence/ # EMA 带宽收敛
│
├── current/                    # 当前主力实现 + 评估实验
│   ├── multi_comm_slo/         # Python + C 多 Communicator（当前主力）
│   │   ├── src/multi_comm.c    # C库：预创建7个communicator，运行时切换索引
│   │   ├── src/slo_scheduler.py# Python调度器：π计算、优先级映射
│   │   └── DESIGN.md           # 架构文档
│   ├── experiments_evaluation/ # P4/V6/ExpA/B/C 评估实验
│   │   ├── P4_dumbbell_slo/    # 哑铃SLO对比 + V6物理实验
│   │   └── experiments_supplementary/  # 补充实验 A/B/C
│   │       ├── 00_prerequisites/       # 前置事项
│   │       ├── 01_exp_A_static_anchor/ # 实验A：静态锚点
│   │       ├── 02_exp_B_tier_swap/     # 实验B：动态锚点tier swap
│   │       ├── 03_exp_C_scale_ladder/  # 实验C：规模/稀缺锚点
│   │       ├── 04_bonus/               # 加分项
│   │       └── Evaluation.tex          # 论文评估章节
│   └── results/                # 实验结果汇总 + 论文图表
│       ├── testbed/            # 物理床结果（QUOTA_EXPERIMENT_RESULTS.md等）
│       ├── physical_result/    # 物理实验原始数据
│       └── fig_arch.*          # 架构图
│
├── README.md                   # 本文件
├── PROJECTS_SUMMARY.md         # 项目汇总（含依赖关系）
├── LongLiu_补充实验方案.md       # 补充实验规划
├── LongLiu_INFOCOM_Draft.md    # 论文草稿
├── LONGLIU_PAPER_PLAN.md       # 论文计划
└── LONGLIU_IMPLEMENTATION.md   # 实现说明
```

---

## Hardware Environment

| Node | GPU | NIC | RDMA | OS |
|------|-----|-----|------|-----|
| guolab-226 | 2× Quadro RTX 5000 | **BlueField-3 B3220** (mlx5_0) | **100GbE** RoCEv2 | Ubuntu 22.04 |
| guolab-10  | 1× Quadro RTX 4000 | **ConnectX-6 Dx** (mlx5_0) | **50GbE** RoCEv2 | Ubuntu 20.04 |

- **Active link**: mlx5_0 only (mlx5_1 has RDMA connectivity issues)
- **NCCL_IB_HCA**: `mlx5_0`
- **NCCL_IB_GID_INDEX**: `3` (RoCEv2 IPv4)
- **Switch**: Mellanox Spectrum SN2700 (32×100GE QSFP28, Cumulus Linux 5.1.0)
- **Link asymmetry**: 226 侧 100Gbps，10.1 侧 50Gbps，有效带宽受限于 50G
- **NIC asymmetry**: 10.1 (CX-6 Dx) 支持 DSCP 分类；226 (BF-3) **不支持** DSCP→prio 分类

---

## Key Constraints

| Constraint | Detail |
|------------|--------|
| NCCL_IB_HCA | Must be `mlx5_0` (mlx5_1 non-functional for RDMA) |
| glibc compatibility | NCCL must be compiled on 10.1 (glibc 2.31) for cross-node compatibility |
| CUDA version | NCCL must be compiled with CUDA 11.8 (PyTorch requirement) |
| 226 NIC | Does **not** classify DSCP into priority queues (all traffic on prio0) |
| mlx5_1 | Link configured but RDMA connection times out |
| MPS | 10.1 uses MPS for GPU sharing (only one NCCL communicator at a time) |

---

## Quick Start

### 1. Build Multi-Comm SLO Scheduler (当前主力实现)

```bash
cd current/multi_comm_slo
./build.sh
```

### 2. Run a V6 Experiment

```bash
# On guolab-10 (master)
cd current/experiments_evaluation/P4_dumbbell_slo
bash run_v6_full.sh

# Sync script to guolab-226 first
bash sync_to_226.sh
```

### 3. Regenerate Fig-6

```bash
cd current/results/testbed/FIGURE_REGISTRY/fig6_v6physical
python3 plot_fig6.py
# Outputs: fig6_testbed.pdf, fig6_testbed_600.png, fig6_self_check.csv
```

### 4. 旧实现 (nccl-dscp)

```bash
# 已弃用，仅供历史参考
cd legacy/nccl-dscp
make -j$(nproc) CUDA_HOME=/usr/local/cuda-11.8
```

---

## Paper Evidence

All paper evidence is archived in `current/results/testbed/PAPER_EVIDENCE/`. See `MANIFEST.md`
for the complete registry. Key results:

- **EQ5 (§6.6)**: LongLiu achieves 1.11× mean slowdown vs CRUX 1.26–1.36×
  (13.7–16.8% advantage) in the stable contention window (epochs 7–11),
  replicated across 3/4 independent rounds with non-overlapping intervals.
- **DSCP→TC mapping inversion**: The default `trafficClass = priority × 8`
  formula silently inverts priority under SP queueing. Corrected via full-class
  probe and NIC counter verification.
- **226 NIC asymmetry**: Only prio0 records traffic on the 226 NIC, sinking
  per-priority enforcement to the first switch hop.

---

## Citation

```bibtex
@inproceedings{longliu2026,
  title={LongLiu: Sender-Side RDMA Priority Scheduling for Distributed DNN Training},
  author={...},
  booktitle={IEEE INFOCOM},
  year={2026}
}
```

---

## License

NCCL source code is derived from NVIDIA NCCL v2.18.3 (BSD-licensed).
Modifications, experiment scripts, and documentation are provided for
academic review.
