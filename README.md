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
├── testbed/                    # Experiment orchestration & paper evidence
│   ├── PAPER_EVIDENCE/         # Archived evidence (5 pillars, read-only)
│   │   ├── 01_V6_main/         # V6-P4 primary results (console logs, CSV)
│   │   ├── 02_replications/    # Statistical replications (×2)
│   │   ├── 03_class_probe/     # Full-class DSCP-to-TC probe data
│   │   ├── 04_226_probe/       # NIC classification capability probes
│   │   ├── 05_tc_map/          # DSCP-to-TC mapping correction records
│   │   ├── 06_calibration/     # T_target calibration & background flow
│   │   ├── 07_V5_diagnosis/    # Environment drift attribution
│   │   └── 00_documents/       # Aggregated tables & §6.6 LaTeX source
│   ├── FIGURE_REGISTRY/        # Production figures & tables
│   │   ├── fig6_v6physical/    # Fig-6: testbed epoch-level slowdown
│   │   └── t3_topo/            # T-3: DSCP mapping & NIC asymmetry
│   ├── quota_bench.py          # NCCL AllReduce benchmark driver
│   ├── dscp_tc_sweep.py        # DSCP → traffic class sweep utility
│   └── experiment/             # Legacy experiment launch scripts
├── experiments/                # Experiment scripts & logs (P1–P4)
│   ├── P4_dumbbell_slo/        # Main V6 experiment scripts
│   ├── P3_ema_convergence/     # EMA convergence calibration
│   ├── P2_bimodal_interval/    # Iteration interval distribution
│   └── P1_dscp_injection/      # DSCP injection validation
├── physical_result/            # Raw physical experiment outputs
├── nccl-dscp/                  # Modified NCCL v2.18.3 source
│   ├── src/misc/dscp_adapter.cc    # Epoch management & DSCP mapping
│   ├── src/transport/net_ib.cc     # QP traffic class injection
│   ├── src/include/dscp_adapter.h  # Data structures
│   └── tools/                      # Statistics analysis utilities
├── multi_comm_slo/             # Multi-communicator SLO scheduler (C library)
│   ├── src/                    # C source with Python ctypes bindings
│   ├── test_scheduler_ema.py   # EMA scheduler unit tests
│   └── DESIGN.md               # Architecture documentation
├── DESIGN_REFERENCE.md         # Code-level design reference (for §3 writing)
├── LONGLIU_IMPLEMENTATION.md   # Architecture red lines & terminology
├── LONGLIU_PAPER_PLAN.md       # Paper outline & assignment
├── DSCP_VERIFICATION_LOG.md    # DSCP end-to-end verification record
├── 实验环境与实现方案.md         # Hardware & software environment (Chinese)
└── 实现进度与结果汇总.md          # Implementation progress summary (Chinese)
```

---

## Hardware Environment

| Node | GPU | NIC | RDMA | OS |
|------|-----|-----|------|-----|
| guolab-226 | 2× Quadro RTX 5000 | BlueField-3 B3220 (mlx5_0) | 100GbE RoCEv2 | Ubuntu 22.04 |
| guolab-10  | 1× Quadro RTX 4000 | BlueField-3 B3220 (mlx5_0) | 100GbE RoCEv2 | Ubuntu 20.04 |

- **Active link**: mlx5_0 only (mlx5_1 has RDMA connectivity issues)
- **NCCL_IB_HCA**: `mlx5_0`
- **NCCL_IB_GID_INDEX**: `3` (RoCEv2 IPv4)
- **Switch**: Supports DSCP→CoS priority queueing

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

### 1. Build NCCL with DSCP Support

```bash
cd nccl-dscp
make -j$(nproc) CUDA_HOME=/usr/local/cuda-11.8
```

### 2. Build Multi-Comm SLO Scheduler

```bash
cd multi_comm_slo
./build.sh
```

### 3. Run a V6 Experiment

```bash
# On guolab-10 (master)
cd experiments/P4_dumbbell_slo
bash run_v6_full.sh

# Sync script to guolab-226 first
bash sync_to_226.sh
```

### 4. Regenerate Fig-6

```bash
cd testbed/FIGURE_REGISTRY/fig6_v6physical
python3 plot_fig6.py
# Outputs: fig6_testbed.pdf, fig6_testbed_600.png, fig6_self_check.csv
```

---

## Paper Evidence

All paper evidence is archived in `testbed/PAPER_EVIDENCE/`. See `MANIFEST.md`
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
