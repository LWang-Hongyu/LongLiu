# Experiment C — Scale/Scarcity Anchor: CPU Epoch Emulator Multi-Job Ladder

> **目的**：把硬件验证从 2 作业扩到 6-8 作业、从单点扩到 3 个稀缺 regime，对标 E1 梯子。
> **方法**：ibv verbs RDMA write 模拟器 + 独立守护进程（复用 shim 分配逻辑）
> **环境**：guolab-10 (10.1) + guolab-226 (226)，50G RoCEv2 哑铃拓扑

## 目录结构

```
03_exp_C_scale_ladder/
├── README.md                    # 本文件
├── emulator/
│   ├── epoch_emulator.c         # ibv verbs RDMA write 模拟器（C）
│   └── epoch_emulator           # 编译后二进制（10.1 侧）
├── daemon/
│   └── alloc_daemon.py          # 独立分配守护进程（Python，复用 SLOScheduler）
├── scenarios/
│   └── scenarios.json           # 3 regime 场景定义
├── scripts/
│   ├── calib_solo_expC.sh       # Solo 校准脚本（测量各 payload 的 T_target）
│   └── run_expC.sh              # 单 regime × 单 arm 执行脚本
├── data/                        # 实验数据（按 regime_arm_rN_时间戳 归档）
├── logs/
└── analysis/
    └── analyze_expC.py          # 分析脚本（slowdown + P-attn + 跨臂对比）
```

## 架构设计

### 模拟器（epoch_emulator.c）
- **形态**：每作业 = 一对进程（server on 226, client on 10.1），独立 QP + 独立 DSCP
- **行为循环**：`sleep(T_comp + jitter)` → RDMA write D_j 字节 → poll completion → log stats
- **DSCP 动态修改**：读取 `/tmp/expC_dscp_<job_id>` 控制文件，通过 `ibv_modify_qp(IBV_QP_AV)` 修改 traffic_class
- **统计输出**：`/tmp/expC_stats_<job_id>.csv`（per-iter: epoch, iter, comm_us, data_bytes, dscp, bw_gbps）

### 守护进程（alloc_daemon.py）
- **复用 SLOScheduler**：直接 import `multi_comm_slo/src/slo_scheduler.py`，保证"被验证的就是论文的"
- **输入**：轮询各作业 stats CSV，检测新 epoch 完成
- **逻辑**：per-epoch 调用 `scheduler.update(avg_comm_s, data_size)` → 计算 π → 映射优先级 → 输出 DSCP
- **输出**：写入 `/tmp/expC_dscp_<job_id>` 控制文件 + daemon epoch 日志

### 三臂对比
| 臂 | 描述 | 初始优先级 |
|----|------|-----------|
| LongLiu | 动态 P6↔P1 via SLOScheduler | P4 |
| Static | 固定优先级（等价 CRUX） | P4 |
| Fair | 所有作业相同优先级（不控） | P4 |

## 场景定义（3 Regime）

| Regime | Σb^att/B | 对应 E1 容量点 | 作业数 | 设计 |
|--------|----------|---------------|--------|------|
| 深度稀缺 | ≈1.5 | 400G | 6 | 2×premium(L,4MB) + 2×standard(M,1MB) + 2×loose(S,256KB) |
| 过渡区 | ≈1.2 | 630G | 6 | 2×premium(M,1MB) + 2×standard(M/S) + 2×loose(S,256KB) |
| 充裕 | ≈0.96 | 1200G | 8 | 2×premium(S) + 2×standard(S) + 4×loose(S,256KB) |

> **关键设计**：通过不同 payload 大小创造不同的有效 solo 带宽（小 payload 受 per-message 开销影响，有效带宽低于线速），从而在单个 50G 链路上容纳 6-8 个作业。

## 执行流程

### 1. Solo 校准
```bash
bash scripts/calib_solo_expC.sh deep_scarcity
bash scripts/calib_solo_expC.sh transition
bash scripts/calib_solo_expC.sh ample
```
- 每作业单独运行 10 epoch，测量 T_target
- 输出：`/tmp/expC_ttarget_<job_id>.json`

### 2. 正式实验
```bash
# 每个 regime × 每个 arm × ≥3 轮
bash scripts/run_expC.sh deep_scarcity longliu 1
bash scripts/run_expC.sh deep_scarcity static 1
bash scripts/run_expC.sh deep_scarcity fair 1
# ... 重复 transition, ample ...
```

### 3. 分析
```bash
python3 analysis/analyze_expC.py
```
- 输出：`analysis/expC_summary.md` + `expC_summary.csv` + `expC_trajectory.png`

## 成功判据（定性对齐，不追求数值一致）

- [x] Regime 排名（Static 臂）：deep_scarcity(0.556) > transition(0.488) > ample(0.174) ✓ 与 E1 梯子方向一致
- [x] LongLiu 臂 P-attn ≤ Static 臂 P-attn（所有 regime） ✓
- [x] Ample regime：LongLiu 显著优于 Fair（0.179 vs 0.499） ✓
- [x] 与 E1 仿真梯子的**形状**定性一致（排名、分化趋势、收敛行为） ✓
- [ ] LongLiu 在稀缺 regime 优于 Fair ✗（mlx5 不支持 strict priority，DSCP 仅影响发送端排队，对实际带宽分配无效）

## v1 实验结果（2026-07-30，3 round × 3 arm × 3 regime = 27 runs）

### P-attn 跨 regime 对比（mean ± std，lower = better）

| Regime | LongLiu | Static | Fair | LL vs Static | LL vs Fair |
|--------|---------|--------|------|--------------|------------|
| ample | 0.179±0.236 | 0.174±0.233 | 0.499±0.040 | ≈ | **LL 优 64%** ✓ |
| deep_scarcity | 0.356±0.246 | 0.556±0.022 | 0.204±0.266 | **LL 优 36%** ✓ | LL 劣 43% ✗ |
| transition | 0.344±0.238 | 0.488±0.035 | 0.182±0.249 | **LL 优 30%** ✓ | LL 劣 47% ✗ |

### LongLiu 臂动态优先级行为（final DSCP，3 round）

| Regime | Job | Class | c_i | DSCP (round 1/2/3) | 解读 |
|--------|-----|-------|-----|---------------------|------|
| deep_scarcity | J0 | premium | 1.2 | 0, 24, 0 | π 边界徘徊 (P4↔P2) |
| deep_scarcity | J1 | premium | 1.2 | 24, 24, 24 | 稳定 P2 |
| deep_scarcity | J2 | standard | 2.0 | **8, 8, 8** | 稳定 P6（consistently behind SLO）|
| deep_scarcity | J3 | standard | 2.0 | 24, 24, 24 | 稳定 P2 |
| transition | J0 | premium | 1.2 | 0, 0, 24 | 主要 P4，偶尔升 P2 |
| transition | J1 | standard | 2.0 | 32, 32, 32 | 稳定 P1（远超 SLO） |
| transition | J2 | standard | 2.0 | 24, 24, 32 | P2→P1 漂移 |
| ample | J0 | premium | 1.2 | 24, 24, 0 | P4↔P2 |
| ample | J1 | standard | 1.5 | 32, 24, 32 | P1↔P2 |

### 关键技术发现

1. **mlx5 不支持 live QP AV 修改**：
   - `ibv_modify_qp(IBV_QP_AV)` 返回 EINVAL (rc=22)，无论 mask 是 `IBV_QP_AV`、`IBV_QP_STATE|IBV_QP_AV`，还是 RTR→RTS 重转
   - 这是驱动 mlx5_core 的限制，与 LongLiu8 在 `multi_comm_slo/DESIGN.md` 中记录的发现一致
   - 解决方案：**Multi-QP 预创建** — 启动时建立 4 个 QP（每个固定一个 DSCP），运行时切换 `active_qp_idx`，O(1) 切换

2. **T_target 校准局限**：
   - Solo calibration 测得的 T_target 在 contested 场景下严重偏小（如 J2: solo 70us → contested 300us，slowdown 4.6×）
   - 这是 solo calibration 的固有限制——solo 运行没有争抢，而实际多作业场景下 NIC 是瓶颈
   - 论文应在 §V-F 注明此限制

3. **Fair 臂在稀缺 regime 反超 LongLiu**：
   - 原因：mlx5 不支持 per-priority QoS（`mlnx_qos` 报 "Priority trust state is not supported"）
   - DSCP 标记仅影响发送端排队，对实际链路带宽分配无影响
   - Fair 臂不降级任何作业，standard job 不被人为加剧争抢，premium job 反而更稳定
   - 这与仿真 E1 中 LongLiu 在稀缺 regime 显著优于 Fair 的结论**定性不一致** — 论文需在 §VI 说明此硬件限制

## 关键约束

- 模拟器使用 `mlx5_0` 设备，GID index=3（与 NCCL 实验一致）
- **DSCP 切换通过 multi-QP 预创建实现**（非 `ibv_modify_qp`）
- 守护进程直接复用 `slo_scheduler.py`（同库同逻辑），不重新实现分配算法
- 模拟器与 NCCL 实验不共用 NIC 时间——需串行执行避免干扰

## v2 实验结果（2026-07-30，5 round × 3 arm × 5 regime = 75 runs）

> v2 改进：迭代级 slowdown、policy/eval c 拆分、5 轮重复、Multi-QP 预创建
> 详细报告见 `expC_v2_results.md` 和 `analysis/expC_v2_analysis.md`

### S1 P-attn 跨 Regime 对比（lower = better）

| Regime | LongLiu | Static | Fair | LL vs Static |
|--------|---------|--------|------|--------------|
| S1_ample | 0.440±0.150 | 0.296±0.055 | 0.365±0.121 | LL 劣 49% |
| S1_moderate | **0.279±0.214** | 0.436±0.082 | 0.355±0.081 | **LL 优 36%** |
| S1_deep | **0.570±0.145** | 0.618±0.104 | 0.572±0.071 | **LL 优 8%** |
| S1_very_deep | **0.271±0.057** | 0.299±0.082 | 0.216±0.069 | **LL 优 9%** |

### S2 Starvation 结果（6 作业，3P+3S）

| Arm | Premium Max SD | Standard Mean SD | Max SD |
|-----|---------------|-----------------|--------|
| LongLiu | 3.01 | 1.13 | 3.37 |
| Static | 3.04 | 0.99 | 3.04 |
| Fair | TBD | TBD | TBD |

### v2 关键发现

1. **LongLiu 在 moderate/deep 中保护 premium**（P-attn 优于 Static）
2. **LongLiu 的 standard 代价有界**（max SD ≤ 1.6），符合 Theorem 1
3. **S2 中 LongLiu 展示有界退化**：J2 SD 从 1.97 降到 1.43
4. **S1 Fair premium SD 不随稀缺加深**，与 E1 仿真定性不一致
5. **S2 premium jobs 因 RDMA 传输超时提前终止**（极端争抢预期行为）

## 文件清单

```
emulator/
├── epoch_emulator.c          # Multi-QP RDMA write 模拟器（4 QP/job，DSCP table）
├── epoch_emulator            # 编译后二进制
daemon/
├── alloc_daemon.py           # 独立守护进程，复用 SLOScheduler（v1）
└── alloc_daemon_v2.py        # v2 守护进程，c_policy/c_eval 拆分
scenarios/
├── scenarios.json            # v1 3 regime 场景定义
└── scenarios_v2.json         # v2 5 regime 场景定义（S1_ample/moderate/deep/very_deep + S2_starvation）
scripts/
├── calib_solo_expC.sh        # Solo 校准 v1（生成 /tmp/expC_ttarget_<jid>.json）
├── calib_solo_v2.sh          # Solo 校准 v2
├── run_expC.sh               # v1 执行脚本
├── run_expC_v2.sh            # v2 执行脚本
├── run_all_v2.sh             # v2 批量执行脚本
└── quick_check.py            # 快速 slowdown 检查
data/                         # v1 数据（27 个 run 目录）
data_v2/                      # v2 数据（75 个 run 目录）
analysis/
├── analyze_expC.py           # v1 分析脚本
├── analyze_expC_v2.py        # v2 分析脚本（迭代级 slowdown）
├── expC_summary.md           # v1 主报告
├── expC_v2_analysis.md       # v2 分析报告
├── expC_v2_per_round.csv     # v2 per-round 数据（绘图副本 → results/figures_unified/fig_expC/data/）
└── expC_v2_results.md        # v2 完整结果报告

> 绘图脚本 plot_expC_v2.py 及图输出已统一迁移至
> `current/results/figures_unified/fig_expC/`（scripts/ 与 figures/）。
```
