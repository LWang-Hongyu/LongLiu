# Experiment A: Static Anchor — HW-vs-Sim Fidelity Evidence Table

> 目标：把论文中 "within 0.2%" 的断言变成可核查的结果表。
> 规划文档：`LongLiu_补充实验方案.md` §A
> 起始日期：2026-07-29

## 设计决策

### 容量维度实现方式

**问题**：方案要求 `{容量：50G / per-QP 限速到 35G / 25G}`，但测试床无 sudo、无 per-QP 限速 API（`multi_comm_slo` 仅支持 DSCP 优先级，不支持速率限制），且 iperf3 UDP 在 6G 以上出现 86% 接收端丢包。

**解决方案**：通过 **payload 大小** 调制有效容量。solo 校准证实 payload 与 solo BW 单调相关：
- 1024MB → solo BW ≈ 40.9 Gbps（双向，~82% of 50G link）
- 768MB → solo BW ≈ 58.1 Gbps（双向，calib: A=28.70, B=29.38 Gbps 单向）
- 512MB → 理论 50G（hold-out，不校准）

仿真器逐比特镜像相同 payload × sleep × c_i，因此 HW-vs-Sim 比较公平。

### 6 场景选取

从 {jobs: 2/2+bg} × {cap: 50G/35G/25G} × {c_i: 1.2/1.5} = 12 格中选 6 个覆盖全维度：

| # | Payload | c_i | BG | 容量 | 角色 |
|---|---------|-----|-----|------|------|
| S1 | 1024MB | 1.2 | 无 | 50G | baseline |
| S2 | 1024MB | 1.5 | 无 | 50G | c_i 效应 |
| S3 | 1024MB | 1.2 | 6G | ~44G | bg 效应 |
| S4 | 1024MB | 1.5 | 6G | ~44G | c_i + bg |
| S5 | 512MB | 1.2 | 无 | ~25G | **HOLD-OUT**（理论 anchor） |
| S6 | 768MB | 1.5 | 无 | ~35G | cap + c_i |

### Hold-out 设计 (S5)

S5 的 anchor 不取自当次 HW solo 校准，而用**理论 BW**（50G × payload_ratio）：
- 理论 T_target = payload_bits / (50G × 1e9) × iters_per_epoch
- 512MB 理论 T_target_epoch = 512×1024×1024×8 / 50e9 × 20 = 1718 ms

若 Sim 在 S5 上仍与 HW 匹配（尽管 anchor 不精确），则模型具备泛化能力。

## 目录结构

```
01_exp_A_static_anchor/
├── README.md                    # 本文件
├── scripts/
│   ├── expA_config.sh           # 共享配置（路径、env、helper）
│   ├── expA_scenarios.json      # 冻结的场景定义（含 md5 校验）
│   ├── calib_solo_expA.sh       # solo BW 校准（768MB for S6）
│   ├── run_expA_hw.sh           # HW 6 场景运行器
│   ├── run_expA_sim.py          # Sim 6 场景运行器（LongLiu 分配逻辑）
│   └── analyze_expA.py          # 误差分析与证据表生成
├── data/
│   ├── run_<timestamp>/         # 每次 HW 运行的归档
│   │   └── S<#>_<label>/       # 每场景目录（CSV + 日志 + manifest）
│   ├── sim_run_<pid>/           # Sim 运行归档
│   └── latest_run.txt           # 指向最新 HW 运行
├── logs/                        # 运行日志
└── analysis/                    # 分析产出
    ├── expA_evidence_table.md   # 证据表（进论文）
    ├── expA_evidence_table.csv  # 原始数据
    └── expA_trajectory.png      # 轨迹图
```

## 关键约束

1. **双机并发启动顺序**：必须先启动 Job A rank 1 (226) → 5s → Job A rank 0 (10.1) → 15s → Job B rank 1 (226) → 5s → Job B rank 0 (10.1)。否则 NCCL init 冲突（10.1 MPS 限制）。
2. **文件系统不共享**：10.1 和 226 的 `/home/why` 是独立文件系统。T_target 文件需 scp 到 226；CSV 需 scp 回 10.1。
3. **`env` 前缀必须**：bash 变量展开的 `KEY=VAL` 对需用 `env` 前缀才能作为环境变量传递给 `timeout` 命令。
4. **静态模式**：`--reverse-epoch 999 --ci-phase1 X --ci-phase2 X` 实现全程不 swap。

## 仿真器模型

Sim 显式实现 LongLiu 分配逻辑（与 HW shim 的 `slo_scheduler.py` 同构）：

1. **π 计算**：π = A / (c × T × k) - 1
2. **π → priority**（4-tier，per project_memory）：π>0.3→P6, -0.1<π≤0.3→P4, -0.5<π≤-0.1→P2, π≤-0.5→P1
3. **SPQ 带宽分配**：最高优先级作业独占链路；同优先级均分
4. **背景流**：eff_link = 50G - bg_rate
5. **slowdown** = comm_time / (c_i × T_target_per_iter)

**已知局限**：Sim 假设 100% duty cycle（两作业始终同时活跃），不建模相位互斥效应。HW 中 30ms sleep 使 duty ≈ 87.5%，两作业重叠概率 ≈ 76.6%，有效争抢低于 Sim 预测。
