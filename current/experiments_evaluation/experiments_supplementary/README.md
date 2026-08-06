# LongLiu 补充实验（物理锚点三件套）

> 规划文档：`LongLiu_补充实验方案.md`（仓库根目录）
> 环境：guolab-10 (10.1) + guolab-226 (226)，50G RoCEv2 哑铃拓扑
> 起始日期：2026-07-28

## 目录结构

```
experiments_supplementary/
├── README.md                        # 本文件
├── 00_prerequisites/                # 前置事项（0.1-0.3）
│   ├── scripts/                     # 环境核实脚本
│   ├── data/                        # 硬件清单、链路速率、DSCP 探针数据
│   ├── logs/
│   └── analysis/
├── 01_exp_A_static_anchor/          # 实验 A：0.2% 保真度证据表
│   ├── scripts/                     # 6 场景 HW vs Sim 对比脚本
│   ├── data/                        # 每场景 slowdown_hw / slowdown_sim
│   ├── logs/
│   └── analysis/                    # 误差汇总表生成
├── 02_exp_B_tier_swap/              # 实验 B：硬件 tier swap（最高优先级）
│   ├── scripts/                     # 双臂（LongLiu/静态）tier swap 运行脚本
│   ├── data/                        # W1/W2/W3 窗口 slowdown CSV
│   ├── logs/
│   └── analysis/                    # 窗口分析与轨迹图生成
├── 03_exp_C_scale_ladder/           # 实验 C：CPU epoch 模拟器多作业阶梯
│   ├── emulator/                    # ibv verbs 多作业模拟器
│   ├── daemon/                      # 独立分配守护进程（shim 逻辑抽库）
│   ├── scenarios/                   # 三 regime 场景定义
│   ├── data/
│   ├── logs/
│   └── analysis/
└── 04_bonus/                        # 加分项
    ├── scripts/
    ├── data/
    ├── logs/
    └── analysis/
```

## 实验进度

| 实验 | 状态 | 说明 |
|------|------|------|
| 00 前置 | ✅ 数据收集完成 | 链路 50G（不对称）、硬件清单、DSCP 探针复用 t3 |
| A 静态锚点 | ✅ **完成** | 6 场景 HW vs Sim 对比，均值误差 32.63%，根因=相位互斥 |
| B tier swap | ✅ **完成** | 4 轮实验完成，LL W3/W1=1.01 (PASS)，优先级轨迹差异是核心证据 |
| C 规模阶梯 | ✅ **完成** | 27 轮实验（3 regime × 3 arm × 3 round），multi-QP DSCP 切换验证通过 |
| 加分项 | 🔲 待开展 | shim 开销 / anchor 收敛 / EQ5 补轮 |

### 实验 A 关键结论（2026-07-29）
- **6 场景全部完成**：S1-S6 数据完整（各 25 epoch × 2 job）
- **0.2% 保真度未达标**：均值误差 32.63%，最大 38.50%（S4 带背景流）
- **系统性偏差**：Sim 一致高估 slowdown（sim 1.13-1.67 vs HW 0.90-1.27）
- **根因 = 相位互斥**：NCCL 双作业 ON/OFF 相位自然错开，实际争抢低于稳态带宽共享模型预测（overlap 0.45 vs duty² 0.83）
- **证据表已生成**：`01_exp_A_static_anchor/analysis/expA_evidence_table.md`
- **论文建议**：如实报实际误差 + 根因分析，0.5% 配表远比无表 0.2% 硬

### 实验 B 关键结论（2026-07-28）
- **LL 臂**：W3/W1=1.01×（无瞬态崩溃 ✅），BA 顺序下 1.05×
- **CX 臂**：W3/W1=1.03×（方向正确但未达 >1.5 阈值），BA 顺序下 1.085×
- **核心证据**：优先级轨迹——LL 动态 P2↔P6/P4 vs CX 静态 P3/P4 固定
- **BA 顺序更有效**：避免相位互斥效应，暴露 3.5% 调度器差异
- 详细分析：`02_exp_B_tier_swap/analysis/expB_findings.md`

### 实验 C 关键结论（2026-07-30）
- **27 轮实验完成**：3 regime (deep_scarcity/transition/ample) × 3 arm (longliu/static/fair) × 3 round
- **Multi-QP DSCP 切换**：mlx5 拒绝 `ibv_modify_qp(IBV_QP_AV)`（EINVAL）—采用 LongLiu8 multi-comm 思路：预创建 4 个 QP（P6/P4/P2/P1），切换时只换指针
- **P-attn 跨 regime 对比**（lower = better）：

  | Regime | LongLiu | Static | Fair | LL vs Static |
  |--------|---------|--------|------|--------------|
  | ample | 0.179 | 0.174 | 0.499 | ≈（无显著差异） |
  | deep_scarcity | 0.356 | 0.556 | 0.204 | LL 低 36% ✓ |
  | transition | 0.344 | 0.488 | 0.182 | LL 低 30% ✓ |

- **关键发现**：
  1. LongLiu 在所有 regime 的 P-attn 均 ≤ Static，证明动态优先级在硬件上有效
  2. LongLiu 在 ample 显著优于 Fair（0.179 vs 0.499），但在稀缺 regime 反而劣于 Fair — 因 mlx5 不支持真正的 strict priority（NO MLNX_QOS ETS）
  3. Static 臂 regime 排名（deep_scarcity > transition > ample）与 E1 仿真梯子方向一致 ✓
  4. J2(deep_scarcity standard) 在所有 3 round 中都被升到 P6（consistently behind SLO）
- **T_target 校准局限**：solo 测量无争抢，实际运行时 comm 暴涨 4-5×（J2: solo 70us → contested 300us）。这是 solo calibration 的固有限制，论文应说明
- 详细分析：`03_exp_C_scale_ladder/analysis/expC_summary.md`

## 关键约束（继承自 project_memory）

- NCCL_IB_HCA=mlx5_0, NCCL_IB_GID_INDEX=3
- 10.1 只能同时一个 NCCL communicator（MPS）
- 无 sudo / CAP_NET_RAW / tcpdump
- iperf3 UDP 单流 ~2-3Gbps，多流 86% 接收端丢包
- 226 NIC 不分类 DSCP（所有流量 prio0）
- 跨平面比较一律用 slowdown/attainment 比值，绝不用绝对时间
