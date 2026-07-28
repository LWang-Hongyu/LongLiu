# FIGURE_REGISTRY — 论文图表数据与脚本归档

> 生成日期：2026-07-27
> 对应：PAPER_EVIDENCE/（原始证据归档） → FIGURE_REGISTRY/（按图号组织的可审查包）

## 总览

```
FIGURE_REGISTRY/
├── README.md                          # 本文件 — 主注册表
├── [共享脚本]                          # 可在多图间复用的实验/分析脚本
│   ├── p4_job1.py / p4_job2.py        # Job 实验运行器
│   ├── p4_job1_crux.py / p4_job2_crux.py
│   ├── p4_job1_asym.py / p4_job2_asym.py
│   ├── diagnose_v5_anomaly.py         # V5 异常诊断
│   ├── diagnose_isolation.py          # V3 隔离诊断
│   ├── analyze_v4_deep.py / analyze_v5.py
│   └── bench_solo_bw.py / bench_bw_direct.py / bench_payload.py
│
├── fig1_arch/                         # Fig-1：系统架构图（论文绘）
├── fig2_e1_ladder/                    # Fig-2：E1 Ladder（仿真）
├── fig3_e2_orthogonal/                # Fig-3：E2/E2' 正交（仿真）
├── fig4_e3_swap/                      # Fig-4：E3 角色反转（仿真）
├── fig5_d1_mechanism/                 # Fig-5：D1 机制轨迹（仿真+物理）
├── fig6_v6physical/                   # Fig-6：V6-P4 物理床对照（物理）
├── t1_ttarget_calib/                  # T-1：T_target 校准表
├── t2_sim_config/                     # T-2：仿真配置表
└── t3_topo/                           # T-3：拓扑与映射探测（物理）
```

## 图→数据映射

| 图号 | 标题 | 数据源 | 状态 |
|------|------|--------|------|
| Fig-1 | System architecture | 论文绘制 | 🔵 待绘 |
| Fig-2 | E1 bandwidth ladder | 仿真 E1 5-seed | 🔵 仿真归档后开画 |
| Fig-3 | E2/E2' orthogonal | 仿真 E2 5-seed+E2' 5-seed | 🔵 同上 |
| Fig-4 | E3 role-reversal swap | 仿真 E3 3-seed | 🔵 同上 |
| Fig-5 | D1 mechanism trajectories | 仿真 D1 3-seed + 物理 V5 π 轨迹 | 🔵 同上 |
| Fig-6 | Physical bed V6-P4 comparison | 物理 V6 4 轮 console.log | 🟢 数据就绪 |
| T-1 | T_target calibration | 物理 06_calibration | 🟢 数据就绪 |
| T-2 | Simulation configuration | 仿真配置参数 | 🔵 仿真归档后 |
| T-3 | Topology mapping evidence | 物理 03/04/05 探测 | 🟢 数据就绪 |

## 审查流程

1. 每个图/表目录读 README → 确定数据源与脚本
2. 数据 CSV 可逐行还原
3. 实验脚本可复现运行（需物理床设备）
4. 仿真侧数据（Fig-2~5, T-2）需等仿真归档完成后补充

## 依赖

- **物理侧依赖**：226/10.1 双机 + NCCL LongLiu + RDMA mlx5_0
- **仿真侧依赖**：仿真器（独立环境），数据在另一台机器
- **绘图统一由仿真侧执行**以保证双栏格式全局一致
