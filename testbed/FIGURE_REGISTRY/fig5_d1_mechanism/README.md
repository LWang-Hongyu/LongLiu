# Fig-5：D1 机制轨迹

> **论文用途**：展示 D1 调度器的 π→priority 动态调整轨迹，以及 V5 物理床的 π 轨迹对照。
> **数据就绪**：🟡 仿真 D1 3-seed 已验收，物理 V5 π 轨迹已就绪
> **数据源**：仿真侧 D1 轨迹 + 物理侧 V5 diagnosis

## 内容

```
fig5_d1_mechanism/
├── README.md                           # 本文件
├── 07_V5_diagnosis/
│   ├── v5_drift_attribution.txt        # V5 漂移归因分析
│   └── phase1_control_group.csv        # Phase 1 控制组数据
│
└── [仿真侧待补]
    ├── d1_sas_trajectory_e3.csv        # D1 E3 臂轨迹
    └── d1_sas_trajectory_e3p.csv       # D1 E3' 臂轨迹
```

## π 轨迹验证（V5 物理床）

LongLiu v1(π):
- Job A: [2, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2] (P2→P4→P2)
- Job B: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6] (全程 P6)

→ **π→DSCP 环路在真实 RoCE 上按设计工作**。

## 待补内容

仿真归档完成后，需从仿真侧补充：

- D1 双臂轨迹 CSV（含 π(t)、priority(t)、slowdown(t)）
- π 提取脚本与 per-job π 表
- D1 臂的 16 个 run_meta
