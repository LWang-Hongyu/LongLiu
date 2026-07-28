# Fig-1：系统架构图

> **论文用途**：展示 LongLiu 系统架构（调度器、NCCL proxy、RoCE 网络栈）。
> **数据就绪**：🔵 论文绘制
> **数据源**：架构设计文档（非实验数据）

## 占位

此图为论文作者绘制，无需实验数据。

包含元素期望：
- LongLiu SLO Scheduler → NCCL proxy → multi_comm → DSCP marking → RoCE switch
- CRUX baseline 对比路径
- 物理床和仿真器的逻辑关系
