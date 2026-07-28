# Fig-2：E1 Bandwidth Ladder

> **论文用途**：展示 LongLiu 在不同带宽超订程度下的 SLO 满足率 vs CRUX 基线。
> **数据就绪**：🔵 仿真归档后开画（E1 5-seed 表已验收通过）
> **数据源**：仿真侧机器（不在此机器上）

## 待补内容

仿真归档完成后，需从仿真侧补充：

- E1 5-seed ladder CSV（30 行 × 5 seed）
- sas_eval 输出的 SLO violation rate 对比
- 绘图脚本

## 关键参数

| 参数 | 值 |
|------|-----|
| 载荷点 | 30 个配置点（200-1000G，步长不等） |
| seed | 5（种子列表待补充） |
| 指标 | SLO violation rate | 
| 对比 | LongLiu / CRUX / v4S / D1 |
