# Fig-3：E2/E2' Orthogonal

> **论文用途**：展示 LongLiu 在多维配置空间的鲁棒性（正交实验 E2 + E2'）。
> **数据就绪**：🔵 仿真归档后开画（E2 5-seed + E2' 5-seed 已验收通过）
> **数据源**：仿真侧机器（不在此机器上）

## 待补内容

仿真归档完成后，需从仿真侧补充：

- E2 5-seed 正交表
- E2' 5-seed 正交表
- 关键关注点验证：
  - E2'@500G v4 vs CRUX gap: 20.0pp ≥ 10pp ✅
  - E2'@630G v4=91.1%（非 100%，但仍具优势）
  - E2-pro 打平 sanity: PASS ✅
- 绘图脚本

## 关键参数

| 参数 | E2 | E2' |
|------|----|-----|
| 载荷 | 500-800-1000G | 200-500-630-800G |
| seed | 5 | 5 |
| 指标 | SLO violation rate | SLO violation rate |
| 对比 | LongLiu/CRUX/v4S/D1 | LongLiu/CRUX/v4S/D1 |
