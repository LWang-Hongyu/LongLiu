# D1@400G 失效机制定案

## 裁决
D1@400G P-attn=37.5% 不是锁入，是错峰+路由伪影 + 单层无 tier 隔离。

## 决定性证据

### 1. J1 与 J5 从未同链路竞争
- 28,893 个 epoch 中，J1 与 J5 head-to-head=0
- 7.1× 份额差是不同链路不同竞争环境的产物

### 2. 真争抢 epoch 中控制律精确执行
- N_active ≥ 3 的 7,294 个真争抢 epoch 中
- J0/J1/J2 的 obs-exp error = 0.0000（完美匹配）
- 反馈定律本身被精确执行

### 3. Standard 计入重算（6,286 混合 epoch）
- J5 obs-exp error = -0.0184
- J6 obs-exp error = -0.0184
- 全 job 均值 error = -0.0000 → **CONVERGED**

## 定稿表述

> 单层反馈定律在极端稀缺下无法隔离 tier：standard 的违约信号与 premium 同池竞争，
> premium 保护结构性失效；v4 的两级分层（premium 池 + standard floor）是结构性解法。
> 反馈定律本身被精确执行（全 job 误差≈0），失效的是单层设计，不是实现。

## 分析数据源
- Trace: `traces/trace_s0.jsonl` (E1_D1_400g, seed=0, 28,893 epochs)
- 分析脚本: `_d1_standard_recalc.py` (standard 计入重算)
- 分析脚本: `_d1_mechanism_audit.py` (条件化对照 + π 语义核查)
- 输出: `standard_recalc_s0.json`, `mechanism_audit_s0.json`

## 论文素材 #3 状态
✅ 定稿（2026-07-27，用户批准）
