# HANDOFF：项目关键决策与约束

## 2026-07-24：数字核查与场景修订

### ci 变更历史

| 日期 | 变更 | 批准状态 |
|------|------|---------|
| 2026-07-19 | ci=1.2 → 1.3（AI 提议，未显式批准） | 追溯接受 |
| 2026-07-24 | ci=1.3/2.0 → 2.0/3.0（用户批准） | 已批准 |

**约束**：c_i 是场景定义参数，变更必须用户显式批准。

### 工作点修正历史

| 日期 | 变更 | 依据 |
|------|------|------|
| 2026-07-24 | 主评估点 @1000G → @800G | Σattain=879 Gbps，按 attain 口径真边界为 1.10× |

### 建模假设

- **overlap=0.85**：仅用于 iter_solo 基准计算（缩短通信时间），不影响实际迭代的 barrier 时间记录
- **iter_solo = comp + comm_solo × (1 - overlap)**：基准迭代时间
- **iter_actual = comp + comm_actual**：实际迭代时间，通信必须完全完成
- **attain_bw = bits / comm_budget**：attainment 带宽需求，comm_budget = ci × iter_solo - comp

### attain_bw 计算口径（唯一合法）

```python
comm_solo_ms = mb_per_iter * 8 / host_bw_gbps  # ms
iter_solo_ms = comp_ms + comm_solo_ms * (1 - overlap_factor)
comm_budget_ms = ci * iter_solo_ms - comp_ms
attain_bw_gbps = bits_per_iter / (comm_budget_ms * 1e-3) / 1e9
```

### 新工作点（ci=2.0/3.0）

- **主评估点**：spine=1000G（Σattain=879G，负载 0.88×）
- **负载阶梯**：1200/1000/800/630G ≈ 0.73/0.88/1.10/1.40×

### 禁止事项

- 禁止漏算小 job 的需求合计
- 禁止 ci 未经批准变更
- 禁止 G0 相对带（顶端塌缩）
- 禁止 git clean

---

## 2026-07-21：D1 重生主表与机制诊断

### T_target 修复（根因级）

旧 D1 控制目标：`T_target = ci × comm_solo × overhead_factor(2.0)`
- 对 13B premium 计算：1248 ms
- 正确值（sas_eval 同源）：189.6 ms
- **差 6.6 倍**，导致 π 排序整个错乱

修复后：`T_target = ci × (comp + comm_solo × (1 - overlap))`

### D1 轨迹关键发现

"S1 爬进 P6 同室操戈"：Premium π>0 进入 P6，但 S1 被剥夺后 π 也升至 ~1.0 进入 P6，形成类内同权竞争。

### D1G 失败根因

1. 参照系失真：pace-demand 包含 comp 时间，掩盖真正的通信瓶颈
2. 无 demand 封顶：P1/P2 的 pace-demand 相近，类内同权竞争

---

## 2026-07-16：项目约束

### Hard Constraints

- Simulator 仅支持 DDP，排除 TP/PP/EP/CP
- AllReduce barrier 语义必须建模（多流 + tail flow 判定）
- 工作负载 tiered SLO：大模型 ci=1.5，中等 ci=2.0，小模型 ci=3.0（已修订为 2.0/3.0）
- LongLiu K 参数：2.0（16-node FatTree）
- Spine 带宽：400Gbps（16-node）或按负载点调整
- 首轮所有任务 DSCP 38（最高优先级）
- 任务到达：Poisson 分布

### Engineering Conventions

- FatTree 拓扑：host_bw / tor_bw / spine_bw / rack_oversub
- 通信量：mb_per_iter = 2 × params × bpp / dp
- SLO attainment：累计平均迭代时间，非完成迭代数
- 权重分配：类间 DWRR，类内 exp(pi·K) 加权