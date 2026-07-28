# CONVENTION.md: 唯一约定链

## 语义版本
SEMANTICS_VERSION = "anchor-v2"

## 锚点公式（paper-baseline-v2 tag，逐字）

```python
# 唯一合法的 SAS / target_iter_ms 计算
# comm_solo_ms = logical_bits / host_bw（逻辑口径，13B=520ms）
# 与代码变量语义一致
comm_budget = slo_ci * comm_solo_ms * overhead_factor   # overhead=1.3
if overlap_factor > 0:
    target_iter_ms = max(comp_ms, comm_budget) + (1 - overlap_factor) * min(comp_ms, comm_budget)   # overlap=0.85
else:
    target_iter_ms = comp_ms + comm_budget
sas = target_iter_ms / avg_iter_ms
```

## 完整约定链（全量标注口径：逻辑/wire）

```
logical_bits       = mb_per_iter × 1e6 × 8                  （mb_per_iter 为十进制 MB，13B = 52.0 Gb）
wire_bits          = logical_bits × overhead_factor(1.3)   （逻辑 → wire）
    |
comm_solo_ms(逻辑) = logical_bits / host_bw × 1000         （13B = 520ms，逻辑口径）
comm_solo_ms(wire) = wire_bits / host_bw × 1000            （13B = 676ms，wire 口径）
    |
comm_budget        = ci × comm_solo_ms(逻辑) × overhead    （= ci × wire_solo，13B ci=2.0 → 1352ms）
    |
target             = max(comp, comm_budget) + (1−overlap) × min(comp, comm_budget)
    |
sas                = target / avg_iter
    |
attain_bw          = wire_bits / (target − comp)            （分子 wire，分母 eff budget）
```

### 链图自检（文档内嵌一致性断言）

以 attain 表中 J3 (LLaMA-2-13B, dp=8, ci=2.0) 为实例，逐级重算：

| 步骤 | 公式 | 数值 | 口径 |
|---|---|---|---|
| logical_bits | 6500 × 1e6 × 8 | 52.0 Gb | 逻辑 |
| wire_bits | 52.0 × 1.3 | 67.6 Gb | wire |
| comm_solo_ms(逻辑) | 52.0e9 / 100e9 × 1000 | 520 ms | 逻辑 |
| comm_solo_ms(wire) | 67.6e9 / 100e9 × 1000 | 676 ms | wire |
| comm_budget | 2.0 × 520 × 1.3 | **1352 ms** | — |
| target | max(80,1352) + 0.15×min(80,1352) | 1364 ms | — |
| eff_budget | 1364 − 80 | 1284 ms | — |
| attain_bw | 67.6e9 / 1.284 | **52.6 Gbps** | — |

**断言**：comm_budget = 1352ms 与 attain 表 J3 一致；attain_bw = 52.6Gbps 与 attain 表 J3 一致。任一不一致即链图或代码有 bug，禁止合并。

## 冻结参数

| 参数 | 值 | 来源 |
|---|---|---|
| overhead_factor | 1.3 | v2 tag 锚点（wire 因子，非发明常数） |
| overlap_factor | 0.85 | v2 tag 锚点 |
| K (DWRR) | 2.0 | 16-node FatTree 约定 |

## 锚点证据

- 裁决实验（2026-07-24）：48/48 逐值精确匹配 v2_test 数据
- 锚点文件：outputs/quickfix/v2_test/per_job.json
- 锚点 tag：paper-baseline-v2 (35be61b)

## 历史误拔事件

2026-07-24：口径审计中将 overhead=1.3 误判为"第三种 SAS 公式"并拔除。
实际 1.3 是 v2 tag 冻结的 wire 因子。现行公式（iter_solo = comp + comm_solo*(1-overlap)）
从未经过锚点验证，其合法地位只来自误判。现已复位。

## 历史锚点值永久不可复现

v2 tag → HEAD 之间存在 `_advance` bug-fix（future flow skip），该修复改变仿真语义。
**v2 历史基线值（Fair 0.897 等）在 HEAD 上永久不可复现**，即使重跑 v2 workload 也无法逐位匹配。

后果：
- 论文引用的所有基线数字，必须来自当前代码重生成的 run
- v2 历史值降级为历史档案，禁止作为现行基线引用
- 新锚点 以 `gatekeeper --init` 在当前代码上建立，该次 run 为新锚点的出生证，完整入档

## attain 表（锚点语义，FEAS_BOUNDARY_V2_WORKLOAD）

| Job | Tier | ci | wire_bits (Gb) | comp (ms) | comm_budget (ms) | target (ms) | eff_budget (ms) | attain (Gbps) |
|-----|------|----|----------------|-----------|------------------|-------------|-----------------|---------------|
| J0  | S    | 3.0| 57.2           | 60        | 1716.0           | 1725.0      | 1665.0          | 34.4          |
| J1  | P    | 2.0| 7.1            | 50        | 141.4            | 148.9       | 98.9            | 71.5          |
| J2  | P    | 2.0| 36.4           | 40        | 728.0            | 734.0       | 694.0           | 52.4          |
| J3  | P    | 2.0| 67.6           | 80        | 1352.0           | 1364.0      | 1284.0          | 52.6          |
| J4  | S    | 3.0| 3.5            | 50        | 106.1            | 113.6       | 63.6            | 55.6          |
| J5  | S    | 3.0| 67.6           | 80        | 2028.0           | 2040.0      | 1960.0          | 34.5          |
| J6  | S    | 3.0| 1.8            | 40        | 53.7             | 59.7        | 19.7            | 91.0          |

- Sigma_attain_P = 176.6 Gbps
- Sigma_attain_S = 215.4 Gbps
- beta * Sigma_attain_S = 107.7 Gbps (beta=0.5)
- 可行域边界 C* = 284.3 Gbps

## 场景说明

当前 7-job workload 在锚点语义下总需求 392 Gbps——@800G 是 0.49 的深可行区。
feas_boundary 场景必须重设计（feas_boundary_v3），使 Sigma_attain_P in [0.85, 0.95] * 800G。

## solo 不变量门禁期望值（锚点语义）

13B (ci=2.0): target/(comp+wire_solo) = 1364/756 = 1.80 (容差 1%)

## 纪律

1. 任何审计出现预设分支外的值，禁止自行决策，必须上报
2. 公式单实现（metrics.py），禁止复制粘贴
3. 配置集中（config.yaml），代码内禁止字面量
4. solo 不变量门禁期望值按锚点语义标注
5. workload profile 变更视为一级变更，必须在 HANDOFF 记录并触发门禁重跑

## 语义钉 #5：聚合语义与物理合理性约束（2026-07-24 裁定）

### 5a. attain_bw 是 job 级 spine 需求，sim 不建模 per-NIC 上限

**证据**：ViT-Base dp=2 solo alloc_bw=466G 被 sim 实际执行（per-flow 233G > 100G NIC 上限仍全速推进）——sim 的带宽分配仅在 spine link 层面建模，不检查 per-NIC 物理上限。

**后果**：
- sim 允许的场景不等于物理可行场景
- 场景设计必须自行维护物理合理性约束

### 5b. 场景设计约束：per_flow_attain = attain/dp ≤ 100G

**性质**：物理合理性约束，非 sim 约束。

**生效范围**：场景设计阶段——选取 (model, dp, ci) 组合时必须检查 per_flow_attain。超过 100G 的组合（如 ViT-Base dp=2 ci=1.5 的 222G/flow）在物理床上无法复现，禁止进入实验设计。

**不影响**：sim 内部执行——sim 不建模 per-NIC 上限，约束由场景设计者自行保证。

---

## 冻结令清单

以下文件/参数的任何变更需显式批准并在 HANDOFF 记录：

| 文件/参数 | 冻结范围 | 备注 |
|---|---|---|
| `config.yaml` | 全文 | 唯一配置源 |
| `config.yaml → frozen.*` | overhead_factor=1.3, overlap_factor=0.85, K=2.0 | 锚点冻结 |
| `config.yaml → topology.*` | k=4, host=100G, spine=400G | 16-node 约定 |
| `config.yaml → v2_anchor_workload` | 24 job 定义 | 门禁复现基准 |
| `config.yaml → semantics_version` | "anchor-v2" | 语义版本号 |
| `longliu_sim/utils/metrics.py` | SAS/target 公式 | 公式单实现 |
| `longliu_sim/network/topology.py` | 路由/链路模型 | 严禁修改 |
