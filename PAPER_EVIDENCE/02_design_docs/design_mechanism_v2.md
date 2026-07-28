# Mechanism v2 设计文档（修订版，2026-07-24）

## 0. 模型审计关键发现

### 迭代时间公式（从 simulator.py）

- `iter_ms = comp_ms + comm_actual_ms`
- `comm_actual_ms = flow.rem_bits / flow.rate_bps × 1000`
- `flow.rate_bps` 由策略的 `allocate` 方法决定

### overlap 0.85 的作用点

- **metrics.py**：`iter_solo = comp + comm_solo × (1 - overlap)`，缩短通信时间
- **simulator.py**：仅影响下一轮迭代开始时间，不影响当前迭代的记录时间
- **关键**：迭代时间记录的是实际的 barrier 时间，不包括 overlap 折扣

### attain_bw 计算口径（唯一合法）

```python
comm_solo_ms = mb_per_iter * 8 / host_bw_gbps  # ms
iter_solo_ms = comp_ms + comm_solo_ms * (1 - overlap_factor)
comm_budget_ms = ci * iter_solo_ms - comp_ms
attain_bw_gbps = bits_per_iter / (comm_budget_ms * 1e-3) / 1e9
```

### Σattain_bw 汇总

| ci 配置 | Σattain_bw (Gbps) | @1000G | @800G |
|---------|-------------------|--------|-------|
| 1.3/2.0 | 1651 | 1.65× | 2.06× |
| 2.0/3.0 | 879 | 0.88× | 1.10× |

---

## 1. 新工作点（ci=2.0/3.0）

### attain_bw 逐 job（ci=2.0/3.0）

| jid | model | dp | ci | attain_bw (Gbps) |
|-----|-------|----|----|------------------|
| J3 | LLaMA-2-13B | 8 | 2.0 | 231 |
| J2 | LLaMA-2-7B | 8 | 2.0 | 237 |
| J1 | BERT-Large | 2 | 2.0 | 86 |
| J5 | LLaMA-2-13B | 8 | 3.0 | 138 |
| J0 | T5-11B | 8 | 3.0 | 145 |
| J4 | BERT-Large | 4 | 3.0 | 25 |
| J6 | ViT-Base | 2 | 3.0 | 17 |

**Σattain_bw = 879 Gbps**
**@1000G 负载 = 0.88×（可达）**

### 负载阶梯

| spine_bw | Σattain / capacity | 可行性 |
|----------|--------------------|--------|
| 1200G | 0.73× | 充裕 |
| 1000G | 0.88× | 甜点 |
| 800G | 1.10× | 轻微超订 |
| 630G | 1.40× | 中度超订 |

---

## 2. Mechanism v3 设计

### 核心改进

1. **gap_i = max(0, attain_bw_i − bw_i)**
   - attain_bw 仅考虑通信时间，不含 comp
   - 与 D1G 的 pace-demand（含 comp）不同

2. **类映射：固定 log2 带**
   - gap > 1,2,4,8,16,32,64 Gbps → P0-P6
   - 与硬件 DSCP 类对齐，无量纲常数

3. **分配封顶 + 水填充**
   - `bw_alloc = min(share, attain_bw)`
   - 盈余按 gap 权重水填充

4. **starvation-free**
   - floor_w 保底（默认 2.0）
   - 封顶不阻塞低权 job 的 min-share

### 预期效果（@1000G）

- P1/P2/P3 全达标（Σattain_premium = 554 Gbps < 1000G）
- S1/S2/S3/S4 有界降级（sas ≈ 0.67，S-cont-cap ≥ 0.5）
- starv = 0%

---

## 3. 预登记判定

### @1000G（ci=2.0/3.0）

| 策略 | P-attn | P-cap | S-cont-cap | starv |
|------|--------|-------|------------|-------|
| Fair | 0% | 0.65 | - | 0% |
| CRUX | ≤33% | 0.70 | - | 0% |
| v3 | **100%** | **1.0** | **≥0.5** | **0%** |

**任一条不成立，报数据不报结论。**

---

## 4. 执行序列（修订：主评估点 @800G）

### 负载阶梯（按 attain 口径）

| spine_bw | Σattain / capacity | 口径 | 备注 |
|----------|--------------------|------|------|
| 1200G | 0.73× | attain | 充裕 |
| 1000G | 0.88× | attain | 轻载 |
| **800G** | **1.10×** | **attain** | **主评估点（真边界）** |
| 630G | 1.39× | attain | 重载 |

**旧 pace-demand 口径标注作废**：原 1.02/1.21/1.54× 标注基于含 comp 的 pace-demand，不反映真实 attainment 压力。

### 预登记判定（@800G，ci=2.0/3.0）

| 策略 | P-attn | P-cap | S-cont-cap | starv |
|------|--------|-------|------------|-------|
| Fair | 0% | ~0.65 | - | 0% |
| CRUX | ≤33% | ~0.70 | - | 0% |
| v3 | **>0%** | **>0.8** | **≥0.5** | **0%** |

**验收标准**：
- P 站稳高类（P5/P6）
- S 低类（P2/P3）
- 封顶生效（bw ≤ attain_bw）
- 无振荡

---

## 1. 主键：绝对带宽缺口 gap_i

### 定义

```
gap_i = max(0, demand_i − bw_i)

demand_i = bits_per_iter / (ci · iter_solo)    [Gbps]
bw_i     = 分配器上一轮分配给 job i 的瓶颈带宽    [Gbps]
```

其中 `bits_per_iter = 2 × params × bytes_per_param`（AllReduce 总通信量）。

### 关键性质

- **绝对缺口携带 tier 信息**：premium 的 `demand_i` 天然大于 standard（因 ci 更小，denominator 更小）。公平份额下 P1 缺口 147.6 Gbps vs S1 缺口 42.9 Gbps，天然 3.4:1 排序。
- **禁止使用相对缺口** `gap/demand`：这恒等于 `1 − sas`，就是 π 本身，tier 信息立即消失。
- **gap=0 标识达标**：此时 job 落在 low-priority floor，出让带宽。

### 物理可实现性

仿真内使用连续 gap 比例权重做探索；论文机制描述与最终验证使用 **7 类对数带量化**：

```
DSCP quantize(gap):
  gap ≤ 0    → P0 (floor, low weight)
  gap ∈ (0, G0/8) → P1
  gap ∈ (G0/8, G0/4) → P2
  gap ∈ (G0/4, G0/2) → P3
  gap ∈ (G0/2, G0)   → P4
  gap ∈ (G0, 2·G0)   → P5
  gap > 2·G0         → P6
```

量化后复用既有 DWRR 权重表 `{1,2,4,8,16,32,64}`，测试床 DSCP 类可直接映射。

---

## 2. 权重映射：gap 比例 + 保底地板

### 连续版（仿真用）

```
w_i = max(floor_w, gap_i / G0)
```

- `w_i`：job i 在类内分配中的权重（跨类 DWRR 仍然按照 DSCP 类权重表分配）。
- `floor_w ∈ {1, 2}`：gap=0 时的保底权重。starvation-free 由 floor + work-conserving 联合保证。
- `G0 ∈ {10, 25, 50} Gbps`：缺口归一化因子。G0 越小，缺口灵敏度越高。

### 与改进 1 的关系

非对称 exp 映射（`exp(α·π)`）**仅作为组合件 D1G+asym 消融**，不单独主用。原因见 §4。

---

## 3. 类内调度

类间：保留既有 7 类 DWRR（权重表 `{1,2,4,8,16,32,64}`），work-conserving。

类内：按 **gap_i 比例权重**分配，即：
- 类内带宽按 `w_i = max(floor_w, gap_i / G0)` 比例切分。
- `clip_ratio` 仍在（同权依赖已经禁了，仅作安全阀）。

### 与 D1 的差异对比

| 维度 | D1 | D1G |
|------|-----|------|
| 调度键 | π = avg/T_target − 1 | gap = demand − bw |
| tier 感知 | 无（π 不分 tier） | 有（demand 含 ci） |
| class 映射 | π 阈值映射 | gap 对数带映射 |
| 保底 | clip_ratio（顶层限制） | floor_w（底层保底） |
| work-conserving | 类间 | 类间+类内 floor |

---

## 4. 禁止项与暂缓项

### 改进 1（非对称 exp）——仅作组合件

轨迹证据：S1 的 π≈1.0 高于 P1 的 π≈0.2。`exp(α·π)` 下 α=4 时 S1 权重是 P1 的 e^{4×0.8}≈**25 倍** → 斜率越陡 S1 碾压 P1 越狠。因此：

- **禁止**单独主用改进 1。
- **允许**作为 D1G+asym 组合消融（gap 键先纠正排序，asym 作为权重的附加整形器）。

### 改进 3（remaining_bytes 联合键）——暂缓

SRPT 逻辑需要 job 有终点。主场景 DDP job 是 600s 持续流，remaining_bytes 对无限流无良定义。
正确场景：Lingjun trace（有限 job，有明确退出点）。列入后续。

---

## 5. 成功标准（双边判定矩阵）

| 负载点 | Premium 侧预期 | Standard 侧约束 |
|--------|---------------|-----------------|
| 1.02× (1000G) | P-attn ≥67% | starv=0 |
| 1.21× (800G)  | P-attn ≥67% | S-cont-cap ≥0.35, starv=0 |
| 1.54× (630G)  | P-attn ∈[33%,67%]（禁止 100%，那是加冕） | starv=0, 无近饿死个体（sas<0.1） |

**物理约束**：@1.54×（630G），三 premium 全达标的代价是 624 Gbps，留给四个 standard 只剩 6G（sas≈0.02，近饿死）。因此重载点必须限制 premium attainment。

---

## 6. 验证序列

### Phase 1：快检（当前任务）
1. 实现 D1G（gap 比例权重版）为 `dwrr.py` 新策略类 `LongLiuDWRRGap`，不动 D1 既有代码。
2. D1G × 1 seed @1.21× 轨迹跑 → 验证 P1/P2 持高权、S1/S2 居低权、gap→0 后权重回地板。

### Phase 2：3 seeds 判决性实验
3. Fair / CRUX / LongLiu-SP / D1G × 3 seeds @1.21× → 对比 P-cap / P-attn / S-cont-cap。
4. 达标后扩展至三负载点。

### Phase 3：消融与扫描
5. G0 扫描（10/25/50 Gbps）、floor_w 扫描（1/2）、D1G+asym 组合。
6. 动态 ci 变体（t=300s 两 job 互换 ci）— 对 CRUX 的致命实验。

---

## 7. CRUX 竞争力归因

本场景 premium = 大模型 = 计算密集。CRUX 的计算强度键（`params / bits_per_iter`）恰好与 tier 正相关。**这是场景的相关性巧合，不是 D1 的失败。**

真正杀 CRUX 的实验：**动态 ci 变体**（t=300s 两个 job 互换 SLO 等级，即物理 V4/V6 的仿真版）。静态键在 SLO 变化后失明，gap 键会跟着 π/demand 翻转。此变体列入后续队列。

---

## 8. 禁止事项

- 禁止使用相对缺口（gap/demand ≡ 1−sas，tier 信息消失）。
- 禁止改动 D1 既有类（`LongLiuDWRR`、`LongLiuDWRRFair`）。
- 禁止发明常数（G0、floor_w 必须扫描，不许硬编码最优值）。
- 禁止单变量改进 1 作为主策略运行。
- 禁止 git clean。
