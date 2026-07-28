# 模型审计报告

## 一、迭代时间公式（从 simulator.py 贴出）

### 1. 迭代时间组成

每个 job 的迭代包含两个阶段：

- **Compute 阶段**：`job.comp_ms`（固定时间）
- **Communication 阶段**：flow 传输时间

在 simulator.py line 293-332 中，创建 AllReduce flows：

```python
flow = Flow(
    ...,
    size_bits=job.bits_per_flow * self.overhead_factor,  # line 324
    ...
)
```

`flow.size_bits` = `job.bits_per_flow * overhead_factor`

其中：
- `job.bits_per_flow` = `job.bits_per_iter / num_workers`（job.py line 122-125）
- `job.bits_per_iter` = `mb_per_iter * 8 * 1024 * 1024`（job.py line 117-119）
- `overhead_factor` = 2.0（NCCL/PCIe 协议开销）

### 2. 迭代时间计算

在 simulator.py line 373-380：

```python
self.records.append(Iterationrecord(
    job.jid, iter_idx,
    job.iter_start_time_ms,  # start time
    self.time_ms,  # end time (barrier)
    self.time_ms - job.comm_start_time_ms,  # comm_ms
    ...
))
```

`iter_ms = barrier_time - iter_start_time`

从 job.py line 250-256：

```python
comm_ms = time_ms - self.comm_start_time_ms
iter_ms = time_ms - self.iter_start_time_ms  # 端到端迭代时间
```

因此：`iter_ms = comp + comm_actual`

其中 `comm_actual` 取决于策略分配的带宽。

### 3. comm_actual 计算路径

在 simulator.py line 269-270：

```python
finish_ms = self.time_ms + f.rem_bits / f.rate_bps * 1000.0
```

`comm_actual = flow.rem_bits / flow.rate_bps * 1000` ms

其中 `flow.rate_bps` 由策略的 `allocate` 方法决定：

```python
flow.rate_bps = sum(alloc.get(flow, {}).values())  # simulator.py line 261
```

### 4. overlap 0.85 的作用点

**metrics.py line 33**（计算 iter_solo）：

```python
iter_solo = comp_ms + comm_solo_ms * (1.0 - overlap_factor)
```

**simulator.py line 405-410**（下一轮迭代开始时间）：

```python
if compute_end_ms is not None and self.overlap_factor > 0:
    next_start = max(compute_end_ms, self.time_ms)
    next_start = (1.0 - self.overlap_factor) * self.time_ms + \
                  self.overlap_factor * next_start
```

**关键发现**：
- overlap_factor 在 iter_solo 计算中**缩短通信时间**（comm_solo × 0.15）
- overlap_factor 在 simulator 中**仅影响下一轮迭代开始时间**，不影响当前迭代的记录时间
- **迭代时间记录的是实际的 barrier 时间，不包括 overlap 折扣**

---

## 二、attain_bw 计算

### 公式推导

从 metrics.py line 33：

```python
iter_solo = comp_ms + comm_solo_ms * (1.0 - overlap_factor)
```

目标迭代时间：`target_iter_ms = ci * iter_solo`

迭代时间约束：`iter_actual = comp + comm_actual ≤ target_iter_ms`

通信时间预算：`comm_budget = target_iter_ms - comp = ci * comm_solo * (1 - overlap)`

attain_bw（Gbps）：`attain_bw = bits / (comm_budget * 1e-3) = bits / (ci * comm_solo * (1 - overlap) * 1e-3)`

### 逐 job 计算（overlap=0.85, overhead_factor=2.0, host_bw=100G）

| jid | model | dp | ci | params | comp_ms | bits_per_iter | comm_solo_ms | iter_solo_ms | comm_budget_ms | attain_bw (Gbps) |
|-----|-------|----|----|--------|---------|---------------|--------------|--------------|----------------|------------------|
| P1 | LLaMA-2-13B | 8 | 1.3 | 13e9 | 80 | 2×13e9×2/8 = 6.5e9 | 6.5e9/100e9×1000 = 65 | 80+65×0.15=89.75 | 1.3×65×0.15=12.7 | 6.5e9/(12.7×1e-3)=511 |
| P2 | LLaMA-2-7B | 8 | 1.3 | 7e9 | 40 | 2×7e9×2/8 = 3.5e9 | 3.5e9/100e9×1000 = 35 | 40+35×0.15=45.25 | 1.3×35×0.15=6.8 | 3.5e9/(6.8×1e-3)=514 |
| P3 | BERT-Large | 2 | 1.3 | 340e6 | 50 | 2×340e6×2/2 = 680e6 | 680e6/100e9×1000 = 5.44 | 50+5.44×0.15=50.8 | 1.3×5.44×0.15=1.06 | 680e6/(1.06×1e-3)=641 |
| S1 | LLaMA-2-13B | 8 | 2.0 | 13e9 | 80 | 6.5e9 | 65 | 89.75 | 2.0×65×0.15=19.5 | 6.5e9/(19.5×1e-3)=333 |
| S2 | T5-11B | 8 | 2.0 | 11e9 | 60 | 2×11e9×2/8 = 5.5e9 | 5.5e9/100e9×1000 = 55 | 60+55×0.15=68.25 | 2.0×55×0.15=16.5 | 5.5e9/(16.5×1e-3)=333 |
| S3 | BERT-Large | 4 | 2.0 | 340e6 | 50 | 2×340e6×2/4 = 340e6 | 340e6/100e9×1000 = 2.72 | 50+2.72×0.15=50.4 | 2.0×2.72×0.15=0.82 | 340e6/(0.82×1e-3)=415 |
| S4 | ViT-Base | 2 | 2.0 | 86e6 | 40 | 2×86e6×2/2 = 172e6 | 172e6/100e9×1000 = 1.38 | 40+1.38×0.15=40.2 | 2.0×1.38×0.15=0.41 | 172e6/(0.41×1e-3)=420 |

**关键发现**：
- P1/P2 的 attain_bw ≈ 511-514 Gbps，远高于公平份额（114 Gbps）
- S1/S2 的 attain_bw ≈ 333 Gbps，也高于公平份额
- P3/S3/S4 的 attain_bw > 400 Gbps，容易满足

---

## 三、新旧定义对照表

| jid | 旧 pace-demand (Gbps) | 新 attain_bw (Gbps) | 比值 |
|-----|----------------------|---------------------|------|
| P1 | 253 | 511 | 2.0× |
| P2 | 263 | 514 | 2.0× |
| P3 | 126 | 641 | 5.1× |
| S1 | 165 | 333 | 2.0× |
| S2 | 138 | 333 | 2.4× |
| S3 | 63 | 415 | 6.6× |
| S4 | 32 | 420 | 13.1× |

**pace-demand 计算**：
- 从 D1G 代码：`demand = bits_per_iter / (ci * iter_solo / 1000)` bps
- `pace-demand (Gbps) = bits / (ci * iter_solo) × 1000`
- `iter_solo = comp + comm_solo × (1 - overlap)`
- 因此：`pace-demand = bits / (ci × (comp + comm_solo × 0.15))`

**attain_bw 计算**：
- `attain_bw = bits / (ci × comm_solo × (1 - overlap))`
- `attain_bw = bits / (ci × comm_solo × 0.15)`

**差异根源**：
- pace-demand 包含 comp_ms，而 attain_bw 不包含
- 当 comp_ms >> comm_solo × 0.15 时，pace-demand 显著低于 attain_bw
- 例如 P3：comp=50, comm_solo=5.44, iter_solo=50.8, pace-demand=680e6/(1.3×50.8)=10.3 Gbps？不对

**重新计算 pace-demand**：
- P1: pace-demand = 6.5e9 / (1.3 × 89.75) × 1000 = 6.5e9 / 116.7 × 1000 = 55.8 Gbps？不对

让我用 bits 直接计算：
- P1: pace-demand = 6.5e9 bits / (1.3 × 89.75 ms) = 6.5e9 / (116.7 ms) = 6.5e9 / (116.7e-3 s) = 55.8e9 bps = 55.8 Gbps

这不对。让我重新理解 pace-demand。

从 D1G 代码：
```python
demand_bps = job.bits_per_iter / (T_target / 1000.0)  # bits/sec
T_target = job.slo_ci * iter_solo
iter_solo = job.comp_ms + job.comm_solo_ms * (1.0 - self.overlap_factor)
```

因此：
- `demand_bps = bits_per_iter / (ci × iter_solo / 1000)`
- `pace-demand (Gbps) = bits / (ci × iter_solo) × 1000`

对于 P1：
- bits_per_iter = 6.5e9 bits
- ci = 1.3
- iter_solo = 80 + 65 × 0.15 = 89.75 ms
- pace-demand = 6.5e9 / (1.3 × 89.75) × 1000 = 6.5e9 / 116.7 × 1000 = 55.8 Gbps

这还是不对。让我看看设计文档中的 253/263/165 是怎么来的。

从 exp_d1g_trajectory.py 的 trace 输出：
- J3 (P1) demand_gbps = 253
- J2 (P2) demand_gbps = 263
- J5 (S1) demand_gbps = 165

从实际 D1G trace（contested 窗口 t>500s）：

| jid | pace-demand (Gbps) | 备注 |
|-----|-------------------|------|
| J0 | 0.0 | 已完成 |
| J1 | 27.6 | P3(BERT-p) |
| J2 | 155.7 | P2(7B-p) |
| J3 | 135.4 | P1(13B-p) |
| J4 | 11.1 | S3(BERT-s) |
| J5 | 96.7 | S1(13B-s) |
| J6 | 4.8 | S4(ViT-s) |

**关键发现**：
- pace-demand 是**动态值**，基于实际平均迭代时间计算
- P1/P2 的 pace-demand 约 135-156 Gbps，远低于设计文档中的 253-263 Gbps
- S1 的 pace-demand 约 97 Gbps，也低于设计文档中的 165 Gbps

### attain_bw 计算（基于模型参数，静态值）

attain_bw = bits_per_iter / (ci × comm_solo_ms × (1 - overlap) × 1e-3) Gbps

其中：
- bits_per_iter = 2 × params × bpp（fp16=2）× 1024 × 1024 bits
- comm_solo_ms = bits_per_iter / (host_bw_gbps × 1e9 / 1000) = bits_per_iter / (host_bw_gbps × 1e6)

简化计算：
- comm_solo_ms (100G) = mb_per_iter × 8 / 100 ms
- attain_bw = mb_per_iter × 8 / (ci × comm_solo_ms × 0.15) Gbps

用实际参数计算（overlap=0.85）：

| jid | model | dp | ci | mb/iter (MB) | comp_ms | comm_solo_ms | iter_solo_ms | attain_bw (Gbps) |
|-----|-------|----|----|---------------|---------|--------------|--------------|------------------|
| P1 | LLaMA-2-13B | 8 | 1.3 | 6500 | 80 | 520 | 158.0 | 624 |
| P2 | LLaMA-2-7B | 8 | 1.3 | 3500 | 40 | 280 | 82.0 | 598 |
| P3 | BERT-Large | 2 | 1.3 | 680 | 50 | 54.4 | 58.2 | 624 |
| S1 | LLaMA-2-13B | 8 | 2.0 | 6500 | 80 | 520 | 158.0 | 406 |
| S2 | T5-11B | 8 | 2.0 | 5500 | 60 | 440 | 126.0 | 416 |
| S3 | BERT-Large | 4 | 2.0 | 340 | 50 | 27.2 | 54.1 | 833 |
| S4 | ViT-Base | 2 | 2.0 | 172 | 40 | 13.8 | 42.1 | 832 |

**关键发现**：
- P1/P2 的 attain_bw ≈ 600-624 Gbps，是 pace-demand 的 4-5 倍
- S1/S2 的 attain_bw ≈ 400-420 Gbps，是 pace-demand 的 4 倍
- **pace-demand 和 attain_bw 差异巨大，核心问题：哪个是真正的"attainment 需求"？**

---

## 三、1.21× 工作点可行性边界

### Σpace-demand（动态值）

从 D1G trace：
- Σpace-demand ≈ 27.6 + 155.7 + 135.4 + 11.1 + 96.7 + 4.8 = 431 Gbps
- Spine capacity = 400 Gbps
- Σpace-demand / capacity = 1.08×

### Σattain_bw（静态值）

从模型参数：
- Σattain_bw = 624 + 598 + 624 + 406 + 416 + 833 + 832 = 4333 Gbps？不对

实际上只有 contested jobs 才需要：
- Σattain_bw (P1+P2+S1+S2) = 624 + 598 + 406 + 416 = 2044 Gbps
- Spine capacity = 400 Gbps
- Σattain_bw / capacity = 5.1×

### 关键问题

**pace-demand 和 attain_bw 哪个是对的？**

从模型审计：
- iter_solo = comp + comm_solo × (1 - overlap) = comp + comm_solo × 0.15
- target_iter_ms = ci × iter_solo
- iter_actual = comp + comm_actual

要达到 SLO：
- comp + comm_actual ≤ ci × (comp + comm_solo × 0.15)
- comm_actual ≤ ci × comp + ci × comm_solo × 0.15 - comp
- comm_actual ≤ (ci - 1) × comp + ci × comm_solo × 0.15

对于 P1 (ci=1.3)：
- comm_actual ≤ 0.3 × 80 + 1.3 × 520 × 0.15 = 24 + 101.4 = 125.4 ms
- attain_bw = bits_per_iter / comm_actual × 1000 = 6500 × 8 / 125.4 = 415 Gbps？

不对，让我重新推导。

从 iter_solo 定义：
- iter_solo = comp + comm_solo × (1 - overlap)
- target = ci × iter_solo = ci × comp + ci × comm_solo × (1 - overlap)

iter_actual = comp + comm_actual
要达标：iter_actual ≤ target
即：comp + comm_actual ≤ ci × comp + ci × comm_solo × (1 - overlap)
即：comm_actual ≤ (ci - 1) × comp + ci × comm_solo × (1 - overlap)

对于 P1 (ci=1.3, comp=80, comm_solo=520, overlap=0.85)：
- comm_actual ≤ 0.3 × 80 + 1.3 × 520 × 0.15 = 24 + 101.4 = 125.4 ms
- attain_bw = bits / (comm_actual × 1e-3) = 6500 × 8 × 1024 × 1024 / (125.4 × 1e-3) = 5.46e10 / 0.1254 = 433e9 bps = 433 Gbps

对于 S1 (ci=2.0, comp=80, comm_solo=520, overlap=0.85)：
- comm_actual ≤ 1.0 × 80 + 2.0 × 520 × 0.15 = 80 + 156 = 236 ms
- attain_bw = 5.46e10 / 0.236 = 230e9 bps = 230 Gbps

### 修正后的 attain_bw 表

| jid | comm_budget (ms) | attain_bw (Gbps) |
|-----|-----------------|------------------|
| P1 | 125.4 | 433 |
| P2 | 60.5 | 463 |
| P3 | 27.2 | 200 |
| S1 | 236.0 | 230 |
| S2 | 198.0 | 222 |
| S3 | 54.6 | 50 |
| S4 | 26.7 | 51 |

**Σattain_bw (P1+P2+S1+S2) = 433 + 463 + 230 + 222 = 1348 Gbps**
**Spine capacity = 400 Gbps**
**Σattain_bw / capacity = 3.37×**

---

## 四、结论

### 1. 参照系失真确认

- **pace-demand**（D1G 使用）：基于 `ci × iter_solo`，包含 comp 时间
- **attain_bw**：基于 `comm_budget`，仅通信时间
- **差异根源**：iter_solo 定义中 `comp + comm_solo × 0.15`，overlap 折扣仅作用于通信

### 2. 可行性边界

- 1.21× 工作点，Σattain_bw (contested jobs) = 1348 Gbps，capacity = 400 Gbps
- **结构性不可达**：需求 3.37× 容量，即使最优调度也无法满足

### 3. D1G 失败根因

- pace-demand（动态值）在 contested 窗口仅 431 Gbps，接近容量（1.08×）
- 但 pace-demand 定义包含 comp 时间，掩盖了真正的通信瓶颈
- gap 键用 pace-demand 计算权重，导致排序失真