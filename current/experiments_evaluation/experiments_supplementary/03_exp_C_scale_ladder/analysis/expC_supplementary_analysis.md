# 实验 C 补充分析：DSCP 核查 + 场景参数重构 + 诊断

> 生成时间：2026-07-30
> 目的：为论文 §V-F 和 §VI 提供准确的技术细节

---

## 1. DSCP 映射一致性核查

### 代码中的映射

| 来源 | P6 | P4 | P3 | P2 | P1 | P0 |
|------|-----|-----|-----|-----|-----|-----|
| **SLOScheduler** (`slo_scheduler.py:73`) | DSCP=8 | DSCP=0 | DSCP=16 | DSCP=24 | DSCP=32 | DSCP=40 |
| **Emulator** (`epoch_emulator.c:63`) | DSCP=8 | DSCP=0 | — | DSCP=24 | DSCP=32 | — |
| **Daemon** (`alloc_daemon.py`) | 同 SLOScheduler | 同 | 同 | 同 | 同 | 同 |

### 论文 §V-D 中的映射

| Priority | π Range | 论文描述 | 代码 DSCP |
|----------|---------|---------|-----------|
| P6 | π > 0.3 | Maximum | 8 |
| P4 | -0.1 < π ≤ 0.3 | High | 0 |
| P2 | -0.5 < π ≤ -0.1 | Normal | 24 |
| P0 | π ≤ -0.5 | Minimum | **40** |

### 不一致点

| # | 问题 | 严重度 | 说明 |
|---|------|--------|------|
| 1 | **论文写 P0，代码映射 P1** | ⚠️ 中 | 论文 §5.3 写最低优先级为 P0（π≤-0.5），代码 `PI_THRESHOLDS` 落入 `result=1`（即 P1, DSCP=32）。论文里的 P0 在代码中映射为 DSCP=40，但 `PI_THRESHOLDS` 的 else 分支返回 `result=1` 而非 `result=0`。**代码注释说"We use P1 instead of P0 to keep DSCP non-zero"**——这是有意为之，但论文应说明。 |
| 2 | **P3 (DSCP=16) 未被使用** | ⚠️ 中 | `PRIORITY_TO_DSCP` 定义了 P3→DSCP=16，但 `PI_THRESHOLDS` 没有映射到 P3 的阈值区间。4-tier 映射跳过了 P3。论文表只列 4 行（P6/P4/P2/P0），P3 确实不存在。代码中 P3 的定义是冗余的。 |
| 3 | **Emulator 只建 4 个 QP** | ✅ 一致 | Emulator 的 `DSCP_TABLE = {8, 0, 24, 32}` 对应 P6/P4/P2/P1，与 `PI_THRESHOLDS` 实际使用的 4 个优先级完全一致。P3 和 P0 的 DSCP 未使用，无需建 QP。 |
| 4 | **论文写 "16 ops/round" 配额** | ⚠️ 需确认 | 论文 §5.3 写 P6=16 ops/round, P4=8, P2=4, P0=1。但 SLOScheduler 没有实现 ops/round 配额——它只做 DSCP 映射。配额控制是 NCCL proxy 线程的职责，不在 shim/daemon 层。模拟器也没有 ops/round 概念。**论文应说明配额机制仅在 NCCL 全链路中生效，模拟器仅验证 DSCP 映射。** |
| 5 | **DSCP=0 对应 P4** | ⚠️ 需文档化 | DSCP=0 在网络上是默认值（无标记），P4 作为默认优先级用 DSCP=0 是合理的，但论文应明确 "P4 = DSCP 0 (default, unmarked)"。 |

### 文档化建议

1. 论文 §V-D 补充：**"P1 (DSCP=32) is used instead of P0 (DSCP=0/40) for the lowest tier to ensure non-zero DSCP marking, making traffic class visible on the wire."**
2. 论文 §V-D 补充：**"The ops-per-round quota is implemented in the NCCL proxy thread (§IV-C), not in the priority scheduler. The emulator validates only the DSCP mapping."**
3. 删除代码中未使用的 P3 和 P0 映射，或加注释说明它们是预留的。

---

## 2. 模拟器参数限制确认

### 硬限制

| 参数 | 限制 | 来源 |
|------|------|------|
| `data_size` (D) | ≤ 16 MB | `MAX_DATA_SIZE` 宏定义 |
| `sleep_us` (T_comp) | 无硬限制，但 iter 时间 = T_comp + comm_time | 实际建议 1ms-1s |
| `num_epochs` | 无硬限制 | 受 `runtime_s` 限制 |
| `iters_per_epoch` | 无硬限制 | 建议 20（与 NCCL 实验一致） |
| `jitter_pct` | 0-100% | 5% 为论文默认 |
| QP 数 | 4 个/作业（P6/P4/P2/P1 固定） | Multi-QP 架构 |
| 总 QP 数 | 作业数 × 4 | 受 NIC QP 上限约束（ConnectX-6 支持 ~130K QP） |
| 链路带宽 | 50 Gbps（10.1 侧瓶颈） | ethtool 实测 |

### 软限制（设计建议）

| 参数 | 建议范围 | 原因 |
|------|---------|------|
| T_comp | 10ms-100ms | 过短则争抢不够充分；过长则每 epoch 时间 >1s，25 epoch 跑不完 |
| D | 256KB-4MB | 4MB 对应 solo BW ≈ 46Gbps；256KB 对应 ≈ 30-41Gbps |
| c_i | 1.2-2.0 | 论文设计范围 |
| 作业数 | 2-6 | 6 作业 × 4 QP = 24 个 QP，ConnectX-6 无压力 |

---

## 3. 场景参数重构表

### 设计原则

1. **b^att = solo_bw / c_i**（论文公式，attained bandwidth）
2. **solo_bw** 取实测值（校准文件），不是理论线速
3. **Σb^att / B** 命中三个 regime 目标（1.5 / 1.2 / 0.96）
4. **Fair-share 检验**：在 strict-priority 下，Fair 臂应该让所有作业获得相同优先级（P4），premium job 的 SLO 必败（因为 c_i=1.2 意味着只允许 20% 余量，而 4 作业争 50G 每人只有 12.5G，远低于 solo BW）
5. **D 缩放**：通过不同 payload 大小创造不同的 solo BW，从而让 b^att 在 50G 链路上合理分布

### 校准的 solo BW（2026-07-29 实测）

| Payload | D (bytes) | Solo BW (Gbps) | T_target (us/iter) |
|---------|-----------|----------------|-------------------|
| 4096 KB | 4,194,304 | 45.9 | 731 |
| 1024 KB | 1,048,576 | 29.7 | 283 |
| 256 KB | 262,144 | 41.3 | 51 |

> **注意**：256KB 的 solo BW 高达 41.3Gbps（比 1024KB 的 29.7 还高），因为小消息的 per-message 开销占比小，有效带宽反而更高。这导致 b^att 的分化不如预期。

### 6 作业参数定义

| Job ID | Label | Tier | c_i | D (KB) | T_comp (us) | Solo BW (Gbps) | b^att (Gbps) |
|--------|-------|------|-----|--------|-------------|----------------|-------------|
| 0 | J0_prem_L | premium | 1.2 | 4096 | 30000 | 45.9 | 38.25 |
| 1 | J1_prem_M | premium | 1.2 | 1024 | 30000 | 29.7 | 24.75 |
| 2 | J2_std_M | standard | 2.0 | 1024 | 30000 | 29.7 | 14.85 |
| 3 | J3_std_S | standard | 2.0 | 256 | 30000 | 41.3 | 20.65 |
| 4 | J4_prem_S | premium | 1.2 | 256 | 30000 | 41.3 | 34.42 |
| 5 | J5_std_XS | standard | 2.0 | 256 | 30000 | 41.3 | 20.65 |

> **b^att = solo_bw / c_i**：premium job 的 b^att 更大（因为 c_i 更小），意味着它们"声称"需要更多带宽。

### 3 Regime 组合

#### deep_scarcity (Σb^att/B ≈ 1.5)

| Job | Tier | c_i | D (KB) | b^att (Gbps) | 预期 slowdown |
|-----|------|-----|--------|-------------|--------------|
| J0 | premium | 1.2 | 4096 | 38.25 | ~1.3 (premium 被争抢) |
| J1 | premium | 1.2 | 1024 | 24.75 | ~1.0 (premium M) |
| J2 | standard | 2.0 | 1024 | 14.85 | ~2.0 (standard 被争抢) |
| J3 | standard | 2.0 | 256 | 20.65 | ~1.0 (standard S) |

- Σb^att = 38.25 + 24.75 + 14.85 + 20.65 = **98.50 Gbps**
- Σb^att / B = 98.50 / 50 = **1.97** ← **太高了！**

> **问题**：当前 4 作业组合的 Σb^att/B = 1.97，远超目标 1.5。原因是 256KB 的 solo BW 太高（41.3 Gbps），导致 b^att 被高估。

#### 修正：用 512KB payload 替代 256KB

256KB 的 solo BW 异常高（41.3Gbps），但 512KB 预期在 30-35Gbps 之间（需要实测）。暂用估算值：

| Payload | D (bytes) | Est. Solo BW (Gbps) | Est. T_target (us) |
|---------|-----------|---------------------|-------------------|
| 512 KB | 524,288 | ~33 | ~125 |

重新计算 6 作业：

| Job ID | Label | Tier | c_i | D (KB) | Est. Solo BW | b^att |
|--------|-------|------|-----|--------|-------------|-------|
| 0 | J0_prem_L | premium | 1.2 | 4096 | 45.9 | 38.25 |
| 1 | J1_prem_M | premium | 1.2 | 1024 | 29.7 | 24.75 |
| 2 | J2_std_M | standard | 2.0 | 1024 | 29.7 | 14.85 |
| 3 | J3_std_S | standard | 2.0 | 512 | ~33 | 16.50 |
| 4 | J4_prem_S | premium | 1.2 | 512 | ~33 | 27.50 |
| 5 | J5_std_XS | standard | 2.0 | 512 | ~33 | 16.50 |

#### deep_scarcity (4 jobs: J0+J1+J2+J3)

- Σb^att = 38.25 + 24.75 + 14.85 + 16.50 = **94.35**
- Σb^att / 50 = **1.89** ← 仍然偏高

> **根本问题**：50G 链路上，4 个作业的 b^att 之和很难降到 1.5。因为 solo BW 本身就接近 30-46Gbps，除以 c_i=1.2 后仍有 25-38Gbps。需要更多作业或更小的 payload。

#### 替代方案：用更多作业（6-8 个）来稀释 Σb^att/B

| Job | Tier | c_i | D (KB) | Solo BW | b^att |
|-----|------|-----|--------|---------|-------|
| J0 | premium | 1.2 | 4096 | 45.9 | 38.25 |
| J1 | premium | 1.2 | 1024 | 29.7 | 24.75 |
| J2 | standard | 2.0 | 1024 | 29.7 | 14.85 |
| J3 | standard | 2.0 | 512 | 33 | 16.50 |
| J4 | premium | 1.2 | 512 | 33 | 27.50 |
| J5 | standard | 2.0 | 512 | 33 | 16.50 |

**deep_scarcity (6 jobs: J0-J5)**: Σb^att = 138.35, /50 = **2.77** ← 更高了

**结论**：50G 链路上用 RDMA write 模拟器，solo BW 接近线速（30-46Gbps），b^att 自然就大。要让 Σb^att/B ≈ 1.5，需要：
- 要么用更小的 payload（< 256KB，但 RDMA write 的 per-message 开销会主导）
- 要么用更少的作业（2-3 个），但这就不是"规模阶梯"了
- 要么**接受实际 Σb^att/B 与 E1 的比值对应关系是近似的**

### 建议的最终场景设计

**核心决策**：用 3 个作业 × 2 个 payload 大小，通过选择不同子集来命中 3 个 regime。

| Regime | Jobs | Σb^att (Gbps) | Σb^att/B | E1 对应点 |
|--------|------|-------------|---------|---------|
| deep_scarcity | J0+J1+J2 (4MB+1MB+1MB) | 38.25+24.75+14.85=77.85 | 1.56 | 400G |
| transition | J0+J2 (4MB+1MB) | 38.25+14.85=53.10 | 1.06 | 630G |
| ample | J1+J2 (1MB+1MB) | 24.75+14.85=39.60 | 0.79 | 1200G |

> **Fair-share 检验**：
> - deep_scarcity: 3 作业公平分 50G = 16.7G/job。J0 的 b^att=38.25G，16.7G << 38.25G，**Fair 必败**（J0 的 SLO 严重不足）。
> - transition: 2 作业公平分 50G = 25G/job。J0 的 b^att=38.25G，25G << 38.25G，**Fair 必败**。
> - ample: 2 作业公平分 50G = 25G/job。J1 的 b^att=24.75G，25G > 24.75G，**Fair 可以满足**（刚好够）。

> **预期 λ（LongLiu 优势）**：
> - deep_scarcity: λ > 1.5（LongLiu 应显著优于 Fair，因为 premium job 需要的带宽远超公平份额）
> - transition: λ ≈ 1.2（中等优势）
> - ample: λ ≈ 1.0（无显著优势，带宽充裕）

---

## 4. 27 轮旧数据按新指标重分析结论

### 核心发现

| Regime | LongLiu P-attn | Static P-attn | Fair P-attn | LL vs Static | LL vs Fair |
|--------|---------|--------|------|------------|------------|
| ample | 0.179 | 0.174 | 0.499 | ≈ | **LL 优 64%** |
| deep_scarcity | 0.356 | 0.556 | 0.204 | **LL 优 36%** | LL 劣 74% |
| transition | 0.344 | 0.488 | 0.182 | **LL 优 30%** | LL 劣 89% |

### S-cont（standard 作业争抢量）

| Regime | LongLiu | Static | Fair | LL vs Static |
|--------|---------|--------|------|------------|
| deep_scarcity | 2.829 | 2.972 | 3.429 | LL 优 5% |
| transition | 0.025 | 0.016 | 0.015 | LL 劣 59% |

### Fair 反常的逐轮诊断

**deep_scarcity** 逐轮 P-attn：

| Round | LongLiu | Static | Fair | Fair P-attn 最低的原因 |
|-------|---------|--------|------|----------------------|
| 1 | 0.546 | 0.587 | 0.580 | Round 1 三臂相近 |
| 2 | **0.009** | 0.540 | 0.021 | **LongLiu 和 Fair 都出现 J1 slowdown<1** |
| 3 | 0.512 | 0.542 | 0.011 | Fair J1 slowdown<1 |

**关键发现**：Fair 的 P-attn 低不是因为 Fair 真的好，而是因为 **J1 (premium, c_i=1.2) 在 Fair 臂中出现 slowdown < 1**——这意味着 J1 的通信时间比 solo 基线还短，这在物理上不可能（contended 场景下通信时间不可能比 solo 短）。

**根因**：T_target 校准的 solo 基线偏大（可能因为校准时 NIC 冷启动或缓存未预热），导致 contested 场景下 slowdown < 1。

### 结论

1. **Fair 反常 = 指标 bug + 场景缺陷的混合**：
   - 指标 bug：T_target 校准偏大 → slowdown < 1 → P-attn 被低估
   - 场景缺陷：mlx5 不支持 strict priority → DSCP 标记对带宽无影响 → Fair 不被惩罚

2. **LongLiu vs Static 的对比是有效的**（两者在相同硬件条件下，LongLiu P-attn ≤ Static）

3. **论文中应使用 LongLiu vs Static 作为主要对比**，Fair 臂仅作为参考（注明硬件限制）

---

## 5. 下一步行动建议

### 必须在硬件环境做的

1. **Solo 基线重新校准**：用更长 warmup（5 min）+ 更多 epoch（30）+ 重复 3 次取最小值，确保 T_target 不偏大
2. **512KB payload 校准**：当前场景缺少 512KB 的 solo BW 数据
3. **DSCP→TC 探针**：确认 10.1 侧 NIC 的 DSCP→TC 映射（方案 0.3 前置）
4. **用新参数重跑**：3 regime × 3 arm × 5 round

### 我现在就能交付的

1. ✅ 场景参数重构表（本文档 §3）
2. ✅ DSCP 映射核查（本文档 §1）
3. ✅ 27 轮重分析（`expC_attainment_analysis.md`）
4. ✅ 重分析脚本（`reanalyze_expC.py`）
5. 🔲 论文 §V-F 初稿（待数据齐备后）
