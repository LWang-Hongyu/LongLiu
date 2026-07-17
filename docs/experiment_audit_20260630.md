# LongLiu 仿真实验设置审查

> 审查日期：2026-06-30
> 审查版本：修正 comp_ms + dp=1 跳过 AllReduce 之后的版本

---

## 一、已完成的修正

| 修正项 | 文件 | 修正前 | 修正后 |
|--------|------|--------|--------|
| 模型特定计算时间 | model_params.py | 所有模型统一 comp_ms=50ms | 按参数量区分 5~250ms |
| dp=1 跳过 AllReduce | simulator.py, event.py | dp=1 仍创建 AllReduce flow | 直接触发 ITERATION_COMPLETE |
| target_iters 不提前停止 | synthetic.py, lingjun.py | 按 iter_interval 算 target_iters | 固定 999999，由 duration 硬截止 |

## 二、实验结果（修正后）

### Table 3 — 16 节点 × 24 Job（2 seeds × 600s）

| Policy | Total(K) | Tight(1.5) | Medium(2.0) | Loose(3.0) | Overall |
|--------|----------|------------|-------------|------------|---------|
| Fair   | 26.28    | 0.0%       | 0.0%        | 50.0%      | 8.3%    |
| SRPT   | 29.39    | 0.0%       | 25.0%       | 100.0%     | 25.0%   |
| CRUX   | 28.61    | 0.0%       | 25.0%       | 100.0%     | 25.0%   |
| LongLiu| 30.98    | 0.0%       | 43.8%       | 100.0%     | 31.2%   |

### Table 4 — 128 节点单链路（10 seeds）

| Policy | Total(K) | Tight | Medium | Loose | Overall |
|--------|----------|-------|--------|-------|---------|
| Fair   | 208.8    | 0.0%  | 0.0%   | 100%  | 16.9%   |
| SRPT   | 216.6    | 20.2% | 65.1%  | 100%  | 49.3%   |
| CRUX   | 224.7    | 0.0%  | 100%   | 100%  | 51.6%   |
| LongLiu| 224.8    | 46.3% | 98.1%  | 100%  | 73.4%   |

### Table 5 — 128 节点 TwoTier（10 seeds）

| Policy | Total(K) | Tight | Medium | Loose | Overall |
|--------|----------|-------|--------|-------|---------|
| Fair   | 223.4    | 8.9%  | 41.2%  | 100%  | 34.7%   |
| SRPT   | 227.2    | 62.7% | 81.2%  | 100%  | 75.0%   |
| CRUX   | 228.1    | 29.1% | 100%   | 100%  | 64.5%   |
| LongLiu| 228.0    | 69.8% | 100%   | 100%  | 84.9%   |

### 关键结论

- LongLiu 在所有场景 Overall SLO attainment 最高
- 规模越大、拓扑越真实，LongLiu 优势越明显（+6pp → +20pp）
- 16 节点 Tight=0% 是 workload 结构性限制（大模型单 iter 最短 ~240ms）

---

## 三、拓扑设置审查

| 项目 | 设置 | 真实性评估 |
|------|------|-----------|
| Host 带宽 40Gbps | A100/H100 集群 40~100G NIC | 合理 |
| Spine 带宽 1680G (128 节点) | 取决于超售比 | 合理 |
| 超售比 3.05:1 | 云厂商 2:1~4:1，HPC 1:1 | 合理 |
| Fat-Tree 实现 | 退化为单 spine link | **有问题** — 多路径优势丢失 |
| TwoTier 8×16 | 真实 Rail-Optimized 更复杂 | 简化但可接受 |

**问题**：FatTreeTopology.get_path() 返回单条 spine link，与 SingleLinkTopology 无区别。
128 节点实验等同于所有流量共享一条链路，竞争被放大。

## 四、任务分布审查

| 项目 | 设置 | 评估 |
|------|------|------|
| 大/中/小模型比例 | 50%/33%/17% | 大模型占比偏高，真实集群通常 <30% |
| DP 度 1~8 | 合理 | 小模型 DP=1~4，大模型 DP=4~8 |
| SLO 分档 ci=1.5/2.0/3.0 | 物理含义不明确 | 缺乏真实 trace 校准 |
| 24 并发 job / 128 GPU | GPU 利用率 ~19% | 合理 |

## 五、模型特征审查

### comp_ms 准确性

| 模型 | 当前 comp_ms | 真实值（A100, bs=32） | 偏差 |
|------|-------------|---------------------|------|
| LLaMA-2-13B | 200ms | ~30-50ms | **偏大 4-6x** |
| LLaMA-2-7B | 120ms | ~20-30ms | **偏大 4x** |
| T5-11B | 180ms | ~30-50ms | **偏大 4-5x** |
| BERT-Large | 40ms | ~10-20ms | 偏大 2-3x |
| ResNet-50 | 8ms | ~5-15ms | 基本合理 |
| ResNet-18 | 5ms | ~3-8ms | 基本合理 |

### AllReduce 通信量

公式：`2 × params × bytes_per_param / dp` — 正确（Ring AllReduce）

overhead_factor = 2.0（NCCL+PCIe 开销）— 偏保守但可接受

## 六、通信模型审查

| 项目 | 当前建模 | 评估 |
|------|---------|------|
| AllReduce 数据量 | 正确 | |
| Compute-Comm 重叠 | **无 — 完全串行** | **最大失真源** |
| 网络延迟 (RTT) | 未建模 | 小消息场景影响大 |
| 多路径/ECMP | 未建模 | Fat-Tree 核心优势丢失 |
| 网络抖动/排队延迟 | 未建模 | 简化可接受 |

## 七、瓶颈链路审查

| 配置 | 瓶颈 | 问题 |
|------|------|------|
| 16 节点 SingleLink | 共享单链路 | 竞争放大，真实有 ToR 上行 |
| 128 节点 FatTree | 单 spine link | 同上，多路径优势丢失 |
| 128 节点 TwoTier | spine link | **最真实**，但仍为单条 spine |

## 八、总结：3 个核心问题

按严重度排序：

### 高：Compute-Comm 无重叠

当前假设 compute 和 communication 完全串行（iter_time = comp_ms + comm_ms）。
真实训练中 backward pass 与 AllReduce 流水线重叠，实际 iter_time ≈ max(comp_ms, comm_ms) + overhead。

影响：高估迭代时间，comp_ms 大的 job（LLaMA-13B）被过度惩罚。
修改方案：`effective_iter_ms = max(comp_ms, comm_ms) + overlap_overhead`

### 高：Fat-Tree 拓扑退化为单链路

FatTreeTopology.get_path() 返回单条 link，ECMP 多路径未建模。
128 节点实验中所有跨 pod 流量共享一条链路。

影响：竞争被系统性放大，Fat-Tree 的带宽可扩展性优势丢失。
修改方案：实现 ECMP 多路径，将 spine 带宽均分到 k/2 条等价路径。

### 中：大模型 comp_ms 偏大 4-6x

LLaMA-2-13B comp_ms=200ms vs 真实 ~30-50ms。
叠加无重叠假设，大模型 job 的迭代时间被严重高估。

影响：改变了 job 之间的 compute-bound vs comm-bound 相对关系。
修改方案：参考 MLPerf 或实际 benchmark 数据校准 comp_ms。

---

## 九、实施的修改（2026-06-30）

### 已完成

| 修改 | 文件 | 状态 |
|------|------|------|
| Fat-Tree ECMP 多路径 | topology.py | ✅ k/2 条 spine link，flow hash 分配 |
| Compute-Comm 重叠模型 | simulator.py | ✅ overlap_factor 参数（0=串行, 1=完全重叠） |
| comp_ms 校准 | model_params.py | ✅ LLaMA-13B: 200→80ms 等 |
| dp=1 迭代批量计算 | simulator.py | ✅ 无通信 job 只模拟首次迭代 |
| TwoTier 带宽修复 | simulator.py | ✅ cross-rack flow 包含 rack link |
| 迭代版本号 | job.py, flow.py, simulator.py | ✅ 防止重叠迭代 barrier 干扰 |
| target_iters 恢复合理值 | synthetic.py, lingjun.py | ✅ 基于 comp_ms + comm 计算 |

### LongLiu < CRUX 排查结果

新 comp_ms 下 LongLiu Overall (41.7%) < CRUX (50.0%)，原因：

1. LongLiu 的 deficit 公式 `pi = avg_comm_ms / T_target - 1` 给大模型（comm_solo=650ms）比小模型（comm_solo=34ms）高得多的优先级
2. 带宽集中在大模型 → 中模型（BERT, ViT）被饿死 → 从 SLO 内掉到 SLO 外
3. 旧 comp_ms（统一50ms）掩盖了这个问题，因为所有模型 comm_solo 差不多
4. 这是算法在真实 workload 下的结构性问题，不是代码 bug

### 结果对比

| 配置 | CRUX | LongLiu | LongLiu 优势？ |
|------|------|---------|---------------|
| 旧 comp_ms（统一50ms） | 25.0% | 31.2% | **是 (+6.2pp)** |
| 新 comp_ms + overlap=0.0 | 50.0% | 41.7% | 否 (-8.3pp) |
| 新 comp_ms + overlap=1.0 | 50.0% | 37.5% | 否 (-12.5pp) |

---

## 十、后续行动

- [x] P0：实现 compute-comm 重叠模型
- [x] P0：实现 Fat-Tree ECMP 多路径
- [x] P0：校准大模型 comp_ms 到真实值
- [ ] **决策：新 comp_ms vs 旧 comp_ms**（影响论文结论）
- [ ] P1：LongLiu 算法改进（防止大模型饿死中模型）
- [ ] P1：Table 3 补跑 10 seeds
- [ ] P1：T_target 校准实验（exp_calibration）
- [ ] P2：不同 workload 比例敏感性分析
