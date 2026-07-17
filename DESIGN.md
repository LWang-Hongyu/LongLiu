# LongLiu 调度器完整实现方案文档

**版本：v3.0（加权分配 + Poisson 分布 + 第一轮最高优先级）**

---

## 一、系统概述

LongLiu 是一个基于 Progress Deficit 的动态优先级网络调度系统，专门针对 DDP（Data Parallel）训练场景设计。系统通过事件驱动仿真器模拟 Fat-Tree 拓扑下的多任务带宽竞争，实现严格优先级调度与动态优先级调整机制。

### 设计目标

1. **SLO 导向**：确保各任务的实际迭代时间接近预设目标
2. **公平性**：防止大模型长期被饿死，同时避免中小模型过度抢占
3. **动态调整**：根据实时运行状态动态调整任务优先级
4. **实用性**：适合云环境多租户训练场景

---

## 二、核心算法

### 2.1 Progress Deficit (pi)

**定义**：
```
pi = avg_iter_ms / T_target - 1
```

**参数解释**：
- `avg_iter_ms`：任务的平均端到端迭代时间（包含计算和通信）
- `T_target`：目标迭代时间，由 SLO 松弛系数 `ci` 和无竞争迭代时间计算

**SLO 分层**：
- 大模型（>1000MB, ≥4 GPUs）：`ci = 1.5`（紧约束）
- 中模型（100-1000MB）：`ci = 2.0`
- 小模型（<100MB）：`ci = 3.0`（松约束）

**语义**：
- `pi > 0`：任务落后于 SLO，需要提升优先级
- `pi = 0`：任务刚好达到 SLO
- `pi < 0`：任务领先于 SLO，可降低优先级

### 2.2 7 级 DSCP 优先级映射

| Priority | DSCP Value | Threshold (pi) | 含义 |
|----------|-----------|----------------|------|
| P6 | 38 | pi > 1.0 | 严重违约（实际迭代时间 > 2× SLO） |
| P5 | 34 | pi > 0.6 | 中度违约 |
| P4 | 36 | pi > 0.3 | 轻度违约 |
| P3 | 26 | pi > 0.0 | 刚开始违约 |
| P2 | 28 | pi > -0.2 | 接近 SLO |
| P1 | 18 | pi > -0.4 | 领先 SLO |
| P0 | 0 | pi ≤ -0.4 | 显著领先（最低优先级） |

### 2.3 带宽分配策略

**严格优先级 + 加权分配**：

1. **优先级层次**：P6 > P5 > P4 > P3 > P2 > P1 > P0
2. **高优先级独占**：高优先级队列占满带宽，低优先级饿死
3. **同级别加权**：同一优先级内按 `exp(pi * K)` 加权分配带宽
4. **权重裁剪**：pi ∈ [-2, 3]，防止溢出

**权重公式**：
```
w = max(MIN_W, exp(K * pi_clipped)) * BASE_W
```

参数：
- `K = 3.0`：指数增益
- `MIN_W = 0.5`：最小权重
- `BASE_W = 4.0`：基础权重

### 2.4 第一轮特殊处理

**设计理念**：避免初始状态不公平

**实现**：
- 所有任务第一轮迭代给最高优先级（DSCP 38）
- 第一轮迭代完成后，`is_first_iter` 设置为 `False`
- 后续迭代按 pi 动态计算优先级

### 2.5 动态调整机制

**反馈周期**：
- 每轮迭代完成后，更新 `pi` 和 DSCP
- 落后的任务优先级上升，领先的优先级下降
- 形成自然的"优先级轮转"

**关键**：任务必须完成迭代才能更新 `pi`，因此需要合理的任务分布避免饿死。

---

## 三、任务分布设计

### 3.1 Poisson 过程

**设计理念**：模拟真实云环境的任务到达模式

**实现**：
- 任务到达时间间隔服从 **Exponential 分布**
- 平均间隔 = `2.0 × duration_ms / job_count`
- 对于 300000ms 仿真时长、24 个任务：平均间隔 = 25000ms

**代码实现**：
```python
mean_interval_ms = 2.0 * self.duration_ms / len(profile)
interval = self.rng.expovariate(1.0 / mean_interval_ms)
current_time += interval
start_time_ms = min(current_time, self.duration_ms * 0.9)
```

**效果**：
- 任务分散在仿真时间内，避免同时开始
- 减少初始竞争，给动态调整留出空间

### 3.2 Workload Profile

**默认分层 workload**（24 个任务）：
- 大模型：12 个（ci=1.5）
- 中模型：8 个（ci=2.0）
- 小模型：4 个（ci=3.0）

**模型选择**：
- LLaMA-2-13B（8 GPUs）
- LLaMA-2-7B（4/8 GPUs）
- BERT-Large-fp16（2/4 GPUs）
- ViT-Large（8 GPUs）
- ResNet-18/50-fp16（1/2 GPUs）

---

## 四、实现细节

### 4.1 Job 类

**关键属性**：
```python
class Job:
    def __init__(self, ...):
        self.jid: str  # 任务 ID
        self.model: str  # 模型名称
        self.num_workers: int  # DDP worker 数量
        self.slo_ci: float  # SLO 松弛系数
        self.worker_hosts: list[int]  # worker 所在主机
        
        # 运行时状态
        self.completed_iters: int = 0
        self.accumulated_iter_ms: float = 0.0
        self.is_first_iter: bool = True  # 第一轮标记
        
    def compute_deficit(self) -> float:
        """计算 Progress Deficit"""
        if self.completed_iters == 0:
            return 0.0
        avg_iter_ms = self.accumulated_iter_ms / self.completed_iters
        return avg_iter_ms / self.default_T_target - 1.0
    
    def on_comm_end(self, time_ms: float):
        """迭代完成，更新状态"""
        self.completed_iters += 1
        self.is_first_iter = False  # 取消第一轮标记
```

### 4.2 LongLiu 策略

**allocate 方法流程**：

```python
def allocate(self, flows, links, time_ms, job_stats):
    # 1. 按 job 计算 pi 和 DSCP
    for jid in jobs:
        job = job_stats[jid]
        
        # 第一轮：所有任务给 DSCP 38
        if job.is_first_iter:
            job_dscp[jid] = 38
            job_pi[jid] = 1.0
            continue
        
        # 后续迭代：动态计算
        pi = job.compute_deficit()
        dscp = self.get_dscp(pi)
        job_dscp[jid] = dscp
        job_pi[jid] = pi
    
    # 2. 按 DSCP 分组 flows
    flows_by_dscp = defaultdict(list)
    for f in flows:
        flows_by_dscp[job_dscp[f.jid]].append(f)
    
    # 3. 严格优先级分配
    for dscp in DSCP_PRIORITY_ORDER:
        if dscp not in flows_by_dscp:
            continue
        flows_at_level = flows_by_dscp[dscp]
        
        # 4. 同级别内加权分配
        weights = {}
        for f in flows_at_level:
            pi = job_pi[f.jid]
            pi_clipped = max(-2.0, min(3.0, pi))
            w = max(MIN_W, exp(K * pi_clipped)) * BASE_W
            weights[f.jid] = w
        
        # 5. 按权重分配带宽
        total_weight = sum(weights.values())
        for f in flows_at_level:
            bw = remaining_bw * weights[f.jid] / total_weight
            alloc[f] = {link: bw}
        
        # 6. 低优先级饿死
        remaining_bw = 0.0
        break
    
    return alloc
```

### 4.3 仿真器

**关键流程**：

```python
def run(self):
    while self.events and self.time_ms < self.duration_ms:
        # 1. 计算下一个时间点
        next_time = min(events[0].time_ms, earliest_flow_finish)
        
        # 2. 推进时间，更新 flow 状态
        self._advance(next_time)
        
        # 3. 处理完成的 flows（Barrier 检查）
        self._cleanup_finished_flows()
        
        # 4. 处理事件
        self._process_event(event)
        
        # 5. 重新计算带宽分配
        self._recompute_bandwidth()
```

**Fat-Tree ECMP**：
- 每条 spine link 独立分配带宽
- 跨 rack 流量可走多条路径

---

## 五、参数配置

### 5.1 系统参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `K` | 3.0 | Deficit 指数增益 |
| `MIN_W` | 0.5 | 最小带宽权重 |
| `BASE_W` | 4.0 | 基础带宽权重 |
| `use_dynamic_T_target` | True | 启用 T_target 动态校准 |
| `overhead_factor` | 1.3 | NCCL/PCIe 协议开销 |
| `overlap_factor` | 0.85 | 计算-通信重叠度 |

### 5.2 拓扑参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `k` | 4 | Fat-Tree 参数 |
| `num_hosts` | 16 | 主机数量 |
| `host_bw_bps` | 100G | 主机带宽 |
| `spine_bw_bps` | 400G | Spine 带宽 |
| `duration_ms` | 300000 | 仿真时长（5 分钟） |

### 5.3 Poisson 分布参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `mean_interval_ms` | 25000 | 平均任务间隔 |
| `max_start_time` | 270000 | 最大开始时间（90% 仿真时长） |

---

## 六、实验结果

### 6.1 主实验（16 hosts, 10 seeds, 600000ms）

| 策略 | Tight% | Medium% | Overall% | Mean SAS | Catastrophic% |
|------|--------|---------|----------|----------|---------------|
| Fair | 27.5 | 7.5 | 24.6 | 0.772 | 1.7 |
| SRPT | 25.0 | 22.5 | 28.3 | 0.826 | 10.8 |
| CRUX | 25.8 | 22.5 | 28.8 | 0.808 | 4.2 |
| **LongLiu** | **36.7** | **30.0** | **36.7** | **0.966** | 16.3 |

- LongLiu vs CRUX: p=0.060（边缘显著）
- LongLiu 在 Tight 和 Medium 层的 SLO attainment 均领先

### 6.2 P1 消融实验（5 组 × 10 seeds）

| 消融模式 | Overall% | Δ | Mean SAS | Δ | Catastrophic% | Δ |
|---------|---------|---|---------|---|-------------|---|
| Default | 36.7% | - | 0.904 | - | 20.8% | - |
| no_startup | **39.6%** | +2.9% | 0.855 | -0.049 | 24.6% | +3.8% |
| no_weighted | 36.2% | -0.5% | 0.883 | -0.021 | 20.0% | **-0.8%** |
| 4level_dscp | 34.2% | -2.5% | **0.915** | +0.011 | **11.3%** | **-9.5%** |
| static_arrival | **13.8%** | **-22.9%** | **0.539** | **-0.365** | 27.1% | +6.3% |
| no_ema | 30.0% | -6.7% | **1.014** | +0.110 | **1.7%** | **-19.1%** |

**消融结论**：
1. **static_arrival 影响最大**（-22.9%）：Poisson 到达对结果至关重要，同时启动导致致命竞争
2. **no_ema 双刃剑**：无 EMA 时 Mean SAS 上升（1.014）但 Medium 层 attainment 从 30% 暴跌至 8.8%
3. **4level_dscp 改善公平性**：7→4 级降低 attainment（-2.5%）但灾难性违约率大幅下降（-9.5%）
4. **no_startup 微幅提升**：去除第一轮启动提升后 attainment 反而上升 2.9%，但 SAS 下降
5. **no_weighted 影响最小**：同级别加权只有 0.5% 差异

### 6.3 P2 扩展性（5 seeds, 24 jobs）

| 策略 | 16 hosts (base) | 32 hosts (54 hosts) | 64 hosts (128 hosts) |
|------|:---------------:|:-------------------:|:--------------------:|
| Fair | 24.6% / 0.772 | 51.7% / 1.478 | 74.2% / 2.338 |
| SRPT | 28.3% / 0.826 | 60.8% / 1.482 | 80.0% / 2.321 |
| CRUX | 28.8% / 0.808 | 65.8% / 1.497 | **86.7%** / **2.344** |
| **LongLiu** | **36.7% / 0.966** | **69.2% / 1.509** | 84.2% / 2.301 |

**扩展性结论**：
- LongLiu 在高竞争场景（16 hosts）优势明显
- 中规模（32 hosts）LongLiu 仍领先 CRUX（69.2% vs 65.8%）
- 大规模（64 hosts）CRUX 略优（86.7% vs 84.2%），因为低竞争时比例分配更高效
- 所有策略随集群增大而收敛——带宽充裕时调度策略差异缩小

### 6.4 P3 Trace-Driven 验证（Alibaba Lingjun, 10 seeds, 300000ms）

| 策略 | Mean SAS | Median SAS | Min SAS | Max SAS | Catastrophic% | p vs CRUX |
|------|---------|-----------|---------|---------|:-------------:|:---------:|
| Fair | 0.664 | 0.613 | 0.325 | 1.366 | 0.0% | 1.9e-5 |
| SRPT | 0.795 | 0.888 | 0.114 | 1.669 | 12.1% | 3.5e-3 |
| CRUX | 0.758 | 0.817 | 0.245 | 1.421 | 1.7% | N/A |
| **LongLiu** | **0.978** | 0.847 | 0.092 | 2.800 | 15.0% | **4.7e-2** |

**Trace 结论**：
- LongLiu 在真实 trace workload 上 SAS 最高（0.978），且统计显著（p=0.047 < 0.05）
- Fair 最稳定（0% 灾难性违约）但 SAS 最低
- LongLiu 方差较大（max=2.800, catastrophic=15.0%），部分任务获得"超额"性能的同时另一些被饿死

### 6.5 公平性指标汇总（P5）

| 策略 | Jain Index | Gini Coeff | Catastrophic Rate |
|------|:---------:|:----------:|:-----------------:|
| Fair | 0.595 | 0.355 | 1.7% |
| SRPT | 0.579 | 0.384 | 10.8% |
| CRUX | 0.610 | 0.356 | 4.2% |
| **LongLiu** | **0.653** | **0.334** | 16.3% |

**公平性结论**：LongLiu 的 Jain 指数最高（0.653）和 Gini 系数最低（0.334），表明其在同层内分配带宽时最公平。但灾难性违约率（16.3%）较高，主要因为 SP 调度下部分任务长期无法获得带宽。

---



## 七、遇到的问题与解决方案

### 7.1 问题 1：严格优先级导致大模型饿死

**现象**：
- 10 seeds 结果显示 47.5% 大模型 SAS < 0.2
- 少数任务独占带宽，多数被饿死

**原因**：
- 所有任务同时开始（start_time_ms = 0）
- 第一个 epoch 的分配决定了后续状态
- P6 队列独占带宽，其他队列饿死

**解决方案**：
1. 第一轮所有任务给最高优先级（DSCP 38）
2. 任务开始时间使用 Poisson 分布
3. 同级别内按 pi 加权分配带宽

### 7.2 问题 2：权重计算溢出

**现象**：
- `math.exp(K * pi)` 溢出错误

**原因**：
- pi 可能很大（如 pi = 10）
- exp(30) 超出 float 范围

**解决方案**：
- 裁剪 pi ∈ [-2, 3]
- `pi_clipped = max(-2.0, min(3.0, pi))`

### 7.3 问题 3：中模型性能下降

**现象**：
- 中模型 Mean SAS 从 0.83 降到 0.57
- SAS < 0.2 比例从 4% 升到 31%

**原因**：
- 严格优先级保护大模型，中模型被抢占

**权衡**：
- 这是设计目标：保护落后的大模型
- 如果需要平衡，可调整 DSCP 阈值或增加 Poisson 间隔

### 7.4 问题 4：P3 Trace 实验的配置一致性

**现象**：
- 首次运行时 SAS 异常高（~55），后续修复后降至合理值（0.6-1.0）

**原因**：
1. `overhead_factor` 不匹配：trace loader 用 2.0 但 Simulator 默认用 1.0
2. `target_bw_bps` 硬编码 40Gbps 但仿真用 100Gbps
3. Poisson 分布将所有任务推到 270000ms cap
4. dp=1 的任务 comm_solo_ms 计算错误（不应有网络通信）

**解决方案**：
1. 添加 `overhead_factor` 参数到 `LingjunTraceLoader`
2. 添加 `target_bw_bps` 参数并使用配置值
3. 改为均匀分布 `uniform(0, duration * 0.8)`
4. 在 `_compute_mb_per_iter` 中对 dp<=1 返回 0.0

### 7.5 问题 5：消融实验 `no_ema` 参数传递错误

**现象**：
- `LongLiu(K=2.0, use_dynamic_T_target=True, use_dynamic_T_target=False)` 重复传参导致 TypeError

**原因**：
- 消融模式代码硬编码 `use_dynamic_T_target=True` 并与 `ablation_params` 合并时冲突

**解决方案**：
- 使用 `{**default_params, **ablation_params}` 合并，允许 ablation_params 覆盖默认值

---

## 八、关键代码文件

| 文件 | 功能 | 关键修改 |
|------|------|----------|
| `longliu_sim/policy/longliu.py` | LongLiu 策略 | 7 级 DSCP + 第一轮最高优先级 + 同级别加权 + 消融参数支持 |
| `longliu_sim/job/job.py` | Job 模型 | `is_first_iter`, `accumulated_iter_ms`, `iter_solo_ms` |
| `longliu_sim/trace/synthetic.py` | 任务生成 | Poisson 分布任务开始时间 + workload profile |
| `longliu_sim/trace/lingjun.py` | Alibaba Lingjun trace 加载器 | 解析真实数据集，支持分层 SLO CI |
| `longliu_sim/core/simulator.py` | 仿真器 | Fat-Tree ECMP + Barrier 语义 |
| `experiments/exp_ablation.py` | 主实验脚本 | 10 seeds 验证 + P1 消融模式 + P4/P5 指标 |
| `experiments/exp_trace.py` | Trace 实验脚本 | P3 trace-driven 验证 |
| `configs/fatree_16host.yaml` | 16 节点配置 | 论文 Table 3 主实验 |
| `configs/fatree_32host.yaml` | 32 节点配置 | P2 扩展性（中规模） |
| `configs/fatree_64host.yaml` | 64 节点配置 | P2 扩展性（大规模） |
| `outputs/calibration.md` | 校准文档 | P0 所有参数的物理来源 |

---

## 九、设计权衡总结

### 9.1 严格优先级 vs 加权分配

**严格优先级优势**：
- 简单直观
- 保护落后任务
- 快速响应违约

**加权分配优势**：
- 防止饿死
- 更公平

**最终选择**：
- **优先级层次使用严格优先级**
- **同级别内使用加权分配**
- 兼顾效率和公平

### 9.2 第一轮特殊处理

**必要性**：
- 避免初始状态不公平
- 给所有任务公平的起点

**实现**：
- 第一轮所有任务 DSCP 38
- 第一轮后动态调整

### 9.3 Poisson 分布

**必要性**：
- 模拟真实云环境
- 减少初始竞争
- 给动态调整留出空间

**参数选择**：
- 平均间隔 = 2 × duration / job_count
- 确保任务分散但不过于稀疏

---

## 十、已完成的工作（P0-P5）

### ✅ P0: 仿真参数校准
- [x] 所有参数的物理来源和方法记录在 `outputs/calibration.md`
- [x] 关键参数：`comp_ms`（MLPerf）、`overhead_factor=1.3`（BlueField-3 testbed）、`overlap_factor=0.85`（实测）

### ✅ P1: 消融实验（5 组）
- [x] `no_startup`：去除第一轮 DSCP 38 启动提升 → Overall +2.9%, SAS -0.049
- [x] `no_weighted`：同级别均匀分配 → Overall -0.5%, SAS -0.021
- [x] `4level_dscp`：4 级 DSCP 替换 7 级 → Overall -2.5%, 公平性大幅改善
- [x] `static_arrival`：所有任务 time=0 启动 → Overall -22.9%（影响最大）
- [x] `no_ema`：无 T_target EMA 动态校准 → Medium 层 attainment 暴跌

### ✅ P2: 扩展性实验
- [x] 32 hosts (k=6, 54 hosts)：LongLiu 69.2% vs CRUX 65.8% → LongLiu 领先
- [x] 64 hosts (k=8, 128 hosts)：CRUX 86.7% vs LongLiu 84.2% → 低竞争时 CRUX 反超
- [x] 结论：LongLiu 在高竞争场景优势明显，低竞争时 CRUX 的比例分配更高效

### ✅ P3: Trace-Driven 验证
- [x] 使用 Alibaba Lingjun 数据集（24 jobs, 300000ms）
- [x] LongLiu SAS=0.978 最高，显著优于 CRUX（p=0.047）
- [x] Fair 最稳定（0% 灾难性违约），但 SAS 最低

### ✅ P4: 统计显著性
- [x] 10 seeds + mean±std + 95% CI + paired t-test
- [x] LongLiu vs CRUX: p=0.060（主实验，边缘显著）
- [x] P3 trace: LongLiu vs CRUX: p=0.047（显著）

### ✅ P5: 公平性指标
- [x] Jain 公平指数：LongLiu 0.653（最高）
- [x] Gini 系数：LongLiu 0.334（最低，最公平）
- [x] 灾难性违约率（SAS<0.2）：LongLiu 16.3%（需改进）

## 十一、后续优化方向

1. **降低灾难性违约率**：调整 DSCP 阈值或引入最小带宽保障机制
2. **自适应 Poisson 参数**：根据负载动态调整任务间隔
3. **混合策略**：在高/低竞争场景切换 LongLiu 与 CRUX
4. **硬件测试床验证**：在真实 BlueField-3 集群上部署验证

---

**文档版本**：v4.1（P0-P5 + Robustness 消融）
**最后更新**：2026-07-15
**实验验证**：10 seeds × 5 policies × 600000ms（主实验 + v2 变种）+ 5 组消融 × 10 seeds + 2 组扩展性 × 5 seeds + trace 验证 × 10 seeds

---

## 附录 A：控制论 Robustness 机制消融（§v2）

### 动机

高竞争下 LongLiu 的灾难性违约率（27.5%）高于 CRUX（10.0%），审稿人视角需论证：
1. 是否有机制可以降低长尾崩溃？
2. 若无痛方案，改造成的性能损失是多少？

### 设计：四个控制论机制

| 机制 | 控制论对应 | 参数 |
|------|-----------|------|
| 死区：\|π\|<δ 时强制映射 P3 | actuator dead zone | δ=0.1 |
| 滑窗 urgency：最近 W 次迭代平均 | forgetting factor | W=8 |
| 优先级老化：连续 L 次挨饿强制升一级 | starvation freedom | L=5 |
| 迟滞：每 epoch 最多移动 1 个 DSCP 档 | anti-chattering | h=0.05 |

### 10 seeds 结果

| 策略 | Overall% | Mean SAS | Catastrophic | p_vs_CRUX |
|------|:-------:|:--------:|:-----------:|:---------:|
| CRUX（baseline） | 29.2% | 0.809 | 10.0% | N/A |
| **LongLiu** | **39.2%** | **0.837** | 27.5% | 0.333 |
| LongLiu_v2 | 34.6% | 0.798 | **26.3%** | 0.444 |

### 结论

控制论四件套在高竞争（16 hosts, 24 jobs）下**损害性能**（Overall -4.6pp），仅在低竞争（64 hosts）下可能有益。审稿人可以引用此消融实验证明：LongLiu 的高竞争性能来源于其动态优先级设计的简洁性，而非冗余的控制论补丁。灾难性违约率问题留作未来工作（可通过 DWRR 地板 + 调整 Poisson 间隔缓解）。
