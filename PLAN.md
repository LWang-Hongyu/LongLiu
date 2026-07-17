# LongLiu 仿真器重构与增强规划

> 本目录存放下一代 LongLiu 仿真器的规划与实现代码，与 `/home/why/LongLiu_rebuild/experiments/sim/` 等现有代码保持独立，便于逐步迁移和对比验证。

---

## 1. 项目目标

构建一个**面向多租户 AI 训练网络调度**的 flow-level 事件驱动仿真器，核心目标：

1. **服务论文**：为 LongLiu 提供拓扑感知、可校准、可复现的仿真证据。
2. **填补社区空白**：成为首个公开支持"迭代级 SLO + DSCP 优先级 + 多租户并发"的训练网络仿真器。
3. **易于扩展**：模块化设计，支持新拓扑、新策略、新 trace 的快速接入。

### 1.4 版本范围

**v1 聚焦 Data Parallel (DDP) 训练场景**，原因：
- 论文物理原型、核心机制与评估均基于 PyTorch DDP + NCCL AllReduce。
- LongLiu 的 Progress Deficit 信号直接来自迭代级 AllReduce 完成时间。
- TP/PP/EP/CP 等混合并行留给未来工作（Phase 3）。
- 不实现论文未引用的基线（如 D2TCP、MLTCP），避免过度扩展。

---

## 2. 功能设计

### 2.1 核心能力

| 功能 | 说明 | 优先级 |
|------|------|--------|
| 拓扑感知网络模拟 | 支持 Fat-Tree / Clos / 单瓶颈链路，per-link 竞争 | P0 |
| 迭代级 workload | 每 job 按迭代生成 compute + comm 事件 | P0 |
| 多策略调度 | Fair / SRPT / CRUX / CASSINI / LongLiu（仅论文引用基线） | P0 |
| DSCP 优先级映射 | LongLiu 连续权重 → 4 级 DSCP | P0 |
| AllReduce 多流 + Barrier 语义 | 每 job 每轮迭代含 N 个 flow，迭代完成时间由最慢 flow 决定 | P1 |
| 物理原型机校准 | 用 2 节点 RoCEv2 数据校准 overhead | P0 |
| 真实 trace 驱动 | Alibaba Lingjun / 自定义 trace | P1 |
| 集合通信算法模型 | AllReduce Ring/Tree 拆解为多阶段 flow | P2 |
| Chakra / AICB 集成 | 导入真实 layer-by-layer trace | P2 |

### 2.2 输出指标

- **SLO attainment**：达到目标吞吐率的 job 比例
- **Jain fairness index**：job 间吞吐量公平性
- **Tail iteration latency**：p95/p99 迭代时间
- **Per-job iteration time series**：用于绘图和细粒度分析
- **链路利用率热力图**：拓扑感知后的附加输出

---

## 3. 文件目录结构

```
/home/why/LongLiu_rebuild/sim-nextgen/
├── PLAN.md                 # 本规划文档
├── README.md               # 使用说明（后续补充）
├── requirements.txt        # Python 依赖
├── longliu_sim/            # 核心包
│   ├── __init__.py
│   ├── core/               # 仿真引擎
│   │   ├── __init__.py
│   │   ├── simulator.py    # 主事件循环
│   │   ├── event.py        # 事件定义
│   │   └── clock.py        # 离散事件时钟
│   ├── network/            # 网络模型
│   │   ├── __init__.py
│   │   ├── topology.py     # 拓扑基类 + FatTree + Clos
│   │   ├── link.py         # 链路模型
│   │   ├── flow.py         # flow 定义
│   │   └── path.py         # 路由 / ECMP
│   ├── job/                # 任务模型
│   │   ├── __init__.py
│   │   ├── job.py          # Job 类
│   │   └── workload.py     # 迭代级 workload
│   ├── policy/             # 调度策略
│   │   ├── __init__.py
│   │   ├── base.py         # Policy 基类
│   │   ├── fair.py
│   │   ├── srpt.py
│   │   ├── crux.py
│   │   ├── cassini.py
│   │   └── longliu.py      # LongLiu + DSCP 映射
│   ├── trace/              # Trace 解析
│   │   ├── __init__.py
│   │   ├── lingjun.py      # Alibaba Lingjun trace loader
│   │   └── synthetic.py    # 合成 workload 生成
│   ├── metrics/            # 指标与可视化
│   │   ├── __init__.py
│   │   ├── stats.py        # 统计指标
│   │   └── plot.py         # 绘图
│   └── utils/              # 工具
│       ├── __init__.py
│       └── config.py       # 配置解析
├── experiments/            # 论文实验脚本
│   ├── exp_ablation.py     # 与论文 Table 3 对应
│   ├── exp_topology.py     # 拓扑影响分析
│   ├── exp_calibration.py  # 物理原型机校准
│   └── exp_compare.py      # 多策略对比
├── configs/                # 实验配置
│   ├── fatree_k4.yaml
│   ├── single_bottleneck.yaml
│   └── lingjun_24jobs.yaml
├── tests/                  # 单元测试
│   ├── test_topology.py
│   ├── test_policy.py
│   └── test_simulator.py
└── outputs/                # 实验输出（gitignore）
    ├── figures/
    └── traces/
```

---

## 4. 核心模块接口设计

### 4.1 Job

```python
class Job:
    def __init__(self, jid: str, model: str,
                 mb_per_iter: float, iter_interval_ms: float,
                 target_iters: int, slo_ci: float = 1.5,
                 num_workers: int = 1,
                 allreduce_algo: str = "aggregate",
                 compute_ms: float | None = None,
                 comm_solo_ms: float | None = None,
                 start_time_ms: float = 0.0,
                 comm_offset_ms: float = 0.0):
        """
        jid: job 唯一标识
        model: 模型名（用于 trace 和 display）
        mb_per_iter: 每轮迭代 AllReduce 数据量（MB）
        iter_interval_ms: 无竞争时单轮迭代时间（ms）
        target_iters: 目标迭代次数
        slo_ci: SLO 松弛系数 c_i
        num_workers: DDP worker 数量，决定每轮迭代生成的 flow 数
        allreduce_algo: 集合通信算法
            "aggregate" — 单流 aggregate（默认，向后兼容旧 sim.py）
            "ring" — 预留，Ring AllReduce 多阶段串行
            "tree" — 预留，Tree AllReduce
        compute_ms: 显式计算时间（ms），为 None 时从 iter_interval 推导
        comm_solo_ms: 无竞争通信时间（ms），为 None 时从带宽推导
        start_time_ms: job 开始时间
        comm_offset_ms: CASSINI time-shift 偏移（ms）
        """
```

**关键方法**：
- `compute_deficit() -> float`：计算 Progress Deficit `pi = avg_comm_ms / (c_i * comm_solo_ms) - 1`
- `gpu_intensity -> float`：CRUX 所需的 `Ij = comp_ms / comm_solo_ms`
- `start_allreduce(n_flows)` / `on_flow_complete() -> bool`：AllReduce 多流 barrier 控制
- `on_iter_start()` / `on_comm_start()` / `on_comm_end()`：迭代生命周期回调

**AllReduce Barrier 语义**：一个 job 的某轮迭代在**其所有 flow 都完成**时才结束。仿真器默认 `num_workers=1`（aggregate flow，与旧 sim.py 一致）；当 `num_workers>1` 时，每轮迭代生成 N 个 parallel flow，每个承载 `mb_per_iter / N` MB 数据，迭代完成时间取最慢 flow 的完成时间。这直接体现论文 "tail flow determines iteration time" 的核心论点。

**Trace 推导**：对于 trace-driven 场景，`mb_per_iter` 从模型参数推导，避免 magic number：
```python
# 在 longliu_sim/utils/model_params.py 中
MODEL_PARAMS = {
    "GPT-3-175B":       {"params": 175e9, "fp16": False},
    "ResNet-50":        {"params": 25e6,  "fp16": True},
    "LLaMA-2-7B":       {"params": 7e9,   "fp16": True},
}
# mb_per_iter = 2 * param_count * bytes_per_param / dp / 1e6  (MB)
```

### 4.2 Topology

```python
class Topology(ABC):
    @abstractmethod
    def get_path(self, src: int, dst: int) -> List[Link]:
        pass

    @abstractmethod
    def get_links_for_flow(self, flow: Flow) -> List[Link]:
        pass

class FatTree(Topology):
    def __init__(self, k: int,
                 host_bw_bps: float,
                 tor_bw_bps: float,
                 spine_bw_bps: float,
                 rack_oversub: float = 1.0):
        """
        分层 Fat-Tree：
        - host_bw_bps: 主机到 ToR 的链路带宽
        - tor_bw_bps: ToR 到 Spine 的链路带宽
        - spine_bw_bps: Spine 到 Core 的链路带宽
        - rack_oversub: 机架超售比，用于表达 rack 内/外带宽差异
        """
        pass
```

### 4.3 Policy

```python
class Policy(ABC):
    @abstractmethod
    def allocate(self, flows: List[Flow], links: List[Link],
                 time_ms: float, job_stats: Dict[str, JobStats]) -> Allocation:
        """
        返回每个 flow 在每条链路上的带宽分配（bps）。
        策略按 **job** 计算权重，同一 job 的所有 flow 共享该权重，
        以符合 AllReduce barrier 语义和 DSCP 的 per-job 优先级。
        """
        pass

class LongLiu(Policy):
    K: float = 3.0
    MIN_W: float = 0.5
    BASE_W: float = 4.0

    def __init__(self, K: float = 3.0, min_w: float = 0.5,
                 base_w: float = 4.0):
        self.K = K
        self.MIN_W = min_w
        self.BASE_W = base_w

    def get_dscp(self, job: Job) -> int:
        """将 deficit pi 映射到 4 级 DSCP。"""
        pi = job.compute_deficit()
        if pi > 0.2:   return 46   # EF, P6
        if pi > -0.1:  return 34   # AF41, P4
        if pi > -0.5:  return 18   # AF21, P2
        return 0                    # BE, P0
```

**基线策略说明**：
- **Fair**：每 job 均分带宽。
- **SRPT**：优先剩余数据量小的 flow，权重 `w_f = 1 / rem_bits`。
- **CRUX**：基于 GPU intensity `Ij = compute_ms / comm_solo_ms` 分配权重 `w_j = I_j^alpha`，高 intensity job 获得更多带宽。
- **CASSINI**：通过 `comm_offset_ms` 偏移各 job 的通信相位，实现 time-shift 交错。偏移量由 `CASSINI.compute_offsets()` 静态计算（基于迭代周期 LCM）。带宽分配等同于 Fair。
- **LongLiu**：基于 Progress Deficit `pi = avg_comm_ms / (c_i * comm_solo_ms) - 1`，权重 `w = max(MIN_W, exp(K * pi) * BASE_W)`，DSCP 映射见 `get_dscp()`。

### 4.4 Simulator

```python
class Simulator:
    def __init__(self, topology: Topology, policy: Policy,
                 duration_ms: float, trace_mode: bool = False):
        pass

    def submit(self, job: Job):
        pass

    def run(self) -> SimulationResult:
        pass
```

### 4.5 使用示例

```python
from longliu_sim.network import FatTree
from longliu_sim.job import Job
from longliu_sim.policy import LongLiu, Fair
from longliu_sim.core import Simulator
from longliu_sim.trace import LingjunTraceLoader

topo = FatTree(k=4, host_bw_bps=40e9, spine_bw_bps=100e9)

jobs = LingjunTraceLoader(
    "trace/job.csv", "trace/worker.csv",
    max_gpus=128, duration_ms=30000
).load()

sim = Simulator(topo, LongLiu(K=3.0), duration_ms=30000)
for job in jobs:
    sim.submit(job)

result = sim.run()
print(result.slo_attainment)
result.plot_iteration_times("outputs/figures/iter_times.pdf")
```

---

## 5. 与现有代码的关系

| 现有代码 | 处理方式 | 说明 |
|----------|----------|------|
| `experiments/sim/sim.py` | 参考/逐步替换 | 旧核心，新实现完成后可弃用 |
| `experiments/sim/policies/__init__.py` | 参考算法逻辑 | LongLui 权重公式直接迁移 |
| `experiments/trace_loader.py` | 参考解析逻辑 | Lingjun trace loader 重构 |
| `experiments/sim/network/__init__.py` | 参考拓扑定义 | Fat-Tree 结构复用 |
| `experiments/sim_v*.py` | 不迁移 | 旧独立脚本，功能已并入新设计 |
| `testbed/` 物理原型 | 作为校准数据来源 | 反推仿真器 overhead 等参数 |

---

## 6. 实施路线图

### Phase 1：基础框架（1 周）

| 天数 | 任务 | 产出 |
|------|------|------|
| 1-2 | 创建目录结构，实现 `Job`, `Flow`, `Link`, `Topology` 基类 | 可运行单链路仿真 |
| 3-4 | 实现事件循环 `Simulator`，集成 Fair / LongLiu 策略 | 复刻旧 sim.py 能力 |
| 5-6 | 实现 Fat-Tree 拓扑和 per-link 竞争 | 拓扑感知仿真 |
| 7 | 单元测试 + 与旧 sim.py 结果对比 | 一致性验证报告 |

### Phase 2：论文增强（1 周）

| 天数 | 任务 | 产出 |
|------|------|------|
| 1-2 | AllReduce 多流 + Barrier 语义 + DSCP 离散映射 | 支持 `num_workers > 1`，迭代时间取 tail flow |
| 3-4 | CRUX / CASSINI / SRPT 策略实现 | 完整策略对比 |
| 5 | Alibaba Lingjun trace loader 接入 | trace-driven 实验 |
| 6 | 物理原型机校准 + T_target 两阶段校准模拟 | 校准脚本和验证 |
| 7 | 实验脚本 `exp_ablation.py`, `exp_topology.py` | 论文图表数据 |

**物理原型机校准方法**：
```python
# exp_calibration.py 逻辑
# 1. 在 2 节点原型机上跑 solo AllReduce（不同数据量）
# 2. 记录实际迭代时间 t_solo（含 NCCL 和 RoCEv2 协议开销）
# 3. 在仿真器中跑相同配置，调整 overhead factor 使 t_sim ≈ t_solo
# 4. 拟合公式：overhead = f(data_size) 或固定值 2.0
```

**T_target 两阶段校准**（论文核心机制）：
- 阶段一：RTT probing 获取网络延迟基准 → 设置初始 `T_target`
- 阶段二：EMA 更新 `T_target = alpha * T_target + (1-alpha) * T_actual` 适应负载变化
- 仿真器中简化为：每轮迭代后按 `alpha` 更新 `T_target`（当前默认 `T_target = ci * comm_solo_ms`）

### Phase 3：社区化（可选，2-4 周）

- 集合通信算法模型（Ring/Tree）
- Chakra / AICB workload 集成
- Docker + reproduction artifact
- 开源到 GitHub

---

## 7. 关键技术决策

### 7.1 为什么继续用 flow-level，不做 packet-level？

- 论文核心是**调度策略设计**，flow-level 足够表达迭代级 SLO。
- packet-level（SimAI NS3）单次实验数小时，不适合大规模参数扫描。
- 物理原型机已覆盖 packet-level 的真实行为验证。

### 7.2 拓扑感知是否必要？

**必要。** 审稿人最可能质疑旧 sim.py 的"单链路假设"。Fat-Tree 多链路竞争能显著增强说服力。

### 7.3 基线策略如何选择？

严格按论文引用实现：**Fair, SRPT, CRUX, CASSINI, LongLiu**。不实现论文未引用的基线（如 D2TCP、MLTCP），以避免无依据的对比。

**D2TCP / MLTCP 排除的 contingency**：如果论文定稿后 Evaluation 或 Related Work 中明显批评 D2TCP/MLTCP 但不给对比，需在论文中明确说明排除理由：
> "We focus on comparing with schedulers that directly compete for SLO attainment (CRUX, CASSINI) and classical baselines (Fair, SRPT). D2TCP and MLTCP are discussed in §Related Work but not evaluated because they target different deployment assumptions (kernel-level congestion control vs. NCCL-level priority injection)."

如果论文 Evaluation 必须对比 D2TCP/MLTCP，则增加 Phase 2 中的实现。

CRUX 复现其 GPU-intensity 权重；CASSINI 复现其 time-shift 通信交错。

### 7.4 是否保留旧代码？

保留但不扩展。新目录 `sim-nextgen/` 独立演进，旧代码作为 baseline 用于结果对比。

---

## 8. 预期交付物

1. 可复现的拓扑感知仿真器
2. 与论文 Table 3 对应的 ablation 实验脚本
3. 物理原型机校准报告
4. （可选）社区版 `longliu-sim` 包

---

## 9. 下一步

等待用户确认本规划后，开始 **Phase 1 第一天** 的实现：创建目录结构并实现 `Job`, `Flow`, `Link`, `Topology` 基类。
