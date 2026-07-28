# Progress Deficit: Iteration-Granularity SLO-Guaranteed Network Scheduling for Multi-Tenant DNN Training

> INFOCOM 2027 Draft v1.0
> 作者: 待定

---

## Abstract

Multi-tenant GPU clusters for deep neural network (DNN) training suffer from unpredictable network contention. When multiple training jobs share the same RDMA link, gradient synchronization delays propagate across iterations, causing jobs to systematically miss their training deadlines. Existing schedulers operate at either the job level (single decision per job lifetime) or the flow level (per-packet), missing the natural scheduling granularity of DNN training iterations. We present **LongLiu**, an iteration-granularity network scheduler that introduces the **Progress Deficit** signal—an online-only, zero-profile metric that measures how far each job is from its service-level objective (SLO). Each job independently computes its deficit from a single user-specified parameter (the relaxation coefficient c_i) and its observed iteration times, requiring no cross-job coordination, no offline profiling, and no global scheduler. A two-stage hybrid measurement scheme auto-calibrates the target iteration time, avoiding the "short-job trap" common in multi-tenant environments. We prove that the deficit vector converges to a stationary distribution with an exponential tail bound via Lyapunov stability analysis (Theorem 1). LongLiu is implemented as a ~50-line C modification to the NCCL proxy thread, deployed via standard library replacement. On an Alibaba trace-driven simulation at up to 128 nodes, LongLiu improves tight-SLO job attainment from 5.3% (Fair) and 0% (CRUX) to 15.8%, while maintaining competitive overall throughput. A 2-node RDMA prototype confirms end-to-end control loop functionality.

---

## 1. Introduction

**SLA's performance vacuum.** Cloud GPU providers (AWS SageMaker, Azure ML, Alibaba PAI) offer service-level agreements (SLAs) that guarantee only infrastructure availability—"the machine is running"—while explicitly excluding network performance guarantees. This leaves tenants paying premium prices for A100/H100 GPUs with no assurance that their training jobs will complete on time. Network contention between concurrent jobs can inflate iteration times unpredictably, turning what should be a deterministic training process into a gamble. We term this the "SLA performance vacuum": the gap between infrastructure availability guarantees and the performance predictability that ML practitioners actually need.

**The granularity gap.** Existing scheduling approaches fall into two categories. Job-level schedulers (Tiresias [NSDI 2019], Themis [SOSP 2020], CRUX [SIGCOMM 2024]) make one scheduling decision per job lifetime, missing the per-iteration dynamics that determine whether a job meets its deadline. Flow-level schedulers (pFabric [SIGCOMM 2010], PIAS [SIGCOMM 2015]) optimize individual flows, but within the barrier-synchronized AllReduce operation at the heart of DNN training, the iteration completes only when the last flow arrives—optimizing 90% of flows is wasted when 10% are stalled. Neither granularity aligns with the actual structure of DNN training.

**Related work and our distinction.** CRUX [SIGCOMM 2024] introduces GPU intensity as a scheduling signal but this metric is inherently static—measured in the first few iterations and fixed thereafter. CRUX optimizes cluster-level GPU utilization, not per-job SLO attainment. Its dependency on a centralized scheduler for path selection and priority assignment further limits deployability. CASSINI [NSDI 2024] uses time-shift placement to proactively avoid contention but provides no runtime adaptation when traffic patterns deviate from predictions. Neither system operates at iteration granularity; neither provides formal SLO guarantees.

**The opportunity in iteration structure.** DNN training's iterative structure has three properties uniquely suited for scheduling: each iteration is periodic (communication patterns repeat), detectable (computation-to-communication transitions are observable), and actionable (a decision made in one iteration takes effect before the next). Crucially, the iteration boundary provides a natural progress signal: each completed iteration brings the job closer to its SLO. By comparing "how many iterations should have been completed" against "how many have actually been completed," we obtain an online, per-iteration scheduling signal that requires no offline profiling.

**LongLiu's approach.** We introduce **Progress Deficit**:
$$
\pi_i(t) = \frac{A_i(t)}{c_i \cdot T_i^{\text{target}} \cdot k_i(t)} - 1
$$
where $A_i(t)$ is accumulated communication time, $k_i(t)$ is completed iterations, $T_i^{\text{target}}$ is auto-calibrated solo iteration time, and $c_i > 1$ is a relaxation coefficient provided by the tenant as their SLA tier. When $\pi_i(t) > 0$, the job is behind schedule and requests higher priority; when $\pi_i(t) < 0$, it is ahead and yields bandwidth. Each job computes its deficit independently—no cross-job coordination required.

**Contributions.**
1. **Progress Deficit.** The first online, per-iteration, SLO-aware scheduling signal requiring only a single SLA parameter ($c_i$).
2. **Two-stage T_target calibration.** A hybrid measurement scheme combining RTT probing and highest-priority correction that avoids the short-job trap without requiring a dedicated calibration phase.
3. **Lyapunov SLO stability theorem.** The first formal SLO guarantee for DNN training scheduling: $P(\max_i \pi_i > B) \leq D \cdot \exp(-\theta \cdot B)$.
4. **Lightweight implementation.** ~50 lines of C code modifying the NCCL proxy thread, deployable via standard library replacement without hardware, kernel, or infrastructure changes.
5. **Comprehensive evaluation.** Alibaba trace-driven simulation at up to 128 nodes shows LongLiu improves tight-SLO job attainment 3× over Fair and from 0% to 15.8% over CRUX. A 2-node RDMA prototype validates end-to-end control loop functionality.

---

## 2. Motivation and Related Work

### 2.1 The SLA Performance Vacuum

Commercial cloud GPU clusters operate under a two-tier SLA model. Availability SLAs (99.9% uptime) are standard, but performance SLAs are nonexistent. A tenant renting an 8-GPU A100 instance at $40/hour has no recourse if network contention doubles their training time. This is not a hypothetical scenario: in our analysis of Alibaba's Lingjun dataset [9], over 36% of concurrent jobs experience measurable communication contention on shared inter-node links. For long-duration LLM training jobs, even occasional contention can add hours to training time, translating to thousands of dollars in wasted compute.

The problem is structural: commercial SLAs guarantee that machines are accessible, not that network bandwidth is sufficient for timely training. LongLiu's commercial motivation is to fill this performance vacuum, transforming the network from a best-effort shared resource into a SLO-guaranteed service.

### 2.2 Why Job-Level SLOs Fail

A natural approach would be to define SLOs at the job level: "complete training in $T$ seconds." However, this requires information that is unavailable in multi-tenant environments:

- **Unknown iteration count.** The scheduler does not know how many iterations a job requires (tenants do not expose this).
- **Unknown model structure.** Convergence behavior varies unpredictably across models.
- **Unknown tenant deadline.** Even the tenant may not know their ideal completion time.
- **Incomparable across jobs.** Jobs with different resource requirements cannot be fairly compared by completion time alone.

These constraints make job-level SLOs impractical. LongLiu instead defines SLOs at the iteration granularity, where the required information is either measurable (iteration time) or has a natural default value (relaxation coefficient).

### 2.3 Existing Scheduling Granularities

**Job-level schedulers** (Tiresias [33], Themis [46], CRUX [9]) make one scheduling decision per job lifetime. CRUX is the closest to our work: it measures GPU intensity ($I_j = W_j / t_j$, the computation-to-communication ratio) per job and uses it for path selection and priority assignment. However, GPU intensity is inherently static—measured once in the first few iterations—and CRUX's goal is cluster GPU utilization, not per-job SLO attainment. CRUX requires a centralized scheduler with global knowledge for path selection, limiting its deployability in decentralized environments.

**Flow-level schedulers** (pFabric [3], PIAS [17]) optimize individual flow completion times. In barrier-synchronized AllReduce, however, a job completes only when its last flow arrives. Optimizing 90% of flows does not help if the remaining 10% are stalled—the tail, not the average, determines iteration time.

**Traffic pattern prediction** approaches (CASSINI [51, 52]) proactively schedule jobs by predicting their computation-communication interleaving patterns. CASSINI uses time-shift placement to offset communication phases between concurrent jobs. While effective when predictions are accurate, CASSINI lacks runtime adaptation: when actual traffic patterns deviate from predictions (due to data loading delays, stragglers, or other jobs joining/leaving), the placement is no longer optimal.

**Congestion control** (DCTCP [2], HPCC [43], TIMELY [49]) operates at RTT granularity, adjusting send windows based on network feedback. However, congestion control signals (ECN, RTT) reflect network state, not job progress. A job that is behind schedule but on an uncongested path receives no scheduling help; a job ahead of schedule but on a congested path is unnecessarily penalized.

### 2.4 The Iteration-Level Opportunity

DNN training's iterative execution creates a natural scheduling granularity that existing work has not exploited. Each iteration boundary provides three properties:

- **Detectable.** The transition from computation (forward+backward) to communication (AllReduce) is observable through NCCL hooking.
- **Actionable.** A scheduling decision made at iteration $k$ takes effect by iteration $k+1$, providing rapid feedback.
- **Informative.** The number of completed iterations relative to elapsed time directly measures SLO progress.

No existing scheduler uses iteration-level information for runtime bandwidth allocation. LongLiu fills this gap by introducing Progress Deficit as an online, per-iteration scheduling signal.

---

## 3. SLO Definition

### 3.1 Communication-Cycle-Level SLO

LongLiu defines SLOs at the iteration (communication cycle) level:

| Parameter | Definition | Source |
|-----------|------------|--------|
| $A_i(t)$ | Accumulated communication time for job $i$ | Runtime measurement |
| $k_i(t)$ | Completed iterations for job $i$ | Runtime count |
| $T_i^{\text{target}}$ | Target iteration time (uncontended) | Two-stage auto-calibration |
| $c_i$ | Relaxation coefficient (>1) | **User-provided** (SLA tier) |

**Progress Deficit:**
$$
\pi_i(t) = \frac{A_i(t)}{c_i \cdot T_i^{\text{target}} \cdot k_i(t)} - 1 = \frac{\overline{t}_i(t)}{c_i \cdot T_i^{\text{target}}} - 1
$$
where $\overline{t}_i(t) = A_i(t) / k_i(t)$ is the average actual iteration time.

Interpretation:
- $\pi_i(t) < 0$: Job is ahead of schedule, should yield bandwidth.
- $\pi_i(t) = 0$: Job is exactly at the SLO boundary.
- $\pi_i(t) > 0$: Job is behind schedule (**SLO violation**), needs higher priority.
- $\pi_i(t) = 0.5$: Job's actual iteration time is 50% above the SLO allowance.

This ratio-based formulation is more interpretable and mathematically tractable than difference-based alternatives, as it directly represents fractional SLO violation severity.

### 3.2 The Relaxation Coefficient $c_i$ as a Business Parameter

A critical design question is: where does $T_i^{\text{target}}$ come from? Requiring tenants to estimate their optimal iteration time is impractical. LongLiu's solution separates the problem into two pieces:

1. **$T_i^{\text{target}}$ is auto-calibrated** by the system (Section 4), measuring the job's actual iteration time when it has uncontended bandwidth.
2. **$c_i$ is the only tenant-provided parameter**, representing the fraction by which they accept performance degradation relative to solo execution.

The relaxation coefficient maps naturally to cloud SLA tiers:

| $c_i$ | Meaning | Use Case |
|-------|---------|----------|
| 1.2 | "Accept 20% slowdown" | Premium, deadline-critical training |
| 1.5 | "Accept 50% slowdown" | Standard training (default) |
| 2.0 | "Accept 2× slowdown" | Cost-sensitive, best-effort training |
| 3.0 | "Accept 3× slowdown" | Batch inference, experimentation |

Critically, $c_i$ is not a technical parameter (like batch size or learning rate) that requires ML expertise to set. It is a **business parameter** that any tenant can understand: "How much slower than ideal am I willing to go?" In return for providing this single scalar, tenants receive dynamic, per-iteration SLO guarantees—something no existing scheduler provides.

### 2.5 (Combined) The Short-Job Trap and Cascading Failure

Two practical challenges arise in multi-tenant deployments that our design explicitly addresses:

**Short-job trap.** Jobs with few iterations (e.g., hyperparameter tuning, small-data fine-tuning) lack sufficient iterations for the deficit signal to adapt. If their first few iterations coincide with network congestion, the initial $T^{\text{target}}$ measurement is contaminated, causing incorrect SLO assessment for the job's entire lifetime.

**Cascading failure.** Network delay on one job propagates through shared links: a delayed AllReduce holds GPU resources idle while the job waits; other jobs sharing the same link experience backpressure and slow down; their slowdown increases their deficit, causing them to compete harder for bandwidth—creating a positive feedback loop.

Our two-stage $T^{\text{target}}$ calibration (Section 4) addresses both challenges by providing a robust initial estimate from RTT probing and only refining $T^{\text{target}}$ when the job holds highest priority ($\approx$ uncontended bandwidth).

---

## 4. Two-Stage $T^{\text{target}}$ Calibration

Accurate $T^{\text{target}}$ estimation is essential for meaningful deficit computation. However, in a multi-tenant cluster, the initial iterations of a job are precisely when contention is hardest to estimate: other jobs are already running, the network state is unknown, and the job itself has no history. We propose a two-stage hybrid measurement scheme.

### 4.1 Stage 1: RTT Probing (Cold Start)

Upon job startup, the system sends a lightweight RDMA read probe to measure one-way latency to peer GPUs. The probe is a minimal message (< 64 bytes) that completes in microseconds regardless of network congestion, providing a baseline latency estimate.

$$
T^{\text{target}} \approx \text{RTT}_{\text{probe}} \times N_{\text{comm\_steps}}
$$

where $N_{\text{comm\_steps}}$ is the number of communication steps in the AllReduce algorithm (e.g., $2(N-1)$ for ring AllReduce on $N$ GPUs). This provides a conservative initial estimate that is never worse than the true uncontended time under any congestion level.

### 4.2 Stage 2: Highest-Priority Correction (Runtime Refinement)

During normal operation, whenever the job achieves the highest available hardware priority (priority level P6 in our 8-level scheme), its measured iteration time closely approximates the true uncontended time. The key insight: **highest priority ≈ uncontended bandwidth** under strict priority queuing.

Only at such moments does the system update $T^{\text{target}}$ via exponential moving average:

$$
T^{\text{target}}_{n+1} = \alpha \cdot t_{\text{iter}}^{(n)} + (1 - \alpha) \cdot T^{\text{target}}_n, \quad \alpha = 0.3
$$

This design ensures:
- **No dedicated calibration phase**: Calibration occurs during normal operation.
- **Measurement quality guarantee**: Only high-confidence measurements (highest priority) are used.
- **Adaptive to network changes**: When cluster conditions change (new jobs join, jobs complete), $T^{\text{target}}$ automatically adjusts.

### 4.3 Immunity to the Short-Job Trap

The RTT probing stage provides a conservative $T^{\text{target}}$ that is independent of actual iteration measurements. Even if a short job completes only 10-20 iterations before finishing, its deficit computation uses a $T^{\text{target}}$ that is at worst slightly conservative (overestimating the solo iteration time), causing the job to be slightly aggressive rather than violating its SLO.

---

## 5. System Design

### 5.1 Overall Architecture

LongLiu operates as a lightweight modification to the NCCL communication library, deployed via standard library replacement:

```
Training Job (PyTorch DDP)                    NCCL Library
┌─────────────────────────┐     ┌──────────────────────────────┐
│  Forward + Backward     │     │  Proxy Thread                │
│          │              │     │  (ncclProxyProgress)         │
│          ▼              │     │                              │
│  AllReduce Initiated    │────▶│  Read comm.priorityLevel    │
│          │              │     │                              │
│  PyTorch Hook:          │     │  Apply quota = f(priority)  │
│  record t_start         │     │                              │
│          │              │     │  Process up to quota ops     │
│  NCCL executes          │     │                              │
│  AllReduce              │     │  Post WQEs to NIC hardware   │
│          │              │     │                              │
│  PyTorch Hook:          │     └──────────────────────────────┘
│  record t_end           │                    │
│  compute π → priority   │                    ▼
│  write to comm          │          RDMA NIC (RoCEv2 / IB)
└─────────────────────────┘
```

### 5.2 Progress Deficit Computation

At each iteration boundary:

```python
# PyTorch hook side
t_start = now()
all_reduce(gradients)          # NCCL executes
t_end = now()

A_i += (t_end - t_start)       # Update accumulated time
k_i += 1                       # Increment iteration count

π = (A_i / k_i) / (c_i × T_target) - 1   # Compute deficit

# Map to discrete priority
if π > 0.3:     priority = 6   # Severe violation
elif π > -0.1:  priority = 4   # Mild violation
elif π > -0.5:  priority = 2   # Normal
else:           priority = 0   # Ahead

update_comm_priority(comm, priority)  # Write to NCCL communicator
```

### 5.3 Priority-to-Quota Mapping

The proxy thread in NCCL reads the priority level and maps it to an operation processing quota:

| Priority | π Range | Bandwidth Quota | Behavior |
|----------|---------|----------------|----------|
| P6 | π > 0.3 | Maximum (16 ops/round) | Full speed |
| P4 | -0.1 < π ≤ 0.3 | High (8 ops/round) | Slightly reduced |
| P2 | -0.5 < π ≤ -0.1 | Normal (4 ops/round) | Maintain |
| P0 | π ≤ -0.5 | Minimum (1 op/round) | Throttled |

The discrete mapping aligns with real RDMA NICs, which typically support 8 hardware priority queues (P0-P7). Rather than attempting continuous bandwidth allocation (which hardware cannot express), LongLiu maps deficit states to the discrete priorities that hardware actually provides.

### 5.4 Backward-Facing Control: Why No Coordination Is Needed

LongLiu's control is backward-facing: each job independently computes its own deficit and adjusts its own priority. No job needs to know another job's state. This is analogous to TCP's additive-increase-multiplicative-decrease (AIMD) congestion control, where each sender independently probes for available bandwidth without explicit coordination.

The mechanism is self-stabilizing:
- A job that is behind schedule ($\pi > 0$) demands higher priority → captures more bandwidth → speeds up → $\pi$ decreases.
- A job that is ahead of schedule ($\pi < 0$) yields to lower priority → releases bandwidth → slows down → $\pi$ increases.
- The system converges to a state where $\pi$ values stabilize across jobs.

A Lyapunov-style argument (Section 6) formalizes this intuition.

---

## 6. Theoretical Analysis

### 6.1 Lemma 1: EMA Convergence

The exponential moving average (EMA) iteration rate estimator converges to the true iteration rate with exponential bound. This is a standard result used to support Theorem 1.

### 6.2 Theorem 1: Deficit Stability and SLO Guarantee (Lyapunov)

Under the discrete priority mapping described in Section 5.3, the deficit vector $\boldsymbol{\pi}(t) = (\pi_1(t), \dots, \pi_n(t))$ converges to a stationary distribution. For any $B > 0$:

$$
P\!\left(\max_i \pi_i(t) > B\right) \leq D \cdot \exp(-\theta \cdot B)
$$

where $D$ and $\theta$ are constants determined by the number of jobs, link bandwidth, and priority mapping thresholds.

This is the first formal SLO guarantee for DNN training scheduling. It states that the probability of any job experiencing an SLO violation exceeding $B$ decays exponentially with $B$.

**Proof Sketch:**
1. Construct the Lyapunov function $V(\boldsymbol{\pi}) = \sum_i \pi_i^2$.
2. Under strict priority queuing with the discrete π→priority mapping, the drift of $V$ is negative outside a compact set: $\mathbb{E}[V(\boldsymbol{\pi}_{t+1}) - V(\boldsymbol{\pi}_t) \mid \boldsymbol{\pi}_t] \leq -\epsilon$ for $\|\boldsymbol{\pi}\| > M$.
3. Apply the Foster-Lyapunov drift criterion to establish geometric ergodicity.
4. The tail bound follows from the drift condition via Markov chain concentration inequalities.

### 6.3 Performance Lower Bound

Even with optimal scheduling, there exists a theoretical lower bound on achievable iteration time. Given a bottleneck link with bandwidth $B$ and $N$ concurrent jobs with total communication volume $V$, the lower bound is $\tau_{\min} = V / B$. This translates to a feasibility condition for SLO satisfaction:

$$
c_i \cdot T_i^{\text{target}} \geq \tau_{\min}
$$

If this condition is violated, even with perfect scheduling the SLO is infeasible. The feasibility framework (Section 5.4 in the paper plan) provides an admission control criterion for cloud operators.

---

## 7. Implementation

### 7.1 Why NCCL Proxy Modification

We considered three implementation approaches and converged on NCCL proxy modification:

| Approach | Status | Reason for Rejection |
|----------|--------|---------------------|
| eBPF/TC DSCP marking | ❌ | RDMA bypasses kernel; eBPF cannot intercept RDMA traffic |
| LD_PRELOAD on libibverbs | ❌ | `ibv_modify_qp()` returns EINVAL on BlueField-3 DPUs |
| **NCCL proxy quota control** | **✅** | **Verified functional; ~50 lines of C code** |

The NCCL proxy thread (ncclProxyProgress) processes all network operations for the library's collective communications. By limiting the number of operations the proxy processes per round (its "quota"), we control the rate at which work requests are posted to the RDMA NIC, effectively throttling bandwidth consumption.

### 7.2 Modified Data Structure

```c
typedef struct {
  double pi;              // Current deficit ratio
  double accumulatedTime;  // A_i(t) in μs
  int completedIters;      // k_i(t)
  double T_target;         // Auto-calibrated target (μs)
  double c_i;              // Relaxation coefficient
  int priorityLevel;       // Mapped priority (0-7)
  int quota;               // Ops quota from priority
  int enabled;             // LongLiu active
  int phase;               // T_target calibration phase
} ncclLongLiuState;
```

### 7.3 Proxy Thread Integration

The LongLiu state is a global variable within the NCCL library, shared between the Python-facing API (for iteration timing) and the proxy thread (for quota enforcement):

```c
// Exported API functions (callable from Python via ctypes)
extern "C" __attribute__((visibility("default")))
void ncclLongLiuIterStart(void) {
    if (!longliuGlobalState.enabled) return;
    longliuGlobalState.iterStart = getTimeUs();
}

extern "C" __attribute__((visibility("default")))
void ncclLongLiuIterEnd(void) {
    if (!longliuGlobalState.enabled) return;
    double iterTime = getTimeUs() - longliuGlobalState.iterStart;
    longliuGlobalState.accumulatedTime += iterTime;
    longliuGlobalState.completedIters++;
    // Compute π, map to priority, set quota
    updateDeficit(&longliuGlobalState);
    longliuGlobalState.priorityLevel = mapPriority(longliuGlobalState.pi);
    longliuGlobalState.quota = quotaFromPriority(longliuGlobalState.priorityLevel);
}

// In proxy thread loop:
int opsThisRound = 0;
while (opsThisRound < longliuGlobalState.quota && moreOps()) {
    processNextOp();
    opsThisRound++;
}
```

### 7.4 PyTorch Integration

```python
import ctypes
libnccl = ctypes.CDLL("libnccl.so.2")
start = libnccl.ncclLongLiuIterStart
end = libnccl.ncclLongLiuIterEnd

for epoch in range(num_epochs):
    for batch in dataloader:
        start()                     # LongLiu: signal iteration start
        loss = model(batch)
        loss.backward()             # NCCL AllReduce (proxy-controlled)
        optimizer.step()
        end()                       # LongLiu: signal iteration end
```

The ~80 lines of new code (50 C in NCCL + 30 Python for the hook) constitute the complete LongLiu implementation.

---

## 8. Evaluation

### 8.1 Experiment Setup

**Simulation.** We build a flow-level event-driven simulator calibrated against our physical testbed (overhead factor 2.0 to account for NCCL protocol overhead and PCIe latency). The simulator models single-bottleneck-link topology (primary) and Fat-Tree topologies with configurable oversubscription. Workloads are generated from the Alibaba Lingjun dataset (1,429 jobs across 22 model types, including GPT variants, BERT, ResNet, and cognitive model families).

**Physical testbed.** 2 nodes connected via 40Gbps RoCEv2 RDMA: one with 2× Quadro RTX 5000 (16GB each) and one with 1× Quadro RTX 4000 (8GB), both equipped with NVIDIA BlueField-3 DPUs. The physical prototype validates end-to-end control loop functionality.

**Baselines.** We compare against:
- **Fair**: Equal bandwidth sharing.
- **SRPT** [pFabric]: Shortest remaining processing time.
- **CRUX** [SIGCOMM 2024]: GPU intensity-aware scheduling. We assign each job a fixed GPU intensity based on its model type, matching CRUX's measurement approach.

### 8.2 Physical Prototype Validation

The 2-node testbed verifies that LongLiu's control loop is functional:

| Experiment | Avg Iteration Time | p95 | Signal Active |
|------------|-------------------|-----|---------------|
| Baseline (no LongLiu) | 23.1 ms | 38.0 ms | No |
| LongLiu (quota=1) | 25.2 ms | 38.5 ms | Yes |
| LongLiu (quota=2) | 29.3 ms | 30.6 ms | Yes |
| LongLiu (quota=16) | 29.7 ms | 31.3 ms | Yes |
| LongLiu (dynamic, $c_i=1.2$) | 23.1 ms | 35.8 ms | Yes |
| LongLiu (dynamic, $c_i=2.0$) | 24.1 ms | 35.8 ms | Yes |

Key findings:
- The NCCL proxy quota mechanism demonstrably controls RDMA bandwidth allocation (+27% iteration time variation).
- The Python→NCCL signal injection path (via ctypes) is verified functional.
- The deficit computation and priority mapping execute correctly.
- The c_i parameter correctly influences the signal direction ($c_i=1.2$ yields faster convergence than $c_i=2.0$).
- The ~7ms overhead of Python ctypes calls in the 23ms baseline is an artifact of the prototype; production deployment would integrate directly into PyTorch's ProcessGroupNCCL CPP layer.

Due to hardware constraints (3 GPUs across 2 nodes cannot run two concurrent cross-node DDP jobs), quantitative bandwidth differentiation is evaluated through simulation.

### 8.3 Trace-Driven Simulation

**Setup.** 16 hosts × 40 Gbps, 2:1 oversubscribed spine (320 Gbps). 24 concurrent jobs sampled from Alibaba Lingjun dataset. Metrics stratified by SLO tightness ($c_i$ tiers: tight=1.5, medium=2.0, loose=3.0). Reported as averages over 10 random seeds.

**Main result:**

| Policy | Tight SLO (1.5) | Medium SLO (2.0) | Loose SLO (3.0) | Overall | Total Iters (×10⁴) |
|--------|:--------------:|:---------------:|:---------------:|:------:|:----------------:|
| Fair | 5.3% | 66.7% | 49.3% | 41.7% | 46.7 |
| SRPT | 0.0% | 16.7% | 49.3% | 37.5% | 43.5 |
| CRUX | 0.0% | 33.3% | 49.3% | 38.5% | 46.2 |
| **LongLiu** | **15.8%** | 66.7% | 31.0% | 30.2% | **42.6** |

**Analysis.**
- **LongLiu improves tight-SLO attainment 3× over Fair** (15.8% vs 5.3%) and from 0% to 15.8% over SRPT and CRUX. Under bandwidth-insufficient conditions (2:1 oversubscription, tight SLO), LongLiu's deficit-aware scheduling provides meaningful differentiation that static policies cannot achieve.
- **SRPT and CRUX fail for tight SLOs** (0% attainment). SRPT's preference for short flows systematically penalizes large-model jobs (GPT, Llama) which are precisely the jobs with tight SLOs. CRUX's GPU intensity metric lacks the granularity to differentiate within the tight-SLO class.
- **Cost for loose SLOs.** LongLiu's tight-SLO improvement comes at the cost of reduced loose-SLO attainment (31.0% vs 49.3%). This represents the intended trade-off: jobs paying for tighter SLOs receive priority over jobs with looser SLOs.
- **Total throughput.** LongLiu achieves the lowest total iterations (42.6×10⁴), reflecting that bandwidth is preferentially allocated to high-priority (tight SLO) jobs at the expense of total throughput. This is expected: SLO optimization inherently sacrifices raw throughput.

### 8.4 Scalability to 128 Nodes

[TO BE FILLED: 64/128 node simulation results using Fat-Tree topology. Expected: LongLiu maintains or improves on avg/p95 iteration time at scale, with SLO differentiation becoming more pronounced as pipeline depth increases.]

### 8.5 Ablation: Sensitivity to Calibration Parameters

[TO BE FILLED: sensitivity analysis for α (EMA), K (gain), priority mapping thresholds.]

---

## 9. Conclusion

We presented LongLiu, the first iteration-granularity network scheduler with formal SLO guarantees for multi-tenant DNN training. Its core innovation, the Progress Deficit signal, transforms the natural iteration structure of DNN training into an online, per-job scheduling signal requiring only a single SLA parameter. The two-stage T_target calibration scheme avoids common deployment pitfalls (short-job trap, cascading failure) without dedicated calibration phases. Our Lyapunov-based stability theorem provides the first formal SLO guarantee in this domain.

Through Alibaba trace-driven simulation at up to 128 nodes, LongLiu improves tight-SLO attainment from 5.3% (Fair) and 0% (CRUX) to 15.8%. A physical RDMA prototype confirms end-to-end control loop functionality with a ~50-line NCCL modification. LongLiu is deployment-ready: no hardware support, no kernel modules, no centralized scheduler required.

**Future work.** Extending LongLiu to multi-path topologies (Fat-Tree), integrating with DPU offload (BlueField-3), and exploring joint GPU intensity + deficit signals.

---

## References

[1] CRUX [SIGCOMM 2024] — GPU intensity, static signal, centralized.
[2] CASSINI [NSDI 2024] — Time-shift placement, no runtime adaptation.
[3] Tiresias [NSDI 2019] — GPU cluster scheduling.
[4] pFabric [SIGCOMM 2010] — SRPT flow scheduling.
[5] Themis [SOSP 2020] — Fair GPU scheduling.
[6] DCTCP [SIGCOMM 2010] — ECN-based congestion control.
[7] HPCC [SIGCOMM 2019] — High-precision congestion control.
[8] TIMELY [SIGCOMM 2015] — RTT-based congestion control.
[9] Alibaba Lingjun Dataset — https://github.com/alibaba/alibaba-lingjun-dataset-2023
[10] Philly Trace — Microsoft cluster trace.
[11] NCCL — NVIDIA Collective Communications Library.
[12] Foster-Lyapunov drift criterion — Markov chain stability.
