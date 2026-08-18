# 物理线 Evidence Package（论文支撑材料）

> 最后更新：2026-07-24
> 对应文档：`QUOTA_EXPERIMENT_RESULTS.md`

---

## a. 机制环路证据

**V4/V5 优先级交叉图** — 物理床首要定位是机制验证而非统计性 outcome。

- V4 同 payload 设计绕开主机 NIC 瓶颈，证明 DSCP 优先级在 RoCE 路径上有效
- V5 优先级轨迹清晰显示 π→priority 闭环：Progress Deficit 正确触发优先级翻转
- **核心证据**：QUOTA_EXPERIMENT_RESULTS.md V4/V5 段的优先级交叉图和 π 轨迹表

**V6 闭环确认**：
- TC 映射修正后，LongLiu 紧 job 升到 P4(DSCP=0→tc:1)，比 CRUX 静态 P3(DSCP=16→tc:2) 高一档 TC
- 13-15% slowdown 差距来自这一个 TC 等级差
- P6(DSCP=8→tc:0) 全程可用但 π 未越阈值（未触发），不影响结论

---

## b. 映射反转发现 + 修正

**系统洞见**：NIC 的 TC 映射与软件优先级命名空间解耦，`trafficClass = priority × 8` 的默认映射在严格优先级调度下产生 **优先级反转**（P6→DSCP=48→tc:6 最低）。

**发现路径**：
1. V6 旧映射实验发现 LongLiu 升频反而 slowdown 更高（倒挂）
2. `mlnx_qos -i mlx5_0 --trust dscp` 读取硬件 TC 映射
3. 全类探测（P0-P7 vs 6G P3 背景流）证实 `p×8` 映射下 P6 劣于 P4/P3
4. 修正映射为 P6→DSCP=8→tc:0（最高）, P4→DSCP=0→tc:1, P3→DSCP=16→tc:2

**论文价值**：此发现值得独立一小节——默认 DSCP→TC 映射在 RoCEv2 多优先级场景下不保证按命名排序。

---

## c. V6 Outcome：Phase 2 tight 13-15% 无重叠优势

| 指标 | 值 |
|------|-----|
| 条件 | 6 Gbps 背景流（P3→tc:2）, c_i tight=1.2/loose=3.0, warmup 5 min |
| 对比 | LongLiu (P4→tc:1) vs CRUX (P3→tc:2) |
| Phase 2 tight B | LL **1.08-1.19×** vs CRUX **1.29-1.33×**（V6-P4 原始） |
| 优势幅度 | **13-15%**，无区间重叠 |
| 统计重演 | **5/6 轮决定性**，1/6 重叠不定（噪声假设） |
| 重演范围 | 2 独立复制 × 2 方向 = 6 数据点 |

**关键发现**：
- Phase 2（后启动 Job）才是调度器优势的放大镜
- Phase 1（先启动 Job）持平略优，因争抢还未建立
- 6G 背景流下争抢强度不足以触发 P6（π≤0.24），P4→tc:1 高一档即足够

---

## d. 226 分类能力 — 结论修正（2026-08-10）

> **修正声明**：2026-07-24 曾基于 iperf3 反向探针判定"226 不分类"。
> 2026-08-10 用更严格的手段复查，**推翻了该结论：226 出口同样按 DSCP 分类**。
> 下方保留旧证据并记录推翻过程，论文以修正后结论为准。

### 旧结论（2026-07-24，已推翻）

判定"226 不分类"的依据：
1. 反向模式 iperf3 探针（226→10.1, DSCP 0-56）：226 全部 8 个 tx_prio 计数器仅 tx_prio0 有增量
2. 6G P3 背景流接收时（10.1→226）：226 仅 rx_prio0 有增量（~43 GB over 65s）
3. 226 与 10.1 NIC 型号/驱动不同（BlueField-3 vs ConnectX-6 Dx）

### 修正后证据（2026-08-10，决定性）

**A. 反向 UDP 探针（Python socket，绕过 iperf3 控制连接）**：
从 226 直接以 `setsockopt(IP_TOS)` 发 UDP（4s，~1.4MB 包）到 10.1，采样 226 出口
`ethtool -S enp59s0f0np0` 的 tx_prio 计数器增量：

| ToS | DSCP | 226 tx_prio 增量 | 判定 |
|:---:|:----:|:----------------:|:----:|
| 32 | 8  | tx_prio1 +2.00 GB | 分类 ✓（DSCP8→tc:0→prio1） |
| 64 | 16 | tx_prio2 +1.60 GB | 分类 ✓（DSCP16→tc:2→prio2） |
| 0  | 0  | tx_prio0 +1.49 GB | 分类 ✓（DSCP0→tc:1→prio0） |

与 10.1 NIC 的硬件映射表完全一致（§e）。

**B. 并发 NIC 采样（test1 R5，2026-08-10）**：
P6(P3 jobA) 与 P3(jobB) 并发期间，226 与 10.1 两端 tx_prio 增量几乎完全一致：
- 226: tx_prio2 +98.2 GB, tx_prio1 +41.7 GB；10.1: tx_prio2 +95.1 GB, tx_prio1 +40.5 GB

**旧证据为何不可信（归因）**：
- iperf3 反向探针依赖 TCP 控制连接；10.1 防火墙对 5205 端口 REJECT
  （ICMP 通但 TCP SYN 被拒 → "No route to host"），探针实际未建立数据流，
  仅极少量包发出 → 计数器增量低于阈值，误判"不分类"
- 或期间 226 QoS 配置曾被重置（无法追溯，以当前硬件计数器为准）

**影响（更新 v6，2026-08-10）**：
- **两端 NIC 均按 DSCP 分类**，物理床不存在"单向分类"的不对称拓扑
- **SP 队列严格性（三组受控实验，全部 3 轮）**：
  - test3（NCCL 连续通信）：并发窗口内抢占度 **58.7%±0.0**，P6:P3 ~58.5:41.5，
    P3 残存 solo 44.6%（`analysis/exp2_test3_report.md`）
  - 对照（两流同为 DSCP16）：**精确 50:50** 均分（`exp2_ctrl_dscp16_report.md`）
  - perftest（持续饱和流）：高优先级满速 45.92±0.10 Gb/s，低优先级 **饿死
    99.9%**（`exp2_perftest_report.md`）
- **v6 判定**：SP 队列**严格 per-packet**（perftest 实锤）；test1/test3 的 58%
  系 **NCCL 流量包级突发间隙**所致（高优先级 tc:0 队列微秒级空闲被低优先级利用），
  **非硬件非严格、非配置错配**。早期 v5 判定（"SP 非严格/物理上限"）已推翻。
- 论文 §拓扑约束 应改为：双端分类能力一致；SP 严格（持续流下饿死低优先级），
  NCCL 场景分配受流量占空比/间隙结构限制（突发 vs 持续 → 59:41 vs 99.9% 饿死）

---

## e. 探测实证 DSCP→TC 映射表

`mlnx_qos -i mlx5_0 --trust dscp` 确认的硬件映射：

| 优先级 | DSCP | TC | 严格优先级次序 | 用途 |
|:-----:|:----:|:--:|:------------:|:----:|
| P6 | 8 | tc:0 | 最高 | LongLiu 最高优先级（本轮未触发） |
| P4 | 0 | tc:1 | 第二 | LongLiu 紧 job 实际使用 |
| P3 | 16 | tc:2 | 第三 | 背景流 + CRUX 静态 |
| P2 | 24 | tc:3 | 第四 | LongLiu 松 job/初始化 |
| P1 | 32 | tc:4 | 第五 | LongLiu 松 job |
| P0 | 40 | tc:5 | 第六 | 未使用 |
| — | 48 | tc:6 | 第七 | 旧 P6（反转） |
| — | 56 | tc:7 | 最低 | 未使用 |

**NIC 计数器验证**：
- 10.1 `tx_prio3`（DSCP=24→P3 背景流）= 5.3 TB
- 10.1 `tx_prio4`（DSCP=32→P4 P4 天花板实验）= 49 GB
- 10.1 `tx_prio6`（DSCP=48→旧 P6）= 1.2 GB
- 226 全部 8 个 prio 计数器 = 0（不分类）

---

## 遗留对齐项（论文动笔前必做）

### 5a. T_target 的 SLO 定义与锚点语义

**问题**：物理侧 T_target 使用"per_epoch_ms"单位，其语义为"epoch 内纯通信时间"。但 SLO 实际由 "communication SLO + overlap synthesis" 组合构成——即通信可以与计算重叠，真正的 end-to-end SLO 是通信时间和计算时间的联合分布。

**需要对齐**：
1. **定义统一**：T_target 是"纯通信时间"还是"通信 + overlap 合成时间"？
2. **锚点选择**：V5 校准基于 solo 1024MB AllReduce 的 epoch 级通信时间，此锚点是否与仿真器建模的 T_target 定义一致？
3. **修正路径**：若定义不一致，需要在论文动笔前统一语义——最简单的方案是保留当前"纯通信时间"定义，在论文中明确声明 T_target 仅覆盖通信分量，overlap 效应在 SLO 松弛量中吸收。

**这是两线（物理床 + 仿真器）合流的最后一处血缘不一致**，解决后方可统一撰写论文的 eval 章节。
