# LongLiu 补充实验方案（物理锚点三件套）

> 目标：为仿真结论补齐三个物理锚点——**静态（转正 0.2%）、动态（tier swap）、规模/稀缺（多作业阶梯）**。
> 环境基准：2 台 GPU 服务器（下称 N1/N2），1 台交换机，50G RoCEv2 链路，NCCL shim 已部署（EQ5 现有 setup）。除标注外全部实验在此环境内完成，无新增设备。

---

## 0. 前置事项（动手前必做，1 天）

### 0.1 核实链路速率与论文口径
论文写 "100Gbps RoCEv2 testbed"，当前环境为 50G。二选一：
- 若论文数据来自另一套环境 → §V-B/§VI-A5 写明该环境配置（交换机/NIC 型号、速率）；
- 若就是这套 50G → **全文 100G 改为 50G**，包括 §IV-E 的 "per-flow line rate (100 Gbps in our testbed)"。

### 0.2 环境清单（填入 §V-B 的硬件表，即现在缺失的 `tab:testbed-hw`）

| 项 | guolab-10 (N1, 10.1) | guolab-226 (N2, 226) |
|---|---|---|
| **GPU** | 1x Quadro RTX 4000 (8GB) | 2x Quadro RTX 5000 (2×16GB) |
| **NIC 型号** | ConnectX-6 Dx (vendor_part_id=4125) | BlueField-3 B3220 (vendor_part_id=41692) |
| **NIC 固件** | 22.41.1000 | 32.41.1000 |
| **MLNX_OFED** | 5.8-3.0.7 | 25.10-1.2.8 |
| **链路速率** | **50 Gbps** (enp130s0f0np0) | **100 Gbps** (enp59s0f0np0) |
| **OS** | Ubuntu 20.04 / Python 3.8 | Ubuntu 22.04 / Python 3.10 |
| **NCCL** | 2.18.3 (DSCP 版本) | 2.18.3 (DSCP 版本) |
| **交换机** | Mellanox Spectrum SN2700 — 32×100GE QSFP28, Cumulus Linux 5.1.0, 2 RU | (同左，共用) |

**关键约束**：
- 有效带宽受限于 50G 侧（N1 链路速率）
- NIC 型号不对称：N1 为 ConnectX-6 Dx，N2 为 BlueField-3
- OFED 版本差异大：N1 用 5.8，N2 用 25.10
- N2 (BlueField-3) 的 DSCP 分类能力需通过探针确认（见 0.3 节）

### 0.3 DSCP→TC 探针（一切优先级实验的前置）
按 Pitfall I 的方法在 **N1、N2 各自的 NIC** 上先跑 full-class probe：P3 背景流打满，逐个标记 DSCP 0–56，读 `tx_prio` 计数器确认真映射。
**没确认映射正确之前，不做任何优先级实验**——你们自己的 pitfall 故事说的就是这一步会静默出错。
同时确认：交换机 strict-priority 配置生效、PFC 关闭。

---

## 实验 A：静态锚点转正——0.2% 保真度证据表（2–3 天）

**目的**：把"within 0.2%"从断言变成可核查的结果；同时拆掉 calibration/validation 混淆。

### A.1 方法
1. 在测试床上运行 **≥6 个静态场景**（不要只用 EQ5 那一个）：组合维度 = {作业数：2 / 2+背景流} × {容量：50G / per-QP 限速到 35G / 25G} × {c_i：1.2 / 1.5}；
2. 每个场景记录测试床实测值：**分配结果**（每作业所得带宽）与**达成率**（slowdown / attainment）；
3. 将每个场景**逐比特**搬进仿真器（相同作业、anchor、容量），输出同两名目；
4. **Hold-out**：6 个场景中留 1 个，其 anchor 不取自被对比的那次硬件运行（用另一次运行的 anchor 或理论值），单独标注。

### A.2 误差口径（写进 caption）
- slowdown 相对误差 |slowdown_sim − slowdown_hw| / slowdown_hw，报 **max 和 mean**；
- attainment 差值以**百分点**报。

### A.3 产出（进论文的表）
表：`场景 | 容量 | c_i | slowdown_hw | slowdown_sim | 误差 | attain_hw | attain_sim | 差值`，hold-out 行打 †。
放置：§V-F（替换现在 0.2% 的一句话），§VI-A1 改引用它。若误差确实在 0.2% 量级，原数字保留并注明 max/mean 和 N=6；若不到，**如实写实际数字**——0.5% 配一张表也远比无表的 0.2% 硬。

---

## 实验 B：动态锚点——硬件 tier swap（最高优先级，3–5 天）

**目的**：给 EQ3（全文最强结论：swap 后 LongLiu 100% vs 静态 10%）补第一个硬件动态证据。

### B.1 场景设计（E3 的最小硬件版）
- 两个真实 NCCL 训练作业 J_A、J_B（各跨 N1/N2 两节点，复用 EQ5 的作业与 5 分钟 warmup）；
- 初始 tier：J_A = premium（c=1.2），J_B = standard（c=2.0）；
- **t = T_swap 时反转**：J_A → standard（c=2.0），J_B → premium（c=1.2），通过 shim 环境变量/API 在线改；
- 两臂：**LongLiu 臂**（动态 P6→P2）与**静态臂**（开工贴一次标签不动，等价于 CRUX 在此拓扑的行为）。

### B.2 测量窗口（与仿真 EQ3 严格对齐）
- W1 = [T_swap−100s, T_swap]，W2 = [T_swap, T_swap+100s]，W3 = [T_swap+200s, T_swap+300s]；
- 指标：紧 SLO 作业的 per-epoch slowdown（无单位比值，遵守语义对齐原则）；从 shim JSON trace 直接算。

### B.3 轮次与统计
≥4 轮，交替 {哪作业先启动} × {哪臂先跑}；每轮配置冻结、md5 校验、日志归档（沿用你们 rep1 排除的惯例）。

### B.4 成功判据（定性对齐，不追求数值一致）
- LongLiu 臂：swap 后 J_B 的 slowdown 在 W3 回到与 W1 中 J_A 相当的水平——**无瞬态崩溃**；
- 静态臂：swap 反转后失锁，W3 slowdown 显著高于 W1；
- 与仿真 Fig. 9/10 的**形状**对比（升降方向、有无瞬态），caption 里明说 "qualitative shape, not absolute values"。

### B.5 产出
进论文：EQ5 新增一段（或 EQ3 末尾加 hardware cross-check 段），一张双臂 slowdown 轨迹图（沿用 fig6 的画法）。

### B.6 风险预案
- 若 shim 暂不支持在线改 c_i → 最小改动：支持 epoch 边界重读环境变量，或重启作业续 checkpoint 模拟 swap（需在论文注明实现方式）；
- 若静态臂在双作业下退化不明显 → 加一路 P3 iperf3 背景流抬高竞争强度（EQ5 已有此机制）。

---

## 实验 C：规模/稀缺锚点——CPU epoch 模拟器多作业阶梯（1–2 周，含开发）

**目的**：把硬件验证从 2 作业扩到 8–12 作业、从单点扩到 3 个稀缺 regime，对标 E1 梯子。

### C.1 模拟器设计（主要工程量）
- **形态**：每作业 = 一对进程（N1、N2 各一，互为对端），独立 QP 集合 + 独立 DSCP 标记；双机各跑 4–6 对，共 8–12 个"作业"；
- **行为循环**（严格按串行迭代模型）：
  `sleep(T_comp_j + jitter)` → 双端 RDMA write 互推共 D_j 字节（当前 DSCP）→ 双向完成确认（作业内 barrier）→ 记录本轮起止时间戳 → 下一轮；
- **jitter**：注入已知分布计算噪声（如 σ = 5%·T_comp 的高斯抖动）——顺带受控验证 SNR/epoch 平均论证；
- **同步**：作业间不需要任何同步（各自的 T_comp 和随机相位偏移自然产生 phase staggering）；
- **实现路线**：优先 ibv verbs（可打 DSCP、与 shim 机制同构）；快速原型可用 perftest 包装，但 DSCP 动态修改需要 verbs 级的 `ibv_modify_qp`。

### C.2 执行平面（关键工程决策）
shim 挂在 NCCL 上，模拟器不是 NCCL，因此需要把 **anchor 估计 + 闭式分配 + DSCP 更新**抽成一个**独立守护进程**：
- 输入：各作业 epoch 统计（进程写共享日志/UDP 上报）；
- 逻辑：完全复用 shim 的分配代码（保证"被验证的就是论文的"）；
- 输出：对模拟器 QP 执行 `ibv_modify_qp`。
工作量估计：shim 逻辑抽库 ~2–3 天，守护进程 + 模拟器 ~3–4 天，联调 ~2 天。

### C.3 场景（与 E1 同比值，需求放大法）
固定 50G 链路，用作业集让 Σ_P b^att + βΣ_S b^att 落在三个比值：
| Regime | Σb^att/B | 对应 E1 容量点 |
|---|---|---|
| 深度稀缺 | ≈1.5 | 400G 点 |
| 过渡区 | ≈1.2 | 630G 点 |
| 充裕 | ≈0.96 | 1200G 点 |
做法：从冻结的 E1 作业集里选 8–10 个作业，按 50G 链路**等比缩放 D_i** 使 b^att 之和命中比值（b^att 公式直接反解）。两臂：LongLiu 守护进程 vs 静态优先级（可加 fair 臂=不控）。

### C.4 测量与判据
- 指标：每作业 per-epoch slowdown → P-attn（窗口与判定口径同仿真）；
- ≥3 轮/场景，报 mean±std；
- **交叉验证**：同场景进仿真器，对比 {各臂排名、regime 边界位置、优势幅度趋势}——判定为**定性一致**（排名一致、稀缺区出现分化、充裕区收敛），不做绝对值比对；
- 若一致 → 论文里 E1 加一句 "the ladder's regime structure is reproduced on hardware with 10 emulated jobs (§VI)"；若不一致 → 这本身就是重要发现，先查执行平面，再查仿真器。

### C.5 产出
进论文：§V-F 或 §VI 新增小节 + 一张三点的硬件-仿真对照表（场景 × 臂 × P-attn_hw × P-attn_sim）。

---

## 加分项（穿插做，每个 1 天左右）

1. **跨 NIC 代际 DSCP 探针表**：full-class probe 在 CX-5/6/7/BF-3 各跑一遍 → "型号 × 分类行为 × TC 映射"表，pitfall 升级为跨代际调查（纯网络，guolab 各机直接做）；
2. **shim 开销**：proxy 线程 CPU%、`ibv_modify_qp` 延迟分布、epoch 处理耗时（现 setup 直接测）→ 一句话进 §V 或 Discussion；
3. **anchor 收敛图**：从已有 JSON trace 画 EMA 带宽估计随 epoch 的收敛曲线 → 进 §IV-B；
4. **EQ5 补轮**：再跑 2–4 轮（交替顺序），加厚统计，复查 rep2_r1 打平轮。

## 实验记录规范（所有实验统一）
- 每个场景定义先冻结再跑，与产物同版本（沿用现有 evidence 惯例）；
- 每轮：配置文件 + md5 + JSON 日志三件套归档；
- 跨平面比较一律用 slowdown / attainment 比值，绝不用绝对时间；
- 打平/异常轮次如实记录，不删。

## 时间表（串行约 3 周，可并行压缩）
| 周 | 内容 |
|---|---|
| W1 | 前置（0.1–0.3）+ 实验 A；启动守护进程抽库 |
| W2 | 实验 B 全量；模拟器开发 |
| W3 | 实验 C 联调 + 场景扫描；加分项穿插 |
| W4 弹性 | 补轮、复盘、进论文（表格/图/段落） |
