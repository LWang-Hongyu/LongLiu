# 实验 C v2：场景参数表（50G 链路，单机多进程 epoch 模拟器）

设计目标：让三种臂各自暴露出**不同的、可预测的失效模式**，且失效由场景参数决定、而非由测量噪声决定。

环境常量：B = 50 Gbps，β = 0.5，W = 20（epoch 长度），分析窗口 epochs 5–19。

---

## 0. 三条构造铁律（上一轮失败的原因）

| # | 铁律 | 上一轮的问题 |
|---|------|--------------|
| 1 | **真实争用**：总供给负载 Σb̄（b̄ = φ·B，φ = Tcomm/(Tcomp+Tcomm)）在稀缺 regime 必须 ≥ 1.0×B | 旧数据 slowdown 普遍 < 1，说明链路根本没打满，各臂无差异可寻 |
| 2 | **fair 失效条件**：premium b^att ≥ 1.5 × (B/N) | 旧场景 premium b^att ≈ B/N，fair 恰好能满足 premium，于是"反常获胜" |
| 3 | **LongLiu 达标条件**：λ ≥ 0.8 时 premium  slowdown 才 ≤ c_eval（见 §1 的 policy/eval 拆分） | 旧场景没有核算 λ，指标口径又是错的，无法归因 |

**结论先行：旧 27 轮数据的"fair 反常"≈100% 是场景构造问题（铁律 1、2 同时违反），指标 bug 只是让报告数字更难看。**

---

## 1. 关键技巧：policy c 与 eval c 拆分

- **policy c_i = 1.35**：写进作业注册、用于算 b^att 的 c。
- **eval c_i = 1.5**：论文/报告里宣称的 SLO，用于判 attainment。

这样 premium 在 λ=1 时实际 slowdown = 1.35，离 eval 线 1.5 有 0.15 的余量，
可吸收 λ 下降到 ≈ 0.8 以及 ±5% 的测量噪声，binary attainment 不会贴在边界上翻硬币。
standard 不做拆分（c = 2.0 同时用于 policy 和 eval）。

---

## 2. 场景 S1：premium 保护梯子（复现仿真 EQ1 的主场景）

5 作业（2 premium + 3 standard），B/N = 10 G。

### 基础作业集（moderate，×1.0）

| 作业 | tier | c_policy | c_eval | Tcomp | Tcomm,solo | D/epoch | φ | b̄ | b^att | b^att/(B/N) |
|---|---|---|---|---|---|---|---|---|---|---|
| J0 | P | 1.35 | 1.5 | 25 ms | 4.9 ms | 31 MB | 0.16 | 8.2 G | **16.0 G** | 1.60× |
| J1 | P | 1.35 | 1.5 | 35 ms | 6.9 ms | 43 MB | 0.16 | 8.2 G | **16.0 G** | 1.60× |
| J2 | S | 2.0 | 2.0 | 30 ms | 4.7 ms | 30 MB | 0.14 | 6.8 G | 6.0 G | 0.60× |
| J3 | S | 2.0 | 2.0 | 40 ms | 5.0 ms | 31 MB | 0.11 | 5.6 G | 5.0 G | 0.50× |
| J4 | S | 2.0 | 2.0 | 50 ms | 4.8 ms | 30 MB | 0.09 | 4.3 G | 4.0 G | 0.40× |

Σ_P = 32.0 G，Σ_S = 15.0 G，C\* = 39.5 G（0.79B），λ = 1.0，offered = 33.2 G（0.66B）。

### regime 由 D 统一缩放（Tcomp 不变，其余列按比例重算）

| regime | D 缩放 | Σ_P | Σ_S | C\*/B | λ | offered | 定位 |
|---|---|---|---|---|---|---|---|
| ample | ×0.6 | 23.2 G | 9.8 G | 0.56 | 1.00 | 0.42B | 人人达标，验证"LongLiu 无代价" |
| moderate | ×1.0 | 32.0 G | 15.0 G | 0.79 | 1.00 | 0.66B | fair 开始伤 premium |
| deep | ×1.3 | 36.8 G | 18.4 G | 0.92 | 1.00 | 0.83B | fair premium 明显违约 |
| very-deep | ×2.0 | 44.7 G | 23.3 G | 1.13 | 0.84 | 1.13B | 唯一 λ<1 的点，考验 policy/eval 余量 |

### 预测（流体近似，方向性，±10%）

| regime | 指标 | LongLiu | fair | static |
|---|---|---|---|---|
| ample | 全部 | 全员达标 | 全员达标 | 全员达标 |
| moderate | P-attn / J0 s | **2/2 (s=1.35)** | 0/2 (s=1.66) | 2/2 (s=1.16) |
| deep | P-attn / J0 s | **2/2 (s=1.35)** | 0/2 (s=1.82) | 2/2 (s=1.20) |
| very-deep | P-attn / J0 s | **2/2 (s=1.47)** | 0/2 (s=2.13) | 2/2 (s=1.28) |
| deep | J3(standard) s | 3.2（floor 有界） | 2.0 | ~2.0（余量大） |

叙事：**S1 里 LongLiu 与 static 并列最优（premium 都达标），fair 的 premium 违约随稀缺加深（1.66→1.82→2.13），复现仿真的梯子。**
S1 里 standard 轴上 LongLiu 劣于 fair/static —— 这是模型内生的代价（Σ_P 占走后 standard 只剩 floor），
不要说"LongLiu standard 也更好"，要说"LongLiu 的 standard 退化是**有界的**（floor β·b^att 保底，s ≤ ~3.3），
而 fair/static 对另一阶层的伤害是**无界的**（随负载增长）"。S2 专门证明后一句。

注意：S1 里 static 不伤 premium 是因为只有 2 个 premium、互不挤兑。static 的失效由实验 B（tier swap）和 S2 展示，不要指望 S1 同时打倒 static。

---

## 3. 场景 S2：饥饿对照（证明"有界 vs 无界"）

6 作业（3 个通信密集型 premium + 3 standard）。premium φ=0.5，瞬时需求 3×50G ≫ B。

| 作业 | tier | c_policy | c_eval | Tcomp | Tcomm,solo | D/epoch | φ | b̄ | b^att |
|---|---|---|---|---|---|---|---|---|---|
| J0–J2 | P | 1.35 | 1.5 | 20 ms | 20.0 ms | 125 MB | 0.50 | 25.0 G | 29.4 G |
| J3 | S | 2.0 | 2.0 | 30 ms | 7.1 ms | 44 MB | 0.19 | 9.5 G | 8.0 G |
| J4 | S | 2.0 | 2.0 | 40 ms | 9.4 ms | 59 MB | 0.19 | 9.5 G | 8.0 G |
| J5 | S | 2.0 | 2.0 | 50 ms | 11.8 ms | 74 MB | 0.19 | 9.5 G | 8.0 G |

Σ_P = 88.2 G，Σ_S = 24.0 G，C\* = 100.2 G（2.00B），λ = 0.43，offered = 103.5 G（2.07B）。

### 预测

| 指标 | LongLiu | fair | static |
|---|---|---|---|
| premium s（连续） | 2.47 | **3.50** | 2.00 |
| standard s（连续） | 3.19（floor 有界） | 1.95 | **≥ 5.6（被饿死）** |
| **max slowdown（全文眼）** | **3.19** | 3.50 | ≥ 5.6 |

叙事：**S2 里没有任何臂能救 premium（结构性不可行），比的是"谁的代价有界"。
LongLiu 的最坏 slowdown ≈ 3.2 封顶；static 把 standard 饿到 5.6+ 且无上限；fair 把 premium 拖到 3.5 且随负载继续涨。**
这一条直接对应 Theorem 1 的 controlled degradation。

调节旋钮：若 static standard 没饿起来（premium 相位错开太巧），把 premium 的 r 从 1.0 提到 1.3（D=163MB），或加第 4 个重 premium。

---

## 4. 校准流程（每个 regime 开跑前必做）

1. **solo 基线（on-wire 实测，禁止用配置理论值）**：每个作业单独跑 ≥ 3 个 epoch，
   记录实际 Tcomp、实际 Tcomm,solo、实际速率。旧数据 slowdown<1 就说明 T_target_solo 是拍脑袋值。
2. **调 D 到目标 r**：实测 r = Tcomm,solo/Tcomp 与表中目标 r 偏差 > 10% 就调 D。
3. **供给负载核验**：Σb̄ = Σ φ·B（φ 用实测值）。scarce regime 若 < 1.0B，整体上调 D 后重测。
4. **DSCP→TC 探针**：开跑前确认守护进程发的 DSCP 落在预期 TC（旧 pitfall 复防）。

## 5. 日志规范（每作业每 epoch 一行）

`run_id, regime, arm, round, job_id, tier, c_eval, epoch, t_comp_ms, comm_start_ms, comm_end_ms, iter_ms, dscp`

- **slowdown 必须按迭代口径算**：s = (t_comp + comm) / (t_comp + Tcomm,solo)，
  与论文 c_i 的定义一致。旧数据用的是通信口径（avg_comm / T_target_solo），偏严且与模型语义不符。
- attainment = 窗口（epochs 5–19）内平均 s ≤ c_eval 的 premium 比例；同时报连续 s 与 p95。
- 同时报 **max slowdown across jobs**（S2 的眼）与 S_cont（standard 平均 s）。

## 6. 跑法

- 每 regime × 每臂 ≥ 5 轮（旧数据 3 轮的组间方差很大，ample 组 S_cont 的 std/mean 达 16%）。
- 臂：longliu / fair（同 DSCP）/ static（tier 固定映射）。
- S1 跑 4 个 regime，S2 跑 1 个 regime，合计 5 regime × 3 臂 × 5 轮 = 75 轮。
- 每轮 24 epoch（沿用现状），先 5 epoch 预热，守护进程在 epoch 5 前完成首次重算。
