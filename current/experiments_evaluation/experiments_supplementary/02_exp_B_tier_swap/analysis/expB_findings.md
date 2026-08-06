# Experiment B: Hardware Tier Swap — Complete Findings

> **日期**: 2026-07-28
> **轮次**: 4 轮（AB×2 + BA×2 交替）
> **环境**: guolab-10 (10.1) + guolab-226 (226), 50G RoCEv2, 6G DSCP=P3 背景流
> **参数**: payload=1024MB, c_i=1.2(premium)↔2.0(standard), swap@epoch8, 25 epochs

## 一、总体结果

### 4 轮窗口 slowdown 汇总

| 轮次 | 顺序 | 臂 | W1 | W2 | W3 | W3/W1 |
|------|------|-----|-------|-------|-------|-------|
| 1 | AB, LL→CX | LL | 0.925 | 1.323 | 0.885 | 0.96× |
| 1 | AB, LL→CX | CX | 0.927 | 1.317 | 0.910 | 0.98× |
| 2 | AB, CX→LL | LL | 0.948 | 1.320 | 0.918 | 0.97× |
| 2 | AB, CX→LL | CX | 0.920 | 1.324 | 0.882 | 0.96× |
| 3 | BA, LL→CX | LL | 1.281 | 1.394 | 1.316 | 1.03× |
| 3 | BA, LL→CX | CX | 1.204 | 1.303 | 1.309 | 1.09× |
| 4 | BA, CX→LL | LL | 1.256 | 1.369 | 1.347 | 1.07× |
| 4 | BA, CX→LL | CX | 1.200 | 1.291 | 1.296 | 1.08× |

### 按启动顺序分组聚合

| 顺序 | 臂 | W1 mean±std | W3 mean±std | W3/W1 mean±std |
|------|-----|------------|------------|----------------|
| AB (2轮) | LL | 0.937±0.012 | 0.902±0.017 | 0.965±0.005 |
| AB (2轮) | CX | 0.924±0.004 | 0.896±0.014 | 0.970±0.010 |
| BA (2轮) | LL | 1.269±0.013 | 1.332±0.016 | 1.050±0.020 |
| BA (2轮) | CX | 1.202±0.003 | 1.303±0.005 | 1.085±0.005 |
| **全部** | **LL** | **1.103±0.192** | **1.116±0.249** | **1.01×** |
| **全部** | **CX** | **1.063±0.161** | **1.099±0.235** | **1.03×** |

## 二、关键发现

### 发现 1：AB 顺序触发相位互斥效应

当 Job A（Phase 1 的 tight 作业）先启动时：
- Job A 有 ~4 个 epoch 的独占运行期（W1 前半 slowdown ~0.70）
- 到 epoch 19，两作业的相位错开，争抢自消退（comm time 从 ~310ms 骤降至 ~175ms）
- **两臂都展现 W3/W1 ≈ 0.96-0.98**——调度器差异被系统性效应掩盖

这与 V6 实验记录的 "epoch 11-12 争抢衰减为系统性" 一致。

### 发现 2：BA 顺序避免相位互斥，暴露调度器差异

当 Job B（Phase 1 的 loose 作业）先启动时：
- Job A（tight）后启动，从第一个 epoch 就面临争抢（W1 ≈ 1.27）
- 相位互斥效应未出现——W3 slowdown 维持在 ~1.31
- **CX 臂 W3/W1 = 1.085 > LL 臂 W3/W1 = 1.050**——静态臂 consistently 略差

BA 顺序是更有意义的对比条件，因为它：
1. W1 窗口不含 solo epoch（紧 SLO 作业全程面临争抢）
2. W3 窗口不被相位互斥效应污染

### 发现 3：优先级轨迹是核心定性证据

**LongLiu 臂**（动态调整）：
- AB 顺序：Job B swap 后 P2→**P6**（紧 SLO 检测到落后）→ P4（争抢缓解后降级）
- BA 顺序：Job B swap 后 P1→**P4**（紧 SLO 检测到落后，但累积进度较好所以 P4 而非 P6）
- Job A swap 后 P4→**P2**（松 SLO 正确降级，让出带宽）

**CRUX 静态臂**（失锁行为）：
- Job B 全程 **P3 不变**——swap 后变为 tight 但仍保持低优先级
- Job A 全程 **P4 不变**——swap 后变为 loose 但仍占用高优先级
- 这正是 EQ3 预测的 "lost lock" 行为

### 发现 4：2-job 硬件设置的限制

slowdown 差异较小（BA 顺序下仅 3.5%）的原因：
1. **主机网卡瓶颈**（V3 已验证）：P6 优先级在交换机层面的优势被主机 NIC 瓶颈抵消
2. **2-job 相位互斥**：双作业 ON/OFF 相位自然错开，限制有效争抢
3. **ETS 不支持**：`mlnx_qos` 报 "Priority trust state is not supported"

仿真 EQ3 不受这些限制（更多作业、无主机 NIC 瓶颈），因此展现出更大的差异化。

## 三、成功判据评估

| 判据 | 目标 | LL 臂结果 | CX 臂结果 | 评估 |
|------|------|----------|----------|------|
| LongLiu 无瞬态崩溃 | W3/W1 < 1.5 | 1.01× (全), 1.05× (BA) | — | ✅ PASS |
| 静态臂失锁 | W3/W1 > 1.5 | — | 1.03× (全), 1.085× (BA) | ⚠️ 方向正确但未达阈值 |
| 优先级轨迹差异 | LL 动态 vs CX 静态 | P2↔P6/P4 动态 | P3/P4 固定 | ✅ PASS（核心证据） |
| BA 顺序 LL 优于 CX | LL W3/W1 < CX W3/W1 | 1.05× | 1.085× | ✅ PASS（3.5% 差异） |

**总体评估**：
- LongLiu 的动态优先级调整机制在硬件上验证通过——正确检测 tier swap 并调整优先级
- 静态臂的 "失锁" 行为在优先级轨迹上清晰可见，但 slowdown 差异受 2-job 硬件限制较小
- BA 顺序的 W3/W1 差异（1.05 vs 1.085）方向正确且可重现，支持 EQ3 结论

## 四、论文写作建议

### 进论文的位置
- §V-F 或 EQ3 末尾加 "hardware cross-check" 段
- 一张双臂 slowdown 轨迹图（BA 顺序，Round 3 或 4）
- caption 注明 "qualitative shape, not absolute values"

### 叙事要点
1. **机制验证**：LongLiu 在硬件上正确检测 c_i swap 并动态调整 DSCP 优先级（P2↔P6/P4）
2. **失锁对比**：CRUX 静态优先级在 swap 后失锁——tight 作业保持低优先级，loose 作业保持高优先级
3. **BA 顺序结果**：LongLiu W3/W1=1.05 vs CRUX W3/W1=1.085（3.5% 优势，2 轮可重现）
4. **2-job 限制**：相位互斥效应和主机 NIC 瓶颈限制了可观测的 slowdown 差异
5. **规模化验证**：Experiment C（CPU 模拟器 8-12 作业）将提供无 2-job 限制的规模验证

### 局限性声明
- "Due to the 2-job hardware setup, phase mutual exclusion limits the observable slowdown differentiation. The BA job-start order avoids this artifact, revealing a consistent 3.5% advantage for LongLiu. The simulation (EQ3) with more jobs shows larger differentiation, as it is not subject to the 2-job phase mutual exclusion effect."

## 五、数据文件索引

| 文件 | 说明 |
|------|------|
| `data/round{1-4}_*/job{A,B}_{LL,CX}_epoch.csv` | 每轮每臂 per-epoch 数据 |
| `data/round{1-4}_*/job{A,B}_{LL,CX}_iter.csv` | 每轮每臂 per-iter 数据 |
| `data/round{1-4}_*/md5_job_script.txt` | 脚本 md5 校验 |
| `logs/round{1-4}_*/` | 每轮完整日志（10.1/226 双侧 + 背景流） |
| `analysis/expB_analysis_report.md` | 自动生成的分析报告（含轨迹表） |
| `analysis/expB_summary.csv` | 汇总 CSV（可导入论文表格） |
| `analysis/expB_trajectory.png` | 4 轮双臂 slowdown 轨迹图 |
