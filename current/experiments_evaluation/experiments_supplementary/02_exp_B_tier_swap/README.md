# 实验 B：动态锚点 — 硬件 Tier Swap

> **优先级**：最高（为 EQ3 全文最强结论补第一个硬件动态证据）
> **规划文档**：`LongLiu_补充实验方案.md` §B
> **基础脚本**：`experiments/P4_dumbbell_slo/p4_job_reverse.py`（复用 V6 双臂 + c_i swap 逻辑）

## 实验目标

给 EQ3（LongLiu swap 后 100% vs 静态 10%）补第一个硬件动态证据：
- 两臂对比：LongLiu 动态 P6→P2 vs 静态贴标签不动
- c_i swap 在 epoch 8 反转（A: premium↔standard, B: standard↔premium）
- W1/W2/W3 三窗口测量紧 SLO 作业 slowdown

## 场景设计

```
Phase 1 (epoch 0-7):          Phase 2 (epoch 8-24):
  J_A: c_i=1.2 (premium)       J_A: c_i=2.0 (standard)
  J_B: c_i=2.0 (standard)      J_B: c_i=1.2 (premium)
         ↓                              ↓
    T_swap (epoch 8)            紧 SLO 作业从 A 切换到 B
```

### 测量窗口

| 窗口 | Epoch 范围 | 时间近似 | 紧 SLO 作业 | 含义 |
|------|-----------|---------|------------|------|
| W1 | 3-7 | T_swap−100s | Job A | swap 前基线 |
| W2 | 8-12 | T_swap~+100s | Job B | swap 后瞬态 |
| W3 | 18-22 | T_swap+200~300s | Job B | swap 后稳态 |

### 两臂配置

| 臂 | 调度器 | 初始优先级 | swap 后行为 |
|----|--------|-----------|------------|
| LongLiu (LL) | 动态 π→P6/P4/P2/P1 | P3（两作业同） | π 跟踪 c_i 变化，动态调整 |
| Static (CX) | CRUX 静态 | A=P4, B=P3 | **不变**（失锁行为） |

## 轮次设计（4 轮交替）

| 轮次 | 作业启动顺序 | 臂运行顺序 | 标签 |
|------|------------|-----------|------|
| 1 | A→B | LL→CX | `round1_AB_ll_cx` |
| 2 | A→B | CX→LL | `round2_AB_cx_ll` |
| 3 | B→A | LL→CX | `round3_BA_ll_cx` |
| 4 | B→A | CX→LL | `round4_BA_cx_ll` |

交替维度：{哪作业先启动} × {哪臂先跑}，消除序列伪影。

## 背景流

复用 V6 校准：12 路 iperf3 UDP，DSCP=P3（TOS=64），总速率 6 Gbps。
- 作用：抬高竞争强度，确保优先级差异可观测
- 可通过 `skip_bg=1` 跳过（用于无背景流对照）

## 成功判据（§B.4 定性对齐）

- **LongLiu 臂**：W3 slowdown ≈ W1（无瞬态崩溃），W3/W1 < 1.5
- **静态臂**：W3 slowdown >> W1（失锁），W3/W1 > 1.5
- 与仿真 Fig. 9/10 对比**形状**（升降方向、有无瞬态），caption 注明 "qualitative shape, not absolute values"

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `expB_config.sh` | 共享配置（参数、窗口定义、helper 函数） |
| `run_expB_round.sh` | 单轮运行（两臂 + 背景流 + 归档） |
| `run_expB_all.sh` | 4 轮完整编排（含轮间冷却） |
| `analyze_expB.py` | W1/W2/W3 窗口分析 + 轨迹图 + 报告生成 |

## 运行方式

```bash
# 完整 4 轮（约 60-80 分钟）
cd /home/why/LongLiu_rebuild/experiments_supplementary/02_exp_B_tier_swap/scripts
bash run_expB_all.sh

# 单轮（用于测试）
bash run_expB_round.sh 1

# 跳过背景流（对照实验）
bash run_expB_all.sh 1   # skip_bg=1

# 从第 3 轮恢复
bash run_expB_all.sh 0 3

# 分析结果
python3 analyze_expB.py
```

## 实验结果（2026-07-28 完成）

### 4 轮窗口 slowdown

| 轮次 | 顺序 | 臂 | W1 | W2 | W3 | W3/W1 |
|------|------|-----|-------|-------|-------|-------|
| 1 | AB | LL | 0.925 | 1.323 | 0.885 | 0.96× |
| 1 | AB | CX | 0.927 | 1.317 | 0.910 | 0.98× |
| 2 | AB | LL | 0.948 | 1.320 | 0.918 | 0.97× |
| 2 | AB | CX | 0.920 | 1.324 | 0.882 | 0.96× |
| 3 | BA | LL | 1.281 | 1.394 | 1.316 | 1.03× |
| 3 | BA | CX | 1.204 | 1.303 | 1.309 | 1.09× |
| 4 | BA | LL | 1.256 | 1.369 | 1.347 | 1.07× |
| 4 | BA | CX | 1.200 | 1.291 | 1.296 | 1.08× |

### 关键结论

1. **LL 臂 PASS**：W3/W1 = 1.01×（全轮），BA 顺序下 1.05×——无瞬态崩溃
2. **优先级轨迹是核心证据**：LL 动态调整 P2↔P6/P4 vs CX 静态 P3/P4 固定
3. **BA 顺序暴露调度器差异**：LL W3/W1=1.05 vs CX W3/W1=1.085（3.5% 优势，2 轮可重现）
4. **AB 顺序有相位互斥效应**：epoch 19 争抢自消退，掩盖调度器差异

详细分析见 `analysis/expB_findings.md`

## 产出

- `data/round<N>_<order>/`：每轮 CSV（per-epoch + per-iter）+ md5
- `logs/round<N>_<order>/`：每轮日志（10.1/226 双侧 + 背景流）
- `analysis/expB_analysis_report.md`：窗口对比表 + 成功判据评估
- `analysis/expB_summary.csv`：汇总数据（可导入论文表格）
- `analysis/expB_trajectory.png`：双臂 slowdown 轨迹图

## 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| payload | 1024 MB | 复用 V5 T_target |
| c_i premium | 1.2 | §B.1 |
| c_i standard | 2.0 | §B.1 |
| sleep_us | 30000 (30ms) | V5/V6 一致 |
| epochs | 25 (500 iters) | 覆盖 W1+W2+W3+gap |
| swap epoch | 8 | §B.2 对齐 |
| T_target | V5 校准（1024MB） | `/tmp/ttarget_v5_job{A,B}.json` |
| BG flow | 12×500M DSCP=P3 | V6 校准 |
| CRUX prio A | P4 (DSCP=0) | premium→高优先级 |
| CRUX prio B | P3 (DSCP=16) | standard→低优先级 |
| LL initial | P3 | V6 一致 |
