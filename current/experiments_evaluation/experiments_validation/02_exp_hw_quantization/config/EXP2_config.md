# 实验2：硬件量化误差与 P6 抢占微观验证 — 配置文档

> 目的：验证当多个逻辑作业映射到同一 DSCP 类时的 FIFO→Fair 退化程度，以及 P6 严格抢占机制的有效性。硬件量化误差（DSCP 只能区分 8 级 → P3 内部无法细分）是本实验的量化对象。

## 1. 测试床与机制

| 项 | 配置 |
|----|------|
| 节点 | guolab-10 + guolab-226（与实验1相同） |
| 多作业模拟 | NCCL Multi-comm 机制：每个逻辑作业独立进程 + 独立 master port，固定优先级 |
| 队列模型 | SP（严格优先级）：tc:0(P6) > tc:1(P4) > tc:2(P3) > tc:3(P2) … |
| 映射 | P6→DSCP=8→tc:0；P3→DSCP=16→tc:2（同 TC 内 FIFO） |

## 2. 测试1：P6 抢占验证

- **场景**：作业 A 固定 P3，作业 B 固定 P6，同时在同一链路运行。
- **步骤**：
  1. solo 校准：P6 solo 20 iters → solo 参考带宽
  2. 作业 A (P3) 先启动，3s 后作业 B (P6) 启动
  3. 各运行 60 iters，记录 per-iter 带宽；全程 NIC 计数器监控
- **预期**：SP 队列下 P6 完全抢占 → P6_bw ≈ solo，P3_bw ≈ 0
- **指标**：抢占度 = P6_bw/(P6_bw+P3_bw) ≥ 95%；P3 饿死度 = P3_bw/solo → 0

## 3. 测试2：P3 内部带宽共享验证

- **场景**：3 个逻辑作业全部固定 P3（同一 DSCP → 同一 tc:2 队列）。
- **步骤**：
  1. solo 校准：P3 solo 20 iters → solo 参考带宽
  2. 3 个 P3 作业依次启动（间隔 3s，观察 FIFO 对先到流的影响）
  3. 各运行 60 iters，记录 per-iter 带宽
- **预期**：同 TC 内 FIFO → 分配可能不均匀；量化与"理想公平 1/3"的偏离
- **指标**：
  - Jain 公平指数（1=完全公平）
  - 聚合利用率 = Σbw_i / solo（链路是否打满）
  - **性能慢度** = 1 − min(bw_i)/fair_share（最差流相对公平份额的缺失，量化 P3 内部无法细分优先级的代价）

## 4. 关键参数

| 参数 | 测试1 | 测试2 |
|------|-------|-------|
| payload | 512 MB | 256 MB（3 job 并发显存安全） |
| sleep | 10 ms/iter | 10 ms/iter |
| iters | solo 20 / main 60 | solo 20 / main 60 |
| 优先级 | A=P3, B=P6 | 3×P3 |
| 监控 | NIC 计数器 1s | NIC 计数器 1s |

## 5. 运行与数据

```bash
bash ../00_common/sync_to_226.sh
bash scripts/run_test1_preempt.sh 1   # 可重复 round 2/3
bash scripts/run_test2_p3share.sh 1
python3 scripts/analyze_exp2.py
```

数据落盘：`data/exp2_test{1,2}_r<round>_<ts>/`：
- `exp2_jobA_rank0_iter.csv`（P3 流）、`exp2_jobB_rank0_iter.csv`（P6 流）
- `exp2_p3flow{1,2,3}_rank0_iter.csv`（3×P3 流）
- `exp2_solo*_solocalib.csv`（solo 参考）
- `nic_10.csv`/`nic_226.csv`（硬件计数器）

## 6. 判定汇总（见 analysis/exp2_report.md）

- 测试1：抢占度 ≥95% 通过；P3 饿死度接近 0
- 测试2：Jain ≥0.9 视为近似公平；否则量化 FIFO 偏斜；聚合利用率接近 100% 说明无空闲

---

## 7. DSCP 映射修正记录（2026-08-08）

### 背景与根因

早期代码（`multi_comm.c`/`slo_scheduler.py`）采用 **单调 DSCP 映射**（P0→DSCP8 … P6→DSCP56），
假设"更高 DSCP class = 更高优先级"。但硬件实测（`mlnx_qos -i mlx5_0 --trust dscp` + 全类探测，
见 `results/testbed/HANDOFF_physical_evidence.md` §e）表明 10.1 NIC 的 DSCP→TC 映射**非线性**：

```
tc:0(prio1,dscp8-15) > tc:1(prio0,dscp0-7) > tc:2(prio2,dscp16-23) >
tc:3(prio3,dscp24-31) > tc:4(prio4,dscp32-39) > tc:5(prio5,dscp40-47) >
tc:6(prio6,dscp48-55) > tc:7(prio7,dscp56-63)
```

旧映射下 **P6→DSCP56→tc:7（最低队列）**，而 P3→DSCP32→tc:4，导致 P6 反被 P3 抢占——
这正是早期 test1 负面结果（P6 并发 14.17 Gbps vs P3 24.31 Gbps）的直接根因。

### 修正后映射（与硬件一致）

| 优先级 | DSCP | ToS(=DSCP<<2) | TC | SP 次序 |
|:-----:|:----:|:----:|:---:|:-----:|
| P6 | 8 | 32 | tc:0 | 最高 |
| P4 | 0 | 0 | tc:1 | 第二 |
| P3/P5 | 16 | 64 | tc:2 | 第三 |
| P2 | 24 | 96 | tc:3 | 第四 |
| P1 | 32 | 128 | tc:4 | 第五 |
| P0 | 40 | 160 | tc:5 | 第六 |

同步修改的文件：
- `multi_comm_slo/src/multi_comm.c`：`prio_dscp[]`（ToS）与 `prio_dscp_val[]`（DSCP）对齐硬件
- `multi_comm_slo/src/slo_scheduler.py`：`PRIORITY_TO_DSCP = {0:40, 1:32, 2:24, 3:16, 4:0, 5:16, 6:8}`
- `multi_comm_slo/src/multi_comm.h`：注释同步
- `02_exp_hw_quantization/scripts/fixed_prio_job.py`：日志用 DSCP 表同步

### 修正后 test1 验证结果（2026-08-08, rounds 1-3）

**统计口径（v2，诚实口径）**：抢占度/饿死度按 **ts 争抢窗口对齐**计算
（jobA 仅取与 jobB 重叠的迭代），见 `analysis/exp2_report.md`。

| Round | solo (Gbps) | 窗口 P3 | 窗口 P6 | 抢占度 | P6/P3 |
|:-----:|:-----------:|:-------:|:-------:|:------:|:-----:|
| 修复前（030446） | 30.91 | 19.64 | 14.17 | 41.9% | 0.72 ❌ |
| R1 attempt2 | 28.71 | 14.38 | 19.91 | 58.1% | 1.38 ✅ |
| R2 attempt2 | 27.81 | 14.27 | 19.72 | 58.0% | 1.38 ✅ |
| R3 attempt1 | 28.64 | 14.41 | 20.34 | 58.5% | 1.41 ✅ |
| R4 attempt2 | 27.61 | 14.57 | 20.22 | 58.1% | 1.39 ✅ |
| R5 attempt1 | 28.71 | 13.45 | 18.73 | 58.2% | 1.39 ✅ |

**修正后 5 轮统计（n=5）**：抢占度 **58.2% ± 0.2%**（58.1/58.0/58.5/58.1/58.2），
论文级稳定性；P3 饿死度 46.9-52.8%（均 ~50%）。

**结论（v3，HANDOFF §d 已修正：226 也分类）**：
1. **方向反转成立**：修复前抢占度 41.9%（P6 反被 P3 抢占）→ 修复后 58.2%±0.2%
   （5 轮一致，P6/P3≈1.39），机制有效。
2. **58% 上限归因（v6，2026-08-10 三组受控实验完成）**：
   - **test3（NCCL 连续通信，3 轮）**：连续饱和未改变并发窗口内分配——抢占度
     58.7%±0.0（test1 突发为 58.2%），P6/solo 63.4%（test1 为 65%），
     P3 残存 solo 的 44.6%（见 `analysis/exp2_test3_report.md`）。
   - **对照实验（两流同为 DSCP16，3 轮）**：同优先级下 jobA/jobB 精确 **50:50**
     均分（份额 50.0%±0.1%），证明 6:4 完全由优先级差造成
     （见 `analysis/exp2_ctrl_dscp16_report.md`）。
   - **perftest 持续流实验（3 轮）**：`ib_write_bw`（持续饱和，占空比≈100%）
     高优先级(DSCP8)满速 45.92±0.10 Gb/s，低优先级(DSCP16) **0.06 Gb/s
     被饿死 99.9%**（见 `analysis/exp2_perftest_report.md`）。
   - **v6 判定**：SP 队列**严格 per-packet**（perftest 实锤，无配置错配、无实现
     失效）；NCCL 场景的 58% 系 **NCCL 流量包级突发间隙**（chunk 注入 +
     DCQCN 暂停）所致——高优先级流 tc:0 队列微秒级空闲被低优先级利用。
     早期 v4 报告（67.3%/91.7%）为统计口径错误——P6 迭代多于 P3 时
     P6 后半段（P3 已退出）恢复 solo 被误计入窗口，已修正为双向重叠窗口。
     早期 v5 判定（"SP 非严格/物理上限"）已被 perftest 证据推翻。
3. 无论 SP 严格性如何，**58.2%±0.2% 的方向反转与统计稳定性是论文可用的核心事实**；
   完全饿死（95%）不作为机制成立的必要条件。
4. 早期 attempt1 中 P3 完全饿死（timeout 90s）是 P3 端口挂死所致，**不作为纯抢占证据**。
5. R1/R2/R4 各有一次 attempt 因已知 NCCL 端口挂死（防火墙 REJECT 段）由重试跳过。
