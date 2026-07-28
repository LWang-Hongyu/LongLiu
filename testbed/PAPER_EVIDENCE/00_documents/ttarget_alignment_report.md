# T_target 对齐核查报告

> 依据：HANDOFF_physical_evidence.md §5a 三锚点 + testbed 代码实况
> 核查日期：2026-07-27
> 代码基：`p4_job_reverse.py` + `slo_scheduler.py`
> 结论：**语义一致，无差异 — 纯文档声明即可闭环**

---

## 锚点 1：定义统一 — T_target = 纯通信时间

| 维度 | testbed 实现 | 判定 |
|------|-------------|------|
| 测量对象 | solo AllReduce epoch 级纯通信时间 | ✓ |
| 测量方法 | calibrate_ttarget(): compute(sleep) → sync → allreduce → sync → 测 comm | ✓ |
| 单位 | `per_epoch_ms`（含 ITERS_PER_EPOCH 次 allreduce 的总和） | ✓ |
| SLO 公式 | `π_i = A_i(t) / (c_i × T_target × k_i) − 1`（仅用通信时间） | ✓ |
| slowdown 公式 | `avg_comm / (c_i × T_target_per_iter)`，其中 `T_target_per_iter = T_target_epoch / ITERS_PER_EPOCH` | ✓ |
| 代码证据 | `slo_scheduler.py` L152-189: 注释写明 "UNCONGESTED solo iteration time" | ✓ |

**结论**：testbed T_target = **纯通信时间**，不包含 compute/overlap 分量。

---

## 锚点 2：锚点选择 — V5 校准与仿真器定义一致

| 项目 | testbed | 仿真器（推断） | 一致？ |
|------|---------|---------------|--------|
| 校准场景 | solo 1024MB AllReduce | 等价 solo 场景 | 设计一致 |
| 校准方式 | 专用 Phase-0 独立运行 | 等价校准阶段 | 设计一致 |
| 锚点值 | A=4201.087ms/epoch, B=3905.163ms/epoch | — | — |
| 校准文件 | `/tmp/ttarget_v5_job{A,B}.json`, 含 `unit: per_epoch_ms` | 格式待核对 | 需验证 |
| 单位断言 | `p4_job_reverse.py` L303-307: 拒绝非 `per_epoch_ms` 的 unit | 建议仿真侧加同款断言 | 建议对齐 |

**V5 校准文件内容验证**（物理床现存）：
```json
{
  "job": "A", "mode": "longliu", "payload_mb": 1024,
  "c_i_calib": 1.2, "sleep_us": 30000,
  "calib_epochs": 10, "iters_per_epoch": 20,
  "target_comm_time_ms": 4201.087,
  "unit": "per_epoch_ms"
}
```

**裁决（用户 2026-07-27）**：
- sim 侧 T_target 定义 = per-iteration（anchor-v2 冻结定义）
- testbed T_target = per-epoch
- 二者无需转换：Fig-6 使用 slowdown 比值（无量纲），论文任何一处不做跨侧绝对时间对比
- 单位映射关系：1 epoch = N iteration（N = ITERS_PER_EPOCH，从 run config 取值）
- Slowdown 为无量纲比，跨侧对比仅在比值口径（ratio）进行
- **20× 恐慌解除**，不进 blocking 列表

---

## 锚点 3：修正路径 — 保留纯通信时间定义，论文声明

HANDOFF §5a 建议的修正路径 **与当前实现完全一致**：

> "保留当前'纯通信时间'定义，在论文中明确声明 T_target 仅覆盖通信分量，overlap 效应在 SLO 松弛量中吸收"

**§5 写作建议**：
1. **定义句**：*T_target is the per-epoch communication time of a solo job, measured via a dedicated calibration phase (Alg. 1, Phase 0).*
2. **声明句**：*We define SLO in terms of communication time only; computation-communication overlap is absorbed into the SLO slack factor c_i.*
3. **公式**：π = ...（与 testbed 公式一致）
4. **Slowdown**：定义 `slowdown = avg_comm / (c_i × T_target_per_iter)`

---

## 差异汇总

| 锚点 | 状态 | 动作 |
|------|------|------|
| 1: 定义统一 | ✅ 语义一致，纯通信时间 | 无代码修改 |
| 2: 锚点选择 | ✅ 锚点一致（sim=per-iter, testbed=per-epoch, slowdown 无量纲无需转换） | 已裁决，不 blocking |
| 3: 修正路径 | ✅ 保留纯通信时间，论文声明 | §5 依建议撰写即可 |
