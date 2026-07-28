# HANDOFF: 锚点公式复位 + 复现门根因分析

## 日期
2026-07-24

## 一、公式复位事件

### 事件
锚点 SAS 公式复位：现行公式（iter_solo = comp + comm_solo×(1-overlap)）被确认为误判，
恢复为 paper-baseline-v2 tag 的锚点公式（comm_budget = ci×comm_solo×overhead, overlap 模式）。

### 根因
口径审计中将 overhead=1.3 误判为"第三种 SAS 公式"并拔除。实际 1.3 是 v2 tag 冻结的 wire 因子。
第四次约定事故，根因=校准表与运行配置血缘从未一次性钉死。

### 裁决证据
- overhead=1.3 + 锚点公式：48/48 逐值精确匹配 v2_test 数据
- overhead=1.0 + 锚点公式：0/48 匹配
- 公式复位后 metrics.py 自洽性验证：48/48 PASS

## 二、v2 复现门根因分析

### 问题
v2 复现门禁声明的 0.908 vs 0.830（整体 mean SAS）差异被前次 session 错误归因于"seed 随机性"，
实际是 workload 定义漂移。

### 调查过程

#### 1. Seed 确认
- 锚点 run seed = 0（从 `outputs/quickfix/v2_test/per_job.json` 逐条 `"seed": 0` 确认）
- 门禁重跑 seed = 0（`task0_gate_keeper.py` seed=0）
- **结论：seed 相同，不是 seed 不匹配。**

#### 2. Workload 逐 job diff（同 seed=0）
锚点 Fair/seed=0 的 24 个 job 模型/ci 分布：
```
Models: BERT-Large-fp16×4, LLaMA-2-13B×4, ViT-Large×2, LLaMA-2-7B×6,
        ResNet-50-fp16×2, ResNet-18×2, ViT-Base×2, T5-11B-fp16×2
CI:     1.5×12, 2.0×8, 3.0×4
```

v2 tag `DEFAULT_TIERED_WORKLOAD`（ci=1.5/2.0/3.0）：**24/24 MATCH**。

HEAD `DEFAULT_TIERED_WORKLOAD`（ci=1.2/2.0/3.0）：
```
CI:     1.2×6, 2.0×14, 3.0×4
Delta:  ci=1.2: 0→6 (NEW premium tier)
        ci=1.5: 12→0 (ELIMINATED)
        ci=2.0: 8→14 (+6 from standard)
        ci=3.0: 4→4 (unchanged)
```
**24/24 MISMATCH** — 不同 workload 产生不同 job 列表。

#### 3. git diff v2-tag..HEAD 影响 RNG/仿真语义的改动

| 文件 | 改动 | 影响 |
|------|------|------|
| `trace/synthetic.py` | `DEFAULT_TIERED_WORKLOAD` ci 从 (1.5/2.0/3.0) 改为 (1.2/2.0/3.0) | **主因**：workload 定义变了，同 seed 不同 job |
| `core/simulator.py` `_advance` | 增加 `if flow.start_time_ms > self.time_ms: continue` | 次因：未来 flow 不再被推进 |
| `core/simulator.py` `_collect_links` | 从只取 spine_links 改为收集活跃流链路 | FatTree 不受影响（走 `_recompute_bandwidth_twotier`） |
| `core/simulator.py` link_utilization | 新增链路利用率跟踪 | 增量特性，不影响核心仿真 |
| `policy/dwrr.py` | 新增 DWRR 策略文件 | 不影响现有策略 |

#### 4. `_advance` 改动影响评估
v2 tag 代码：
```python
for flow in self.active_flows.values():
    flow.advance(dt_ms)
```
HEAD 代码：
```python
for flow in self.active_flows.values():
    if flow.start_time_ms > self.time_ms:
        continue
    flow.advance(dt_ms)
```
此改动会导致：start_time 在 future 的 flow 在 v2 中会被错误地推进（产生负 rem_bits），
在 HEAD 中被正确跳过。这是一个 **bug-fix**，改变仿真语义但方向是纠正。

### 结论：**(a) 漂移可解释，当前代码可信**

1. **主因**：`DEFAULT_TIERED_WORKLOAD` 在 v4 tag（29bf0f4）被异构化（ci=1.5→1.2/2.0），导致同 seed 同 profile 引用产生不同 workload。
2. **次因**：`_advance` flow skip 是仿真 bug-fix，可能轻微影响结果。
3. **仿真核心（调度逻辑、网络模型）未漂移**；变更是 workload 定义和 bug-fix。
4. 当前代码对**新场景**可信；历史基线（Fair/CRUX/SP 锚点值）必须在当前代码上**有意识地重新生成**。

### 门禁链升级

原有门禁链只有"代码同一性"（文件哈希比对 v2 tag），缺少"结果复现"检查。
升级为两段：

1. **代码同一性**（keep）：比对 `longliu_sim/core/` `longliu_sim/policy/` `longliu_sim/trace/` 等
   核心目录中 v2-tag..HEAD 的文件哈希（不含 pycache）。
2. **结果复现**（new）：使用 v2 锚点 workload profile（冻结在 `V2_ANCHOR_WORKLOAD_PROFILE`），
   同 seed 同配置重跑，逐 job SAS 对比锚点值，**容差 0**（bit-exact 断言）。

注意：当前 HEAD 不能再跑 v2 锚点 workload 的结果复现（因为仿真语义已变），
"结果复现"门禁的锚点基准必须从当前代码重新生成。这是一个**首次建立锚点**的过程。

## 三、attain 表（锚点语义）
- Sigma_attain_P = 176.6 Gbps
- Sigma_attain_S = 215.4 Gbps
- C* = 284.3 Gbps
- 当前 7-job workload 在 @800G 下是 0.49 深可行区，战斗场不存在

## 四、新增纪律
1. 任何审计出现预设分支外的值，禁止自行决策，必须上报
2. 公式单实现（metrics.py），禁止复制粘贴
3. 配置集中（config.yaml），代码内禁止字面量
4. solo 不变量门禁期望值按锚点语义标注
5. workload profile 变更视为一级变更，必须在 HANDOFF 记录并触发门禁重跑

## 四-A、config.yaml 配置统一（2026-07-24）

### 创建文件
- `config.yaml`：唯一配置源，含 frozen/topology/simulation/tiered_workload/model_types/v2_anchor_workload
- `longliu_sim/utils/config.py`：配置加载器，提供 get_frozen/get_topology/get_simulation 等接口
- `scripts/ci_lint.py`：CI lint 脚本，检测硬编码字面量（0 errors, 31 warnings 现存量）

### 集中参数
| 参数 | 值 | 来源/备注 |
|---|---|---|
| overhead_factor | 1.3 | frozen（锚点 wire 因子） |
| overlap_factor | 0.85 | frozen（锚点重叠度） |
| K | 2.0 | frozen（16-node DWRR） |
| topology.k | 4 | 16-node FatTree |
| topology.host_bw_bps | 100e9 | 100 Gbps |
| topology.spine_bw_bps | 400e9 | 400 Gbps |
| v2_anchor_workload | 24 jobs | 门禁复现基准 |

### 接入情况
- gatekeeper.py：已从 config.yaml 读取 ANCHOR_CONFIG 和 V2_ANCHOR_WORKLOAD
- Simulator：gatekeeper 的 run_meta 已记录 config_hash + SEMANTICS_VERSION
- CI lint：通过（0 errors），31 warnings 为存量实验脚本，逐步迁移

## 五、待执行
1. v4 轨迹快检（锚点 attain 版 @800G 1 seed，自洽性）
2. feas_boundary_v3 场景设计（报用户批准，禁止先跑）
3. 物理侧 SLO 定义对齐（排入队列）

## 五-A、基线重生成（anchor-regen-v1，2026-07-24）

### 执行配置
- 代码：HEAD（含 `_advance` bug-fix）
- Workload：V2_ANCHOR_WORKLOAD（24 job，ci=1.5/2.0/3.0）
- 策略：Fair / CRUX / SP / D1
- Seeds：0, 1, 2
- 输出：`outputs/anchor_regen_v1/`
- Git tag：`anchor-regen-v1`
- run_meta：config_hash=0140186f4c22fd02，SEMANTICS_VERSION=anchor-v2

### 现行锚点表（3-seed mean）

| Strategy | Mean SAS | SLO Rate | Collapse | Starved |
|----------|----------|----------|----------|---------|
| Fair     | 0.7835   | 23.6%    | 1.4%     | 0.0     |
| CRUX     | 0.8422   | 29.2%    | 13.9%    | 0.0     |
| SP       | 0.7776   | 29.2%    | 37.5%    | 5.7     |
| D1       | **0.8131** | **22.2%** | **0.0%** | **0.0** |

> D1 行 2026-07-24 修订（anchor-regen-v1.1）：原值 0.7240/25.0%/19.4%/0.0 因旧公式
> ci×(comp+comm_solo×(1-overlap)) 缺 overhead_factor 导致 T_target 系统性低估，
> pi 排序失真。复位为锚点公式后 collapse 归零（见 §八 控制侧复位）。

### v2 历史档案（paper-baseline-v2 tag，2 seeds，禁止作为现行基线引用）

| Strategy | Mean SAS | SLO Attain | Collapse | 来源 |
|----------|----------|------------|----------|------|
| Fair     | 0.830    | 27.1%      | 0.0%     | `outputs/quickfix/v2_test/table3.csv` L2: "Fair (baseline)" |
| CRUX     | 0.897    | 33.3%      | 4.2%     | `outputs/quickfix/v2_test/table3.csv` L3: "CRUX" |
| SP       | 0.941    | 41.7%      | 33.3%    | `outputs/quickfix/v2_test/table3.csv` L4: "LongLiu (SP)" |
| D1       | 0.908    | 47.9%      | 29.2%    | `outputs/quickfix/v2_test/table3.csv` L5: "LongLiu-v2 (DWRR)" |

**注意**：v2 历史值在 HEAD 上永久不可复现（`_advance` bug-fix 改变仿真语义），两组数字禁止混用。
论文引用的所有基线数字必须来自现行锚点表。

门禁已更新：
- "结果复现"段基准为 `outputs/anchor_regen_v1/per_policy_results.json`
- D1 行使用 `outputs/anchor_regen_v1/D1_rerun.json`（锚点公式修正版）
- 容差 0，逐 job SAS + 逐 seed 对比

## 八、D1 控制侧公式复位（2026-07-24）

### 事件
LongLiuDWRR (D1/D2/D3/D1G/v3/v3.1) 的 T_target 计算使用了
`ci×(comp+comm_solo×(1-overlap))` 公式（缺 overhead_factor），
与 metrics.py 的锚点公式（`comm_budget=ci×comm_solo×overhead` + `target=max+overlap`）不一致。

### 根因（误判链条闭环）
- 原始 D1 公式结构 `T_target = ci × (comp + comm_solo × (1-overlap))`  本**可**正确——前提是 `comm_solo` 含 wire 口径
- 但 v2 锚点确认 overhead=1.3 挂在 `comm_budget = ci × comm_solo × overhead(逻辑口径)`，而非预乘 wire_solo
- 缺 overhead 导致 T_target 系统性低估，pi 排序失真：最需带宽的 job 被误判"超前"
- **罪不在公式结构，在 2.0 缺 overlap 合成（缺 overhead_factor）**

### 修正
1. `dwrr.py` LongLiuDWRR.__init__ 新增 `overhead_factor` 参数（禁止静默默认值）
2. T_target 统一为锚点公式：
   ```
   comm_budget = ci × comm_solo_ms × overhead
   T_target = max(comp, comm_budget) + (1-overlap) × min(comp, comm_budget)
   ```
3. D1G/v3/v3.1 家族全部审计修复
4. 新增 CI 断言测试 `scripts/ci_lint_formulas.py`：
   340/340 组合验证 D1 T_target == metrics.compute_target_iter_ms（容差 1e-9）

### 影响
- D1 锚点值修正：0.724/19.4% collapse → **0.813/0% collapse**
- git tag anchor-regen-v1.1（仅 D1 行修订，Fair/CRUX/SP 不变）

## 六、CONVENTION.md 约定链修正（2026-07-24）

### 问题
原约定链图中 `comm_solo_ms = wire_bits / host_bw` + `comm_budget = ci × comm_solo × overhead`
导致 1.3 双重应用（13B budget = 1757.6ms vs 正确 1352ms），链图与 attain 表自相矛盾。

### 修正
采用方案 A（逻辑口径 comm_solo，与代码变量语义一致）：
- `comm_solo_ms(逻辑) = logical_bits / host_bw × 1000`（13B = 520ms）
- `comm_budget = ci × comm_solo_ms(逻辑) × overhead`（= ci × wire_solo = 1352ms）
- 全链每个量标注（逻辑/wire）口径
- 嵌入一致性自检：链图推导 13B budget = 1352ms，与 attain 表一致

### 附加
- 声明 v2 历史锚点值（Fair 0.897 等）在 HEAD 上因 `_advance` bug-fix 永久不可复现
- 论文基线必须来自当前代码重生成，v2 历史值降级为档案

## 七、第五次单位制险情（2026-07-24）

### 事件
CONVENTION.md 二次修正稿 L22 写入 `logical_bits = mb_per_iter × 8 × 1024²`，
将十进制 MB 口径的 mb_per_iter 按 MiB 解释（×1024²），导致 13B logical_bits = 54.5 Gb，
与文档自身 attain 表和自检表用的 52.0 Gb 矛盾。

### 根因
6198.88 血缘事件已确认 mb_per_iter 为十进制 MB（6500×1e6×8=52e9），
但宪法文档编写者未将该裁决嵌入链图的 default 写法。

### 修正
- L22：`× 8 × 1024²` → `× 1e6 × 8`（注明 mb_per_iter 为十进制 MB）
- L43：自检公式 `2×1.3e10×2×8/8` → `6500 × 1e6 × 8`（可追溯）
- L10：注释从 wire 反除写法改为正向 `= logical_bits / host_bw（逻辑口径）`
- L52：attuion_bw → attain_bw 拼写修正

### 教训
单位制（MiB vs MB）已是第五次现身的幽灵。

## 八、D1@400G 机制定案 + E3 双 swap 臂（2026-07-27）

### 八-A、D1 失效机制定案（素材 #3）

**裁决**：D1@400G P-attn=37.5% 不是锁入，是错峰+路由伪影 + 单层无 tier 隔离。

**决定性证据**：
- J1 与 J5 在 28,893 个 epoch 中从未同链路竞争（head-to-head=0）
- 7.1× 份额差是不同链路不同竞争环境的产物
- 真争抢 epoch 中控制律对 J0/J1/J2 误差为 0

**Standard 计入重算**（6,286 混合 epoch）：
- J5 obs-exp error = -0.0200, J6 = -0.0200
- 全 job 均值误差 = -0.0000 → **CONVERGED**

**定稿表述**：
> 单层反馈定律在极端稀缺下无法隔离 tier：standard 的违约信号与 premium 同池竞争，premium 保护结构性失效；v4 的两级分层（premium 池 + standard floor）是结构性解法。反馈定律本身被精确执行（全 job 误差≈0），失效的是单层设计，不是实现。

**论文素材 #3 状态**：✅ 定稿

### 八-B、E3 对照臂（CRUX-advantaging swap）

**配置**：E1 workload @800G, t=300s ci swap（大模型→premium/小模型→standard）

| 指标 | 实测 | 判定 |
|------|------|------|
| v4 W1 P-attn | 100% | PASS |
| v4 W3 P-attn | 100% | PASS |
| v4 W3 starv | 0 | PASS |
| v4 W3 S-cont-cap | 1.00（下界 ≥0.62） | PASS |
| CRUX W3 P-attn | 83.3% (5/6) | 与 intensity 键修正分析一致 |

**关键发现**：swap 方向使 CRUX 从"逆 tier"变"顺 tier"（post-swap premium=小模型高 intensity → CRUX 偏爱），双重原因（swap 方向 + 800G 容量过肥）导致 CRUX 不崩。入档为对照臂。

**用户 CRUX 预测错误入档**：预登记写"4 个 BERT premium 被 size 键压死"——用了已被纠正的 size 键错误。按正确的 intensity 键，实测 83.3% 与修正分析完全一致。

### 八-C、E3' 杀伤臂（CRUX-disadvantaging swap）

**配置**：E2-pro workload @630G, t=300s ci swap（大模型→premium ci=1.5 / 小模型→standard ci=3.0）

**Post-swap attain 独立复核通过**：
- ΣP' = 569.6G（13B×3: 71.5×3 + 7B×3: 71.1×3 + T5×2: 70.9×2）
- ΣS' = 230.0G（BERT dp2: 41.7×2 + BERT dp4: 55.6 + ViT: 45.5×2）
- C\*' = 684.6G > 630G → 不可行域（gap=54.6G）

**1-seed 快检结果**：

| 指标 | 预登记 | 实测 | 判定 |
|------|--------|------|------|
| v4 W1 P-attn | 100% | **100%** (5/5) | PASS |
| v4 W3 P-attn | 100% | **100%** (8/8) | PASS |
| v4 W3 starv | 0 | **0** | PASS |
| v4 W2 收敛 | 瞬态≈0 | **W2 S-cap=0.914**（闭式稳定） | PASS |
| CRUX W3 P-attn ≪ v4 (gap≥10pp) | ≥25pp | **100pp** (0.0% vs 100%) | **PASS** |
| CRUX W2/W3 详细 | — | **0/8 premium 达标**，sas=0.30-0.49 | 全崩 |

**论文叙事**：双臂对照
- E3 对照臂：CRUX 顺 tier swap → 存活（83.3%）
- E3' 杀伤臂：CRUX 逆 tier swap → 崩溃（0.0%）
- v4 两臂恒定：100%
- 同一 CRUX，同一 tier 结构，swap 方向翻转 → hero↔zero；v4 不受影响

**论文主图候选**：sas-t 轨迹（双臂，W1/W2/W3 标注），逐 epoch 分辨率数据已保留。

### 八-D、仿真线证据链现状（论文骨架）

| 证据 | 状态 |
|---|---|
| 保障区域：C≥C\* 三场景 v4 P-attn=100±0.0% | ✅ |
| 杀场：E2' CRUX 崩溃（500G 33.3pp / 630G 25.9pp） | ✅ |
| 正交对照：E2-pro CRUX 100% 持平 | ✅ |
| 深饱和排序：v4 ≥ D1（400/500G 大比分） | ✅ |
| 稳定性：v4 std=0.0 vs D1 std=23.6pp（@630G） | ✅ |
| 保障保守性量化（λ=0.60→0.977） | ✅ |
| D1 失效机制（tier 无隔离，standard 摊薄） | ✅ 定稿 |
| 动态 swap 双臂：CRUX hero↔zero，v4 恒定 100% | ✅ E3+E3' 1-seed |
| E3/E3' 3 seeds 正式批 | ✅ | 见 §八-E |

## 八-E、E3/E3' 3 seeds 正式批结果（2026-07-27）

### 执行配置
- 双臂（E3 对照臂 + E3' 杀伤臂）× v4/CRUX × 3 seeds = 12 runs
- E3 对照臂：FEAS_BOUNDARY_V3_WORKLOAD (14 jobs) @800G spine
- E3' 杀伤臂：FEAS_BOUNDARY_V3_PRO_WORKLOAD (13 jobs) @630G spine
- 判定矩阵 v2.2，窗口过滤：start_ms ∈ [window_start, window_end]

### 结果表

| 臂 | 策略 | W3 P-attn (3-seed mean±std) | starv | 判定 |
|----|------|---------------------------|-------|------|
| E3 对照 | v4 | **100.0±0.0%** | 0 | PASS |
| E3 对照 | CRUX | 61.1±19.2% | 0 | 观测行 |
| E3' 杀伤 | v4 | **100.0±0.0%** | 0 | PASS |
| E3' 杀伤 | CRUX | **8.3±14.4%** | 0 | PASS (gap=91.7pp ≫ 10pp) |

### 判定
- E3' 杀伤臂 v4 W3 P-attn=100.0±0.0%（下界 PASS），starv=0（PASS）
- E3' 杀伤臂 CRUX W3 P-attn=8.3±14.4%，gap=91.7pp ≫ 10pp（PASS）
- E3 对照臂 v4 W3 P-attn=100.0±0.0%（PASS）
- 全部 12 runs PASS，无 FAIL 停跑

### sas-t 轨迹图
- 输出：`outputs/e3_swap/sas_t_trajectory.png`
- 双臂四线（v4/CRUX × E3/E3'），W1/W2/W3 标注，逐 epoch 分辨率，3 seeds 均值±标准差带
- 论文主图候选

## 八-F、证据关四步验证（E3'@630G CRUX 0.0% 真实性）

### Step 1: 静态复制品对照（决定性）
- Post-swap 配置（E3' workload @630G, post-swap ci mapping）直接以静态方式运行 CRUX（无 swap 事件）
- 逐 job SAS 对比 swap 实验 W3 vs 静态 run：delta < 0.002
- **结论**：CRUX 0.0% 是该配置下 CRUX 的真实稳态行为，与 swap 机制无关、与窗口无关

### Step 2: 窗口过滤修正
- 原逻辑使用 `end_ms` 过滤 → 可能纳入跨 swap 边界的迭代
- 修正为 `start_ms >= window_start AND end_ms <= window_end`（纯 post-swap 稳态）
- 验证：新旧过滤方法 SAS delta max = 0.0002，零差异（200s gap ≫ guard band）
- `start_ms` 过滤已写入 `exp_e3_swap.py` `stats_for_window()` 方法，入方法论附录

### Step 3: 带宽时间线
- CRUX premium 拿 29–34G（交付率 ~52%）—— "饿着但活着"，不是断供
- J9/J12 的 ECMP 哈希 artifact 已标注，3 seeds 平滑

### Step 4: 0.0% 真实性确认
- 经静态复制品对照 + 窗口过滤修正双重验证，CRUX W3 0.0% 坐实为真

## 八-G、CRUX 场景敏感性（鲁棒性证据）

### 现象
- E2' 静态 @630G CRUX P-attn = 70.4%（9 premium 大模型 + 5 standard 小模型）
- E3' post-swap @630G CRUX P-attn = 0.0%（8 premium 大模型 + 5 standard 小模型）
- 两个"相似"配置下 70.4% 与 0.0% 的极端落差

### 定性
- 静态复制品对照证明这不是 bug，是 **CRUX 对 workload 组合的极端敏感**
- E2' 的 ±27.7pp 种子方差（1-seed）指向同一件事：ECMP 哈希运气主导边际场景
- v4 在两个配置下都是 100%，恒定不变

### 论文叙事价值
- v4 的价值不止"逆 tier 时赢"，还有"在基线表现不可预测的 regime 里恒定"
- 此对比写入论文"鲁棒性"叙事

### v4 premium SAS 盈余确认
- v4 臂 premium sas = 1.84–1.86（>1），符合设计预期
- 封顶后盈余按规则分给 premium，capped 指标免疫
- 这是 CRUX mean 通胀指纹在 v4 这里无害化的体现
