# 新实验规划文档（E10-E15）

本文档描述为回应审稿人意见而规划的 6 个新实验，用于验证 LongLiu 算法的鲁棒性、参数敏感性以及相对于更合理基线的优越性。

---

## 实验概览

| 实验编号 | 名称 | 目的 | 关键对比 | 预期结论 |
|----------|------|------|----------|----------|
| E10 | WFS Baseline | 加权公平共享基线对照 | LongLiu vs WFS (weight=1/ci) | 闭式解优于线性权重映射，特别是在资源稀缺且作业异构时 |
| E11 | Overlap Sensitivity | 串行模型保守性敏感性 | 不同 overlap_factor (0.0-1.0) | 即使 30%-50% 重叠，串行模型推导的分配仍维持高 SLO 达成率 |
| E12 | DSCP Quantization | DSCP 量化误差宏观影响 | v4 (7级) vs LongLiuContinuous (理想连续) | 高度量化下 SLO 达成率下降 <5% |
| E13 | Window Sensitivity | 窗口大小 W 敏感性 | W=5/10/20/50 | W=20 是工程折中点（噪声 vs 响应速度） |
| E14 | Anchor Probe | 锚点冻结与主动探测 | probe_enabled=False vs True | 主动探测可修正冻结锚点，提升系统利用率 |
| E15 | Straggler Injection | Straggler 注入实验 | straggler_factor=1.0/2.0/3.0/5.0 | 窗口平均机制吸收 2-5× 计算膨胀，无雪崩降级 |

---

## 实验详情

### E10: WFS Baseline (Weighted Fair Sharing)

**动机**：审稿人指出现有基线（Fair, SP, CRUX, DF）未能涵盖最自然的选择——直接将 SLO 松弛系数 ci 映射为权重的加权公平调度。

**实验设计**：
- 实现 WFS 策略：weight_j = 1 / ci_j（tighter SLO → higher weight）
- 在 E1 带宽阶梯（400-1200 Gbps）下对比 6 策略：Fair/WFS/CRUX/SP/D1/v4
- 5 seeds，统计 P-attn、P-cap、S-cap

**关键文件**：
- 策略实现：`longliu_sim/policy/wfs.py`
- 实验脚本：`experiments/exp_e10_wfs.py`
- 配置文件：`configs/e10_wfs.yaml`
- 绘图脚本：`outputs/figures/_draw_e10_wfs.py`

**运行方法**：
```bash
python experiments/exp_e10_wfs.py --seeds 5
python outputs/figures/_draw_e10_wfs.py
```

**预期目标**：证明 LongLiu 的闭式解和可行性边界不仅比启发式好，而且比简单的线性权重映射更好，特别是在资源稀缺且作业异构（计算/通信比例不同）的情况下，WFS 会因为无法精确表达"需求上限"而导致 SLO 违约。

---

### E11: Overlap Sensitivity

**动机**：论文公式(2)假设计算与通信串行，这在现代梯度桶化和流水线并行中过于保守，可能导致 b_i^att 虚高，从而误拒可行租户集。

**实验设计**：
- 引入"重叠率"参数：overlap_factor ∈ {0.0, 0.3, 0.5, 0.85, 1.0}
- 在过渡区带宽（500/630 Gbps）测试 5 策略
- 5 seeds，统计 P-attn、starv

**关键文件**：
- 实验脚本：`experiments/exp_e11_overlap.py`
- 配置文件：`configs/e11_overlap.yaml`
- 绘图脚本：`outputs/figures/_draw_e11_overlap.py`

**运行方法**：
```bash
python experiments/exp_e11_overlap.py --seeds 5
python outputs/figures/_draw_e11_overlap.py
```

**预期目标**：展示即使在实际存在 30%-50% 重叠的情况下，LongLiu 基于串行模型推导的带宽分配依然能维持较高的 SLO 达成率，或者量化指出保守模型在特定资源利用率下才会导致准入误判，以此界定算法的安全边界。

---

### E12: DSCP Quantization Error

**动机**：闭式解计算出连续的带宽分配，但硬件只有 7 个优先级队列。当 Premium 作业很多时，必然有大量作业挤在同一队列，削弱精确 differentiation。

**实验设计**：
- 实现 LongLiuContinuous 策略：理想连续分配（无 DSCP 量化）
- 固定总带宽 800 Gbps，增加集群中的作业数量（14/50/100）
- 对比 v4 (7级量化) vs LongLiuContinuous (理想连续)
- 5 seeds，统计 P-attn、P-cap

**关键文件**：
- 策略实现：`longliu_sim/policy/longliu_continuous.py`
- 实验脚本：`experiments/exp_e12_dscp.py`
- 配置文件：`configs/e12_dscp.yaml`
- 绘图脚本：`outputs/figures/_draw_e12_dscp.py`

**运行方法**：
```bash
python experiments/exp_e12_dscp.py --seeds 5
python outputs/figures/_draw_e12_dscp.py
```

**预期目标**：证明由于相位交错效应和严格优先级的 Work-conserving 特性，即使在高度量化（作业数远大于 7）的情况下，SLO 达成率的下降仍在可接受范围内（例如下降不超过 5%）。

---

### E13: Window Size Sensitivity

**动机**：W=20 贯穿全文，但没有探讨调节该参数的影响。

**实验设计**：
- 测试 LongLiu 控制环策略（非 v4 闭式解）在不同 window_size 下的表现
- window_size ∈ {5, 10, 20, 50}
- E3 swap 场景（630 Gbps），观察优先级调整速度
- 5 seeds，统计 P-attn、DSCP 变化频率

**关键文件**：
- 实验脚本：`experiments/exp_e13_window.py`
- 配置文件：`configs/e13_window.yaml`
- 绘图脚本：`outputs/figures/_draw_e13_window.py`

**运行方法**：
```bash
python experiments/exp_e13_window.py --seeds 5
python outputs/figures/_draw_e13_window.py
```

**预期目标**：展示 W 减小（如 5）时，锚点估计受计算噪声影响变大，可能导致优先级抖动；W 增大（如 50）时，对 Tier Swap 的响应变慢。证明 W=20 是一个较好的工程折中点。

---

### E14: Anchor Freeze + Active Probe

**动机**：审稿人关注在长期 95%+ 高负载下，新作业永远等不到无拥塞窗口的问题。

**实验设计**：
- 构造长期满载场景（30 jobs，95%+ 利用率）
- 测试 T_target_ema 冻结在链路容量 B 时的系统表现
- 激活"主动探测"机制（临时提升到 P6 采样一次）
- 对比 probe_enabled=False vs True
- 5 seeds，统计 P-attn、冻结 job 数量

**关键文件**：
- 实验脚本：`experiments/exp_e14_probe.py`
- 配置文件：`configs/e14_probe.yaml`
- 绘图脚本：`outputs/figures/_draw_e14_probe.py`

**运行方法**：
```bash
python experiments/exp_e14_probe.py --seeds 5
python outputs/figures/_draw_e14_probe.py
```

**预期目标**：展示探测后锚点修正带来的优先级调整和系统整体利用率的提升。

---

### E15: Straggler Injection

**动机**：真实集群中普遍存在慢节点，这是网络调度必须面对的。

**实验设计**：
- 在仿真中随机让 2 个作业的计算时间 Ticomp 膨胀 2-5 倍
- straggler_factor ∈ {1.0, 2.0, 3.0, 5.0}（1.0=control）
- 对比 5 策略：Fair/CRUX/SP/D1/v4
- 5 seeds，统计 P-attn、P-cap

**关键文件**：
- 实验脚本：`experiments/exp_e15_straggler.py`
- 配置文件：`configs/e15_straggler.yaml`
- 绘图脚本：`outputs/figures/_draw_e15_straggler.py`

**运行方法**：
```bash
python experiments/exp_e15_straggler.py --seeds 5
python outputs/figures/_draw_e15_straggler.py
```

**预期目标**：观察 LongLiu 的窗口级平均机制如何吸收这种异常，以及优先级调整是否符合预期（即不发生雪崩式降级）。

---

## 项目目录结构

```
sim-nextgen/
├── configs/                          # 实验配置
│   ├── e10_wfs.yaml                  # E10 WFS 基线配置
│   ├── e11_overlap.yaml              # E11 Overlap 敏感性配置
│   ├── e12_dscp.yaml                 # E12 DSCP 量化配置
│   ├── e13_window.yaml               # E13 窗口大小配置
│   ├── e14_probe.yaml                # E14 锚点探测配置
│   └── e15_straggler.yaml            # E15 Straggler 配置
├── experiments/                      # 实验脚本
│   ├── exp_e10_wfs.py                # E10 WFS 基线实验
│   ├── exp_e11_overlap.py            # E11 Overlap 敏感性实验
│   ├── exp_e12_dscp.py               # E12 DSCP 量化实验
│   ├── exp_e13_window.py             # E13 窗口大小实验
│   ├── exp_e14_probe.py              # E14 锚点探测实验
│   └── exp_e15_straggler.py          # E15 Straggler 实验
├── longliu_sim/policy/               # 调度策略
│   ├── wfs.py                        # WFS 策略（新增）
│   └── longliu_continuous.py         # LongLiu 理想连续分配（新增）
├── outputs/
│   ├── e10_wfs/                      # E10 实验结果
│   ├── e11_overlap/                  # E11 实验结果
│   ├── e12_dscp/                     # E12 实验结果
│   ├── e13_window/                   # E13 实验结果
│   ├── e14_probe/                    # E14 实验结果
│   ├── e15_straggler/                # E15 实验结果
│   └── figures/                      # 论文图表
│       ├── _draw_e10_wfs.py          # E10 绘图脚本
│       ├── _draw_e11_overlap.py      # E11 绘图脚本
│       ├── _draw_e12_dscp.py         # E12 绘图脚本
│       ├── _draw_e13_window.py       # E13 绘图脚本
│       ├── _draw_e14_probe.py        # E14 绘图脚本
│       └── _draw_e15_straggler.py    # E15 绘图脚本
└── PAPER_EVIDENCE/                   # 论文证据归档（只读）
    ├── 10_E10_wfs/                   # E10 归档（实验运行后创建）
    ├── 11_E11_overlap/               # E11 归档
    ├── 12_E12_dscp/                  # E12 归档
    ├── 13_E13_window/                # E13 归档
    ├── 14_E14_probe/                 # E14 归档
    └── 15_E15_straggler/             # E15 归档
```

---

## 复现性保证

1. **配置驱动**：所有实验参数集中在 `configs/e*.yaml`，代码内禁止字面量
2. **种子控制**：所有实验支持 `--seeds N` 参数，默认 5 seeds
3. **快速验证**：所有实验支持 `--quick` 参数（2 seeds）用于快速验证
4. **结果归档**：实验结果自动保存到 `outputs/e*/`，包含 run_meta.json 和 summary.csv
5. **绘图脚本**：每个实验都有独立的绘图脚本，输出 PNG 和 PDF 格式

---

## 运行全部实验

```bash
# 快速验证（2 seeds）
for exp in e10 e11 e12 e13 e14 e15; do
    python experiments/exp_${exp}_*.py --quick
done

# 正式运行（5 seeds）
for exp in e10 e11 e12 e13 e14 e15; do
    python experiments/exp_${exp}_*.py --seeds 5
done

# 绘制所有图表
for fig in _draw_e10 _draw_e11 _draw_e12 _draw_e13 _draw_e14 _draw_e15; do
    python outputs/figures/${fig}_*.py
done
```

---

## 论文证据归档流程

实验运行完成后，需要将结果归档到 `PAPER_EVIDENCE/` 目录：

```bash
# 解冻 PAPER_EVIDENCE
chmod -R u+w PAPER_EVIDENCE/

# 创建归档目录
mkdir -p PAPER_EVIDENCE/10_E10_wfs
mkdir -p PAPER_EVIDENCE/11_E11_overlap
mkdir -p PAPER_EVIDENCE/12_E12_dscp
mkdir -p PAPER_EVIDENCE/13_E13_window
mkdir -p PAPER_EVIDENCE/14_E14_probe
mkdir -p PAPER_EVIDENCE/15_E15_straggler

# 复制实验结果
cp -r outputs/e10_wfs/* PAPER_EVIDENCE/10_E10_wfs/
cp -r outputs/e11_overlap/* PAPER_EVIDENCE/11_E11_overlap/
cp -r outputs/e12_dscp/* PAPER_EVIDENCE/12_E12_dscp/
cp -r outputs/e13_window/* PAPER_EVIDENCE/13_E13_window/
cp -r outputs/e14_probe/* PAPER_EVIDENCE/14_E14_probe/
cp -r outputs/e15_straggler/* PAPER_EVIDENCE/15_E15_straggler/

# 复制实验脚本和配置
cp experiments/exp_e10_wfs.py PAPER_EVIDENCE/10_E10_wfs/
cp configs/e10_wfs.yaml PAPER_EVIDENCE/10_E10_wfs/
# ... 其他实验类似

# 重新冻结
chmod -R a-w PAPER_EVIDENCE/
```

---

## 下一步工作

1. 运行所有实验（建议先 `--quick` 验证，再 `--seeds 5` 正式运行）
2. 检查实验结果是否符合预期
3. 绘制论文图表
4. 归档到 PAPER_EVIDENCE
5. 撰写论文相关章节（§6.7-§6.12）

---

**文档版本**：v1.0  
**创建日期**：2026-08-07  
**作者**：LongLiu 项目组
