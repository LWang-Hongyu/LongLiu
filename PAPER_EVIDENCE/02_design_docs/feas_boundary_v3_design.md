# feas_boundary_v3 场景设计稿

> 重构日期: 2026-07-27
> 来源: HANDOFF_formula_restore.md, conversation records, batch3.log
> SEMANTICS_VERSION: anchor-v2, config_hash: 57f57512

---

## 一、三场景 attain 表

| 场景 | 定义 | #Jobs | Premium | Standard | ΣP (Gbps) | ΣS (Gbps) | C* (Gbps) | Regime |
|------|------|-------|---------|----------|-----------|-----------|-----------|--------|
| E1 | 正相关（大模型紧ci） | 14 | 8 (ci≤2.0) | 6 (ci=3.0) | 605.7 | 263.6 | 737.5 | 保障场景 |
| E2' | 逆相关（大模型紧ci） | 14 | 9 (ci≤2.0) | 5 (ci=3.0) | 623.5 | 257.8 | 752.4 | CRUX杀场 |
| E2-pro | CRUX主场（小模型紧ci） | 13 | 5 (ci≤2.0) | 8 (ci=3.0) | 632.9 | 275.4 | 770.6 | 正交对照 |

---

## 二、四点 regime 标注

| Spine BW | Σ/C* | Regime | 期望行为 |
|----------|------|--------|----------|
| 400 Gbps | 0.49-0.53 | 深不可行区 | 全部崩溃，v4 仍保 ≥50% P-attn |
| 500 Gbps | 0.61-0.66 | 不可行区 | 基线重创，v4 保持 ≥80% |
| 630 Gbps | 0.77-0.84 | 边际区 | CRUX 表现波动，v4 维护 |
| 800 Gbps | 0.98-1.06 | 可行/边界 | v4 ≤ baseline 持平 |
| 1000 Gbps | 1.22-1.33 | 深可行区 | 全部达标 |
| 1200 Gbps | 1.46-1.59 | 过保障区 | 全部≥100% |

---

## 三、E1 正相关场景 workload (FEAS_BOUNDARY_V3_WORKLOAD)

```
14 jobs, 8 premium + 6 standard
Premium tier: ci=1.5/2.0 (大模型紧ci → 正相关)
  J0: LLaMA-2-13B, dp=8, ci=1.5
  J1: LLaMA-2-13B, dp=8, ci=1.5
  J2: LLaMA-2-13B, dp=8, ci=1.5
  J3: LLaMA-2-13B, dp=8, ci=1.5
  J4: LLaMA-2-7B,  dp=8, ci=1.5
  J5: LLaMA-2-7B,  dp=8, ci=1.5
  J6: BERT-Large,  dp=4, ci=2.0
  J7: T5-11B,      dp=8, ci=2.0
Standard tier: ci=3.0 (小/中模型松ci → 正相关)
  J8:  LLaMA-2-13B, dp=8, ci=3.0
  J9:  LLaMA-2-13B, dp=8, ci=3.0
  J10: BERT-Large,  dp=4, ci=3.0
  J11: BERT-Large,  dp=4, ci=3.0
  J12: BERT-Large,  dp=2, ci=3.0
  J13: BERT-Large,  dp=2, ci=3.0

ΣP = 605.7G, ΣS = 263.6G, C* = 737.5G
per_flow_attain 全部 ≤ 100G
```

---

## 四、E2' 逆相关场景 workload (FEAS_BOUNDARY_V3_PRIME_WORKLOAD)

```
14 jobs, 9 premium + 5 standard
Premium tier: ci=1.5/2.0 (大模型紧ci → CRUX 低 intensity → 被压底)
  J0-J5: LLaMA-2-13B ×6, dp=8, ci=1.5  (intensity=0.15 → CRUX LOW)
  J6-J7: LLaMA-2-7B  ×2, dp=8, ci=1.5  (intensity=0.14)
  J8:    T5-11B      ×1, dp=8, ci=2.0  (intensity=0.14)
Standard tier: ci=3.0 (小模型松ci → CRUX 高 intensity → 抬顶)
  J9-J11: BERT-Large ×3, dp=4, ci=3.0  (intensity=1.84 → CRUX HIGH)
  J12-J13: ViT-Large ×2, dp=2, ci=3.0  (intensity=1.22)

ΣP = 623.5G, ΣS = 257.8G, C* = 752.4G
预期：CRUX intensity排序与tier完全反向 → 崩溃；v4 不受影响
```

---

## 五、E2-pro 正交对照 workload (FEAS_BOUNDARY_V3_PRO_WORKLOAD)

```
13 jobs, 5 premium + 8 standard
Premium tier: ci=1.5/2.0 (小模型紧ci → CRUX 高 intensity → 高优先级)
  J0-J1: BERT-Large ×2, dp=2, ci=1.5  (intensity=0.92)
  J2:    BERT-Large ×1, dp=4, ci=2.0  (intensity=1.84)
  J3-J4: ViT-Large  ×2, dp=2, ci=1.5  (intensity=1.22)
Standard tier: ci=3.0 (大模型松ci → CRUX 低 intensity → 低优先级)
  J5-J7:  LLaMA-2-13B ×3, dp=8, ci=3.0  (intensity=0.15)
  J8-J10: LLaMA-2-7B  ×3, dp=8, ci=3.0  (intensity=0.14)
  J11-J12: T5-11B     ×2, dp=8, ci=3.0  (intensity=0.14)

ΣP = 632.9G, ΣS = 275.4G, C* = 770.6G
预期：CRUX ≈ 100% 持平 v4（CRUX主场）
```

---

## 六、矩阵 v2.2 判定规则全文

### E1 (正相关阶梯) 判定
| Spine BW | v4 下界 | D1 baseline | Fair baseline |
|----------|---------|-------------|---------------|
| 1200 Gbps | P-attn=100%, starv=0 | 观测行 | 观测行 |
| 1000 Gbps | P-attn=100%, starv=0 | 观测行 | 观测行 |
| 800 Gbps | P-attn=100%, starv=0 | 观测行 | 观测行 |
| 630 Gbps | P-attn=100%, starv=0 | 观测行 | 观测行 |
| 500 Gbps | P-attn≥80%, starv=0 | v4 > D1 | v4 > Fair |
| 400 Gbps | P-attn≥50%, starv=0 | v4 > D1 | v4 > Fair |

### E2' (真·杀场) 判定
| Spine BW | v4 | CRUX | 判定 |
|----------|-----|------|------|
| 800 Gbps | P-attn=100% | 观测行 | CRUX可能存活（充裕容量） |
| 630 Gbps | P-attn=100% | P-attn ≪ v4, gap≥25pp | CRUX 崩溃 |
| 500 Gbps | P-attn≥75% | P-attn ≪ v4 | CRUX 全崩 |
| 400 Gbps | P-attn≥37.5% | P-attn=0% | 全部崩溃 regression |

### E2-pro (CRUX主场) 判定
| Spine BW | v4 | CRUX | 判定 |
|----------|-----|------|------|
| 800 Gbps | P-attn=100% | P-attn≈100% 持平 | 正交确认 CRUX 在其优势场景正常 |
| 630 Gbps | P-attn=100% | P-attn≈100% 持平 | 正交确认 |

### E3 (对照臂) 判定
- v4 W1 P-attn=100%, W3 P-attn=100%, starv=0
- CRUX W3 观测行（顺 tier swap → 存活）

### E3' (杀伤臂) 判定
- v4 W3 P-attn=100% (下界), starv=0
- CRUX W3 P-attn ≪ v4 (gap ≥ 10pp)
- 任一 FAIL 停跑上报

---

## 七、策略列表

| 缩写 | 全称 | 用途 |
|------|------|------|
| v4 | LongLiuAllocatorV4 | 两级分层（premium池 + standard floor） |
| CRUX | CRUX baseline | 基于 byte-progress 的优先级调度 |
| D1 | LongLiuDWRR (DSCP+DWRR+exp(pi*K)) | 单层反馈定律 |
| SP | SRPT (Shortest Remaining Processing Time) | 最短剩余优先 |
| Fair | FairShare | 均等份额 baseline |
