# LongLiu 仿真器核心库（受保护源码）

> **⚠️ 警告**：本目录包含论文所有实验结果的核心仿真器源码。任何修改都会影响 E01-E09 全部已归档实验的可复现性。

## 保护等级

本目录实施**最高保护等级**：

1. **禁止未批准修改**：任何文件变更需先说明动机、评估对既有结果的影响，经确认后修改，并在 `HANDOFF.md` 记录
2. **公式单实现**：`utils/metrics.py` 中的 SAS/target_iter_ms 计算是唯一权威实现，禁止在其他位置复制
3. **拓扑冻结**：`network/topology.py` 的路由/链路模型严禁修改，变更会使所有历史结果作废
4. **归档校验**：任何源码变更后，必须重跑 `scripts/gatekeeper.py` 确认历史 headline 数字不变

## 目录结构

```
longliu_sim/
├── core/           # 事件驱动仿真引擎（simulator.py, event.py）
├── network/        # 网络模型（topology.py, link.py, flow.py）
├── job/            # 任务模型（job.py - 迭代生命周期、barrier）
├── policy/         # 调度策略实现
│   ├── longliu.py  # LongLiu v4（tier-aware DWRR）
│   ├── dwrr.py     # DWRR / DF（deficit-feedback）
│   ├── crux.py     # CRUX（GPU-intensity 权重）
│   ├── fair.py     # Fair（均分带宽）
│   ├── srpt.py     # SRPT / SP（最短剩余处理时间）
│   └── cassini.py  # CASSINI（time-shift 交错）
├── trace/          # Trace 生成与解析
│   ├── synthetic.py    # 合成 workload 生成
│   ├── lingjun.py      # Alibaba Lingjun trace loader
│   └── synthetic_128.py # 128 节点合成 trace
├── metrics/        # 指标计算（stats.py）与可视化（plot.py）
└── utils/          # 工具（config.py, metrics.py, model_params.py）
```

## 实验隔离原则

**实验参数/场景变更一律通过以下方式实现，不得改动本目录：**

- `configs/` 目录：新增实验配置文件（如 `trace_replay.yaml`）
- `experiments/` 目录：新增实验脚本（如 `exp_trace_replay.py`）

## Git 纪律

- 本目录的变更必须显式提交并注明影响范围
- 提交信息必须包含：`[CORE]` 前缀 + 影响的实验编号（如 `E01-E09`）
- 建议的提交信息格式：`[CORE] 修改 utils/metrics.py 中的 XXX 计算 - 影响 E03/E04`

## 验证清单

修改本目录后，必须完成以下验证：

- [ ] 重跑 `scripts/gatekeeper.py` 确认锚点数字不变
- [ ] 重跑受影响实验的 summary 脚本，确认 headline 数字一致
- [ ] 在 `HANDOFF.md` 记录变更动机、影响范围、验证结果
- [ ] 更新 `PAPER_EVIDENCE/MANIFEST.md` 中的 config_hash（如适用）
