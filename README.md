# LongLiu 仿真器（下一代）

面向多租户 AI 训练网络调度的 flow-level 离散事件仿真器。

## 快速开始

```bash
cd /home/why/LongLiu_rebuild/sim-nextgen
python3 tests/test_simulator.py
python3 experiments/exp_compare.py
```

## 目录结构

- `longliu_sim/`：核心仿真包
  - `core/`：事件循环与仿真器
  - `network/`：拓扑、链路、flow
  - `job/`：Job 模型
  - `policy/`：调度策略
- `experiments/`：论文实验脚本
- `tests/`：单元测试

## 已实现功能

- [x] 事件驱动仿真核心
- [x] 单瓶颈链路拓扑
- [x] Fair / LongLiu 调度策略
- [x] 迭代级 workload
- [x] SLO 达成率统计

## 进行中

- [ ] Fat-Tree 拓扑感知
- [ ] CRUX / CASSINI 策略
- [ ] Alibaba Lingjun trace 接入
- [ ] 物理原型机校准
