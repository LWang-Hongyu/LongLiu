# T-1：T_target 校准表

> **论文用途**：SLO 调度器的 T_target 标定方法和结果。
> **对应证据**：PAPER_EVIDENCE/06_calibration
> **数据就绪**：✅

## 内容

```
t1_ttarget_calib/
├── README.md                           # 本文件
├── 06_calibration/
│   ├── 6G_acceptance_record.txt        # 6G 背景流校准验收
│   └── ttarget_calibration.txt         # T_target 标定参数
```

## 关键参数

| 参数 | Job A | Job B |
|------|-------|-------|
| T_target (ms/epoch) | 4201.087 | 3905.163 |
| 校准 payload | 1024 MB | 1024 MB |
| c_i tight/loose | 1.2 / 3.0 | 3.0 / 1.2 |
| ITERS_PER_EPOCH | 20 | 20 |
| 校准方式 | Phase-0 solo Dedicated | Phase-0 solo Dedicated |
| 校准文件 | `/tmp/ttarget_v5_jobA.json` | `/tmp/ttarget_v5_jobB.json` |

## 口径说明

- T_target = **纯通信时间**/epoch（不含 compute overlap）
- 单位：per_epoch_ms（1 epoch = 20 iterations）
- Slowdown = avg_comm / (c_i × T_target_per_iter)，其中 T_target_per_iter = T_target_epoch / 20
- 仿真侧 T_target = per-iteration，二者无冲突（对比仅用无量纲 ratio）
