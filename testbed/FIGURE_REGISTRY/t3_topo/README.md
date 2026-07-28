# T-3：拓扑与映射探测数据表

> **论文用途**：物理床拓扑约束 & DSCP→TC 映射修正的实证列。
> **对应证据**：PAPER_EVIDENCE/03_class_probe, 04_226_probe, 05_tc_map
> **数据就绪**：✅

## 内容

```
t3_topo/
├── README.md                           # 本文件
├── t3_data_table.csv                   # 三表合一主数据
│
├── 03_class_probe/                     # DSCP 0-56 全类探测
│   ├── probe_dscp_priority.sh          # 探测脚本
│   └── run_meta.txt
│
├── 04_226_probe/                       # 226 NIC 分类能力探针（三轮）
│   ├── 226_classify_probe_20260724_155645/
│   ├── 226_classify_probe_20260724_161454/
│   ├── 226_classify_probe_20260724_161816/
│   └── 各轮含：run_meta.txt, tx_before/after_dscp*.txt, 
│              rx_before/after_bg.txt, probe_dscp*.log
│
├── 05_tc_map/                          # tc_map 修正记录
│   ├── discovery_path.txt              # 发现路径
│   ├── old_map_config.txt / new_map_config.txt
│   └── mlnx_qos_commands.txt           # 采集命令
│
└── ib_prio_test.py                     # IB 优先级测试脚本

## 论文输出

```
├── tab_t3.tex                          # T-3 LaTeX 三合一表（tab:testbed-hw）
│   (a) tc_map 修正前后映射对照
│   (b) 全类探测关键计数（5.3TB prio3 / 0.049TB prio4）
│   (c) 226 vs 10.1 NIC 分类能力对照（9.79TB prio0 vs 分布式计数）
└── tab_t3_crossref.tex                 # 逐格对照表（审计件）
```

## 三表概览

| 表 | 内容 | 结论 |
|----|------|------|
| A（03_class_probe） | DSCP 0-56 × 1G 探针 vs 6G P3 背景 | 10.1 NIC 正确分类（tx_prio3=5.3TB） |
| B（04_226_probe） | 反向模式 iperf3 + NIC 计数器 | 226 NIC **不分类**（prio0=9.79TB, prio1-7=0） |
| C（05_tc_map） | 旧(p×8)→新映射修正 | 修正后 P6→tc:0 > P4→tc:1 > P3→tc:2 |

## 论文用途

1. **§5 拓扑约束**：不对称分类能力（10.1 分类 vs 226 不分类）
2. **§5 映射修正**：`trafficClass = priority × 8` 的默认映射在 SP 队列下产生优先级反转
3. **§5 验证**：V6-P4 结果证实修正映射有效（P4→tc:1 优于 P3→tc:2）
