# Archive 索引 — P4_dumbbell_slo 历史归档

本目录存放**已完成的历史实验产物**与**已过时的探索脚本**（2026-08-18 整理）。
所有归档均为 `mv` 移动，未删除任何文件；如需恢复，按下表移回顶层即可。

| 子目录 | 内容 | 来源/说明 |
|:--|:--|:--|
| `probe_runs/` | DSCP 双流探针与 226 分类探针的运行输出 | `fullclass_probe_20260724_archive/`（probe_dscp_priority.sh 及运行结果）、`226_classify_probe_20260724_*/`（7 个 DSCP 值 × 多轮）、`verify_nccl_dscp_{3,6}_20260817_*/`（trafficClass 验证原始输出：eth_delta/tos_summary/tcpdump_samples/rank 日志） |
| `legacy_asym/` | 非对称 payload 实验 | `run_p4_asym.sh`、`run_p4_asym_v2.sh`、`p4_job1_asym.py`、`p4_job2_asym.py`；另含 V6 早期入口 `run_v6_round1_bgfirst.sh`（EXP_DIR 为旧路径，已过时） |
| `legacy_misc/` | 诊断/带宽/IB 优先级探针工具 | `run_diagnose.sh`+`diagnose_isolation.py`+`diagnose_v5_anomaly.py`、`bench_*`、`run_solo_bw.sh`、`run_quick_bw_test.sh`、`ib_prio_test.*`、`ib_prio_strict.sh`、`verify_dscp_queue.sh`、`probe_dscp_priority.sh`、`probe_226_classify.sh`、v3 可视化 `plot_p4.py` |
| `analysis/` | 历史分析脚本与文本结论 | `analyze_results.py`(v3)、`analyze_v4_deep.py`、`analyze_v5.py`、`analyze_reverse{,_v2}.py`、`v4/v5_analysis_deep.txt`、`reverse_v2_summary.txt`、`run_meta_v6_p6_unclamped.txt` |
| `logs/` | 松散运行日志 | v3/v4/v5/asym/V6 历史运行的 node101 日志与 console 日志（数据已提取入 CSV，日志仅备查） |
| `results_legacy/` | 历史 CSV | v3（`p4_job{1,2}_*_rank0.csv`）、asym、诊断、reverse 对比、v4/v5 带宽对比、V6 p4cap 与 iter 粒度 CSV（**注意**：v6 round1/round2 的 iter CSV 可能被后续运行覆盖过，权威数据在顶层 `data_v6_bg*_round*/` 的 window CSV） |
| `misc/` | 历史图 | `p4_bandwidth.png`、`p4_iter_time.png`、`p4_slo_ui.png` |
| `data_backup/` | 数据备份/重复目录 | `v2_backup/`、`v6_dup_20260806_of_rep2/`、`v6_replication_3_pre_archive/`（无脚本引用，仅备份） |
| `README_v3_legacy.txt` | v3 时代旧 README | 原 `README.txt`，已由顶层 `README.md` 取代 |
