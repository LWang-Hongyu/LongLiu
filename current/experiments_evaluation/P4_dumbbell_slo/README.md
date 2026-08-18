# P4 Dumbbell SLO 实验目录 — 复现说明

LongLiu 基于 DSCP 动态优先级的 SLO 调度在 P4 交换机测试床上的物理原型验证。
本目录是全部实验脚本与数据的单一入口；论文侧的数字与三层证据结构见
`LongLiu_INFOCOM_EvalSupplement.md`（位于仓库 `current/` 根目录）。

实验证据按三层组织（每层对应一套可独立复现的脚本）：

| 层 | 内容 | 入口脚本 | 结论 |
|:--|:--|:--|:--|
| Layer 1 | 交换机线速仲裁 + NCCL `trafficClass` DSCP 上线验证 | `verify_nccl_dscp.sh` + 双流探针 | 机制正确（探针 253× 服务比；DSCP 真实打到线缆） |
| Layer 2 | 真实训练任务 train_gpt | `run_gpt_validation.sh` → `run_p4.sh train_gpt` | 主机侧瓶颈，优先级在无线口拥塞时不可见 |
| Layer 3 | 争用部署（V6：双作业 + 背景流） | `run_v6_full.sh <round> <bg>` | 非拥塞区间（总需求 ≤37 Gbps < 100G），优先级无实质仲裁对象 |

---

## 1. 测试床环境

- **节点**：10.1（Quadro RTX 4000，RoCEv2）/ 226（2× RTX 5000，RoCEv2）
- **网络**：100G RoCEv2，P4 可编程交换机，RDMA 实测 ~46 Gbps
- **NCCL**：系统编译版 **2.30.7**（`ncclConfig_t.trafficClass` 支持多优先级 communicator）
- **调度库**：`/home/why/LongLiu_rebuild/current/multi_comm_slo/`（`build/libmulti_comm.so` + `src/slo_scheduler.py`）
- **DSCP→TC 映射**（P4 交换机实测）：P6→DSCP8/tc0、P4→DSCP0/tc1、P3/P5→DSCP16/tc2、P2→DSCP24/tc3、P1→DSCP32/tc4、P0→DSCP40/tc5
- **NIC DSCP→prio 分组**：`prio = DSCP>>3`（Mellanox 默认，经 `mlnx_qos` 实测确认，非配置错误）

## 2. 目录结构

```
P4_dumbbell_slo/
├── README.md                  # 本文件
├── run_v6_full.sh             # [Layer3] V6 主入口：双轮交替 + 12 路背景流
├── run_p4_reverse.sh          # [V5/V6] 校准 + reverse 主实验（V6 前置：生成 T_target）
├── run_v6_calib_atomic.sh     # [V6] ttarget 原子校准（替代入口）
├── run_v6_calibrate.sh        # [V6] 背景流效果校准（注意：EXP_DIR 为旧路径，已过时）
├── run_v6_p4cap_llarm.sh      # [V6] p4cap LL 变体
├── run_calib_sweep.sh         # [V6] 校准扫描
├── verify_nccl_dscp.sh        # [Layer1] NCCL trafficClass → 硬件队列验证
├── mc_solo_prio.py            # [Layer1] 固定优先级 solo AllReduce 驱动
├── run_gpt_validation.sh      # [Layer2] train_gpt 验证（solo/fair/longliu 按序）
├── run_p4.sh                  # [v3/Layer2] 模式入口（solo/fair/longliu/crux/train_gpt）
├── p4_job_reverse.py          # [V6 核心] 双作业 window 粒度调度脚本（被 run_v6_full 等引用）
├── p4_train_gpt.py            # [Layer2] GPT-2 tiny 训练
├── p4_job1.py p4_job2.py      # [v3] Job1/Job2 训练脚本（run_p4.sh 引用）
├── p4_job1_crux.py p4_job2_crux.py  # [v3] CRUX 固定优先级变体
├── v6_background_flow.sh      # [V6] iperf3 UDP DSCP=P3 背景流
├── sync_to_226.sh             # 通用：同步脚本/数据到 226
├── analyze_v6.py              # [V6] window CSV 分析（读顶层 + v6_replication_*）
├── analyze_v6_stats.py        # [V6] 统计复制分析（读 v6_replication_{1..5}）
├── data_v6_bg{15,30}_round{1,2}/  # V6 每窗口 CSV（fig7 数据源）
├── v6_replication_{1..5}/     # V6 统计复制数据（fig6 数据源）
└── archive/                   # 历史归档（见 archive/README_archive.md）
```

## 3. 复现步骤

所有实验需在 **10.1 的真实 shell** 中执行（GPU + RDMA + 免密 ssh 到 226）。

### 3.0 前置检查

```bash
# 1) 同步代码与 226 一致（Layer2/3 都依赖）
bash sync_to_226.sh

# 2) 确认 226 与 10.1 使用同一 NCCL 版本（2.30.7 系统编译版）
#    101 侧启动时须显式设置 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
#    （否则回退到 PyTorch bundled NCCL 2.18.6，带宽慢 ~3×）
ldconfig -p | grep libnccl      # 两侧都应命中 2.30.7

# 3) 网卡接口名（ethtool/tcpdump 判定用，不是 RDMA 设备名 mlx5_0）
#    10.1: enp130s0f0np0     226: enp59s0f0np0
```

### 3.1 Layer 1 — 机制验证

```bash
# a) 双流探针（交换机严格优先级仲裁，253× 服务比）：
#    脚本已归档：archive/probe_runs/fullclass_probe_20260724_archive/probe_dscp_priority.sh

# b) NCCL trafficClass 上线验证（核心）：
cd /home/why/LongLiu_rebuild/current/experiments_evaluation/P4_dumbbell_slo
bash verify_nccl_dscp.sh 6 3
#    预期：P6 → tx_prio1 +41 GB，P3 → tx_prio2 +41 GB
#    （无 CAP_NET_RAW 时自动降级为纯 ethtool 判定）
```

### 3.2 Layer 2 — 真实训练（train_gpt）

```bash
bash run_gpt_validation.sh    # 按序跑 solo → fair → longliu，输出 loss 下降汇总
```

### 3.3 Layer 3 — V6 争用部署

```bash
# 步骤 1：T_target 校准（1024MB payload，生成 /tmp/ttarget_v5_job{A,B}.json）
bash run_p4_reverse.sh v5 both

# 步骤 2：跑 2 轮 × 2 背景速率（每轮 ~18 min）
bash run_v6_full.sh 1 15     # round 1: LL→CX, 15 Gbps bg
bash run_v6_full.sh 1 30     # round 1: LL→CX, 30 Gbps bg
bash run_v6_full.sh 2 15     # round 2: CX→LL, 15 Gbps bg
bash run_v6_full.sh 2 30     # round 2: CX→LL, 30 Gbps bg

# 步骤 3：结果
#   CSV → data_v6_bg{15,30}_round{1,2}/  (p4_job[AB]_v6_*_rank0_window.csv)
#   日志 → 顶层 p4_job[AB]_v6_*_node101.log（汇总时记得归档）
#   背景流 → /tmp/v6_bgflow_*.log
```

## 4. 关键陷阱（历史踩坑记录）

1. **NCCL 版本分裂**：`libmulti_comm.so` 的 libnccl 依赖由 `LD_LIBRARY_PATH` 决定，与 torch 内置 2.18.6 独立加载。101 侧启动必须 `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`，否则慢 3×。
2. **T_target 文件必须同步到 226**（run_v6_full.sh 已内置 scp + 校验）：226 缺失时 rank1 回退本地 EMA 学习，与 rank0 的 preset 分叉 → π 不同 → 两侧用不同 communicator → **AllReduce 永久死锁**。
3. **接口名**：`mlx5_0` 是 RDMA 设备名；ethtool/tcpdump 用 netdev 接口名 `enp130s0f0np0`(10.1)/`enp59s0f0np0`(226)。
4. **无 CAP_NET_RAW**：226 与 10.1 均无法 tcpdump，DSCP 上线判定以 ethtool 每优先级 TX 字节增量为准。
5. **等待超时**：30 Gbps 背景下 ~2.7 s/iter，单模式约 900 s；`run_v6_full.sh` 已放宽至 1080 s。老脚本（420 s）会误杀正常任务。
6. **NIC 出口语义**：prio 越大 → tc 越小 → strict 下越优先，P6(DSCP8,prio1) 在 NIC 出口反而低于 P3(DSCP16,prio2)。这是 Mellanox 队列语义而非配置错误；无线口不拥塞时无实际影响。

## 5. 数据产出与图

- **fig6**（epoch 统计复制）：数据 `v6_replication_{1..5}/`，绘图 `results/figures_unified/fig6_v6stats/`
- **fig7**（V6 window 轨迹 + 汇总，r1r2 合并版）：数据 `data_v6_bg{15,30}_round{1,2}/`，绘图 `results/figures_unified/fig7_v6_round1/scripts/plot_v6_r1r2_supp.py`（数据副本在 `fig7_v6_round1/data/`）

## 6. 历史归档

旧实验脚本、探针运行输出、历史 CSV/日志/分析已移入 `archive/`，索引见 `archive/README_archive.md`。归档为纯 `mv`（未删除任何文件），可按索引恢复。
