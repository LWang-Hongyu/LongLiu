# DWRR 战前准备 Runbook

> **状态**：仅准备文档，不执行任何交换机配置。
> **目标**：为 LongLiu vs CRUX 的 DWRR 对比实验准备配置命令、验证步骤、回滚方案。
> **权重方案**：7 类 TC0–TC6，权重 1:2:4:8:16:32:64
> **创建日期**：2026-07-19

---

## 1. 实验拓扑回顾

```
[10.1 guolab-10]                              [226 guolab-226]
  ConnectX-6 Dx (MT2892)                        BlueField-3 (MT43244, CX-7)
  enp130s0f0np0                                 enp59s0f0np0
  trust=dscp                                    trust=pcp  ← 需改为 dscp
  TSA: strict (全部)                            TSA: vendor (全部)
       │                                              │
       └────────  RoCE  (mlx5_0, GID_INDEX=3) ───────┘
                         │
                  [Cumulus 交换机]
                  (需要 DWRR 配置)
```

**当前状态**：两端 NIC 所有 TC 都是 SP（strict/vendor），交换机未配置 DWRR。
**目标状态**：两端 NIC 启用 ETS（7 类 DWRR），交换机启用 DWRR 调度。

---

## 2. 主机侧 ETS 支持确认（BF-3 / CX-6 Dx）

### 2.1 检查 NIC 型号与 ETS 能力

**10.1 节点（ConnectX-6 Dx）**：
```bash
# 确认 NIC 型号
lspci | grep -i mellanox
# 预期：MT2892 Family [ConnectX-6 Dx]

# 查看当前 QoS 配置
mlnx_qos -i enp130s0f0np0
# 预期字段：
#   DCBX mode: OS controlled
#   Priority trust state: dscp   ← 已是 dscp，无需改
#   tc: 0..7, tsa: strict        ← 当前全 strict（SP）
```

**226 节点（BlueField-3 / ConnectX-7）**：
```bash
ssh 192.10.10.226 "lspci | grep -i mellanox"
# 预期：MT43244 BlueField-3 integrated ConnectX-7

ssh 192.10.10.226 "mlnx_qos -i enp59s0f0np0"
# 预期字段：
#   DCBX mode: OS controlled
#   Priority trust state: pcp    ← 需改为 dscp
#   tc: 0..7, tsa: vendor        ← 当前全 vendor（近似 SP）
```

### 2.2 确认 ETS 算法支持

```bash
# 10.1
mlnx_qos -i enp130s0f0np0 -s help 2>&1 | grep -A2 tsa
# 应输出："Possible algorithms: strict, ets and vendor"

# 226
ssh 192.10.10.226 "mlnx_qos -i enp59s0f0np0 -s help 2>&1 | grep -A2 tsa"
```

**BF-3 ETS 支持结论**：BlueField-3 (CX-7) 与 CX-6 Dx 均通过 `mlnx_qos -s ets,...` 支持 ETS DWRR 调度，TSA 可设为 `ets`。

---

## 3. 主机侧 ETS 配置命令（7 类 DWRR，权重 1:2:4:8:16:32:64）

### 3.1 权重到百分比换算

| TC  | 权重 | 百分比（四舍五入） | DSCP 范围    | 用途                |
|-----|------|------------------|--------------|---------------------|
| TC0 | 1    | 1%               | 0–7          | 最低优先级（BE）    |
| TC1 | 2    | 2%               | 8–15         |                     |
| TC2 | 4    | 3%               | 16–23        |                     |
| TC3 | 8    | 6%               | 24–31        | CRUX Job1 (P3)      |
| TC4 | 16   | 13%              | 32–39        | CRUX Job2 (P4)      |
| TC5 | 32   | 25%              | 40–47        | LongLiu P5          |
| TC6 | 64   | 50%              | 48–55        | LongLiu P6（最高）  |
| TC7 | 0    | 0%               | 56–63        | 保留，strict        |

**校验**：1+2+3+6+13+25+50 = 100% ✓

### 3.2 10.1 节点配置命令

```bash
# 设置 TSA: TC0-TC6 = ets, TC7 = strict
mlnx_qos -i enp130s0f0np0 \
    -s ets,ets,ets,ets,ets,ets,ets,strict

# 设置 ETS 带宽权重（百分比，TC7=0 因为是 strict）
mlnx_qos -i enp130s0f0np0 \
    -t 1,2,3,6,13,25,50,0

# 确认 trust=dscp（已是 dscp，无需改；如需重设）
mlnx_qos -i enp130s0f0np0 --trust=dscp

# 验证配置
mlnx_qos -i enp130s0f0np0
# 预期：tc:0..6 tsa:ets, tc:7 tsa:strict
#       tcbw: 1,2,3,6,13,25,50,0
```

### 3.3 226 节点配置命令

```bash
ssh 192.10.10.226 << 'EOF'
# 设置 TSA: TC0-TC6 = ets, TC7 = strict
mlnx_qos -i enp59s0f0np0 \
    -s ets,ets,ets,ets,ets,ets,ets,strict

# 设置 ETS 带宽权重
mlnx_qos -i enp59s0f0np0 \
    -t 1,2,3,6,13,25,50,0

# 改 trust 从 pcp 到 dscp（关键！226 当前是 pcp）
mlnx_qos -i enp59s0f0np0 --trust=dscp

# 验证配置
mlnx_qos -i enp59s0f0np0
EOF
```

### 3.4 主机侧回滚命令（恢复 SP）

```bash
# 10.1 回滚
mlnx_qos -i enp130s0f0np0 \
    -s strict,strict,strict,strict,strict,strict,strict,strict
mlnx_qos -i enp130s0f0np0 -t 0,0,0,0,0,0,0,0

# 226 回滚
ssh 192.10.10.226 "mlnx_qos -i enp59s0f0np0 \
    -s vendor,vendor,vendor,vendor,vendor,vendor,vendor,vendor && \
    mlnx_qos -i enp59s0f0np0 -t 0,0,0,0,0,0,0,0 && \
    mlnx_qos -i enp59s0f0np0 --trust=pcp"
```

---

## 4. 交换机侧 Cumulus DWRR 配置（**不执行，仅文档**）

> ⚠️ 交换机配置需要人工窗口，本节仅提供命令模板。

### 4.1 Cumulus Linux DWRR 配置

Cumulus Linux 通过 `/etc/cumulus/datapath/traffic.conf` 配置 DWRR。以下为 7 类 DWRR（权重 1:2:4:8:16:32:64）的配置模板：

**步骤 1：编辑 traffic.conf**

```bash
# 登录 Cumulus 交换机
ssh cumulus@<switch-mgmt-ip>

# 备份原配置
sudo cp /etc/cumulus/datapath/traffic.conf \
        /etc/cumulus/datapath/traffic.conf.bak.$(date +%Y%m%d)

# 编辑配置
sudo nano /etc/cumulus/datapath/traffic.conf
```

**步骤 2：在 traffic.conf 中添加/修改以下字段**

```ini
# Enable 8 traffic classes (TC0-TC7)
traffic.class.count = 8

# TC to scheduling algorithm mapping
# 0 = strict, 1 = DWRR
traffic.class.scheduling.algorithm = 1,1,1,1,1,1,1,0
# TC0-TC6 = DWRR(1), TC7 = strict(0)

# DWRR weights for TC0-TC6 (TC7 is strict, weight ignored)
traffic.class.dwrr.weights = 1,2,4,8,16,32,64,0

# DSCP to TC mapping (match host-side mlnx_qos dscp2prio)
# DSCP 0-7   → TC0
# DSCP 8-15  → TC1
# DSCP 16-23 → TC2
# DSCP 24-31 → TC3
# DSCP 32-39 → TC4
# DSCP 40-47 → TC5
# DSCP 48-55 → TC6
# DSCP 56-63 → TC7
cos.0.dscp = 0,1,2,3,4,5,6,7
cos.1.dscp = 8,9,10,11,12,13,14,15
cos.2.dscp = 16,17,18,19,20,21,22,23
cos.3.dscp = 24,25,26,27,28,29,30,31
cos.4.dscp = 32,33,34,35,36,37,38,39
cos.5.dscp = 40,41,42,43,44,45,46,47
cos.6.dscp = 48,49,50,51,52,53,54,55
cos.7.dscp = 56,57,58,59,60,61,62,63
```

**步骤 3：应用配置**

```bash
# 重启 switchd 使配置生效
sudo systemctl restart switchd

# 验证 switchd 状态
sudo systemctl status switchd
```

### 4.2 替代方案：使用 `tc qdisc` 命令（运行时临时配置）

如果不想重启 switchd，可用 `tc` 命令临时配置（重启后失效）：

```bash
# 查看当前 qdisc
tc qdisc show dev swp1

# 设置 DWRR（mq + fq_codel 模拟）
# 注意：Cumulus 的硬件 QoS 通常通过 traffic.conf 配置，
# tc qdisc 主要用于软件路径，硬件 offload 需确认支持
tc qdisc replace dev swp1 root handle 1: mqprio \
    num_tc 8 \
    map 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 \
    queues 1@0 1@1 1@2 1@3 1@4 1@5 1@6 1@7 \
    hw 1 mode channel

# 设置 DWRR 权重
tc qdisc replace dev swp1 parent 1:1 handle 10: dwrr \
    weight 1 quantum 1518
tc qdisc replace dev swp1 parent 1:2 handle 20: dwrr \
    weight 2 quantum 1518
# ... 依此类推
```

> ⚠️ `tc qdisc dwrr` 在 Cumulus 上的支持取决于硬件。生产配置应以 `traffic.conf` 为准。

---

## 5. 生效验证命令

### 5.1 主机侧验证

```bash
# 10.1
mlnx_qos -i enp130s0f0np0 | grep -E "tsa|tcbw|trust"
# 预期：
#   Priority trust state: dscp
#   tc: 0..6 tsa: ets
#   tc: 7 tsa: strict

# 226
ssh 192.10.10.226 "mlnx_qos -i enp59s0f0np0 | grep -E 'tsa|tcbw|trust'"
# 预期同上
```

### 5.2 交换机侧验证

```bash
# 在 Cumulus 交换机上
# 检查 switchd 配置加载
sudo systemctl status switchd

# 查看 TC 配置（如支持）
cat /etc/cumulus/datapath/traffic.conf | grep -E "dwrr|scheduling|class.count"

# 查看接口 QoS 状态
tc qdisc show dev swp1
```

### 5.3 端到端带宽比验证（两 Job 竞争时）

**核心验证**：两 Job 用不同 DSCP 优先级竞争同一条链路，观察带宽比是否接近权重比。

```bash
# Job X 使用 DSCP=24 (TC3, 权重 8)
# Job Y 使用 DSCP=48 (TC6, 权重 64)
# 预期带宽比：8:64 = 1:8 ≈ 11%:89%

# 在 10.1 上启动两个 iperf3 客户端（不同 DSCP）
# 需要先在交换机端口镜像或用 sar/tcpdump 测量

# 简化验证：用我们的 p4_job_reverse.py 跑两个 Job，
# 一个 crux_priority=3 (DSCP=24), 一个 crux_priority=6 (DSCP=48)
# 观察 bw_gbps 比值是否接近 8:64

# 启动验证实验（仅 DWRR 配置生效后执行）
cd /home/why/LongLiu_rebuild/experiments/P4_dumbbell_slo
# 临时修改 p4_job_reverse.py 的 crux_static_priority
# Job A: crux_priority=3 (TC3, weight 8)
# Job B: crux_priority=6 (TC6, weight 64)
bash run_p4_reverse.sh crux 7

# 分析 CSV
python3 -c "
import csv
with open('p4_jobA_reverse_crux_rank0_iter.csv') as f:
    bw_a = [float(r['bw_gbps']) for r in csv.DictReader(f) if r['phase']=='phase1']
with open('p4_jobB_reverse_crux_rank0_iter.csv') as f:
    bw_b = [float(r['bw_gbps']) for r in csv.DictReader(f) if r['phase']=='phase1']
avg_a = sum(bw_a)/len(bw_a)
avg_b = sum(bw_b)/len(bw_b)
print(f'Job A (TC3, weight 8): {avg_a:.2f} Gbps')
print(f'Job B (TC6, weight 64): {avg_b:.2f} Gbps')
print(f'Ratio A:B = 1:{avg_b/avg_a:.2f} (expected 1:8)')
print(f'Deviation: {abs(avg_b/avg_a - 8)/8 * 100:.1f}%')
"
```

**验证通过标准**：观察到的带宽比与权重比的偏差 < 15%（DWRR 近似公平，不要求精确）。

---

## 6. 完整回滚流程

### 6.1 交换机回滚

```bash
ssh cumulus@<switch-mgmt-ip>
# 恢复备份的配置
sudo cp /etc/cumulus/datapath/traffic.conf.bak.YYYYMMDD \
        /etc/cumulus/datapath/traffic.conf
sudo systemctl restart switchd
sudo systemctl status switchd
```

### 6.2 主机回滚

```bash
# 10.1 恢复 SP
mlnx_qos -i enp130s0f0np0 \
    -s strict,strict,strict,strict,strict,strict,strict,strict
mlnx_qos -i enp130s0f0np0 -t 0,0,0,0,0,0,0,0
# trust 保持 dscp（不影响 SP 模式）

# 226 恢复 vendor SP + pcp trust
ssh 192.10.10.226 "mlnx_qos -i enp59s0f0np0 \
    -s vendor,vendor,vendor,vendor,vendor,vendor,vendor,vendor && \
    mlnx_qos -i enp59s0f0np0 -t 0,0,0,0,0,0,0,0 && \
    mlnx_qos -i enp59s0f0np0 --trust=pcp"
```

### 6.3 回滚验证

```bash
# 10.1
mlnx_qos -i enp130s0f0np0 | grep tsa
# 预期：tc: 0..7 tsa: strict

# 226
ssh 192.10.10.226 "mlnx_qos -i enp59s0f0np0 | grep tsa"
# 预期：tc: 0..7 tsa: vendor
```

---

## 7. 实验执行顺序（DWRR 战前准备完成后的流程）

1. **人工窗口**：执行 §3.2 + §3.3（主机 ETS）+ §4.1（交换机 DWRR）
2. **验证**：执行 §5.1 + §5.2 + §5.3（端到端带宽比验证）
3. **跑实验**：
   - `bash run_p4_reverse.sh longliu 7`（LongLiu v1(π) on DWRR）
   - `bash run_p4_reverse.sh crux 7`（CRUX static on DWRR）
4. **分析**：对比 SP vs DWRR 下的 LongLiu/CRUX 性能
5. **回滚**：执行 §6（恢复 SP）

---

## 8. 已知风险与注意事项

1. **226 的 trust=pcp 需改为 dscp**：当前 226 NIC trust=pcp，DSCP 标记不会生效。DWRR 实验前必须改。
2. **BF-3 的 DPU 模式**：BlueField-3 可能运行在 DPU 模式（offload 模式），此时 mlnx_qos 配置可能需要通过 DPU OS 操作，而非主机 OS。需确认 BF-3 工作模式：
   ```bash
   ssh 192.10.10.226 "cat /sys/module/mlx5_core/parameters/num_vfs 2>/dev/null || true"
   ssh 192.10.10.226 "lspci -vvv -s 3b:00.0 | grep -i 'parallel\|single\|dpu' | head -5"
   ```
3. **PFC 与 ETS 的交互**：当前 PFC 全部 disabled，ETS 配置不会与 PFC 冲突。如果后续启用 PFC，需注意 PFC priority 与 ETS TC 的映射。
4. **NCCL 与 DSCP**：NCCL 的 DSCP 标记通过 `NCCL_IB_TC` 或 MultiCommWrapper 的 `multi_comm_set_priority` 设置。当前 MultiCommWrapper 已实现 `priority*8` → DSCP 映射，与 `mlnx_qos dscp2prio` 一致。
5. **DWRR 的严格性**：DWRR 是近似公平，不是严格按权重分配。短流可能获得超过权重比的带宽（packet-level 效应）。验证时允许 ±15% 偏差。
6. **TC7 处理**：TC7 保持 strict（权重 0），用于控制流量（如 LLDP、ARP）。如果 NCCL 误用 TC7，会饿死其他 TC。需确认 NCCL 不会用 DSCP 56-63。

---

## 9. 当前主机状态快照（2026-07-19）

### 10.1 (guolab-10)
```
NIC: ConnectX-6 Dx (MT2892)
Interface: enp130s0f0np0
Trust: dscp
TSA: strict (全部 8 个 TC)
DSCP2prio: 标准映射（每 8 个 DSCP 一个 priority）
PFC: 全 disabled
```

### 226 (guolab-226)
```
NIC: BlueField-3 integrated ConnectX-7 (MT43244)
Interface: enp59s0f0np0
Trust: pcp  ← DWRR 实验前需改为 dscp
TSA: vendor (全部 8 个 TC，近似 SP)
DSCP2prio: 未配置（trust=pcp 时无效）
PFC: 全 disabled
```

### 交换机
```
型号: 待确认（Cumulus Linux）
DWRR 配置: 未配置（当前默认 SP）
traffic.conf: 默认配置
```

---

## 10. 下一步行动项

- [ ] 确认 Cumulus 交换机管理 IP 与登录方式
- [ ] 确认 BF-3 工作模式（DPU mode vs embedded mode）
- [ ] 人工窗口预约：执行 §3 + §4 配置
- [ ] 端到端验证：执行 §5.3 带宽比验证
- [ ] DWRR 实验执行：对比 LongLiu v1(π) vs CRUX on DWRR
- [ ] 实验后回滚：执行 §6
