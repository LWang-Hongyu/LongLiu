# Evaluation Supplement — Physical Prototype Validation (§8.2)

Supplementary evidence pack for §8.2 of the main draft (`LongLiu_INFOCOM_Draft.md`).
This file **supersedes** the Layer-1 numbers of the main draft: the "30.1% vs. fair" figure
in the main draft has no reliable experimental support and should be replaced by the
probe-based quantification below (253× service ratio).

The extended validation is organized in **three layers** that separate mechanism
correctness from real-workload deployment effects:

1. **Layer 1 — Wire-level arbitration + DSCP-on-wire verification** (mechanism is correct)
2. **Layer 2 — Real training task (train_gpt)**: host-side bottleneck
3. **Layer 3 — Contested deployment (V6, two jobs + background flow)**: non-congested regime

---

## §8.2 (Extended) — Physical Prototype Validation

We validate LongLiu's DSCP-based line-rate enforcement on a 2-node P4 testbed
(100G RoCEv2, 7-priority communicator built on NCCL 2.30.7 `trafficClass`).

**Layer 1 — Wire-level arbitration is effective and DSCP marking reaches the wire.**
Two measurements confirm the mechanism end-to-end. First, a dual-flow raw-RDMA probe
(perftest, DSCP 8 vs. DSCP 0) shows strict-priority arbitration at the switch: the
DSCP 8 flow sustains 45.6 Gbps while the DSCP 0 flow is starved to 0.18 Gbps
(253× service ratio), verifying that the P4 data plane enforces the DSCP→TC mapping
under wire congestion. Second, to confirm that NCCL's `config.trafficClass` setting is
not a no-op, we ran a controlled solo AllReduce (1024 MB, `mc_solo_prio`) at P6 and P3
and read the sender's per-priority egress counters (`ethtool -S`). P6 (DSCP 8) traffic
lands on `tx_prio1` (+41 GB) and P3 (DSCP 16) on `tx_prio2` (+41 GB), exactly matching
the NIC's default DSCP-to-priority grouping (`mlnx_qos`: `prio = DSCP>>3`). The apparent
inversion (P6→prio1 vs. P3→prio2) is Mellanox queue semantics rather than a configuration
error, and it has no observable impact in our deployments because the wire is never
congested (total demand ≤ 37 Gbps < 100 Gbps). Together these measurements close the
loop: NCCL DSCP marking reaches the data plane, packets carry the intended DSCP, are
classified into distinct hardware egress queues, and the switch arbitrates strictly
between them.

**Layer 2 — Real training task (host-side bottleneck).** We validate LongLiu's DSCP
enforcement on a realistic training task (GPT-2 tiny). The scheduler correctly triggers
P6 (comm_ratio=1.56), but the communication time does not improve—the bottleneck lies in
host-side resources (NCCL proxy/PCIe), not the switch egress queue. This is not a failure
of the scheduling logic; it reveals a deployment consideration: **DSCP-based line-rate
guarantees are necessary but not sufficient when the bottleneck is above the wire.**

**Layer 3 — Contested deployment (two jobs + background flow).** We run two jobs
(1024 MB AllReduce + 30 ms GPU sleep) against a synthetic background flow (15 and 30 Gbps)
under both LongLiu (dynamic priority; stays at P6 since π>0 throughout) and CRUX
(static P3), across two rounds with alternating job order (R1 LL→CX, R2 CX→LL).
Combined over both rounds, LongLiu is modestly *faster* than CRUX on the SLO-tight job
(+3.5% at 15 Gbps, +14.9% at 30 Gbps) but *slower* on the loose job (−8.6% and −3.9%);
directions are now consistent within each tightness class, yet the magnitudes (a few
percent) are far below the probe's 253× ratio. Across all rows the SLO-tight job's
slowdown consistently exceeds the loose job's (3.4–5.2× vs. 1.8–2.4×), i.e., the effect
of the assigned priority on achieved bandwidth is at most marginal. This corroborates
Layer 2: with total demand (18–37 Gbps) far below the 100 Gbps wire, switch egress queues
are never the bottleneck, so line-rate priority arbitration has almost nothing to
arbitrate. The mechanism is correct (Layer 1) yet nearly invisible in this non-congested
regime — the boundary condition every wire-rate guarantee must acknowledge.

Key findings:
- The full control loop is verified end-to-end: window-granularity scheduling (W=20 iters)
  → comm_ratio emergency signal (1.56 > 1.3) → P2 → P6 → DSCP 8 → TC 0, with a clean solo
  baseline calibrated in window 1.
- NCCL's `config.trafficClass` is verified to reach the wire: ethtool per-priority egress
  counters show P6/P3 landing on distinct hardware queues, exactly matching the NIC's
  `prio = DSCP>>3` mapping.
- The train_gpt case (325 MB gradients, ~7–9 Gbps effective throughput on a 100G wire)
  isolates the boundary condition: DSCP/TC arbitration governs only switch egress queues,
  so it is invisible when the bottleneck is host-side (NCCL proxy / PCIe / shared GPU memory
  bandwidth). V6 confirms the same under a contested two-job deployment.
- The Python→NCCL signal injection path (ctypes + `trafficClass`) is verified functional
  on both nodes.

---

## Evidence Details

### 1. Dual-flow probe (switch strict-priority arbitration)

Raw RDMA (perftest) dual flows, DSCP 8 vs. DSCP 0, through the P4 switch:

| Flow | DSCP | Switch TC | Sustained BW |
|:--|:--:|:--:|:--:|
| High-priority | 8 | tc0 | 45.6 Gbps |
| Low-priority | 0 | tc1 | 0.18 Gbps (starved) |

Service ratio **253×**. Confirms the P4 data plane enforces strict-priority arbitration
between DSCP classes when the wire is the bottleneck.

### 2. `config.trafficClass` on-wire verification (NCCL → hardware queue)

Solo 1024 MB AllReduce at fixed priority (`mc_solo_prio`), sender egress counters
(`ethtool -S enp130s0f0np0`, per-priority TX byte delta over a full run):

| Test | DSCP | ToS | `tx_prioN_bytes` delta | Hardware egress queue |
|:--|:--:|:--:|:--:|:--:|
| P6 | 8 | 0x20 | `tx_prio1` **+41.1 GB** | prio1 |
| P3 | 16 | 0x40 | `tx_prio2` **+41.1 GB** | prio2 |

`mlnx_qos -i enp130s0f0np0` (dscp trust) confirms the mapping is the NIC default grouping
`prio = DSCP>>3` (each 8 consecutive DSCP values map to one priority). The bulk of each
run's traffic lands on exactly one queue — NCCL DSCP marking is **not a no-op**.

Note on semantics: the NIC's egress priority order (`prio` larger → `tc` smaller →
serviced first under strict scheduling) means the paper-intended P6 (DSCP 8) is *lower*
priority than P3 (DSCP 16) at the *NIC egress*. This is Mellanox queue semantics, not a
misconfiguration, and has no observable effect on this testbed because the NIC egress is
never congested. The switch-side DSCP→TC table (the mechanism's arbitration point) is
independently verified correct by the probe above.

### 3. V6 contested deployment (round-1 + round-2)

Two jobs (1024 MB + 30 ms GPU sleep), c_i tight=1.2 / loose=3.0 swapped at window 7,
order alternating across rounds (R1 LL→CX, R2 CX→LL). Background iperf3 UDP DSCP=P3
at 15 / 30 Gbps. Numbers = mean slowdown (window ≥1, excluding cold-start window 0).
Δ = (CX−LL)/CX (positive = LongLiu faster).

**Round 1 (LL→CX) — bg15 (15 Gbps background):**

| Job | Phase (c_i) | LongLiu (P6) | CRUX (P3) | Δ |
|:--|:--|:--:|:--:|:--:|
| A | phase1 **tight** | 4.82 | 4.73 | −1.9% |
| A | phase2 loose | 2.19 | 1.88 | −16.6% |
| B | phase1 loose | 2.38 | 2.03 | −17.1% |
| B | phase2 **tight** | 5.18 | 4.81 | −7.7% |

**Round 1 (LL→CX) — bg30 (30 Gbps background):**

| Job | Phase (c_i) | LongLiu (P6) | CRUX (P3) | Δ |
|:--|:--|:--:|:--:|:--:|
| A | phase1 **tight** | 3.42 | 4.41 | +22.5% |
| A | phase2 loose | 1.97 | 1.78 | −10.9% |
| B | phase1 loose | 2.11 | 1.92 | −9.7% |
| B | phase2 **tight** | 4.00 | 4.45 | +10.1% |

**Round 2 (CX→LL) — bg15 (15 Gbps background):**

| Job | Phase (c_i) | LongLiu (P6) | CRUX (P3) | Δ |
|:--|:--|:--:|:--:|:--:|
| A | phase1 **tight** | 4.18 | 4.96 | +15.7% |
| A | phase2 loose | 2.01 | 1.98 | −1.5% |
| B | phase1 loose | 2.14 | 2.14 | +0.0% |
| B | phase2 **tight** | 4.69 | 5.06 | +7.2% |

**Round 2 (CX→LL) — bg30 (30 Gbps background):**

| Job | Phase (c_i) | LongLiu (P6) | CRUX (P3) | Δ |
|:--|:--|:--:|:--:|:--:|
| A | phase1 **tight** | 4.03 | 4.85 | +16.9% |
| A | phase2 loose | 1.89 | 1.93 | +1.9% |
| B | phase1 loose | 2.04 | 2.08 | +1.5% |
| B | phase2 **tight** | 4.39 | 4.88 | +10.1% |

Round-level directions disagree (R1 favours CX on the tight job at 15 Gbps; R2 favours
LongLiu on the tight job everywhere), but combined over the two rounds LongLiu is
consistently faster on the SLO-tight job and slower on the loose job (see the summary
table under Figures). Across all rows the SLO-tight job's slowdown exceeds the loose job's
(3.4–5.2× vs. 1.8–2.4×) — bandwidth allocation is essentially independent of priority,
confirming the non-congested regime.

---

## Figures

Generated by `results/figures_unified/fig7_v6_round1/scripts/plot_v6_r1r2_supp.py`
(IEEEtran full-width, same style as `fig6_v6physical/plot_fig6.py`:
LongLiu blue #0072B2 solid + square, CRUX orange #D55E00 dashed + circle,
serif/stix, grey tight-window band, dotted grid, 600 dpi PNG + PDF).
Data: `experiments_evaluation/P4_dumbbell_slo/data_v6_bg{15,30}_round{1,2}/`.
Each curve = mean over the two rounds, shaded band = round min–max.

- **fig7_v6_r1r2_trajectory** (2×2): window-by-window slowdown (windows 0–14) for
  15/30 Gbps background × Job A/B. Grey band = SLO-tight window range
  (A: w0–6, B: w7–14; c_i = 1.2).
  `results/figures_unified/fig7_v6_round1/figures/fig7_v6_r1r2_trajectory_600.png`
- **fig7_v6_r1r2_summary**: mean slowdown per (bg rate × SLO tightness);
  bar = mean over 2 rounds × 2 jobs, error bar = full range (window 0 excluded).
  `results/figures_unified/fig7_v6_round1/figures/fig7_v6_r1r2_summary_600.png`

Summary numbers (LL / CX, mean over 2 rounds × 2 jobs, window 0 excluded;
Δ = (CX−LL)/CX, positive = LongLiu faster):

| bg | SLO | LongLiu | CRUX | Δ (CX-relative) |
|:--|:--|:--:|:--:|:--:|
| 15 Gbps | tight | 4.72 | 4.89 | +3.5% |
| 15 Gbps | loose | 2.18 | 2.01 | −8.6% |
| 30 Gbps | tight | 3.96 | 4.65 | +14.9% |
| 30 Gbps | loose | 2.00 | 1.93 | −3.9% |

Combined, LongLiu is faster than CRUX on the SLO-tight job at both background rates
(+3.5% / +14.9%) and slower on the loose job (−8.6% / −3.9%); the effect is a few percent
at most, versus 253× for the wire-level probe — consistent with the non-congested regime.

---

## Reproduction

All commands run on node 10.1 (real shell; GPU + RDMA required).

```bash
# 1. Dual-flow probe (switch SP): existing probe script, DSCP 8 vs DSCP 0
# 2. trafficClass on-wire verification
cd /home/why/LongLiu_rebuild/current/experiments_evaluation/P4_dumbbell_slo
bash verify_nccl_dscp.sh 6 3
#    expects: P6 -> tx_prio1 +41GB, P3 -> tx_prio2 +41GB

# 3. V6 contested deployment
bash run_v6_full.sh 1 15    # round 1: LL->CX, 15 Gbps bg
bash run_v6_full.sh 1 30    # round 1: LL->CX, 30 Gbps bg
bash run_v6_full.sh 2 15    # round 2: CX->LL, 15 Gbps bg
bash run_v6_full.sh 2 30    # round 2: CX->LL, 30 Gbps bg
```

Outputs: `data_v6_bg{15,30}_round{1,2}/` (per-job window CSV: window,
phase, payload_mb, c_i, sleep_us, avg_comm_s, avg_bw_gbps, pi, priority, dscp, slowdown,
t_target_ms).

---

## Status

- [x] Layer 1 (probe 253× + trafficClass on-wire)
- [x] Layer 2 (train_gpt)
- [x] Layer 3 (V6 round 1: bg15 + bg30)
- [x] Layer 3 (V6 round 2: bg15 + bg30) — merged into the tables above
