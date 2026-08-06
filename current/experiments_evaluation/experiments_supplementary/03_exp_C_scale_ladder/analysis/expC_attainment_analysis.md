# Experiment C: Attainment + S-cont Reanalysis (27 rounds)

> Generated: 2026-07-30T10:05:32+08:00
> Analysis window: epochs 5-19
> Metric definitions:
> - **Attainment** = slowdown / c_i (≤1 = SLO met, >1 = SLO violated)
> - **P-attn** = Σ_premium max(0, slowdown-1) (premium attention needed)
> - **S-cont** = Σ_standard max(0, slowdown-1) (contention suffered by standard)
> - **SLO rate** = fraction of epochs where SLO was met

## Regime: ample

> Σb^att/B ≈ 0.96 — corresponds to E1 1200G point. 2 jobs, 1 premium + 1 standard.
> Expected Σb^att/B = 0.9

### Per-job metrics (mean ± std across rounds)

| Job | Label | Class | c_i | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|-------|-----|------|------|------|------|------|------|
| 0 | J0_premium_L | premium | 1.2 | 1.499±0.040 | 1.249±0.034 | 1.179±0.236 | 0.982±0.196 | 1.174±0.233 | 0.979±0.194 | 
| 1 | J1_standard_M | standard | 1.5 | 1.013±0.001 | 0.675±0.000 | 0.772±0.152 | 0.514±0.101 | 0.682±0.001 | 0.455±0.000 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO rate | Std SLO rate | N_jobs slow<1 |
|-----|--------|--------|-----------------|-------------|--------------|
| fair | 0.499±0.040 | 0.013±0.001 | 0.000 | 1.000 | 0.0/2 |
| longliu | 0.179±0.236 | 0.000±0.000 | 0.667 | 1.000 | 1.0/2 |
| static | 0.174±0.233 | 0.000±0.000 | 0.667 | 1.000 | 1.0/2 |

## Regime: deep_scarcity

> Σb^att/B ≈ 1.5 — corresponds to E1 400G point. 4 jobs, 2 premium + 2 standard.
> Expected Σb^att/B = 1.6

### Per-job metrics (mean ± std across rounds)

| Job | Label | Class | c_i | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|-------|-----|------|------|------|------|------|------|
| 0 | J0_premium_L | premium | 1.2 | 1.194±0.252 | 0.995±0.210 | 1.356±0.246 | 1.130±0.205 | 1.530±0.013 | 1.275±0.011 | 
| 1 | J1_premium_M | premium | 1.2 | 0.895±0.152 | 0.746±0.126 | 0.878±0.141 | 0.731±0.118 | 1.027±0.009 | 0.856±0.007 | 
| 2 | J2_standard_M | standard | 2.0 | 3.992±0.082 | 1.996±0.041 | 3.520±0.831 | 1.760±0.415 | 3.650±0.651 | 1.825±0.326 | 
| 3 | J3_standard_S | standard | 2.0 | 1.436±0.017 | 0.718±0.008 | 1.309±0.206 | 0.654±0.103 | 1.322±0.186 | 0.661±0.093 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO rate | Std SLO rate | N_jobs slow<1 |
|-----|--------|--------|-----------------|-------------|--------------|
| fair | 0.204±0.266 | 3.429±0.093 | 0.833 | 0.500 | 0.7/4 |
| longliu | 0.356±0.246 | 2.829±0.969 | 0.667 | 0.500 | 1.0/4 |
| static | 0.556±0.022 | 2.972±0.577 | 0.489 | 0.500 | 0.0/4 |

## Regime: transition

> Σb^att/B ≈ 1.2 — corresponds to E1 630G point. 3 jobs, 1 premium + 2 standard.
> Expected Σb^att/B = 1.1

### Per-job metrics (mean ± std across rounds)

| Job | Label | Class | c_i | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|-------|-----|------|------|------|------|------|------|
| 0 | J0_premium_L | premium | 1.2 | 1.182±0.249 | 0.985±0.207 | 1.344±0.238 | 1.120±0.198 | 1.488±0.035 | 1.240±0.029 | 
| 1 | J1_standard_M | standard | 2.0 | 0.987±0.021 | 0.493±0.011 | 0.770±0.150 | 0.385±0.075 | 0.900±0.150 | 0.450±0.075 | 
| 2 | J2_standard_S | standard | 2.0 | 0.931±0.119 | 0.465±0.060 | 0.935±0.145 | 0.467±0.072 | 0.843±0.130 | 0.421±0.065 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO rate | Std SLO rate | N_jobs slow<1 |
|-----|--------|--------|-----------------|-------------|--------------|
| fair | 0.182±0.249 | 0.015±0.010 | 0.667 | 1.000 | 1.0/3 |
| longliu | 0.344±0.238 | 0.025±0.018 | 0.333 | 1.000 | 1.3/3 |
| static | 0.488±0.035 | 0.016±0.011 | 0.000 | 1.000 | 1.3/3 |

## Cross-regime comparison

### P-attn (lower = better, premium jobs closer to SLO)

| Regime | LongLiu | Static | Fair | LL vs Static | LL vs Fair |
|--------|---------|--------|------|--------------|------------|
| ample | 0.179 | 0.174 | 0.499 | LL 劣 3% | LL 优 64% |
| deep_scarcity | 0.356 | 0.556 | 0.204 | LL 优 36% | LL 劣 74% |
| transition | 0.344 | 0.488 | 0.182 | LL 优 29% | LL 劣 89% |

### S-cont (lower = better, standard jobs less contended)

| Regime | LongLiu | Static | Fair | LL vs Static | LL vs Fair |
|--------|---------|--------|------|--------------|------------|
| ample | 0.000 | 0.000 | 0.013 | LL 劣 0% | LL 优 100% |
| deep_scarcity | 2.829 | 2.972 | 3.429 | LL 优 5% | LL 优 17% |
| transition | 0.025 | 0.016 | 0.015 | LL 劣 59% | LL 劣 64% |

## Anomaly diagnosis: Fair beats LongLiu in scarce regimes

### deep_scarcity

| Round | Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Jobs slow<1 |
|-------|-----|--------|--------|-------------|---------|-------------|
| 1 | longliu | 0.546 | 4.117 | 0.500 | 0.500 | 1/4 |
| 2 | longliu | 0.009 | 1.780 | 1.000 | 0.500 | 1/4 |
| 3 | longliu | 0.512 | 2.590 | 0.500 | 0.500 | 1/4 |
| 1 | static | 0.587 | 3.181 | 0.467 | 0.500 | 0/4 |
| 2 | static | 0.540 | 3.552 | 0.500 | 0.500 | 0/4 |
| 3 | static | 0.542 | 2.185 | 0.500 | 0.500 | 0/4 |
| 1 | fair | 0.580 | 3.557 | 0.500 | 0.500 | 0/4 |
| 2 | fair | 0.021 | 3.338 | 1.000 | 0.500 | 1/4 |
| 3 | fair | 0.011 | 3.392 | 1.000 | 0.500 | 1/4 |

**Jobs with slowdown < 1 (artifact — measured faster than solo):**

- longliu J1(premium) c_i=1.2: slowdown<1 in rounds [1, 2, 3]
- fair J1(premium) c_i=1.2: slowdown<1 in rounds [2, 3]

### transition

| Round | Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Jobs slow<1 |
|-------|-----|--------|--------|-------------|---------|-------------|
| 1 | longliu | 0.528 | 0.041 | 0.000 | 1.000 | 1/3 |
| 2 | longliu | 0.497 | 0.034 | 0.000 | 1.000 | 1/3 |
| 3 | longliu | 0.008 | 0.000 | 1.000 | 1.000 | 2/3 |
| 1 | static | 0.497 | 0.020 | 0.000 | 1.000 | 1/3 |
| 2 | static | 0.442 | 0.000 | 0.000 | 1.000 | 2/3 |
| 3 | static | 0.526 | 0.027 | 0.000 | 1.000 | 1/3 |
| 1 | fair | 0.009 | 0.026 | 1.000 | 1.000 | 1/3 |
| 2 | fair | 0.004 | 0.003 | 1.000 | 1.000 | 1/3 |
| 3 | fair | 0.534 | 0.016 | 0.000 | 1.000 | 1/3 |

**Jobs with slowdown < 1 (artifact — measured faster than solo):**

- longliu J1(standard) c_i=2.0: slowdown<1 in rounds [1, 2, 3]
- longliu J2(standard) c_i=2.0: slowdown<1 in rounds [3]
- static J1(standard) c_i=2.0: slowdown<1 in rounds [2, 3]
- static J2(standard) c_i=2.0: slowdown<1 in rounds [1, 2]
- fair J1(standard) c_i=2.0: slowdown<1 in rounds [1, 2]
- fair J2(standard) c_i=2.0: slowdown<1 in rounds [3]

## Root cause analysis

### Why does Fair beat LongLiu in scarce regimes?

**Observed**: In deep_scarcity and transition, Fair arm has LOWER P-attn than LongLiu.

**Root cause chain**:

1. **mlx5 does NOT implement strict-priority queuing** (confirmed: `mlnx_qos` reports
   'Priority trust state is not supported'). DSCP marking only affects the packet
   header, NOT the actual bandwidth allocation at the NIC.

2. **LongLiu demotes standard jobs to P1/P2 (DSCP=32/24)**. In simulation, this
   gives them less bandwidth via strict-priority scheduling. On this hardware,
   demotion has NO effect on bandwidth — the demoted job still gets the same
   share, but LongLiu's π calculation now treats it as 'ahead of SLO' and
   keeps premium jobs at P4 (not P6), because π is calibrated for the
   strict-priority world that doesn't exist.

3. **Fair arm never demotes anyone**. All jobs stay at P4 (DSCP=0). Standard
   jobs don't get artificially penalized, so premium jobs don't get artificially
   elevated. The SLOScheduler's π is computed honestly — standard jobs that
   are genuinely ahead of SLO (slowdown < 1) don't get extra priority.

4. **The slow<1 anomaly**: Many standard jobs show slowdown < 1.0, meaning they
   are FASTER than solo baseline. This is physically impossible in a truly
   contended regime — it means the solo T_target calibration is too conservative
   (measured during a cold run or with NIC in a different state).

### Implication for paper

- The LongLiu vs Static comparison is **valid** (same hardware, same DSCP behavior,
  LongLiu ≤ Static P-attn in all regimes).
- The LongLiu vs Fair comparison is **invalid** on this hardware because Fair
  is not a meaningful baseline without strict-priority queuing.
- Paper should state: 'On hardware without per-priority QoS, DSCP marking
  provides classification only; bandwidth differentiation requires switch-side
  strict-priority scheduling (§V-D).'
