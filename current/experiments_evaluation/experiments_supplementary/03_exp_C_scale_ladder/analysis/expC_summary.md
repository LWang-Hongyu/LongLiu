# Experiment C: Scale/Scarcity Ladder — Hardware vs LongLiu Analysis

> Generated: 2026-07-30T03:29:33+08:00
> Analysis window: epochs 5-19
> Runs analyzed: 27

## Methodology

- **Slowdown** = `avg_comm_contended / T_target_solo` (per-epoch, mean over window)
- **Attainment** = `slowdown / c_i` (>1 means missing SLO)
- **P-attn** = `Σ_premium max(0, slowdown - 1)` (lower is better)
- 3 regimes × 3 arms × N rounds; mean±std across rounds

## Regime: ample

> Σb^att/B ≈ 0.96 — corresponds to E1 1200G point. 2 jobs, 1 premium + 1 standard.
> Expected Σb^att/B = 0.9

### Per-job slowdown (mean ± std across rounds)

| Job | Label | Class | c_i | fair | longliu | static |
|-----|-------|-------|-----|-----|-----|-----|
| 0 | J0_premium_L | premium | 1.2 | 1.499±0.040 | 1.179±0.236 | 1.174±0.233 |
| 1 | J1_standard_M | standard | 1.5 | 1.013±0.001 | 0.772±0.152 | 0.682±0.001 |
| **P-attn** | (premium attention) | metric | — | **0.499±0.040** | **0.179±0.236** | **0.174±0.233** |

### LongLiu arm — final DSCP per job (mode of last 5 epochs)

| Job | Label | Class | Final DSCP (per round) |
|-----|-------|-------|------------------------|
| 0 | J0_premium_L | premium | 24, 24, 0 |
| 1 | J1_standard_M | standard | 32, 24, 32 |

## Regime: deep_scarcity

> Σb^att/B ≈ 1.5 — corresponds to E1 400G point. 4 jobs, 2 premium + 2 standard.
> Expected Σb^att/B = 1.6

### Per-job slowdown (mean ± std across rounds)

| Job | Label | Class | c_i | fair | longliu | static |
|-----|-------|-------|-----|-----|-----|-----|
| 0 | J0_premium_L | premium | 1.2 | 1.194±0.252 | 1.356±0.246 | 1.530±0.013 |
| 1 | J1_premium_M | premium | 1.2 | 0.895±0.152 | 0.878±0.141 | 1.027±0.009 |
| 2 | J2_standard_M | standard | 2.0 | 3.992±0.082 | 3.520±0.831 | 3.650±0.651 |
| 3 | J3_standard_S | standard | 2.0 | 1.436±0.017 | 1.309±0.206 | 1.322±0.186 |
| **P-attn** | (premium attention) | metric | — | **0.204±0.266** | **0.356±0.246** | **0.556±0.022** |

### LongLiu arm — final DSCP per job (mode of last 5 epochs)

| Job | Label | Class | Final DSCP (per round) |
|-----|-------|-------|------------------------|
| 0 | J0_premium_L | premium | 0, 24, 0 |
| 1 | J1_premium_M | premium | 24, 24, 24 |
| 2 | J2_standard_M | standard | 8, 8, 8 |
| 3 | J3_standard_S | standard | 24, 24, 24 |

## Regime: transition

> Σb^att/B ≈ 1.2 — corresponds to E1 630G point. 3 jobs, 1 premium + 2 standard.
> Expected Σb^att/B = 1.1

### Per-job slowdown (mean ± std across rounds)

| Job | Label | Class | c_i | fair | longliu | static |
|-----|-------|-------|-----|-----|-----|-----|
| 0 | J0_premium_L | premium | 1.2 | 1.182±0.249 | 1.344±0.238 | 1.488±0.035 |
| 1 | J1_standard_M | standard | 2.0 | 0.987±0.021 | 0.770±0.150 | 0.900±0.150 |
| 2 | J2_standard_S | standard | 2.0 | 0.931±0.119 | 0.935±0.145 | 0.843±0.130 |
| **P-attn** | (premium attention) | metric | — | **0.182±0.249** | **0.344±0.238** | **0.488±0.035** |

### LongLiu arm — final DSCP per job (mode of last 5 epochs)

| Job | Label | Class | Final DSCP (per round) |
|-----|-------|-------|------------------------|
| 0 | J0_premium_L | premium | 0, 0, 24 |
| 1 | J1_standard_M | standard | 32, 32, 32 |
| 2 | J2_standard_S | standard | 24, 24, 32 |

## Cross-regime comparison

P-attn (lower = better, premium jobs closer to SLO):

| Regime | LongLiu | Static | Fair |
|--------|---------|--------|------|
| ample | 0.179 | 0.174 | 0.499 |
| deep_scarcity | 0.356 | 0.556 | 0.204 |
| transition | 0.344 | 0.488 | 0.182 |

## Key findings

1. Multi-QP DSCP switching verified working (4 pre-created QPs per job, switched at iter granularity)
2. See per-regime tables above for slowdown and P-attn comparison
3. Compare regime ranking vs E1 simulation ladder (deep_scarcity > transition > ample for P-attn differentiation)
