# Experiment C v2: Iteration-Level Slowdown Analysis

> Generated: 2026-07-30T16:26:07.527506
> Data dir: ../data_v2
> Analysis window: epochs 5–194
> Total runs: 75

## Metric Definitions (v2)

- **Slowdown (iter-level)**: s = (Tcomp + comm) / (Tcomp + Tcomm_solo)
  - This aligns with the paper's c_i definition (total iteration time)
  - v1 used comm-only: comm / Tcomm_solo (incorrect per v2 spec)
- **Attainment**: att = s / c_eval (≤1 = SLO met, >1 = SLO violated)
- **c_policy / c_eval split**: scheduler uses c_policy=1.35 for π; attainment uses c_eval=1.5
- **P-attn** = Σ_premium max(0, s-1) — premium attention needed (lower=better)
- **S-cont** = Σ_standard max(0, s-1) — standard contention (lower=better)
- **Max slowdown** = max across all jobs (key for S2 bounded degradation)

## Regime: S1_ample

> Scenario: S1 | D scale: ×0.6
> Light load — everyone meets SLO, verify 'LongLiu at no cost'

### Per-job metrics (mean ± std across rounds)

| Job | Label | Tier | c_policy | c_eval | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|------|----------|--------|------|------|------|------|------|------|
| 0 | J0_P_L | premium | 1.35 | 1.5 | 1.083±0.047 | 0.722±0.031 | 1.020±0.050 | 0.680±0.033 | 1.104±0.035 | 0.736±0.023 | 
| 1 | J1_P_M | premium | 1.35 | 1.5 | 1.282±0.077 | 0.855±0.051 | 1.407±0.129 | 0.938±0.086 | 1.192±0.025 | 0.795±0.017 | 
| 2 | J2_S_M | standard | 2.0 | 2.0 | 1.111±0.052 | 0.555±0.026 | 0.956±0.031 | 0.478±0.015 | 1.019±0.011 | 0.510±0.006 | 
| 3 | J3_S_S | standard | 2.0 | 2.0 | 1.089±0.038 | 0.545±0.019 | 1.086±0.178 | 0.543±0.089 | 1.010±0.021 | 0.505±0.011 | 
| 4 | J4_S_XS | standard | 2.0 | 2.0 | 1.097±0.036 | 0.549±0.018 | 1.224±0.068 | 0.612±0.034 | 1.008±0.024 | 0.504±0.012 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Max SD | N slow<1 |
|-----|--------|--------|-------------|---------|--------|----------|
| fair | 0.365±0.121 | 0.298±0.125 | 0.974 | 1.000 | 1.28 | 0.0/5 |
| longliu | 0.440±0.150 | 0.335±0.233 | 0.837 | 1.000 | 1.46 | 1.8/5 |
| static | 0.296±0.055 | 0.050±0.043 | 0.999 | 1.000 | 1.19 | 1.0/5 |

## Regime: S1_deep

> Scenario: S1 | D scale: ×1.3
> Fair premium clearly violates SLO

### Per-job metrics (mean ± std across rounds)

| Job | Label | Tier | c_policy | c_eval | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|------|----------|--------|------|------|------|------|------|------|
| 0 | J0_P_L | premium | 1.35 | 1.5 | 1.237±0.052 | 0.825±0.034 | 1.289±0.119 | 0.859±0.080 | 1.288±0.055 | 0.859±0.037 | 
| 1 | J1_P_M | premium | 1.35 | 1.5 | 1.336±0.021 | 0.890±0.014 | 1.281±0.152 | 0.854±0.101 | 1.330±0.059 | 0.887±0.039 | 
| 2 | J2_S_M | standard | 2.0 | 2.0 | 1.140±0.021 | 0.570±0.011 | 0.983±0.019 | 0.492±0.010 | 1.007±0.022 | 0.503±0.011 | 
| 3 | J3_S_S | standard | 2.0 | 2.0 | 1.128±0.012 | 0.564±0.006 | 1.041±0.117 | 0.521±0.058 | 1.000±0.020 | 0.500±0.010 | 
| 4 | J4_S_XS | standard | 2.0 | 2.0 | 1.097±0.005 | 0.549±0.003 | 1.299±0.139 | 0.650±0.070 | 1.019±0.005 | 0.509±0.003 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Max SD | N slow<1 |
|-----|--------|--------|-------------|---------|--------|----------|
| fair | 0.572±0.071 | 0.365±0.033 | 0.998 | 1.000 | 1.34 | 0.0/5 |
| longliu | 0.570±0.145 | 0.355±0.250 | 0.781 | 1.000 | 1.41 | 1.6/5 |
| static | 0.618±0.104 | 0.040±0.019 | 0.876 | 1.000 | 1.34 | 1.0/5 |

## Regime: S1_moderate

> Scenario: S1 | D scale: ×1.0
> Fair starts hurting premium — the ladder begins

### Per-job metrics (mean ± std across rounds)

| Job | Label | Tier | c_policy | c_eval | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|------|----------|--------|------|------|------|------|------|------|
| 0 | J0_P_L | premium | 1.35 | 1.5 | 1.056±0.031 | 0.704±0.021 | 0.938±0.024 | 0.625±0.016 | 1.075±0.037 | 0.717±0.025 | 
| 1 | J1_P_M | premium | 1.35 | 1.5 | 1.299±0.060 | 0.866±0.040 | 1.273±0.222 | 0.849±0.148 | 1.361±0.049 | 0.907±0.033 | 
| 2 | J2_S_M | standard | 2.0 | 2.0 | 1.308±0.066 | 0.654±0.033 | 1.345±0.208 | 0.672±0.104 | 1.124±0.036 | 0.562±0.018 | 
| 3 | J3_S_S | standard | 2.0 | 2.0 | 1.288±0.047 | 0.644±0.024 | 1.376±0.159 | 0.688±0.079 | 1.103±0.018 | 0.551±0.009 | 
| 4 | J4_S_XS | standard | 2.0 | 2.0 | 1.254±0.042 | 0.627±0.021 | 1.397±0.263 | 0.699±0.132 | 1.097±0.018 | 0.548±0.009 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Max SD | N slow<1 |
|-----|--------|--------|-------------|---------|--------|----------|
| fair | 0.355±0.081 | 0.851±0.151 | 0.972 | 1.000 | 1.33 | 0.0/5 |
| longliu | 0.279±0.214 | 1.118±0.547 | 0.873 | 0.979 | 1.54 | 1.2/5 |
| static | 0.436±0.082 | 0.324±0.059 | 0.824 | 0.999 | 1.36 | 0.0/5 |

## Regime: S1_very_deep

> Scenario: S1 | D scale: ×2.0
> Only regime with Σb̄>BW_eff — tests policy/eval margin

### Per-job metrics (mean ± std across rounds)

| Job | Label | Tier | c_policy | c_eval | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|------|----------|--------|------|------|------|------|------|------|
| 0 | J0_P_L | premium | 1.35 | 1.5 | 1.102±0.032 | 0.734±0.021 | 0.992±0.025 | 0.661±0.017 | 1.132±0.049 | 0.755±0.032 | 
| 1 | J1_P_M | premium | 1.35 | 1.5 | 1.114±0.042 | 0.743±0.028 | 1.264±0.051 | 0.842±0.034 | 1.167±0.040 | 0.778±0.027 | 
| 2 | J2_S_M | standard | 2.0 | 2.0 | 1.132±0.020 | 0.566±0.010 | 0.959±0.035 | 0.480±0.017 | 0.999±0.023 | 0.499±0.011 | 
| 3 | J3_S_S | standard | 2.0 | 2.0 | 1.274±0.022 | 0.637±0.011 | 1.493±0.046 | 0.747±0.023 | 1.100±0.026 | 0.550±0.013 | 
| 4 | J4_S_XS | standard | 2.0 | 2.0 | 1.216±0.017 | 0.608±0.009 | 1.561±0.033 | 0.780±0.017 | 1.107±0.022 | 0.554±0.011 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Max SD | N slow<1 |
|-----|--------|--------|-------------|---------|--------|----------|
| fair | 0.216±0.069 | 0.622±0.048 | 1.000 | 1.000 | 1.27 | 0.0/5 |
| longliu | 0.271±0.057 | 0.944±0.215 | 0.998 | 0.999 | 1.55 | 1.2/5 |
| static | 0.299±0.082 | 0.218±0.044 | 0.988 | 1.000 | 1.17 | 0.4/5 |

## Regime: S2_starvation

> Scenario: S2 | D scale: ×1.0
> No arm can save premium — compare bounded vs unbounded degradation

### Per-job metrics (mean ± std across rounds)

| Job | Label | Tier | c_policy | c_eval | fair SD | fair Att | longliu SD | longliu Att | static SD | static Att | 
|-----|-------|------|----------|--------|------|------|------|------|------|------|
| 0 | J0_P_H0 | premium | 1.35 | 1.5 | 2.272±0.024 | 1.515±0.016 | 2.721±0.888 | 1.814±0.592 | 2.944±0.036 | 1.963±0.024 | 
| 1 | J1_P_H1 | premium | 1.35 | 1.5 | 2.286±0.011 | 1.524±0.007 | 3.006±1.138 | 2.004±0.758 | 3.005±0.023 | 2.003±0.016 | 
| 2 | J2_P_H2 | premium | 1.35 | 1.5 | 1.679±0.017 | 1.119±0.011 | 1.428±0.457 | 0.952±0.305 | 1.955±0.011 | 1.303±0.007 | 
| 3 | J3_S_M | standard | 2.0 | 2.0 | 1.298±0.003 | 0.649±0.001 | 1.142±0.199 | 0.571±0.100 | 0.987±0.004 | 0.493±0.002 | 
| 4 | J4_S_L | standard | 2.0 | 2.0 | 1.297±0.005 | 0.649±0.003 | 1.084±0.144 | 0.542±0.072 | 0.989±0.006 | 0.494±0.003 | 
| 5 | J5_S_XL | standard | 2.0 | 2.0 | 1.268±0.003 | 0.634±0.001 | 1.159±0.140 | 0.580±0.070 | 0.986±0.011 | 0.493±0.005 | 

### Aggregate metrics (mean ± std across rounds)

| Arm | P-attn | S-cont | Premium SLO% | Std SLO% | Max SD | N slow<1 |
|-----|--------|--------|-------------|---------|--------|----------|
| fair | 3.238±0.021 | 0.864±0.007 | 0.085 | 1.000 | 2.29 | 0.0/6 |
| longliu | 4.186±1.389 | 0.410±0.230 | 0.441 | 0.956 | 3.37 | 1.6/6 |
| static | 4.903±0.067 | 0.002±0.003 | 0.106 | 1.000 | 3.00 | 2.6/6 |

## Cross-regime comparison

### P-attn (lower = better, premium jobs closer to SLO)

| Regime | LongLiu | Static | Fair | LL vs Static | LL vs Fair |
|--------|---------|--------|------|--------------|------------|
| S1_ample | 0.440 | 0.296 | 0.365 | LL 劣 49% | LL 劣 21% |
| S1_deep | 0.570 | 0.618 | 0.572 | LL 优 8% | LL 优 0% |
| S1_moderate | 0.279 | 0.436 | 0.355 | LL 优 36% | LL 优 21% |
| S1_very_deep | 0.271 | 0.299 | 0.216 | LL 优 9% | LL 劣 25% |
| S2_starvation | 4.186 | 4.903 | 3.238 | LL 优 15% | LL 劣 29% |

### S-cont (lower = better, standard jobs less contended)

| Regime | LongLiu | Static | Fair | LL vs Static | LL vs Fair |
|--------|---------|--------|------|--------------|------------|
| S1_ample | 0.335 | 0.050 | 0.298 | LL 劣 569% | LL 劣 13% |
| S1_deep | 0.355 | 0.040 | 0.365 | LL 劣 785% | LL 优 3% |
| S1_moderate | 1.118 | 0.324 | 0.851 | LL 劣 245% | LL 劣 31% |
| S1_very_deep | 0.944 | 0.218 | 0.622 | LL 劣 333% | LL 劣 52% |
| S2_starvation | 0.410 | 0.002 | 0.864 | LL 劣 27059% | LL 优 53% |

### Max slowdown across jobs (lower = better, S2 key metric)

| Regime | LongLiu | Static | Fair | Narrative |
|--------|---------|--------|------|------------|
| S1_ample | 1.46 | 1.19 | 1.28 | — |
| S1_deep | 1.41 | 1.34 | 1.34 | — |
| S1_moderate | 1.54 | 1.36 | 1.33 | — |
| S1_very_deep | 1.55 | 1.17 | 1.27 | — |
| S2_starvation | 3.37 | 3.00 | 2.29 | — |

## v1 (comm-only) vs v2 (iter-level) Slowdown Comparison

| Regime | Arm | Job | Tier | v1 SD | v2 SD | v2/v1 ratio |
|--------|-----|-----|------|-------|-------|-------------|
| S1_ample | fair | 0(J0_P_L) | premium | 1.310 | 1.083 | 0.83 |
| S1_ample | fair | 1(J1_P_M) | premium | 2.057 | 1.282 | 0.62 |
| S1_ample | fair | 2(J2_S_M) | standard | 1.476 | 1.111 | 0.75 |
| S1_ample | fair | 3(J3_S_S) | standard | 1.487 | 1.089 | 0.73 |
| S1_ample | fair | 4(J4_S_XS) | standard | 1.649 | 1.097 | 0.67 |
| S1_ample | longliu | 0(J0_P_L) | premium | 1.076 | 1.020 | 0.95 |
| S1_ample | longliu | 1(J1_P_M) | premium | 2.524 | 1.407 | 0.56 |
| S1_ample | longliu | 2(J2_S_M) | standard | 0.811 | 0.956 | 1.18 |
| S1_ample | longliu | 3(J3_S_S) | standard | 1.469 | 1.086 | 0.74 |
| S1_ample | longliu | 4(J4_S_XS) | standard | 2.495 | 1.224 | 0.49 |
| S1_ample | static | 0(J0_P_L) | premium | 1.388 | 1.104 | 0.79 |
| S1_ample | static | 1(J1_P_M) | premium | 1.719 | 1.192 | 0.69 |
| S1_ample | static | 2(J2_S_M) | standard | 1.083 | 1.019 | 0.94 |
| S1_ample | static | 3(J3_S_S) | standard | 1.056 | 1.010 | 0.96 |
| S1_ample | static | 4(J4_S_XS) | standard | 1.051 | 1.008 | 0.96 |
| S1_deep | fair | 0(J0_P_L) | premium | 1.887 | 1.237 | 0.66 |
| S1_deep | fair | 1(J1_P_M) | premium | 2.257 | 1.336 | 0.59 |
| S1_deep | fair | 2(J2_S_M) | standard | 1.600 | 1.140 | 0.71 |
| S1_deep | fair | 3(J3_S_S) | standard | 1.699 | 1.128 | 0.66 |
| S1_deep | fair | 4(J4_S_XS) | standard | 1.650 | 1.097 | 0.67 |
| S1_deep | longliu | 0(J0_P_L) | premium | 2.082 | 1.289 | 0.62 |
| S1_deep | longliu | 1(J1_P_M) | premium | 2.053 | 1.281 | 0.62 |
| S1_deep | longliu | 2(J2_S_M) | standard | 0.927 | 0.983 | 1.06 |
| S1_deep | longliu | 3(J3_S_S) | standard | 1.224 | 1.041 | 0.85 |
| S1_deep | longliu | 4(J4_S_XS) | standard | 2.994 | 1.299 | 0.43 |
| S1_deep | static | 0(J0_P_L) | premium | 2.080 | 1.288 | 0.62 |
| S1_deep | static | 1(J1_P_M) | premium | 2.236 | 1.330 | 0.59 |
| S1_deep | static | 2(J2_S_M) | standard | 1.028 | 1.007 | 0.98 |
| S1_deep | static | 3(J3_S_S) | standard | 1.002 | 1.000 | 1.00 |
| S1_deep | static | 4(J4_S_XS) | standard | 1.126 | 1.019 | 0.90 |
| S1_moderate | fair | 0(J0_P_L) | premium | 1.211 | 1.056 | 0.87 |
| S1_moderate | fair | 1(J1_P_M) | premium | 2.121 | 1.299 | 0.61 |
| S1_moderate | fair | 2(J2_S_M) | standard | 2.323 | 1.308 | 0.56 |
| S1_moderate | fair | 3(J3_S_S) | standard | 2.576 | 1.288 | 0.50 |
| S1_moderate | fair | 4(J4_S_XS) | standard | 2.696 | 1.254 | 0.47 |
| S1_moderate | longliu | 0(J0_P_L) | premium | 0.768 | 0.938 | 1.22 |
| S1_moderate | longliu | 1(J1_P_M) | premium | 2.022 | 1.273 | 0.63 |
| S1_moderate | longliu | 2(J2_S_M) | standard | 2.480 | 1.345 | 0.54 |
| S1_moderate | longliu | 3(J3_S_S) | standard | 3.056 | 1.376 | 0.45 |
| S1_moderate | longliu | 4(J4_S_XS) | standard | 3.647 | 1.397 | 0.38 |
| S1_moderate | static | 0(J0_P_L) | premium | 1.281 | 1.075 | 0.84 |
| S1_moderate | static | 1(J1_P_M) | premium | 2.353 | 1.361 | 0.58 |
| S1_moderate | static | 2(J2_S_M) | standard | 1.533 | 1.124 | 0.73 |
| S1_moderate | static | 3(J3_S_S) | standard | 1.561 | 1.103 | 0.71 |
| S1_moderate | static | 4(J4_S_XS) | standard | 1.646 | 1.097 | 0.67 |
| S1_very_deep | fair | 0(J0_P_L) | premium | 1.381 | 1.102 | 0.80 |
| S1_very_deep | fair | 1(J1_P_M) | premium | 1.428 | 1.114 | 0.78 |
| S1_very_deep | fair | 2(J2_S_M) | standard | 1.568 | 1.132 | 0.72 |
| S1_very_deep | fair | 3(J3_S_S) | standard | 2.496 | 1.274 | 0.51 |
| S1_very_deep | fair | 4(J4_S_XS) | standard | 2.437 | 1.216 | 0.50 |
| S1_very_deep | longliu | 0(J0_P_L) | premium | 0.970 | 0.992 | 1.02 |
| S1_very_deep | longliu | 1(J1_P_M) | premium | 1.988 | 1.264 | 0.64 |
| S1_very_deep | longliu | 2(J2_S_M) | standard | 0.825 | 0.959 | 1.16 |
| S1_very_deep | longliu | 3(J3_S_S) | standard | 3.695 | 1.493 | 0.40 |
| S1_very_deep | longliu | 4(J4_S_XS) | standard | 4.739 | 1.561 | 0.33 |
| S1_very_deep | static | 0(J0_P_L) | premium | 1.494 | 1.132 | 0.76 |
| S1_very_deep | static | 1(J1_P_M) | premium | 1.624 | 1.167 | 0.72 |
| S1_very_deep | static | 2(J2_S_M) | standard | 0.995 | 0.999 | 1.00 |
| S1_very_deep | static | 3(J3_S_S) | standard | 1.549 | 1.100 | 0.71 |
| S1_very_deep | static | 4(J4_S_XS) | standard | 1.716 | 1.107 | 0.65 |
| S2_starvation | fair | 0(J0_P_H0) | premium | 3.288 | 2.272 | 0.69 |
| S2_starvation | fair | 1(J1_P_H1) | premium | 3.314 | 2.286 | 0.69 |
| S2_starvation | fair | 2(J2_P_H2) | premium | 2.222 | 1.679 | 0.76 |
| S2_starvation | fair | 3(J3_S_M) | standard | 2.372 | 1.298 | 0.55 |
| S2_starvation | fair | 4(J4_S_L) | standard | 2.370 | 1.297 | 0.55 |
| S2_starvation | fair | 5(J5_S_XL) | standard | 2.237 | 1.268 | 0.57 |
| S2_starvation | longliu | 0(J0_P_H0) | premium | 4.096 | 2.721 | 0.66 |
| S2_starvation | longliu | 1(J1_P_H1) | premium | 4.608 | 3.006 | 0.65 |
| S2_starvation | longliu | 2(J2_P_H2) | premium | 1.770 | 1.428 | 0.81 |
| S2_starvation | longliu | 3(J3_S_M) | standard | 1.652 | 1.142 | 0.69 |
| S2_starvation | longliu | 4(J4_S_L) | standard | 1.385 | 1.084 | 0.78 |
| S2_starvation | longliu | 5(J5_S_XL) | standard | 1.733 | 1.159 | 0.67 |
| S2_starvation | static | 0(J0_P_H0) | premium | 4.497 | 2.944 | 0.65 |
| S2_starvation | static | 1(J1_P_H1) | premium | 4.606 | 3.005 | 0.65 |
| S2_starvation | static | 2(J2_P_H2) | premium | 2.717 | 1.955 | 0.72 |
| S2_starvation | static | 3(J3_S_M) | standard | 0.940 | 0.987 | 1.05 |
| S2_starvation | static | 4(J4_S_L) | standard | 0.949 | 0.989 | 1.04 |
| S2_starvation | static | 5(J5_S_XL) | standard | 0.934 | 0.986 | 1.06 |

## Iron Rule Verification

| Regime | Rule 1: Real contention? | Rule 2: Fair must fail? | Rule 3: LL passes? |
|--------|--------------------------|--------------------------|----------------------|
| S1_ample | ✓ | ✗ | ✓ (LL P att=0.81) |
| S1_deep | ✓ | ✗ | ✓ (LL P att=0.86) |
| S1_moderate | ✓ | ✗ | ✓ (LL P att=0.74) |
| S1_very_deep | ✓ | ✗ | ✓ (LL P att=0.75) |
| S2_starvation | ✓ | ✓ (fair P SD=2.08) | ✗ (att=1.59) |
