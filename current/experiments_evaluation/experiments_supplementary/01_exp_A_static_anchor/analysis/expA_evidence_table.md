# Experiment A: Static Anchor — HW-vs-Sim Fidelity Evidence Table

> Generated: 2026-07-29T17:58:08+08:00
> Analysis window: epochs 5-19 (skip 5 warmup + 5 tail)
> Scenarios: 6 (hold-out: 1)

## Aggregate Fidelity (non-holdout, N=5)

- **Max relative error**: 38.50%
- **Mean relative error**: 32.63%
- Paper claim threshold: 0.2% — **FAIL**

## Evidence Table (§A.3)

| # | Capacity | c_i | slowdown_hw | slowdown_sim | rel.err | attain_hw | attain_sim | diff(pp) | holdout | anchor |
|---|----------|-----|-------------|--------------|---------|-----------|------------|----------|---------|--------|
| S1 | 50G | 1.2 | 1.0776 | 1.4147 | 31.28% | 0.8980 | 1.1790 | +28.09 |  | V5 measured solo BW (1024MB) |
| S2 | 50G | 1.5 | 0.8975 | 1.1318 | 26.10% | 0.5983 | 0.7545 | +15.62 |  | V5 measured solo BW (1024MB) |
| S3 | ~44G (50G - 6G bg) | 1.2 | 1.1698 | 1.6077 | 37.43% | 0.9749 | 1.3397 | +36.49 |  | V5 measured solo BW (1024MB) |
| S4 | ~44G (50G - 6G bg) | 1.5 | 0.9183 | 1.2719 | 38.50% | 0.6122 | 0.8479 | +23.57 |  | V5 measured solo BW (1024MB) |
| S5 | ~25G | 1.2 | 1.2708 | 1.6667 | 31.16% | 1.0590 | 1.3889 | +32.99 | † | Theoretical BW = 50G × 0.5 (half payload) — NOT measured solo |
| S6 | ~35G | 1.5 | 1.1928 | 1.5488 | 29.84% | 0.7952 | 1.0325 | +23.73 |  | ExpA measured solo BW (768MB) |

† = hold-out scenario (anchor from theoretical BW, not measured solo)

## Per-Scenario Breakdown

### S1 — 2job_50G_ci1.2

- Payload: 1024MB, c_i: 1.2, Capacity: 50G
- HW slowdown: A=1.0387, B=1.1166 → mean=1.0776
- Sim slowdown: A=1.3631, B=1.4664 → mean=1.4147
- Relative error: 31.28%
- Attainment: hw=0.8980, sim=1.1790, diff=+28.09pp

### S2 — 2job_50G_ci1.5

- Payload: 1024MB, c_i: 1.5, Capacity: 50G
- HW slowdown: A=0.8631, B=0.9319 → mean=0.8975
- Sim slowdown: A=1.0905, B=1.1731 → mean=1.1318
- Relative error: 26.10%
- Attainment: hw=0.5983, sim=0.7545, diff=+15.62pp

### S3 — 2job_bg_44G_ci1.2

- Payload: 1024MB, c_i: 1.2, Capacity: ~44G (50G - 6G bg)
- HW slowdown: A=1.1276, B=1.2121 → mean=1.1698
- Sim slowdown: A=1.5490, B=1.6664 → mean=1.6077
- Relative error: 37.43%
- Attainment: hw=0.9749, sim=1.3397, diff=+36.49pp

### S4 — 2job_bg_44G_ci1.5

- Payload: 1024MB, c_i: 1.5, Capacity: ~44G (50G - 6G bg)
- HW slowdown: A=0.8880, B=0.9486 → mean=0.9183
- Sim slowdown: A=1.2439, B=1.2998 → mean=1.2719
- Relative error: 38.50%
- Attainment: hw=0.6122, sim=0.8479, diff=+23.57pp

### S5 — 2job_25G_ci1.2_HOLDOUT

- Payload: 512MB, c_i: 1.2, Capacity: ~25G
- HW slowdown: A=1.2658, B=1.2758 → mean=1.2708
- Sim slowdown: A=1.6667, B=1.6667 → mean=1.6667
- Relative error: 31.16%
- Attainment: hw=1.0590, sim=1.3889, diff=+32.99pp

### S6 — 2job_35G_ci1.5

- Payload: 768MB, c_i: 1.5, Capacity: ~35G
- HW slowdown: A=1.1777, B=1.2078 → mean=1.1928
- Sim slowdown: A=1.5304, B=1.5671 → mean=1.5488
- Relative error: 29.84%
- Attainment: hw=0.7952, sim=1.0325, diff=+23.73pp

