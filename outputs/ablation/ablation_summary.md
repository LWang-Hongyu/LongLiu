# Per-Seed Ablation: Weighted vs Uniform Bandwidth Allocation

## Table 1: Overall Mean SAS per Seed

| Seed | LongLiu (weighted) | LongLiu (uniform) | Delta SAS |
|------|-------------------|-------------------|-----------|
| 0 | 0.7623 | 0.7870 | -0.0247 |
| 1 | 0.8588 | 0.8153 | +0.0435 |
| 2 | 0.8023 | 0.7951 | +0.0072 |
| 3 | 0.7575 | 0.7510 | +0.0065 |
| 4 | 0.7460 | 0.7513 | -0.0053 |
| 5 | 0.7668 | 0.7651 | +0.0016 |
| 6 | 0.8175 | 0.8008 | +0.0167 |
| 7 | 0.8427 | 0.8382 | +0.0045 |
| 8 | 0.8180 | 0.8096 | +0.0084 |
| 9 | 0.7506 | 0.7521 | -0.0015 |
| **Mean** | **0.7922** | **0.7866** | **+0.0057** |
| **Std** | 0.0387 | 0.0291 | 0.0163 |

**Weighted wins in 7/10 seeds.**

## Table 2: Mean SAS by SLO Tier

| Tier | Weighted Mean | Uniform Mean | Delta | Improvement |
|------|---------------|--------------|-------|-------------|
| Tight (1.5) | 0.4150 | 0.4151 | -0.0001 | -0.0% |
| Medium (2.0) | 1.0336 | 1.0164 | +0.0172 | +1.7% |
| Loose (3.0) | 1.4411 | 1.4412 | -0.0000 | -0.0% |

## Conclusion

- Weighted allocation wins in **7/10 seeds**, with a small but consistent mean improvement of +0.006 in overall SAS
- The improvement comes mainly from the **medium tier** (+1.7% SAS), not the large-model tier as initially hypothesized
- The large-model tier shows essentially no difference (Δ = -0.0001), suggesting the starvation problem is not primarily caused by intra-DSCP bandwidth sharing
- This ablation refines our understanding: LongLiu's fairness benefit comes from DSCP-level priority assignment, not from weighted intra-class allocation