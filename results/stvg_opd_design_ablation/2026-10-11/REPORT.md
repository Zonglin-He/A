# Component Ablation Study

Absolute parent-macro m_vIoU (%). Cross-domain: both fixed orders; same-domain: five 5% physical corruption families.

| Configuration | Cross HC2 | Cross VidSTG | Corrupt HC2 | Corrupt VidSTG |
|---|---:|---:|---:|---:|
| Full method | 23.502 | 18.337 | 33.024 | 25.769 |
| w/o Event-Scoped Sampling | 22.549 | 17.984 | 32.056 | 25.930 |
| w/o Action-Reward Correspondence | 19.665 | 9.516 | 26.704 | 15.547 |
| w/o On-Policy Refresh | 22.378 | 14.477 | 29.667 | 23.190 |
| w/o Query Residual | 23.672 | 17.873 | 33.015 | 25.349 |
| w/o LayerNorm Adaptation | 22.320 | 15.015 | 29.082 | 24.331 |
| w/o Slow-State Consolidation | 23.264 | 18.079 | 32.807 | 25.289 |
| w/o Fast-State Reset | 23.457 | 17.883 | 32.594 | 25.668 |
| w/o Slow-State Attenuation | 22.853 | 17.325 | 32.411 | 23.716 |
| Frozen source | 22.156 | 14.466 | 29.094 | 23.594 |

## Paired Full-minus-control differences

| Intervention | Setting | Difference (pp) | Paired 95% CI |
|---|---|---:|---|
| w/o Event-Scoped Sampling | A_hc2_cross_clean | +0.953 | [-0.228, +2.257] |
| w/o Event-Scoped Sampling | A_vidstg_cross_clean | +0.352 | [-0.389, +1.058] |
| w/o Event-Scoped Sampling | A_hc2_same_5percent | +0.968 | [-0.295, +2.360] |
| w/o Event-Scoped Sampling | A_vidstg_same_5percent | -0.161 | [-1.394, +1.058] |
| w/o Fast-State Reset | A_hc2_cross_clean | +0.045 | [-0.345, +0.400] |
| w/o Fast-State Reset | A_vidstg_cross_clean | +0.453 | [+0.067, +0.902] |
| w/o Fast-State Reset | A_hc2_same_5percent | +0.430 | [-0.183, +1.114] |
| w/o Fast-State Reset | A_vidstg_same_5percent | +0.101 | [-0.245, +0.451] |
| w/o Slow-State Attenuation | A_hc2_cross_clean | +0.649 | [+0.064, +1.283] |
| w/o Slow-State Attenuation | A_vidstg_cross_clean | +1.011 | [+0.237, +1.821] |
| w/o Slow-State Attenuation | A_hc2_same_5percent | +0.613 | [+0.003, +1.238] |
| w/o Slow-State Attenuation | A_vidstg_same_5percent | +2.052 | [+1.038, +3.233] |

All 5,376 new formal fits and all deployment conditions sealed before the new GT read. Original P2/P3 rows and complete Full streams are reused, not rerun.
Coverage grouping is offline and descriptive; bins and bootstrap were fixed before new predictions. Confidence intervals are conditional on historical exposed parent sources, with no multiplicity adjustment.
Cases without an annotated reference frame have missing expert quality, not a zero score. State-arm expert inference is reused; recorded fit costs are not full cold-start latency.
All per-condition/order results, negative tails, parent effects, costs and expert counts remain in the machine-readable companion.
Actual root plot review and anonymous remote publication are still required; this CPU completion is not final closure.

Historical table readback corrected two display values: shuffled-feedback Cross VidSTG 9.517 → 9.516, and frozen-source Corrupt HC2 29.093 → 29.094. All original predictions, per-query rows and underlying scores remain unchanged.
