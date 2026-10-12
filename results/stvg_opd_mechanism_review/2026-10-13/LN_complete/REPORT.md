# Original LN-only: complete four-arm, three-order result

All 768 new LN-only fits were sealed across both targets before GT. All 2,304 existing Full/reset/full-carry controls are reused exactly, with the same source, pixels, frames, Native event, detector evidence and fixed configuration. Each target has 128 historical parents and three actual full-stream permutations. The three orders are repeated outcomes of the same parents; they do not create 768 independent parents.

## Direct final-output contrasts

Parent averages retain all three actual orders. The 10,000 paired parent bootstrap uses the locked seed 20261011; intervals are descriptive and unadjusted. Values are percentage points.

| Target | Order | Contrast | Difference (pp) | 95% CI | Negative parents |
|---|---|---|---:|---|---:|
| hc2 | pooled | Full - alpha0 | +0.129 | [-0.092, +0.365] | 54 / 128 |
| hc2 | pooled | Full - alpha1 | +0.623 | [+0.155, +1.141] | 46 / 128 |
| hc2 | pooled | Full - LN_only | -0.169 | [-0.679, +0.246] | 57 / 128 |
| hc2 | order3 | Full - alpha0 | +0.147 | [-0.099, +0.421] | 47 / 128 |
| hc2 | order3 | Full - alpha1 | +1.661 | [+0.667, +2.810] | 42 / 128 |
| hc2 | order3 | Full - LN_only | -0.232 | [-0.739, +0.208] | 58 / 128 |
| hc2 | order4 | Full - alpha0 | +0.063 | [-0.280, +0.371] | 50 / 128 |
| hc2 | order4 | Full - alpha1 | +0.265 | [-0.302, +0.876] | 44 / 128 |
| hc2 | order4 | Full - LN_only | -0.207 | [-0.640, +0.153] | 53 / 128 |
| hc2 | order5 | Full - alpha0 | +0.178 | [-0.136, +0.511] | 48 / 128 |
| hc2 | order5 | Full - alpha1 | -0.057 | [-0.545, +0.416] | 51 / 128 |
| hc2 | order5 | Full - LN_only | -0.069 | [-0.751, +0.522] | 57 / 128 |
| vidstg | pooled | Full - alpha0 | +0.300 | [-0.119, +0.705] | 45 / 128 |
| vidstg | pooled | Full - alpha1 | +1.212 | [+0.580, +1.865] | 32 / 128 |
| vidstg | pooled | Full - LN_only | +0.464 | [+0.098, +0.922] | 39 / 128 |
| vidstg | order3 | Full - alpha0 | +0.508 | [+0.018, +1.014] | 42 / 128 |
| vidstg | order3 | Full - alpha1 | +0.938 | [+0.207, +1.705] | 36 / 128 |
| vidstg | order3 | Full - LN_only | +0.627 | [+0.164, +1.209] | 41 / 128 |
| vidstg | order4 | Full - alpha0 | +0.001 | [-0.588, +0.530] | 48 / 128 |
| vidstg | order4 | Full - alpha1 | +1.452 | [+0.760, +2.237] | 29 / 128 |
| vidstg | order4 | Full - LN_only | +0.211 | [-0.221, +0.630] | 44 / 128 |
| vidstg | order5 | Full - alpha0 | +0.392 | [-0.085, +0.882] | 49 / 128 |
| vidstg | order5 | Full - alpha1 | +1.247 | [+0.198, +2.265] | 37 / 128 |
| vidstg | order5 | Full - LN_only | +0.554 | [+0.112, +1.105] | 41 / 128 |

## Absolute outcomes and current-update negative tails

A current-update difference is official vIoU After minus Before. A negative parent counts any negative mean across the three orders; the >5 pp and >20 pp columns separately identify substantive harm. Before minus Frozen is a descriptive entering-state readout, not a direct Full-minus-reset comparison.

| Target | Arm | Frozen (%) | Before (%) | After (%) | Current update (pp) | Negative parents | >5 pp harm | >20 pp harm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hc2 | full | 22.156 | 24.053 | 23.393 | -0.660 | 41 / 128 | 15 | 2 |
| hc2 | alpha0 | 22.156 | 22.156 | 23.264 | +1.108 | 29 / 128 | 11 | 2 |
| hc2 | alpha1 | 22.156 | 20.554 | 22.770 | +2.216 | 26 / 128 | 7 | 1 |
| hc2 | LN_only | 22.156 | 23.969 | 23.562 | -0.407 | 40 / 128 | 11 | 1 |
| vidstg | full | 14.466 | 16.139 | 18.379 | +2.240 | 26 / 128 | 6 | 1 |
| vidstg | alpha0 | 14.466 | 14.466 | 18.079 | +3.613 | 15 / 128 | 5 | 1 |
| vidstg | alpha1 | 14.466 | 13.732 | 17.167 | +3.435 | 18 / 128 | 6 | 0 |
| vidstg | LN_only | 14.466 | 16.027 | 17.915 | +1.887 | 25 / 128 | 7 | 2 |

## Scope and completion boundaries

The original joint Full regime has a positive pooled final-output difference versus LN-only in VidSTG; the HC interval crosses zero. This establishes a target-specific online component effect in the locked historical cohort, not a universal need for query adaptation. It does not isolate one LN state under a fixed-query causal state swap, validate an unlabeled gate, or promote a new production method.

Full-minus-reset intervals cross zero in both domains. Full-minus-unattenuated-carry pooled intervals are positive in both of these 128-parent, three-order component cohorts. These populations and orders differ from C Expanded-State (205/700 parents and two orders); the datasets cannot be pooled or treated as independent replications.

The companion retains all 3,072 anonymous row outcomes, all 256 paired parents, pooled and each-order statistics, all positive and negative tails, neutral values, no-information trajectories and null diagnostics. Original observed/unobserved diagnostics use the legacy DenseTubeGT timeline, not P9-P0 common-U genuine-frame support. Historical evaluation exposure is retained; this is not a fresh untouched test.

The original final CPU scorer counted completed first-formal pairs as if three new arms existed (36). This scope has one new arm: two domains x three orders x two first arrivals = 12. All 12 were already accepted and independently matched. The original source, logs, scoring outputs and science pins were preserved; a separate CPU dispatch corrected only the final aggregate assertion. No fit, GPU, model, detector or new GT scoring was rerun for that recovery.

Complete fit/source/state/input/dense and statistical root passed. The figures must still receive actual PDF pixel view, and all source/anonymous outcomes/closing receipts must receive actual remote byte/SHA/Git blob/mode/tree verification and archive checks before genuine FINAL. Original cost32 and later decisions remain distinct. Cancelled formal baselines remain standby; no manuscript was written.
