# Expanded-State: complete historical non-development result

All 5,430 new formal fits were sealed before GT exposure: 205 HC-STVG-v2 parents and 700 VidSTG parents, three fixed retention rules, two fixed online orders. No new detector calls. This is the complete Expanded-State phase; deployment timing, LN-only and audited cost remain separate authorized work.

## Direct final-output comparisons

Parent averages retain both online orders. Intervals use the original paired 10,000-parent bootstrap (seed 20261011) without multiplicity adjustment. Values are percentage points.

| Target | Order | Paired contrast | Difference (pp) | 95% CI | Negative parents |
|---|---|---|---:|---|---:|
| hc2 | pooled | Full - alpha0 | +0.066 | [-0.171, +0.305] | 77 / 205 |
| hc2 | pooled | Full - alpha1 | +0.337 | [-0.094, +0.790] | 71 / 205 |
| hc2 | order1 | Full - alpha0 | -0.034 | [-0.243, +0.173] | 80 / 205 |
| hc2 | order1 | Full - alpha1 | +0.253 | [-0.351, +0.846] | 75 / 205 |
| hc2 | order2 | Full - alpha0 | +0.166 | [-0.138, +0.506] | 76 / 205 |
| hc2 | order2 | Full - alpha1 | +0.420 | [-0.055, +0.931] | 70 / 205 |
| vidstg | pooled | Full - alpha0 | +0.237 | [-0.015, +0.491] | 229 / 700 |
| vidstg | pooled | Full - alpha1 | +0.809 | [+0.570, +1.055] | 179 / 700 |
| vidstg | order1 | Full - alpha0 | +0.192 | [-0.084, +0.469] | 235 / 700 |
| vidstg | order1 | Full - alpha1 | +0.909 | [+0.630, +1.193] | 166 / 700 |
| vidstg | order2 | Full - alpha0 | +0.282 | [+0.020, +0.552] | 220 / 700 |
| vidstg | order2 | Full - alpha1 | +0.709 | [+0.400, +1.036] | 197 / 700 |

## Absolute outcomes and current-update negative tails

All entries average within parent across the two orders. A negative current-update parent means its mean official vIoU After-minus-Before is below zero; this counts arbitrarily small negative differences as well as substantial harms. The >5 pp and >20 pp columns distinguish the negative tail.

| Target | Rule | Frozen vIoU (%) | Before vIoU (%) | After vIoU (%) | Current update (pp) | Negative parents | >5 pp harm | >20 pp harm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hc2 | full | 20.906 | 22.650 | 22.280 | -0.370 | 65 / 205 | 23 | 4 |
| hc2 | alpha0 | 20.906 | 20.906 | 22.214 | +1.308 | 45 / 205 | 17 | 5 |
| hc2 | alpha1 | 20.906 | 19.599 | 21.943 | +2.344 | 51 / 205 | 14 | 3 |
| vidstg | full | 11.410 | 13.027 | 14.370 | +1.344 | 122 / 700 | 34 | 10 |
| vidstg | alpha0 | 11.410 | 11.410 | 14.134 | +2.724 | 78 / 700 | 29 | 5 |
| vidstg | alpha1 | 11.410 | 11.496 | 13.561 | +2.066 | 86 / 700 | 27 | 0 |

Full-minus-reset pooled intervals cross zero in both domains. Full-minus-full-carry is positive in VidSTG for both orders and the pooled endpoint; HC intervals cross zero. The measured effect therefore supports attenuation versus unattenuated carry in VidSTG, while a stable advantage over reset across both domains is not established.

Before-minus-Frozen describes the state entering the current query; it is not the direct Full-minus-reset contrast. The three rules intervene on an entire online retention regime. They do not isolate the causal effect of one LN state under a single fixed-query state swap, nor validate an unlabeled update selector. P9-P0 natural order/state associations and these regime comparisons must retain these distinct scopes.

The cohort excludes the predeclared development parents but has historical evaluation exposure. It is not a new untouched test set. All 5,430 anonymous scalar rows, all 905 paired parent outcomes, pooled and order-specific absolute/direct statistics, neutral values, unavailable observations and negative tails are exported. No GT-driven cohort or configuration selection was performed.

The optional observed/unobserved_iou_delta columns are the original C DenseTubeGT-timeline diagnostic; unavailable entries remain null. They do not use P9-P0's common-U genuine-frame definition. Cached capture/fit compute fields describe this experiment and do not substitute for the pending complete deployment timing or teacher cost measurements.

The original scientific implementation, runtime pins, accepted fits, qualification and global barrier remain unchanged. No gate, new method, causal state-intervention GPU experiment, production promotion or manuscript was created.
