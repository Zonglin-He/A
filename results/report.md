# Final evaluation

One frozen ReferDINO initializes and supervises spatial OPD; Strong time uses the same frozen TA-STVG Source. Configuration: configs/default.json. Each dataset has128 exposed development parents, one query each, seed41. These are saved results, not a new full-dataset or independent confirmation.

| Target / row | vIoU % | tIoU % | fixed actual sIoU % | unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| hc2 / Source | 25.463541 | 51.439514 | 47.979496 | 47.754147 |
| hc2 / Spatial | 27.297077 | 51.439514 | 51.346351 | 50.997326 |
| hc2 / Temporal | 26.969901 | 55.920522 | 47.979496 | 47.754147 |
| hc2 / Full | 29.066935 | 55.920522 | 51.346351 | 50.997326 |
| hc2 / WithoutOPD | 28.587667 | 55.920522 | 50.161139 | 49.931468 |
| vidstg / Source | 14.582710 | 37.130256 | 36.381530 | 36.343478 |
| vidstg / Spatial | 17.533587 | 37.130256 | 42.066274 | 41.740686 |
| vidstg / Temporal | 16.481081 | 40.790336 | 36.381530 | 36.343478 |
| vidstg / Full | 19.764884 | 40.790336 | 42.066274 | 41.740686 |
| vidstg / WithoutOPD | 19.664027 | 40.790336 | 42.015608 | 41.769393 |

| Target / change | vIoU pp [95% parent bootstrap] | positive / negative / unchanged | losses below−5pp |
|---|---:|---:|---:|
| hc2 / Full-Source | +3.603394 [+2.171601, +5.053013] | 88 / 37 / 3 | 12 |
| hc2 / Full-WithoutOPD | +0.479268 [+0.202244, +0.807143] | 92 / 33 / 3 | 0 |
| vidstg / Full-Source | +5.182174 [+2.585867, +7.799148] | 77 / 30 / 21 | 17 |
| vidstg / Full-WithoutOPD | +0.100857 [-0.119699, +0.362130] | 65 / 34 / 29 | 1 |

HC final configuration: lr0.014/tau0.14/40 updates/gamma0.5/alpha1. Vid: lr0.01/tau0.5/30 updates/gamma1/alpha1. Policy scales0.05/0.25,32 candidates and8×256 per-query reset states remain fixed.

Full−Source includes expert initialization and Strong, not just OPD. The HC OPD development interval is positive; Vid crosses zero. Vid retains BV030−7.116367pp relative to E+Strong. All signed query outcomes, zero-overlap/no-update cases and this severe failure remain in evaluation.json. No new independent Live−Fixed claim is made. Selection-biased development intervals do not establish full-dataset generalization.

runtime.json contains one complete-input profile per dataset, including Source224/448, native ReferDINO8 and adaptation. Profiles are not panel averages. cost.json preserves total development cost; cached work is not a zero-cost deployment claim.

The cleanup preserved core operator ASTs, exact numerical configuration and two previously evaluated full-pipeline outputs. All256 five-row saved scores were recomputed only as a scoring compatibility check. See migration_checks.json, score_regression.json and cleanup_checks.json. Historical protocols, searches and negative findings remain reachable in Git history at7a0a8131733d124c9262db0678260c46f095ead4.
