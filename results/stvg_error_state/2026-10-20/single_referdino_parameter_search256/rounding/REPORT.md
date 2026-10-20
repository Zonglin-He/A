# One rounded HC configuration, unchanged Vid

The user requested shorter decimal representations during publication, before any branch update. This supplement actually changes and reruns HC lr0.0141421356→0.014 and tau0.1414213562→0.14 on all128 exposed parents with seed41. The original exact-value search, selected outputs and profiles remain unchanged at their original paths. Vid is reused exactly. The predeclared one-trial practical tolerance passed; no further precision search follows.

HC rounded full vIoU29.066935%, change versus exact+0.053405pp; meaningful gains60/128, meaningful losses18/128; any positive/negative/tied92/33/3; severe adaptation losses below−5pp0. Prior exact values:29.013530%,60 meaningful gains,19 meaningful losses. The same teacher evidence, policy, gamma0.5 and40 updates are used.

## Final five-row results

| Dataset / row | full vIoU % | tIoU % | fixed actual sIoU % | unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| hc2 / Source | 25.463541 | 51.439514 | 47.979496 | 47.754147 |
| hc2 / Spatial_only | 27.297077 | 51.439514 | 51.346351 | 50.997326 |
| hc2 / Temporal_only | 26.969901 | 55.920522 | 47.979496 | 47.754147 |
| hc2 / Full | 29.066935 | 55.920522 | 51.346351 | 50.997326 |
| hc2 / Without_spatial_adaptation | 28.587667 | 55.920522 | 50.161139 | 49.931468 |
| vidstg / Source | 14.582710 | 37.130256 | 36.381530 | 36.343478 |
| vidstg / Spatial_only | 17.533587 | 37.130256 | 42.066274 | 41.740686 |
| vidstg / Temporal_only | 16.481081 | 40.790336 | 36.381530 | 36.343478 |
| vidstg / Full | 19.764884 | 40.790336 | 42.066274 | 41.740686 |
| vidstg / Without_spatial_adaptation | 19.664027 | 40.790336 | 42.015608 | 41.769393 |

## Paired complete-vIoU changes

| Dataset / comparison | mean pp [95% parent bootstrap] | positive / negative / unchanged | below−5pp |
|---|---:|---:|---:|
| hc2 / Full_minus_Source | +3.603394 [+2.171601, +5.053013] | 88 / 37 / 3 | 12 |
| hc2 / Full_minus_Without_spatial_adaptation | +0.479268 [+0.202244, +0.807143] | 92 / 33 / 3 | 0 |
| hc2 / Full_minus_Spatial_only | +1.769858 [+0.717507, +2.829836] | 75 / 39 / 14 | 13 |
| hc2 / Spatial_only_minus_Source | +1.833536 [+0.810567, +2.759732] | 94 / 28 / 6 | 11 |
| hc2 / Without_spatial_adaptation_minus_original_ERef_Strong | +0.633679 [-0.317686, +1.613297] | 63 / 62 / 3 | 9 |
| hc2 / Full_minus_original_Full | +0.813549 [-0.106356, +1.760907] | 63 / 62 / 3 | 9 |
| vidstg / Full_minus_Source | +5.182174 [+2.585867, +7.799148] | 77 / 30 / 21 | 17 |
| vidstg / Full_minus_Without_spatial_adaptation | +0.100857 [-0.119699, +0.362130] | 65 / 34 / 29 | 1 |
| vidstg / Full_minus_Spatial_only | +2.231297 [+0.970180, +3.724509] | 51 / 33 / 44 | 4 |
| vidstg / Spatial_only_minus_Source | +2.950877 [+0.747229, +5.186708] | 74 / 29 / 25 | 16 |
| vidstg / Without_spatial_adaptation_minus_original_ERef_Strong | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 128 | 0 |
| vidstg / Full_minus_original_Full | +0.035778 [-0.023073, +0.103687] | 53 / 46 / 29 | 0 |

This is parameter-development evidence with selection-biased descriptive intervals. Teacher-reference errors and cross-frame propagation are not universally fixed. Vid coverage choice still has BV030−7.116367pp vs E; its small-lr/early-stop mean alternative is preserved in the pre-rounding full report. HC initialization gain is separated from the subsequent OPD gain. There is no new independent Live-versus-Fixed claim. The final full-input HC verification and cumulative cost receipt are added after completion.

See [all128 rounded signed outcomes](hc2/fine/RESULTS_PUBLIC.json), [independent update/readout audit](hc2/fine/PRE_GT_AUDIT_PUBLIC.json), [independent score audit](hc2/fine/ROOT_AUDIT_PUBLIC.json), [rounding decision and signed changes](ROUNDING_DECISION_PUBLIC.json), [all256 final five rows](FINAL_FIVE_ROW_RESULTS_PUBLIC.json), and [original complete search](../REPORT.md).

## Complete-input verification and cost

Rounded HC first-case RGB→Source224/448→ReferDINO8→E_gamma→40-stepOPD→Strong/MBR completed in24.436050s, peak7.734084GiB, final prediction bitexact to the sealed rounded result; no historical detector/cache reads inside the fresh pipeline. Vid's unchanged prior full-input profile remains37.686893s/20.194110GiB. These are first-case profiles, not panel-average latency.

The complete finite search plus rounding and verification costs4096new primary fits,163990state updates including40compatibility and110complete-input verification updates,6210.026630s cumulative worker wall below7200. Matrix paths reuse cached inputs; the three full-input checks actually add6Source224/448 forwards and3native ReferDINO8 queries. Cache creation is inherited, not free. See FINAL_CUMULATIVE_COST_PUBLIC.json for every invocation. No more search, new seed or full-dataset evaluation is queued.
