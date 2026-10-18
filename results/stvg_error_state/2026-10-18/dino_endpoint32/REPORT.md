# Cached teacher-endpoint versus transported-residual update: original32

Same exposed HC16/Vid16, one query per parent. Only the estimation target changes. E, aligned teacher outputs, admission masks, original24 states, denominator24,0.35 threshold, four planned output nodes and exact final-MBR interval remain fixed. No new model, DINO, feature extraction, training or amplitude search. All32 outputs sealed before new GT scoring.

Residual_local is the exact prior Independent4_observed output. Endpoint_local adds sum(m*(z(T)-z(E)))/24 to E at each planned node, in (cx,cy,log w,log h). All other decoder nodes are bit-exact E. No admission is a no-op; teacher equals E is a no-op; teacher equals sampled input need not be a no-op for the endpoint target. This is not an automatically correct repair or an OPD result.

| Output | HC complete vIoU % | Vid complete vIoU % |
|---|---:|---:|
| E_MBR | 26.861045 | 28.187972 |
| Residual_local | 26.830781 | 28.285157 |
| Endpoint_local | 26.895387 | 28.286621 |

## hc2: paired differences, pp

| Contrast | Complete vIoU [95% interval] | Observed actual sIoU | Unobserved actual sIoU | Benchmark interpolated sIoU |
|---|---:|---:|---:|---:|
| Residual_local_minus_E_MBR | -0.030264 [-0.063122, -0.001052] | -0.291001 [-0.598312, -0.020389] | +0.000000 [+0.000000, +0.000000] | -0.035926 [-0.072261, -0.003368] |
| Endpoint_local_minus_E_MBR | +0.034341 [-0.002470, +0.089014] | +0.289037 [+0.062791, +0.562924] | +0.000000 [+0.000000, +0.000000] | +0.037696 [-0.001719, +0.094152] |
| Endpoint_local_minus_Residual_local | +0.064606 [+0.004755, +0.143690] | +0.580037 [+0.147636, +1.047616] | +0.000000 [+0.000000, +0.000000] | +0.073622 [+0.007849, +0.159514] |

Known observed-node quality is available for 16/16 queries; unknown support stays null, while all16 remain in complete vIoU.

| Output minus E | Observed-node v contribution pp | Unobserved actual contribution pp | Benchmark interpolated contribution pp | Positive / negative / neutral | Loss below -5pp |
|---|---:|---:|---:|---:|---:|
| Residual_local | -0.003265 | +0.000000 | -0.026999 | 3 / 7 / 6 | 0 /16 |
| Endpoint_local | +0.005562 | +0.000000 | +0.028780 | 8 / 2 / 6 | 0 /16 |

| Output | Observed image-center norm | Observed log-size norm |
|---|---:|---:|
| Residual_local | 0.002837157 | 0.014675791 |
| Endpoint_local | 0.001955777 | 0.009647295 |

## vidstg: paired differences, pp

| Contrast | Complete vIoU [95% interval] | Observed actual sIoU | Unobserved actual sIoU | Benchmark interpolated sIoU |
|---|---:|---:|---:|---:|
| Residual_local_minus_E_MBR | +0.097184 [+0.026135, +0.181096] | +0.945308 [+0.097641, +1.778138] | +0.000000 [+0.000000, +0.000000] | +0.102083 [+0.024815, +0.192697] |
| Endpoint_local_minus_E_MBR | +0.098649 [+0.028249, +0.185633] | +0.810981 [-0.076043, +1.720453] | +0.000000 [+0.000000, +0.000000] | +0.107932 [+0.031129, +0.203221] |
| Endpoint_local_minus_Residual_local | +0.001464 [-0.027695, +0.027134] | -0.134327 [-0.659472, +0.370324] | +0.000000 [+0.000000, +0.000000] | +0.005850 [-0.024017, +0.031315] |

Known observed-node quality is available for 14/16 queries; unknown support stays null, while all16 remain in complete vIoU.

| Output minus E | Observed-node v contribution pp | Unobserved actual contribution pp | Benchmark interpolated contribution pp | Positive / negative / neutral | Loss below -5pp |
|---|---:|---:|---:|---:|---:|
| Residual_local | +0.016448 | +0.000000 | +0.080736 | 8 / 1 / 7 | 0 /16 |
| Endpoint_local | +0.013351 | +0.000000 | +0.085298 | 8 / 1 / 7 | 0 /16 |

| Output | Observed image-center norm | Observed log-size norm |
|---|---:|---:|
| Residual_local | 0.004007056 | 0.020298388 |
| Endpoint_local | 0.004072712 | 0.020381742 |

## Per-query signed complete-vIoU changes, pp

| Alias | Domain | Residual minus E | Endpoint minus E | Endpoint minus Residual |
|---|---|---:|---:|---:|
| H007 | hc2 | -0.008113 | -0.003692 | +0.004421 |
| V010 | vidstg | +0.034923 | +0.046315 | +0.011392 |
| H008 | hc2 | +0.056928 | +0.060739 | +0.003811 |
| H010 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| H018 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| H022 | hc2 | -0.121757 | +0.064870 | +0.186627 |
| H023 | hc2 | -0.003461 | +0.025990 | +0.029451 |
| H031 | hc2 | -0.058906 | +0.034204 | +0.093110 |
| H032 | hc2 | -0.161724 | +0.387296 | +0.549020 |
| H035 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| H037 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| H043 | hc2 | +0.016504 | -0.092265 | -0.108769 |
| H045 | hc2 | -0.033748 | +0.024945 | +0.058693 |
| H050 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| H058 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| H060 | hc2 | +0.001324 | +0.014160 | +0.012836 |
| H061 | hc2 | -0.171277 | +0.033217 | +0.204494 |
| V012 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| V015 | vidstg | +0.312228 | +0.413049 | +0.100821 |
| V016 | vidstg | +0.003117 | +0.008113 | +0.004996 |
| V022 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| V024 | vidstg | +0.024815 | +0.116064 | +0.091249 |
| V025 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| V026 | vidstg | +0.028507 | +0.002564 | -0.025943 |
| V036 | vidstg | +0.302445 | +0.136150 | -0.166295 |
| V037 | vidstg | -0.015705 | -0.015705 | +0.000000 |
| V038 | vidstg | +0.395556 | +0.436074 | +0.040518 |
| V042 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| V044 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| V048 | vidstg | +0.469064 | +0.435759 | -0.033305 |
| V056 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| V061 | vidstg | +0.000000 | +0.000000 | +0.000000 |

## Limits and execution boundary

Intervals use paired parent bootstrap10000 with seed20261018; this repeatedly exposed cohort is not independent confirmation. The two targets generally produce different directions and actual amplitudes: matching coefficients and support does not isolate amplitude-independent direction effects. All no-support queries and negative cases remain.

All tIoUs are exact. Both local outputs preserve unobserved actual decoder nodes, so their quality change there is exactly zero by construction. Official interpolation changes neighboring benchmark frames; the contribution columns add exactly to complete-vIoU changes. These are not new visual observations or learned propagation.

No new cohort, center-reference comparison or dynamic OPD has run. A target-level contrast cannot establish sampling or on-policy superiority. The existing production registration and previous results are unchanged.

Cached output construction CPU wall time: 0.355898 s; new GPU/model/DINO/Source calls0. This excludes old Source/E/feedback generation and subsequent scoring/audit, and is not first-use end-to-end latency.
