# Cached mask-aligned local residuals: original32

Original exposed development cohort: HC16 + Vid16, one query per parent. All methods use the exact same E and final-MBR time interval. No model/expert calls, feature extraction, training, resampling or parameter search. All 32 new outputs were sealed before new GT scoring.

| Output | HC complete vIoU % | Vid complete vIoU % |
|---|---:|---:|
| E_MBR | 26.861045 | 28.187972 |
| Shared4_global | 26.642224 | 28.140723 |
| Shared4_observed | 26.827122 | 28.192414 |
| Independent4_observed | 26.830781 | 28.285157 |

## hc2: paired differences, pp

| Contrast | Complete vIoU [95% paired interval] | Observed actual sIoU | Unobserved actual sIoU | Benchmark interpolated sIoU |
|---|---:|---:|---:|---:|
| Shared4_global_minus_E_MBR | -0.218821 [-0.480469, -0.004661] | -0.361121 [-0.848359, +0.108857] | -0.288805 [-0.634208, +0.029727] | -0.284614 [-0.616369, +0.007659] |
| Shared4_observed_minus_E_MBR | -0.033923 [-0.072301, +0.000896] | -0.361121 [-0.848359, +0.108857] | +0.000000 [+0.000000, +0.000000] | -0.037913 [-0.076349, -0.001264] |
| Independent4_observed_minus_E_MBR | -0.030264 [-0.063122, -0.001052] | -0.291001 [-0.598312, -0.020389] | +0.000000 [+0.000000, +0.000000] | -0.035926 [-0.072261, -0.003368] |
| Shared4_observed_minus_Shared4_global | +0.184898 [-0.003858, +0.422111] | +0.000000 [+0.000000, +0.000000] | +0.288805 [-0.029727, +0.634208] | +0.246701 [-0.029662, +0.559920] |
| Independent4_observed_minus_Shared4_observed | +0.003659 [-0.038595, +0.042051] | +0.070120 [-0.330157, +0.542048] | +0.000000 [+0.000000, +0.000000] | +0.001986 [-0.044868, +0.041353] |

Known observed-node quality: 16/16 queries; known unobserved-node quality: 16/16. Null GT support stays unknown; all16 remain in complete-tube means.

| Output minus E | Changed observed contribution to vIoU pp | Unobserved actual contribution pp | Interpolated contribution pp | Positive / negative / neutral queries | Loss below -5pp |
|---|---:|---:|---:|---:|---:|
| Shared4_global | -0.006102 | -0.020456 | -0.192263 | 5 / 9 / 2 | 0 /16 |
| Shared4_observed | -0.006102 | +0.000000 | -0.027821 | 5 / 8 / 3 | 0 /16 |
| Independent4_observed | -0.003265 | +0.000000 | -0.026999 | 3 / 7 / 6 | 0 /16 |

Displacement below is the query mean of mean per-node norms: center in normalized image units, size in log units. Equal support does not imply equal update strength.

| Output | Observed center norm | Observed log-size norm | Unobserved center norm | Unobserved log-size norm |
|---|---:|---:|---:|---:|
| Shared4_global | 0.004444411 | 0.025438247 | 0.004444411 | 0.025438247 |
| Shared4_observed | 0.004444411 | 0.025438247 | 0.000000000 | 0.000000000 |
| Independent4_observed | 0.002837157 | 0.014675791 | 0.000000000 | 0.000000000 |

## vidstg: paired differences, pp

| Contrast | Complete vIoU [95% paired interval] | Observed actual sIoU | Unobserved actual sIoU | Benchmark interpolated sIoU |
|---|---:|---:|---:|---:|
| Shared4_global_minus_E_MBR | -0.047250 [-0.108855, +0.006473] | +0.080599 [-0.261202, +0.425497] | -0.250204 [-0.422344, -0.100760] | -0.198622 [-0.360812, -0.058126] |
| Shared4_observed_minus_E_MBR | +0.004441 [-0.008080, +0.019606] | +0.080599 [-0.261202, +0.425497] | +0.000000 [+0.000000, +0.000000] | +0.003561 [-0.008537, +0.018240] |
| Independent4_observed_minus_E_MBR | +0.097184 [+0.026135, +0.181096] | +0.945308 [+0.097641, +1.778138] | +0.000000 [+0.000000, +0.000000] | +0.102083 [+0.024815, +0.192697] |
| Shared4_observed_minus_Shared4_global | +0.051691 [+0.000774, +0.110771] | +0.000000 [+0.000000, +0.000000] | +0.250204 [+0.100760, +0.422344] | +0.202183 [+0.058080, +0.369037] |
| Independent4_observed_minus_Shared4_observed | +0.092743 [+0.020446, +0.181436] | +0.864709 [+0.085184, +1.761937] | +0.000000 [+0.000000, +0.000000] | +0.098521 [+0.021869, +0.191339] |

Known observed-node quality: 14/16 queries; known unobserved-node quality: 16/16. Null GT support stays unknown; all16 remain in complete-tube means.

| Output minus E | Changed observed contribution to vIoU pp | Unobserved actual contribution pp | Interpolated contribution pp | Positive / negative / neutral queries | Loss below -5pp |
|---|---:|---:|---:|---:|---:|
| Shared4_global | +0.001412 | -0.011583 | -0.037078 | 3 / 7 / 6 | 0 /16 |
| Shared4_observed | +0.001412 | +0.000000 | +0.003029 | 4 / 5 / 7 | 0 /16 |
| Independent4_observed | +0.016448 | +0.000000 | +0.080736 | 8 / 1 / 7 | 0 /16 |

Displacement below is the query mean of mean per-node norms: center in normalized image units, size in log units. Equal support does not imply equal update strength.

| Output | Observed center norm | Observed log-size norm | Unobserved center norm | Unobserved log-size norm |
|---|---:|---:|---:|---:|
| Shared4_global | 0.003036623 | 0.015426530 | 0.003036623 | 0.015426530 |
| Shared4_observed | 0.003036623 | 0.015426530 | 0.000000000 | 0.000000000 |
| Independent4_observed | 0.004007056 | 0.020298388 | 0.000000000 | 0.000000000 |

## Interpretation boundaries

The two local outputs preserve every unobserved actual decoder node bit-for-bit. Their unobserved actual-frame sIoU is therefore exactly unchanged. Changes at benchmark-interpolated frames arise from the official xyxy interpolation of changed neighboring nodes, not new visual evidence or learned propagation. The three contribution columns add exactly to complete-vIoU differences, with the fixed temporal-overlap denominator.

Independent4 uses each accepted teacher-minus-input correction with all24 states in the denominator. It does not average different anchors. Shared4_observed uses the exact prior global-output boxes at the planned nodes; its observed-node performance must equal Shared4_global. E itself is preserved. No GT selects updates or changes threshold, support, weights or method.

All signed cases, no-ops, missing-support queries, displacement maxima/RMS and paired intervals are included in RESULTS_PUBLIC.json. These are repeatedly exposed development data; a positive mean is not independent confirmation. No dynamic OPD or deployment promotion follows this finite experiment automatically.

Cached output-construction CPU wall time: 0.365126 s. New GPU time 0, new Source/DINO/feature calls 0. This excludes the old Source/E and mask-alignment generation cost and is not first-use online latency.
