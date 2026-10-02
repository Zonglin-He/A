# Current query correction and persistent Uniform A: two-round review

Both rounds and matched confirmation completed. Current corrections never enter future state. The baseline is sealed best Uniform A with native temporal Fast, not Frozen TA-STVG. Each dataset has 32 historically exposed development sources and 16 source-disjoint within-batch confirmation sources, one query per source, two orders, clean and five transient 5% corruptions, 25% specialist arrivals. Confirmation is also historically exposed; it does not tune the decision.

Vid K1/lr .033761698432507946/teacher .34902548789596055; HC K8/lr .006097133675874025/teacher1. rho .05/student1/D4/1792 parameters and official same-domain checkpoints remain fixed. All persistent transitions equal saved A, including multi-step HC updates; two live A arrivals per dataset verify prediction, gradient and state bitwise. New full nine-probe suffix replays validate saved supports. Direct selection is a changed spatial readout; temporary one-step SGD is current adapt-then-predict and expires immediately. Neither is cross-query fast memory.

Round1 selected one rule for both datasets: **R_select**. Final screen selection: **CT**; new acquisition eligible: **False**. These are development decisions, not production promotion or statistical guarantees. No confirmation reselection.


The confirmation supports small positive direct-selection means but does not establish a Routed-specific gain over the matched second Uniform5 observation: pooled C minus twoUniform5 is -.000978 pp, with a confidence interval spanning zero. Temporal output is positive on Vid and negative on HC in both development and confirmation. New acquisition does not satisfy the common development eligibility rule and was excluded before confirmation. Specific temporary SGD has negative means on both confirmation datasets. The nominated CT is therefore a completed confirmation experiment, not a proven common improvement or a promoted method.


## Confirmation relative to Frozen and A

| Dataset | Frozen vIoU (%) | A (%) | C (%) | CT (%) | CT minus Frozen, pp [95% CI] |
|---|---:|---:|---:|---:|---:|

| vidstg | 32.9534 | 32.6465 | 32.7082 | 32.8664 | -0.0870 [-0.4502, +0.2742] |

| hc2 | 26.4778 | 26.7630 | 26.8574 | 26.8072 | +0.3294 [-0.5922, +1.4548] |


Vid A is below Frozen on this confirmation cohort; the nominated CT still has a negative mean versus Frozen. Reported positive current increments must not be described as an overall Frozen win. The same-cohort paired intervals and all rows are retained.


## round1 / search: corruption-all source macro incremental vIoU (pp)

| Arm | VidSTG, mean [95% paired source CI] | HC2, mean [95% paired source CI] |
|---|---:|---:|

| A | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

| U_select | +0.0649 [+0.0117, +0.1344] | +0.0476 [-0.0179, +0.1197] |

| R_select | +0.0947 [+0.0279, +0.1755] | +0.1044 [+0.0463, +0.1717] |

| R_temp | +0.1705 [-0.0550, +0.4250] | +0.0258 [+0.0121, +0.0416] |

| Specific_temp | -0.1924 [-0.5395, +0.0386] | +0.0428 [+0.0023, +0.1140] |


vidstg: Frozen/A corruption vIoU 16.2145/17.4529%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- U_select: gross gain/loss 0.0729/0.0080 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0675 [+0.0099, +0.1442]; two order means +0.0450, +0.0847 pp.

- R_select: gross gain/loss 0.0986/0.0040 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0730 [+0.0072, +0.1542]; two order means +0.0830, +0.1063 pp.

- R_temp: gross gain/loss 0.2556/0.0851 pp; >5pp harmful arrivals 1; .3 rescue/destroy 2/0; clean -0.0128 [-0.4060, +0.3599]; two order means +0.3666, -0.0256 pp.

- Specific_temp: gross gain/loss 0.0421/0.2345 pp; >5pp harmful arrivals 5; .3 rescue/destroy 0/0; clean -0.2811 [-0.7711, +0.1401]; two order means -0.0326, -0.3521 pp.


Routed selected a better existing tube but temporary SGD harmed the output in **14** corrupt expert arrivals. This diagnosis distinguishes expert preference from parameter execution; it is not an online GT filter.


hc2: Frozen/A corruption vIoU 29.8667/30.9637%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- U_select: gross gain/loss 0.0780/0.0304 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0486 [-0.0172, +0.1198]; two order means +0.0232, +0.0720 pp.

- R_select: gross gain/loss 0.1101/0.0057 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0970 [+0.0313, +0.1672]; two order means +0.1205, +0.0884 pp.

- R_temp: gross gain/loss 0.0290/0.0033 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0142 [-0.0121, +0.0352]; two order means +0.0290, +0.0225 pp.

- Specific_temp: gross gain/loss 0.0437/0.0009 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0006 [-0.0413, +0.0351]; two order means +0.0799, +0.0057 pp.


Routed selected a better existing tube but temporary SGD harmed the output in **3** corrupt expert arrivals. This diagnosis distinguishes expert preference from parameter execution; it is not an online GT filter.


### Paired differences between alternatives (corrupt-all, pp)

| Contrast | Vid | HC2 | Equal-dataset pooled [95% CI] |
|---|---:|---:|---:|

| R_select_minus_A_v | +0.0947 [+0.0279, +0.1755] | +0.1044 [+0.0463, +0.1717] | +0.0996 [+0.0533, +0.1503] |


R_select_minus_A_v, vidstg: leave-one-source-out mean range [+0.0707, +0.1001] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


R_select_minus_A_v, hc2: leave-one-source-out mean range [+0.0872, +0.1088] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| R_temp_minus_A_v | +0.1705 [-0.0550, +0.4250] | +0.0258 [+0.0121, +0.0416] | +0.0981 [-0.0147, +0.2242] |


R_temp_minus_A_v, vidstg: leave-one-source-out mean range [+0.1001, +0.2342] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


R_temp_minus_A_v, hc2: leave-one-source-out mean range [+0.0216, +0.0266] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| Specific_temp_minus_A_v | -0.1924 [-0.5395, +0.0386] | +0.0428 [+0.0023, +0.1140] | -0.0748 [-0.2516, +0.0495] |


Specific_temp_minus_A_v, vidstg: leave-one-source-out mean range [-0.2331, -0.0525] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


Specific_temp_minus_A_v, hc2: leave-one-source-out mean range [+0.0113, +0.0445] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| R_select_minus_U_select_v | +0.0298 [-0.0314, +0.1040] | +0.0568 [+0.0125, +0.1133] | +0.0433 [+0.0044, +0.0872] |


R_select_minus_U_select_v, vidstg: leave-one-source-out mean range [+0.0047, +0.0474] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


R_select_minus_U_select_v, hc2: leave-one-source-out mean range [+0.0375, +0.0603] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| R_select_minus_R_temp_v | -0.0758 [-0.3038, +0.1225] | +0.0786 [+0.0311, +0.1335] | +0.0014 [-0.1156, +0.1059] |


R_select_minus_R_temp_v, vidstg: leave-one-source-out mean range [-0.1340, -0.0055] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


R_select_minus_R_temp_v, hc2: leave-one-source-out mean range [+0.0646, +0.0828] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| Specific_temp_minus_R_temp_v | -0.3629 [-0.6371, -0.1327] | +0.0171 [-0.0251, +0.0851] | -0.1729 [-0.3170, -0.0552] |


Specific_temp_minus_R_temp_v, vidstg: leave-one-source-out mean range [-0.3753, -0.2866] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


Specific_temp_minus_R_temp_v, hc2: leave-one-source-out mean range [-0.0120, +0.0226] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.


## round2 / search: corruption-all source macro incremental vIoU (pp)

| Arm | VidSTG, mean [95% paired source CI] | HC2, mean [95% paired source CI] |
|---|---:|---:|

| A | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

| C | +0.0947 [+0.0279, +0.1755] | +0.1044 [+0.0463, +0.1717] |

| T | +0.1614 [-0.0108, +0.4903] | -0.0257 [-0.4620, +0.3278] |

| CT | +0.2614 [+0.0308, +0.6740] | +0.0807 [-0.3593, +0.4827] |

| C_newacquisition | +0.0499 [-0.0049, +0.1191] | +0.1125 [+0.0526, +0.1814] |

| CT_newacquisition | +0.2162 [-0.0071, +0.6232] | +0.0901 [-0.3532, +0.4863] |

| twoUniform5 | +0.0754 [+0.0228, +0.1437] | +0.0731 [+0.0258, +0.1295] |

| twoUniform5_newtime | +0.2421 [+0.0236, +0.6450] | +0.0503 [-0.3944, +0.4523] |

| R_acquisition_old | +0.0947 [+0.0279, +0.1755] | +0.1044 [+0.0463, +0.1717] |

| R_acquisition_new | +0.0499 [-0.0049, +0.1191] | +0.1125 [+0.0526, +0.1814] |


vidstg: Frozen/A corruption vIoU 16.2145/17.4529%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- C: gross gain/loss 0.0986/0.0040 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0730 [+0.0072, +0.1542]; two order means +0.0830, +0.1063 pp.

- T: gross gain/loss 0.1668/0.0054 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.1594 [+0.0000, +0.4782]; two order means +0.0044, +0.3185 pp.

- CT: gross gain/loss 0.2708/0.0094 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.2381 [+0.0075, +0.6541]; two order means +0.0882, +0.4347 pp.

- C_newacquisition: gross gain/loss 0.0621/0.0122 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0621 [-0.0013, +0.1423]; two order means +0.0549, +0.0449 pp.

- CT_newacquisition: gross gain/loss 0.2325/0.0163 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.2271 [-0.0010, +0.6433]; two order means +0.0595, +0.3728 pp.

- twoUniform5: gross gain/loss 0.0771/0.0017 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0939 [+0.0280, +0.1762]; two order means +0.0606, +0.0903 pp.

- twoUniform5_newtime: gross gain/loss 0.2478/0.0057 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.2589 [+0.0280, +0.6760]; two order means +0.0660, +0.4182 pp.

- R_acquisition_old: gross gain/loss 0.0986/0.0040 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0730 [+0.0072, +0.1542]; two order means +0.0830, +0.1063 pp.

- R_acquisition_new: gross gain/loss 0.0621/0.0122 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0621 [-0.0013, +0.1423]; two order means +0.0549, +0.0449 pp.


Readout chain, same sources: Frozen to A native +0.5730 [-0.5463, +2.2916] pp; A native to original Fast A +0.6654 [-0.3514, +2.0249] pp. A native includes all inherited state effects; this is a readout decomposition, not a unique causal diagnosis of spatial or temporal learning.


Temporal actual changed-observation fraction: 0.9552. Old/new selected candidate changed in 9/80 expert arrivals. Two views use the existing observed STVG frame grid, with nearest-frame sampling; they are different image observations rather than original full-rate independent videos. Stable agreement can be wrong.

- U: event-frame precision 32.500%; candidate pairwise accuracy 70.394% (mean qualified arrivals); empty 5/80.

- Rnew: event-frame precision 49.000%; candidate pairwise accuracy 70.324% (mean qualified arrivals); empty 5/80.

- U2: event-frame precision 41.250%; candidate pairwise accuracy 72.176% (mean qualified arrivals); empty 0/80.


Old/new temporal replacements 75/69; tIoU-worsening replacements versus the same A native interval 17/11; vIoU-worsening replacements 18/11. These are expert-arrival counts, not error probabilities on independent videos.


New versus old Routed event-frame precision: +0.5000 [-4.5000, +5.0062] pp; fixed-time tube utility remains a separate paired contrast below.


hc2: Frozen/A corruption vIoU 29.8667/30.9637%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- C: gross gain/loss 0.1101/0.0057 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0970 [+0.0313, +0.1672]; two order means +0.1205, +0.0884 pp.

- T: gross gain/loss 0.1624/0.1880 pp; >5pp harmful arrivals 6; .3 rescue/destroy 1/0; clean -0.1857 [-0.6139, +0.0652]; two order means +0.1481, -0.1995 pp.

- CT: gross gain/loss 0.2658/0.1851 pp; >5pp harmful arrivals 6; .3 rescue/destroy 1/0; clean -0.0912 [-0.5339, +0.1842]; two order means +0.2748, -0.1134 pp.

- C_newacquisition: gross gain/loss 0.1159/0.0033 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0965 [+0.0301, +0.1679]; two order means +0.1403, +0.0848 pp.

- CT_newacquisition: gross gain/loss 0.2727/0.1825 pp; >5pp harmful arrivals 6; .3 rescue/destroy 1/0; clean -0.0892 [-0.5339, +0.1873]; two order means +0.2960, -0.1158 pp.

- twoUniform5: gross gain/loss 0.0847/0.0116 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0827 [+0.0322, +0.1412]; two order means +0.1027, +0.0436 pp.

- twoUniform5_newtime: gross gain/loss 0.2413/0.1911 pp; >5pp harmful arrivals 6; .3 rescue/destroy 1/0; clean -0.1030 [-0.5427, +0.1674]; two order means +0.2571, -0.1566 pp.

- R_acquisition_old: gross gain/loss 0.1101/0.0057 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0970 [+0.0313, +0.1672]; two order means +0.1205, +0.0884 pp.

- R_acquisition_new: gross gain/loss 0.1159/0.0033 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0965 [+0.0301, +0.1679]; two order means +0.1403, +0.0848 pp.


Readout chain, same sources: Frozen to A native +1.2847 [-0.8640, +4.5699] pp; A native to original Fast A -0.1878 [-1.0877, +0.6036] pp. A native includes all inherited state effects; this is a readout decomposition, not a unique causal diagnosis of spatial or temporal learning.


Temporal actual changed-observation fraction: 0.8333. Old/new selected candidate changed in 19/80 expert arrivals. Two views use the existing observed STVG frame grid, with nearest-frame sampling; they are different image observations rather than original full-rate independent videos. Stable agreement can be wrong.

- U: event-frame precision 21.250%; candidate pairwise accuracy 64.035% (mean qualified arrivals); empty 5/80.

- Rnew: event-frame precision 57.000%; candidate pairwise accuracy 75.914% (mean qualified arrivals); empty 4/80.

- U2: event-frame precision 31.250%; candidate pairwise accuracy 73.026% (mean qualified arrivals); empty 1/80.


Old/new temporal replacements 79/73; tIoU-worsening replacements versus the same A native interval 38/40; vIoU-worsening replacements 38/40. These are expert-arrival counts, not error probabilities on independent videos.


New versus old Routed event-frame precision: -1.7143 [-5.1429, +1.1429] pp; fixed-time tube utility remains a separate paired contrast below.


### Paired differences between alternatives (corrupt-all, pp)

| Contrast | Vid | HC2 | Equal-dataset pooled [95% CI] |
|---|---:|---:|---:|

| A_minus_Frozen_v | +1.2384 [-0.4576, +3.3505] | +1.0970 [-1.2681, +4.4660] | +1.1677 [-0.3688, +3.1344] |


A_minus_Frozen_v, vidstg: leave-one-source-out mean range [+0.5347, +1.4754] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


A_minus_Frozen_v, hc2: leave-one-source-out mean range [-0.2943, +1.5508] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| C_minus_Frozen_v | +1.3330 [-0.3621, +3.4360] | +1.2014 [-1.1493, +4.5513] | +1.2672 [-0.2598, +3.2270] |


C_minus_Frozen_v, vidstg: leave-one-source-out mean range [+0.6325, +1.5644] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


C_minus_Frozen_v, hc2: leave-one-source-out mean range [-0.1865, +1.6586] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_Frozen_v | +1.4998 [-0.1873, +3.5926] | +1.1777 [-1.0720, +4.5267] | +1.3388 [-0.1591, +3.2782] |


CT_minus_Frozen_v, vidstg: leave-one-source-out mean range [+0.8046, +1.7365] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_Frozen_v, hc2: leave-one-source-out mean range [-0.2110, +1.6341] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| C_minus_A_v | +0.0947 [+0.0279, +0.1755] | +0.1044 [+0.0463, +0.1717] | +0.0996 [+0.0533, +0.1503] |


C_minus_A_v, vidstg: leave-one-source-out mean range [+0.0707, +0.1001] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


C_minus_A_v, hc2: leave-one-source-out mean range [+0.0872, +0.1088] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_A_v | +0.2614 [+0.0308, +0.6740] | +0.0807 [-0.3593, +0.4827] | +0.1711 [-0.0974, +0.4576] |


CT_minus_A_v, vidstg: leave-one-source-out mean range [+0.0716, +0.2722] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_A_v, hc2: leave-one-source-out mean range [-0.0474, +0.2517] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| T_minus_A_v | +0.1614 [-0.0108, +0.4903] | -0.0257 [-0.4620, +0.3278] | +0.0679 [-0.1766, +0.3201] |


T_minus_A_v, vidstg: leave-one-source-out mean range [+0.0004, +0.1722] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


T_minus_A_v, hc2: leave-one-source-out mean range [-0.1329, +0.1465] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| T_minus_A_t | +0.1879 [-0.0548, +0.6278] | -0.1169 [-0.8690, +0.4407] | +0.0355 [-0.3770, +0.4047] |


T_minus_A_t, vidstg: leave-one-source-out mean range [-0.0210, +0.2123] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.


T_minus_A_t, hc2: leave-one-source-out mean range [-0.2620, +0.1973] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_C_v | +0.1668 [-0.0108, +0.5055] | -0.0237 [-0.4660, +0.3388] | +0.0715 [-0.1780, +0.3306] |


CT_minus_C_v, vidstg: leave-one-source-out mean range [+0.0009, +0.1777] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_C_v, hc2: leave-one-source-out mean range [-0.1346, +0.1506] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| C_newacquisition_minus_C_v | -0.0448 [-0.1003, -0.0026] | +0.0081 [-0.0141, +0.0438] | -0.0183 [-0.0488, +0.0095] |


C_newacquisition_minus_C_v, vidstg: leave-one-source-out mean range [-0.0477, -0.0238] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


C_newacquisition_minus_C_v, hc2: leave-one-source-out mean range [-0.0076, +0.0120] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_newacquisition_minus_CT_v | -0.0453 [-0.1015, -0.0026] | +0.0094 [-0.0126, +0.0449] | -0.0179 [-0.0487, +0.0099] |


CT_newacquisition_minus_CT_v, vidstg: leave-one-source-out mean range [-0.0482, -0.0241] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


CT_newacquisition_minus_CT_v, hc2: leave-one-source-out mean range [-0.0063, +0.0133] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.

| C_minus_twoUniform5_v | +0.0192 [-0.0093, +0.0573] | +0.0313 [+0.0052, +0.0667] | +0.0253 [+0.0047, +0.0496] |


C_minus_twoUniform5_v, vidstg: leave-one-source-out mean range [+0.0075, +0.0241] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


C_minus_twoUniform5_v, hc2: leave-one-source-out mean range [+0.0180, +0.0341] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_twoUniform5_newtime_v | +0.0193 [-0.0093, +0.0577] | +0.0305 [+0.0048, +0.0660] | +0.0249 [+0.0045, +0.0495] |


CT_minus_twoUniform5_newtime_v, vidstg: leave-one-source-out mean range [+0.0074, +0.0242] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_twoUniform5_newtime_v, hc2: leave-one-source-out mean range [+0.0171, +0.0333] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_newacquisition_minus_twoUniform5_newtime_v | -0.0259 [-0.0594, -0.0016] | +0.0398 [-0.0020, +0.0952] | +0.0070 [-0.0200, +0.0377] |


CT_newacquisition_minus_twoUniform5_newtime_v, vidstg: leave-one-source-out mean range [-0.0280, -0.0148] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


CT_newacquisition_minus_twoUniform5_newtime_v, hc2: leave-one-source-out mean range [+0.0210, +0.0447] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.

| R_acquisition_new_minus_R_acquisition_old_v | -0.0448 [-0.1003, -0.0026] | +0.0081 [-0.0141, +0.0438] | -0.0183 [-0.0488, +0.0095] |


R_acquisition_new_minus_R_acquisition_old_v, vidstg: leave-one-source-out mean range [-0.0477, -0.0238] pp; sign-changing removals 0/32. The source with largest influence and all source means remain in CONTRASTS.json.


R_acquisition_new_minus_R_acquisition_old_v, hc2: leave-one-source-out mean range [-0.0076, +0.0120] pp; sign-changing removals 1/32. The source with largest influence and all source means remain in CONTRASTS.json.


## round1 / confirm: corruption-all source macro incremental vIoU (pp)

| Arm | VidSTG, mean [95% paired source CI] | HC2, mean [95% paired source CI] |
|---|---:|---:|

| A | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

| U_select | +0.0526 [+0.0070, +0.1082] | +0.0645 [-0.1171, +0.2763] |

| R_select | +0.0617 [+0.0093, +0.1245] | +0.0944 [-0.0637, +0.2937] |

| R_temp | +0.2877 [+0.0163, +0.6493] | -0.0511 [-0.2714, +0.1079] |

| Specific_temp | -0.0934 [-0.2485, +0.0150] | -0.1195 [-0.3263, +0.0078] |


vidstg: Frozen/A corruption vIoU 32.9534/32.6465%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- U_select: gross gain/loss 0.0573/0.0046 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0564 [+0.0033, +0.1196]; two order means +0.0480, +0.0573 pp.

- R_select: gross gain/loss 0.0667/0.0050 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0612 [+0.0032, +0.1300]; two order means +0.0408, +0.0827 pp.

- R_temp: gross gain/loss 0.2991/0.0114 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.2627 [-0.0236, +0.6339]; two order means +0.0835, +0.4919 pp.

- Specific_temp: gross gain/loss 0.0145/0.1080 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean -0.1678 [-0.4809, +0.0171]; two order means -0.1264, -0.0605 pp.


Routed selected a better existing tube but temporary SGD harmed the output in **0** corrupt expert arrivals. This diagnosis distinguishes expert preference from parameter execution; it is not an online GT filter.


hc2: Frozen/A corruption vIoU 26.4778/26.7630%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- U_select: gross gain/loss 0.1340/0.0695 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0873 [-0.0675, +0.2817]; two order means -0.0141, +0.1431 pp.

- R_select: gross gain/loss 0.1314/0.0371 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.1257 [-0.0077, +0.3081]; two order means +0.0465, +0.1422 pp.

- R_temp: gross gain/loss 0.0433/0.0944 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0450 [+0.0007, +0.1192]; two order means +0.0026, -0.1048 pp.

- Specific_temp: gross gain/loss 0.0049/0.1244 pp; >5pp harmful arrivals 1; .3 rescue/destroy 0/0; clean -0.1317 [-0.3740, +0.0115]; two order means -0.0122, -0.2268 pp.


Routed selected a better existing tube but temporary SGD harmed the output in **11** corrupt expert arrivals. This diagnosis distinguishes expert preference from parameter execution; it is not an online GT filter.


### Paired differences between alternatives (corrupt-all, pp)

| Contrast | Vid | HC2 | Equal-dataset pooled [95% CI] |
|---|---:|---:|---:|

| R_select_minus_A_v | +0.0617 [+0.0093, +0.1245] | +0.0944 [-0.0637, +0.2937] | +0.0780 [-0.0046, +0.1810] |


R_select_minus_A_v, vidstg: leave-one-source-out mean range [+0.0411, +0.0688] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


R_select_minus_A_v, hc2: leave-one-source-out mean range [+0.0111, +0.1402] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| R_temp_minus_A_v | +0.2877 [+0.0163, +0.6493] | -0.0511 [-0.2714, +0.1079] | +0.1183 [-0.0585, +0.3168] |


R_temp_minus_A_v, vidstg: leave-one-source-out mean range [+0.1619, +0.3134] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


R_temp_minus_A_v, hc2: leave-one-source-out mean range [-0.0928, +0.0433] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.

| Specific_temp_minus_A_v | -0.0934 [-0.2485, +0.0150] | -0.1195 [-0.3263, +0.0078] | -0.1065 [-0.2364, -0.0066] |


Specific_temp_minus_A_v, vidstg: leave-one-source-out mean range [-0.1090, -0.0330] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


Specific_temp_minus_A_v, hc2: leave-one-source-out mean range [-0.1320, -0.0211] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| R_select_minus_U_select_v | +0.0091 [-0.0302, +0.0496] | +0.0298 [-0.0202, +0.0824] | +0.0195 [-0.0125, +0.0539] |


R_select_minus_U_select_v, vidstg: leave-one-source-out mean range [-0.0045, +0.0237] pp; sign-changing removals 2/16. The source with largest influence and all source means remain in CONTRASTS.json.


R_select_minus_U_select_v, hc2: leave-one-source-out mean range [+0.0094, +0.0426] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| R_select_minus_R_temp_v | -0.2260 [-0.5308, -0.0044] | +0.1455 [+0.0232, +0.2904] | -0.0403 [-0.2015, +0.0962] |


R_select_minus_R_temp_v, vidstg: leave-one-source-out mean range [-0.2454, -0.1208] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


R_select_minus_R_temp_v, hc2: leave-one-source-out mean range [+0.0969, +0.1551] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| Specific_temp_minus_R_temp_v | -0.3811 [-0.8099, -0.0401] | -0.0684 [-0.1539, -0.0056] | -0.2248 [-0.4432, -0.0478] |


Specific_temp_minus_R_temp_v, vidstg: leave-one-source-out mean range [-0.4183, -0.2256] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


Specific_temp_minus_R_temp_v, hc2: leave-one-source-out mean range [-0.0756, -0.0346] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


## round2 / confirm: corruption-all source macro incremental vIoU (pp)

| Arm | VidSTG, mean [95% paired source CI] | HC2, mean [95% paired source CI] |
|---|---:|---:|

| A | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

| C | +0.0617 [+0.0093, +0.1245] | +0.0944 [-0.0637, +0.2937] |

| T | +0.1570 [-0.0002, +0.4422] | -0.0510 [-0.1393, +0.0000] |

| CT | +0.2199 [+0.0214, +0.5224] | +0.0441 [-0.1318, +0.2604] |

| C_newacquisition | +0.0732 [+0.0102, +0.1539] | +0.0815 [-0.0867, +0.2851] |

| CT_newacquisition | +0.2314 [+0.0278, +0.5364] | +0.0217 [-0.1891, +0.2550] |

| twoUniform5 | +0.0584 [-0.0024, +0.1272] | +0.0996 [-0.0734, +0.3036] |

| twoUniform5_newtime | +0.2157 [+0.0351, +0.5119] | +0.0494 [-0.1391, +0.2652] |

| R_acquisition_old | +0.0617 [+0.0093, +0.1245] | +0.0944 [-0.0637, +0.2937] |

| R_acquisition_new | +0.0732 [+0.0102, +0.1539] | +0.0815 [-0.0867, +0.2851] |


vidstg: Frozen/A corruption vIoU 32.9534/32.6465%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- C: gross gain/loss 0.0667/0.0050 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0612 [+0.0032, +0.1300]; two order means +0.0408, +0.0827 pp.

- T: gross gain/loss 0.1572/0.0001 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean -0.0020 [-0.0059, +0.0000]; two order means +0.3141, +0.0000 pp.

- CT: gross gain/loss 0.2250/0.0051 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0593 [-0.0002, +0.1294]; two order means +0.3572, +0.0827 pp.

- C_newacquisition: gross gain/loss 0.0781/0.0050 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0704 [+0.0034, +0.1548]; two order means +0.0425, +0.1039 pp.

- CT_newacquisition: gross gain/loss 0.2364/0.0050 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0685 [+0.0001, +0.1535]; two order means +0.3589, +0.1039 pp.

- twoUniform5: gross gain/loss 0.0713/0.0128 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0495 [-0.0114, +0.1168]; two order means +0.0323, +0.0846 pp.

- twoUniform5_newtime: gross gain/loss 0.2269/0.0112 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0476 [-0.0145, +0.1153]; two order means +0.3469, +0.0846 pp.

- R_acquisition_old: gross gain/loss 0.0667/0.0050 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0612 [+0.0032, +0.1300]; two order means +0.0408, +0.0827 pp.

- R_acquisition_new: gross gain/loss 0.0781/0.0050 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0704 [+0.0034, +0.1548]; two order means +0.0425, +0.1039 pp.


Readout chain, same sources: Frozen to A native +0.0734 [-0.2520, +0.4201] pp; A native to original Fast A -0.3803 [-0.7433, -0.0788] pp. A native includes all inherited state effects; this is a readout decomposition, not a unique causal diagnosis of spatial or temporal learning.


Temporal actual changed-observation fraction: 0.9950. Old/new selected candidate changed in 4/40 expert arrivals. Two views use the existing observed STVG frame grid, with nearest-frame sampling; they are different image observations rather than original full-rate independent videos. Stable agreement can be wrong.

- U: event-frame precision 32.500%; candidate pairwise accuracy 65.139% (mean qualified arrivals); empty 0/40.

- Rnew: event-frame precision 52.000%; candidate pairwise accuracy 70.000% (mean qualified arrivals); empty 4/40.

- U2: event-frame precision 32.500%; candidate pairwise accuracy 70.556% (mean qualified arrivals); empty 0/40.


Old/new temporal replacements 40/39; tIoU-worsening replacements versus the same A native interval 29/26; vIoU-worsening replacements 29/26. These are expert-arrival counts, not error probabilities on independent videos.


New versus old Routed event-frame precision: +1.0000 [-6.5000, +8.0000] pp; fixed-time tube utility remains a separate paired contrast below.


hc2: Frozen/A corruption vIoU 26.4778/26.7630%. Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. This establishes a current-readout increment on top of online A, not new future parameter transfer.

- C: gross gain/loss 0.1314/0.0371 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.1257 [-0.0077, +0.3081]; two order means +0.0465, +0.1422 pp.

- T: gross gain/loss 0.0134/0.0644 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0000 [+0.0000, +0.0000]; two order means -0.0374, -0.0646 pp.

- CT: gross gain/loss 0.1400/0.0959 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.1257 [-0.0077, +0.3081]; two order means +0.0088, +0.0794 pp.

- C_newacquisition: gross gain/loss 0.1356/0.0541 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0260 [-0.1938, +0.2509]; two order means +0.0546, +0.1083 pp.

- CT_newacquisition: gross gain/loss 0.1442/0.1225 pp; >5pp harmful arrivals 1; .3 rescue/destroy 0/0; clean +0.0260 [-0.1938, +0.2509]; two order means +0.0169, +0.0265 pp.

- twoUniform5: gross gain/loss 0.1442/0.0446 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.1129 [-0.0390, +0.3034]; two order means +0.0505, +0.1487 pp.

- twoUniform5_newtime: gross gain/loss 0.1528/0.1034 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.1129 [-0.0390, +0.3034]; two order means +0.0128, +0.0859 pp.

- R_acquisition_old: gross gain/loss 0.1314/0.0371 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.1257 [-0.0077, +0.3081]; two order means +0.0465, +0.1422 pp.

- R_acquisition_new: gross gain/loss 0.1356/0.0541 pp; >5pp harmful arrivals 0; .3 rescue/destroy 0/0; clean +0.0260 [-0.1938, +0.2509]; two order means +0.0546, +0.1083 pp.


Readout chain, same sources: Frozen to A native -0.0419 [-0.1834, +0.0992] pp; A native to original Fast A +0.3272 [-0.5304, +1.2935] pp. A native includes all inherited state effects; this is a readout decomposition, not a unique causal diagnosis of spatial or temporal learning.


Temporal actual changed-observation fraction: 0.7949. Old/new selected candidate changed in 5/40 expert arrivals. Two views use the existing observed STVG frame grid, with nearest-frame sampling; they are different image observations rather than original full-rate independent videos. Stable agreement can be wrong.

- U: event-frame precision 32.500%; candidate pairwise accuracy 66.374% (mean qualified arrivals); empty 0/40.

- Rnew: event-frame precision 74.500%; candidate pairwise accuracy 65.424% (mean qualified arrivals); empty 0/40.

- U2: event-frame precision 45.000%; candidate pairwise accuracy 67.836% (mean qualified arrivals); empty 0/40.


Old/new temporal replacements 38/38; tIoU-worsening replacements versus the same A native interval 15/18; vIoU-worsening replacements 14/17. These are expert-arrival counts, not error probabilities on independent videos.


New versus old Routed event-frame precision: -0.5714 [-8.5714, +9.1429] pp; fixed-time tube utility remains a separate paired contrast below.


### Paired differences between alternatives (corrupt-all, pp)

| Contrast | Vid | HC2 | Equal-dataset pooled [95% CI] |
|---|---:|---:|---:|

| A_minus_Frozen_v | -0.3069 [-0.6228, -0.0066] | +0.2853 [-0.5561, +1.2271] | -0.0108 [-0.4626, +0.4660] |


A_minus_Frozen_v, vidstg: leave-one-source-out mean range [-0.3762, -0.2281] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


A_minus_Frozen_v, hc2: leave-one-source-out mean range [-0.0771, +0.5766] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.

| C_minus_Frozen_v | -0.2452 [-0.5457, +0.0423] | +0.3797 [-0.5507, +1.5050] | +0.0672 [-0.4315, +0.6276] |


C_minus_Frozen_v, vidstg: leave-one-source-out mean range [-0.3103, -0.1772] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


C_minus_Frozen_v, hc2: leave-one-source-out mean range [-0.0660, +0.6758] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_Frozen_v | -0.0870 [-0.4502, +0.2742] | +0.3294 [-0.5922, +1.4548] | +0.1212 [-0.3818, +0.6839] |


CT_minus_Frozen_v, vidstg: leave-one-source-out mean range [-0.1821, -0.0084] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_Frozen_v, hc2: leave-one-source-out mean range [-0.1196, +0.6222] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.

| C_minus_A_v | +0.0617 [+0.0093, +0.1245] | +0.0944 [-0.0637, +0.2937] | +0.0780 [-0.0046, +0.1810] |


C_minus_A_v, vidstg: leave-one-source-out mean range [+0.0411, +0.0688] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


C_minus_A_v, hc2: leave-one-source-out mean range [+0.0111, +0.1402] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_A_v | +0.2199 [+0.0214, +0.5224] | +0.0441 [-0.1318, +0.2604] | +0.1320 [-0.0110, +0.3148] |


CT_minus_A_v, vidstg: leave-one-source-out mean range [+0.0861, +0.2377] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_A_v, hc2: leave-one-source-out mean range [-0.0425, +0.1002] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.

| T_minus_A_v | +0.1570 [-0.0002, +0.4422] | -0.0510 [-0.1393, +0.0000] | +0.0530 [-0.0443, +0.2033] |


T_minus_A_v, vidstg: leave-one-source-out mean range [+0.0310, +0.1676] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


T_minus_A_v, hc2: leave-one-source-out mean range [-0.0544, -0.0147] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| T_minus_A_t | +0.2473 [-0.0031, +0.6976] | -0.1306 [-0.3217, +0.0000] | +0.0584 [-0.1206, +0.3014] |


T_minus_A_t, vidstg: leave-one-source-out mean range [+0.0489, +0.2655] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


T_minus_A_t, hc2: leave-one-source-out mean range [-0.1393, -0.0535] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_C_v | +0.1582 [-0.0002, +0.4452] | -0.0502 [-0.1380, +0.0000] | +0.0540 [-0.0440, +0.2059] |


CT_minus_C_v, vidstg: leave-one-source-out mean range [+0.0314, +0.1688] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_C_v, hc2: leave-one-source-out mean range [-0.0536, -0.0136] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| C_newacquisition_minus_C_v | +0.0114 [-0.0000, +0.0332] | -0.0129 [-0.0393, +0.0082] | -0.0007 [-0.0160, +0.0153] |


C_newacquisition_minus_C_v, vidstg: leave-one-source-out mean range [+0.0010, +0.0122] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


C_newacquisition_minus_C_v, hc2: leave-one-source-out mean range [-0.0182, -0.0037] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_newacquisition_minus_CT_v | +0.0114 [-0.0000, +0.0332] | -0.0224 [-0.0625, +0.0082] | -0.0055 [-0.0277, +0.0145] |


CT_newacquisition_minus_CT_v, vidstg: leave-one-source-out mean range [+0.0010, +0.0122] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


CT_newacquisition_minus_CT_v, hc2: leave-one-source-out mean range [-0.0284, -0.0058] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| C_minus_twoUniform5_v | +0.0033 [-0.0445, +0.0569] | -0.0052 [-0.0304, +0.0160] | -0.0010 [-0.0276, +0.0272] |


C_minus_twoUniform5_v, vidstg: leave-one-source-out mean range [-0.0171, +0.0195] pp; sign-changing removals 2/16. The source with largest influence and all source means remain in CONTRASTS.json.


C_minus_twoUniform5_v, hc2: leave-one-source-out mean range [-0.0117, +0.0040] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_minus_twoUniform5_newtime_v | +0.0042 [-0.0445, +0.0601] | -0.0052 [-0.0304, +0.0160] | -0.0005 [-0.0274, +0.0284] |


CT_minus_twoUniform5_newtime_v, vidstg: leave-one-source-out mean range [-0.0171, +0.0205] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.


CT_minus_twoUniform5_newtime_v, hc2: leave-one-source-out mean range [-0.0117, +0.0040] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.

| CT_newacquisition_minus_twoUniform5_newtime_v | +0.0156 [-0.0375, +0.0760] | -0.0276 [-0.0738, +0.0041] | -0.0060 [-0.0402, +0.0288] |


CT_newacquisition_minus_twoUniform5_newtime_v, vidstg: leave-one-source-out mean range [-0.0049, +0.0318] pp; sign-changing removals 1/16. The source with largest influence and all source means remain in CONTRASTS.json.


CT_newacquisition_minus_twoUniform5_newtime_v, hc2: leave-one-source-out mean range [-0.0317, -0.0097] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.

| R_acquisition_new_minus_R_acquisition_old_v | +0.0114 [-0.0000, +0.0332] | -0.0129 [-0.0393, +0.0082] | -0.0007 [-0.0160, +0.0153] |


R_acquisition_new_minus_R_acquisition_old_v, vidstg: leave-one-source-out mean range [+0.0010, +0.0122] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


R_acquisition_new_minus_R_acquisition_old_v, hc2: leave-one-source-out mean range [-0.0182, -0.0037] pp; sign-changing removals 0/16. The source with largest influence and all source means remain in CONTRASTS.json.


## Measured incremental execution cost

| Work | Suffix replays / new expert invocations | Backwards / live controls | Process wall seconds |
|---|---:|---:|---:|

| vidstg/round1/search | 1243 | 180 | 185.128 |

| hc2/round1/search | 1244 | 182 | 180.042 |

| vidstg/round2/search | 1440 | 276 | 200.334 |

| hc2/round2/search | 1438 | 275 | 186.847 |

| vidstg/round1/confirm | 624 | 95 | 94.998 |

| hc2/round1/confirm | 624 | 96 | 91.870 |

| vidstg/round2/confirm | 720 | 140 | 101.801 |

| hc2/round2/confirm | 720 | 144 | 98.198 |

| temporal/search | 165 | 2 | 309.257 |

| temporal/confirm | 70 | 2 | 187.241 |

| round1/search | 0 | None | 203.359 |

| round2/search | 152 | 1 | 353.370 |

| round1/confirm | 55 | 1 | 195.355 |

| round2/confirm | 85 | 0 | 208.059 |


Total current-readout suffix replays 8053, backwards 1388, new spatial calls 292, new temporal calls 235. Successful worker process wall sums to 43.264 minutes. This includes loading, I/O and cache readback, excludes rejected attempts, CPU scoring, publication and the separately preserved three-setting TF32 head diagnostic. The summed wall is not a deployment per-query benchmark or pure GPU kernel time. Historical backbone/Uniform/phase0 inference is cached, so incremental new-call counts understate end-to-end deployment work.


## Interpretation and limitations

The two-Uniform5 comparator uses original endpoint Uniform5 for persistent learning and distinct bin-midpoint Uniform5 for the current correction, each with five frames. It matches the two observation requests, not a ten-frame merged teacher. Probes and A persistence are shared across readouts. Report both logical budget and actual unique uncached specialist invocations. All process wall times include loading/IO, not pure GPU kernels. Reference selection and outputs are never derived from GT.

Rank-RKL remains unchanged, including nonempty flat-reward entropy effects. Geometry compatibility is not a native tube policy. Current tie policy always falls back to center, even for top ties excluding center. Specific correction additionally requires both evidence branches nonempty. A direct tube selection result does not prove a generic Uniform/event-specific Routed semantic decomposition. Min-view temporal replacement does not imply localization correctness or better acquisition.

All anonymous arrival rows, candidate/reward diagnostics, positive and severe negative cases, order and clean controls, paired source-bootstrap intervals and resource receipts are published. Private captions, media, annotations, weights, H, parameter and gradient tensors are excluded. The HC frame-freeze donor binding failure was rejected by pixel equality before new inference. The UniversalVTG environment import and runtime revision ordering were repaired before affected new predictions. Its development cached-feature head control had a maximum Vid error .02586555 physical frames, confidence 2.77162e-5 and student candidate score 7.63083e-6, with identical winning candidate; HC development errors were zero. Every confirmation control error is separately retained in BARRIERS_AND_CONTROLS.json. Original phase0 remains the exact saved input. A <.1-frame/<1e-4/same-winner interface control is numerical, not a GT gate. Sa2VA original Uniform masks/boxes/text are bitwise reproduced before new calls. CPU diagnostic JSON serialization was fixed without changed scores. All original engineering records remain archived. CURRENT_METHOD and paused queues are unchanged.
