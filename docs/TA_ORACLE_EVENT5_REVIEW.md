# Fixed-A oracle ceilings and GT-event5 observation intervention

Both experiments completed. All values below are vIoU percentage points or absolute percent as labelled; intervals are 95% paired source bootstrap (10000 draws, seed 20261003). This is a GT-assisted diagnostic, not an unlabelled method score.

## Frozen setting and cohort

The predecessor is verified GitHub 3b3ebd297f621667ce03f1291d130b6fd71236db. Each dataset reuses its exact 32 development and 16 within-batch disjoint confirmation sources, one query/source, two fixed orders, clean plus five 5% corruptions and 25% expert arrivals. All sources have historical exposure. Confirmation is a descriptive existing cohort, with no new parameter selection. A is the Uniform persistent spatial learner plus original temporal Fast, rather than Frozen. Its actual pre-arrival states, nine probes and persistent K1/K8 trajectories are read-only.

Vid: lr .033761698432507946, teacher .34902548789596055, K1; HC: lr .006097133675874025, teacher 1, K8. Both rho .05, student 1, D4 and 1792 parameters. Same-domain official checkpoints, original Paper48 observed grid and exact old corruption pixels. GT interval substitution affects evaluation only. No GT is supplied to the decoder.

## Experiment 1: complete-stream ideal readouts

GT space is the true dense annotated tube, rather than sparse GT interpolation. Joint GT scores exactly 100% at every arrival under the literal official dense metric. A-box trajectories are interpolated only inside original sampled support. HC endpoint conventions remain literal. These readout interventions are not additive causal contributions.

| Split / dataset / subset | Cells / sources | A | GT time | GT space | Joint GT |
|---|---:|---:|---:|---:|---:|
| search / vidstg / all | 320 / 32 | +17.4529 [+11.1003, +24.5945] | +34.7422 [+25.6260, +43.7910] | +41.4412 [+30.4881, +52.4565] | +100.0000 [+100.0000, +100.0000] |
| search / vidstg / expert | 80 / 16 | +17.8323 [+7.0981, +30.2077] | +30.9726 [+18.4782, +44.1666] | +42.3285 [+24.4749, +60.1092] | +100.0000 [+100.0000, +100.0000] |
| search / vidstg / nonexpert | 240 / 32 | +16.8797 [+10.8011, +23.5532] | +34.3729 [+25.2066, +43.5114] | +40.0350 [+29.2018, +51.0349] | +100.0000 [+100.0000, +100.0000] |
| search / hc2 / all | 320 / 32 | +30.9637 [+24.1821, +37.9082] | +53.3610 [+47.2018, +59.3026] | +55.5404 [+46.7520, +64.1399] | +100.0000 [+100.0000, +100.0000] |
| search / hc2 / expert | 80 / 14 | +30.1973 [+20.4351, +41.4661] | +53.6170 [+44.4194, +62.2079] | +53.7567 [+41.8364, +65.3187] | +100.0000 [+100.0000, +100.0000] |
| search / hc2 / nonexpert | 240 / 30 | +31.2710 [+23.9776, +38.7990] | +53.4377 [+46.6592, +59.8003] | +55.8718 [+46.3314, +65.0712] | +100.0000 [+100.0000, +100.0000] |
| confirm / vidstg / all | 160 / 16 | +32.6465 [+20.5681, +45.3322] | +55.6330 [+43.1619, +67.3234] | +56.2971 [+40.3486, +71.6771] | +100.0000 [+100.0000, +100.0000] |
| confirm / vidstg / expert | 40 / 8 | +22.0480 [+10.0314, +34.8736] | +49.6024 [+34.1195, +63.7588] | +46.0983 [+22.7644, +69.4825] | +100.0000 [+100.0000, +100.0000] |
| confirm / vidstg / nonexpert | 120 / 16 | +33.2118 [+21.0128, +45.9157] | +55.8919 [+43.3353, +67.6331] | +57.0084 [+40.5689, +72.7157] | +100.0000 [+100.0000, +100.0000] |
| confirm / hc2 / all | 160 / 16 | +26.7630 [+17.9558, +35.3714] | +44.4055 [+31.0727, +56.7562] | +59.2502 [+49.4809, +68.9110] | +100.0000 [+100.0000, +100.0000] |
| confirm / hc2 / expert | 40 / 7 | +33.7574 [+21.2534, +46.0642] | +50.4652 [+35.5160, +64.5896] | +68.8674 [+50.3129, +83.4929] | +100.0000 [+100.0000, +100.0000] |
| confirm / hc2 / nonexpert | 120 / 15 | +25.0498 [+16.2773, +33.4338] | +42.4083 [+28.6138, +55.1697] | +58.5593 [+48.8883, +68.0798] | +100.0000 [+100.0000, +100.0000] |

## Candidate coverage and selection on expert arrivals

Only the already cached 288 expert arrivals have 8 temporal candidates and 9 spatial tubes; all 20736 combinations are evaluated on CPU. No support is generated for nonexpert arrivals. Candidate-native/center outputs and duplicates remain included. Expert source-macro means have different source/order membership from full-stream means; do not multiply them by 25% to estimate full-stream effects.

| Split / dataset | Temporal selection H | Temporal coverage H | Spatial selection H | Spatial coverage H | Joint over best single | T recovered |
|---|---:|---:|---:|---:|---:|---:|
| search / vidstg | +2.1197 [+0.4235, +4.3390] | +11.0206 [+5.0167, +18.1370] | +0.5505 [+0.2869, +0.8454] | +23.9457 [+12.2617, +36.8887] | +0.4865 [+0.1576, +0.8834] | +0.6458 [-0.0414, +1.9555] |
| search / hc2 | +5.5354 [+2.5993, +9.2671] | +17.8843 [+12.0898, +23.8647] | +0.5988 [+0.3821, +0.8256] | +22.9605 [+17.1176, +30.1106] | +0.5509 [+0.2774, +0.8928] | +0.2657 [-0.9542, +1.5666] |
| confirm / vidstg | +3.1790 [+1.7870, +4.4631] | +24.3754 [+9.2308, +41.7397] | +0.4014 [+0.1972, +0.6315] | +23.6488 [+10.1691, +38.3487] | +0.4212 [+0.1770, +0.6970] | +0.6282 [-0.0010, +1.6524] |
| confirm / hc2 | +3.5431 [+1.0239, +6.8607] | +13.1648 [+4.5320, +24.3654] | +0.7450 [+0.2876, +1.4308] | +34.3651 [+20.9954, +48.3496] | +0.3711 [+0.1885, +0.5827] | -0.1481 [-0.3370, +0.0000] |

H_temporal=GT-time−A=H_selection+H_coverage. H_spatial=GT-space−A=H_spatial_selection+H_spatial_coverage. The upper-bound identity holds cellwise under the official full-GT-span denominator. Joint increments and synergy, T remaining selection error, clean, nonexpert and order values and candidate uniqueness are preserved in anonymous rows/SUMMARY. This bounds current cached supports, not all possible temporal decoders or spatial interfaces.

search/vidstg: T recovers 30.47% of existing temporal selection headroom (ratio of macro means), zero-gap cells 50. Negative recovery is retained.
search/hc2: T recovers 4.80% of existing temporal selection headroom (ratio of macro means), zero-gap cells 24. Negative recovery is retained.
confirm/vidstg: T recovers 19.76% of existing temporal selection headroom (ratio of macro means), zero-gap cells 8. Negative recovery is retained.
confirm/hc2: T recovers -4.18% of existing temporal selection headroom (ratio of macro means), zero-gap cells 12. Negative recovery is retained.

## Experiment 2: event-only observations at fixed A state

GT-event is five fixed quantiles of existing observed frames strictly within the saved half-open GT interval. Fewer than five unique event frames means unsupported; no outside padding. All unsupported cells remain in Experiment 1. Primary comparisons below use the same eligible donor subset for all four observations; full-scheduled no-op sensitivity is also published. U/U2/R masks, actual reward selection and ordinary one-step outputs are cached. Only GT-event observes new images. Temporary offsets expire after this query, while persistent A remains unchanged. HC temporary one-step results do not establish its K8 sustained learning value.

| Split / dataset | Eligible / scheduled donors | U selected | U2 selected | R selected | GT-event selected | U update | U2 update | R update | GT-event update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| search / vidstg | 75 / 80 | +0.5705 [+0.1169, +1.1132] | +0.6232 [-0.0827, +1.2907] | +0.7478 [-0.0307, +1.5395] | +1.5095 [+0.4928, +3.0930] | +0.8703 [-0.2967, +2.1421] | +0.7150 [-1.2554, +2.9347] | +0.5370 [-1.1247, +2.1165] | +1.1660 [-0.5261, +2.7637] |
| search / hc2 | 80 / 80 | +0.3927 [-0.1868, +0.9484] | +0.6744 [+0.3281, +1.0877] | +0.8259 [+0.4985, +1.1947] | +0.6299 [+0.2289, +1.0434] | +0.1549 [+0.0251, +0.3057] | +0.2218 [+0.1086, +0.3498] | +0.1909 [+0.1126, +0.2722] | +0.1881 [+0.0619, +0.3333] |
| confirm / vidstg | 35 / 40 | +0.4314 [+0.1959, +0.6468] | +0.3073 [-0.2017, +0.6970] | +0.5430 [+0.2711, +0.7824] | +0.5863 [+0.2895, +0.8298] | +2.0038 [+0.4029, +3.6048] | +1.3300 [-0.2807, +2.9471] | +1.9159 [+0.6005, +3.3610] | +2.6975 [+1.5383, +3.9975] |
| confirm / hc2 | 40 / 40 | +0.2529 [-0.6861, +1.3589] | +0.5655 [-0.2733, +1.5384] | +0.4902 [-0.4183, +1.5243] | +0.3648 [-0.3537, +1.3541] | -0.1400 [-1.2188, +0.6208] | -0.0717 [-0.9808, +0.6065] | -0.1870 [-1.2218, +0.5820] | -0.1053 [-0.8061, +0.5205] |

The previous table reports selected-center and one-step-center increments evaluated at GT time. Spatial output alone changes; GT time is never fed through the decoder.

| Split / dataset / interval | GT-event−R selection | GT-event−U2 selection | GT-event−R update | GT-event−U2 update |
|---|---:|---:|---:|---:|
| search / vidstg / GT | +0.7616 [-0.6438, +2.8508] | +0.8862 [-0.3031, +2.8555] | +0.6291 [-1.3729, +2.9069] | +0.4510 [-1.8029, +2.9422] |
| search / vidstg / A | -0.0509 [-0.3274, +0.1902] | +0.0312 [-0.1236, +0.1943] | +0.0877 [-1.1523, +1.5504] | +0.1367 [-0.9923, +1.6651] |
| search / hc2 / GT | -0.1961 [-0.4611, +0.0300] | -0.0445 [-0.3086, +0.1481] | -0.0028 [-0.1051, +0.1025] | -0.0337 [-0.0948, +0.0176] |
| search / hc2 / A | -0.0930 [-0.2754, +0.0443] | +0.0392 [-0.0871, +0.1428] | -0.0307 [-0.1022, +0.0224] | -0.0046 [-0.0382, +0.0228] |
| confirm / vidstg / GT | +0.0432 [-0.0962, +0.2415] | +0.2790 [-0.1076, +0.9046] | +0.7816 [-0.0316, +1.7204] | +1.3675 [+0.3740, +2.3641] |
| confirm / vidstg / A | +0.0461 [-0.0647, +0.1861] | +0.0851 [-0.0577, +0.2832] | +0.4270 [-0.0351, +1.0946] | +0.6883 [+0.1390, +1.3295] |
| confirm / hc2 / GT | -0.1254 [-0.7322, +0.4644] | -0.2007 [-0.8650, +0.3999] | +0.0817 [-0.1800, +0.4535] | -0.0336 [-0.2875, +0.2220] |
| confirm / hc2 / A | -0.0943 [-0.5213, +0.3127] | -0.1183 [-0.6483, +0.3727] | +0.0691 [-0.1319, +0.3576] | +0.0129 [-0.1803, +0.2402] |

Eligible corrupted donors are Vid development 75/80 (15 distinct expert sources), Vid confirmation 35/40 (7), HC development 80/80 (14), HC confirmation 40/40 (7). The 12 unsupported donors include 10 corrupt and 2 clean Vid inputs; those remain in complete-stream Experiment 1 and the published scheduled no-op sensitivity.

Clean is a matched control, without a corruption-specific gain claim. At GT interval:

| Split / dataset | GT-event−R selection | GT-event−U2 selection | GT-event−R update | GT-event−U2 update |
|---|---:|---:|---:|---:|
| search / vidstg | +0.9540 [-0.1617, +2.8535] | +0.6467 [-0.6411, +2.6581] | +2.2223 [-0.9014, +6.0086] | +0.7935 [-1.5508, +3.5713] |
| search / hc2 | -0.3445 [-0.8806, -0.0153] | -0.0872 [-0.3415, +0.1049] | -0.1010 [-0.3537, +0.0655] | -0.0454 [-0.1607, +0.0283] |
| confirm / vidstg | +0.0860 [-0.1085, +0.3665] | +0.4155 [-0.0201, +1.0222] | +0.6605 [-0.3064, +2.2334] | +1.4304 [-0.0020, +3.2588] |
| confirm / hc2 | -0.2794 [-0.8745, +0.1290] | -0.1993 [-0.8143, +0.3376] | -0.2605 [-0.6414, +0.0037] | +0.0784 [-0.2226, +0.4846] |

## Evidence and execution diagnostics

| Split / dataset / observation | Empty requests | Valid/scorable event frames | Event GT IoU, valid-only | Event GT IoU, empty=0 observed | Better selected tube but harmful update, GT / A |
|---|---:|---:|---:|---:|---:|
| search / vidstg / U | 0/75 | 130/130 | 63.05% | 63.05% | 8 / 8 |
| search / vidstg / U2 | 0/75 | 160/160 | 58.21% | 58.21% | 12 / 12 |
| search / vidstg / R | 1/75 | 187/187 | 64.00% | 61.69% | 10 / 14 |
| search / vidstg / GT_event | 0/75 | 373/373 | 60.62% | 60.30% | 17 / 12 |
| search / hc2 / U | 5/80 | 75/75 | 52.77% | 46.56% | 6 / 2 |
| search / hc2 / U2 | 1/80 | 101/101 | 69.52% | 56.17% | 6 / 0 |
| search / hc2 / R | 3/80 | 191/191 | 64.81% | 52.90% | 8 / 3 |
| search / hc2 / GT_event | 5/80 | 358/358 | 58.60% | 52.45% | 6 / 1 |
| confirm / vidstg / U | 0/35 | 50/50 | 63.58% | 48.90% | 1 / 0 |
| confirm / vidstg / U2 | 0/35 | 41/41 | 60.49% | 41.34% | 1 / 2 |
| confirm / vidstg / R | 1/35 | 80/80 | 73.85% | 57.92% | 0 / 0 |
| confirm / vidstg / GT_event | 0/35 | 158/158 | 66.14% | 59.72% | 0 / 2 |
| confirm / hc2 / U | 0/40 | 64/64 | 61.39% | 60.44% | 4 / 4 |
| confirm / hc2 / U2 | 0/40 | 90/90 | 67.84% | 67.84% | 8 / 9 |
| confirm / hc2 / R | 0/40 | 150/150 | 67.57% | 67.57% | 11 / 11 |
| confirm / hc2 / GT_event | 0/40 | 200/200 | 62.15% | 62.15% | 6 / 6 |

Evidence-quality ratios are frame-count weighted within each observation; task outcomes and pairwise contrasts use source-macro bootstrap. Valid-only IoU excludes empty masks and no-event observation positions, so it cannot be interpreted without the denominators. Empty request/frame rates, event annotation gaps, gross gains/losses, strict .3/.5 correctness changes, >5pp harms and both positive/negative cases remain in machine-readable outputs. A good selected tube is a relative gain; it is not automatically threshold-correct.

## Root decision and limits

**Principal next research variable: temporal candidate position and boundary coverage.** This is a recommendation only; no new candidate generator, method or third experiment was executed. Keep A, the temporal scorer, observation budget and spatial update rule fixed when testing that variable. The rationale is the repeated, positive coverage gap on both datasets and both existing cohorts, rather than selecting a slightly higher-mean combination.
On corrupt expert arrivals, temporal coverage versus selection is 11.021 vs 2.120 pp (Vid development), 17.884 vs 5.535 (HC development), 24.375 vs 3.179 (Vid confirmation), and 13.165 vs 3.543 (HC confirmation). Coverage accounts for 83.9%, 76.4%, 88.5% and 78.8% of temporal headroom respectively, as ratios of source-macro means. Their coverage intervals are all above zero. The existing T recovers only .646, .266, .628 and −.148 pp respectively; it also leaves selection errors, but a better selector over exactly these eight intervals cannot close the larger coverage gap.
The spatial branch also has a candidate-capacity limitation. At original A time the nine-tube oracle adds only .551/.599/.401/.745 pp; even at GT time the same probes offer 2.034/1.212/.835/.950 pp on the eligible matched development/confirmation sets. GT-space ideal readout headroom is much larger. These are limits of the current fixed probes and readout, not an upper bound on future SGD, spatial representations or new candidates. The joint oracle adds only .487/.551/.421/.371 pp above the better single oracle, with no large joint-only reservoir under this support. Spatial candidate generation remains a documented secondary bottleneck, rather than being declared adequate.
Correct event timing alone does not establish a useful observation module. Vid development GT-event−R/U2 selection at GT time is +.762/+.886 pp with both paired intervals crossing zero; one donor can reverse those selection differences under leave-one-source-out analysis. Vid confirmation differences are only +.043/+.279 pp, again crossing zero. HC selection differences are negative in both cohorts, with intervals crossing zero. Thus the required condition of robustly better selection and actual execution against both controls on both datasets was not met.
Retain the positive execution result: Vid confirmation GT-event−U2 one-step update is +1.368 pp [+.374,+2.364] at GT time and +.688 [+.139,+1.330] at original A time. Against R the corresponding +.782 [−.032,+1.720] and +.427 [−.035,+1.095] remain uncertain. HC has no stable corresponding improvement. This gives a specific Vid execution signal, without proving shared routing value, dependence on corruption, or HC K8 transfer.
Evidence validity and execution remain separate. Putting every observation inside the event increases event-frame coverage, but does not guarantee a better localized expert box or candidate ranking. Conditional valid event-box IoU is lower for GT-event than R in all four corrupt cohorts; those are different observed frame sets, so the descriptive ratios alone are not a paired causal estimate of expert degradation. At GT time, GT-event selects a relatively better tube yet its one-step update harms the center in 17/75 Vid development donors, 6/80 HC development donors, 0/35 Vid confirmation donors and 6/40 HC confirmation donors. Eight Vid-development updates exceed 5 pp harm, whereas the other three cohorts have zero such GT-event harms. Correct event acquisition therefore does not remove preference-to-update mismatch.
Order sensitivity, clean controls and source influence limit extrapolation. Vid development GT-event−R selection at GT time is +1.784/−.133 pp across the two orders, versus +.063/+.017 on confirmation; HC confirmation GT-event selection itself is −.129/+.653 pp across orders. HC development clean GT-event−R selection is −.345 pp [−.881,−.015]. All clean outcomes, order values, unsupported no-op sensitivities, positive/negative cases and source influence are retained. Expert subsets have only 7–16 distinct sources; paired intervals are descriptive on historically exposed development data, with many correlated endpoints rather than new held-out benchmark claims.
A minimal later test would keep the eight-interval budget and all downstream modules fixed, predefine a candidate allocation that spans distinct locations and boundary scales, then compare existing versus replacement support before any scorer change. That later test requires separate authorization and a prediction seal before GT. This report does not resume full-query evaluation or choose online thresholds from GT.

## Actual incremental resources and verification

GT-event logical donors 288; eligible 276; unsupported 12; actual new Sa2VA calls 67; input-matched reused requests 209. Extra specialist controls 0. Suffix replays 646; backwards 540 (including Uniform bitwise gradient controls). New TA-STVG backbone calls and regenerated probes are both zero. Sum of successful model worker process wall 14.013 minutes, including model loading and I/O; not pure GPU kernel time or deployment end-to-end latency.

The old confirmation U caches retained exact first-step states and gradients but omitted 96 post-update tubes. These were reconstructed with 96 suffix-only forwards from the saved states plus four bitwise pre-state prediction controls. No learning or raw GT was used in those model calls. This additional readout work is included in the resource totals; it is not a new online run. Valid development results emitted before this cache-contract repair are retained and byte-compared against the completed scoring output.

Those U readouts were reconstructed after the first, failed CPU scoring attempt had read diagnostic GT. All U states and gradients predate that exposure, were bitwise fixed, and no U parameter/output was selected with GT. Both U barriers precede the final complete CPU scoring. Serialization and revision-order failures, as well as the missing-readout cache-contract repair, are recorded in ENGINEERING_HISTORY with original saved sources and valid partial outputs; no scientific rule changed.

Experiment1 official scalar checks and Experiment2 reward/KL/SGD/state seals are recorded in ROOT_AUDIT. The A trajectory and CURRENT_METHOD remain unchanged. All derived anonymous rows, candidate matrices, paired uncertainty and negative cases are exported; captions/media/annotations/GT coordinates, weights and state/gradient/H tensors stay private. No new loss, Specific correction, temporal view, full stream, parameter search or production promotion is authorized by this report.
