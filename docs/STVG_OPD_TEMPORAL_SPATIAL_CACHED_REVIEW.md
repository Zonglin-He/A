# Temporal × spatial cached factor study and Q0 mean readout

Existing Actionness Projection improves OPD under the two native source checkpoints. The interaction with OPD versus frozen Native is not established: both primary interaction intervals span zero. At projected time, the OPD-versus-output-capped-Direct contrast also spans zero in both primary panels; the two-order Vid secondary panel favors capped Direct. These cached measurements do not establish a new joint method or justify Query-level OPD or a temperature search.

## Actual scope and input availability

All 1,792 Common-State cells and 1,536 distinct packed inputs were checked before scoring. Exactly 128 clean cells match the complete P6 temporal captures in source-model state SHA, pixels, physical frame IDs, corruption, Native interval and frozen boxes. They represent the original 64 P6 parents (32 per target), repeated under two orders. The primary panel is their original order1 (64 cells); the secondary is both orders (128 cells), not new independent parents. All 192 panel rows / 1,536 logical eight-way readouts were hash-sealed before the new offline GT scoring.

The remaining 1,664 O cells lack exact two-offset frozen temporal logits, raw actionness and physical-offset frame records. That full extension was not executed. No time signal was borrowed across source checkpoints or corruptions; no new GPU, backbone, optimizer, teacher or model call was made.

T0 is frozen Native WHEN. T1 exactly replays original P6 Actionness Projection (HC center fraction 1; Vid 0.5; original prior 0.1 and epsilon 1e-6). Native is the frozen-source spatial output. OPD, Direct and Capped Direct share the original Full initial state, observations and their existing saved fits. Native-versus-adapted differences therefore include existing state history; the three adapted arms are conditional on Full history. The 576 original P6 Native/projection/SpatialOPD scalar controls were reproduced.

## Complete eight-arm mean vIoU (%)

| Panel | Target | Spatial readout | Native time T0 | Projection T1 | T1−T0 (pp), 95% CI | >5 pp temporal-harm parents |
|---|---|---|---:|---:|---|---:|
| P6_primary | hc2 | Native | 22.663 | 26.417 | +3.755 [+0.620, +6.713] | 4 |
| P6_primary | hc2 | OPD | 22.500 | 26.323 | +3.823 [+0.336, +7.169] | 4 |
| P6_primary | hc2 | Direct | 20.449 | 23.399 | +2.950 [-0.297, +6.164] | 4 |
| P6_primary | hc2 | Capped_Direct | 22.340 | 25.926 | +3.587 [+0.211, +6.870] | 4 |
| P6_primary | vidstg | Native | 12.742 | 14.377 | +1.634 [+0.165, +3.402] | 1 |
| P6_primary | vidstg | OPD | 16.861 | 18.941 | +2.079 [+0.269, +4.259] | 1 |
| P6_primary | vidstg | Direct | 16.546 | 18.785 | +2.239 [+0.423, +4.472] | 1 |
| P6_primary | vidstg | Capped_Direct | 17.209 | 19.412 | +2.203 [+0.396, +4.422] | 1 |
| O_matched_secondary | hc2 | Native | 22.663 | 26.417 | +3.755 [+0.620, +6.713] | 4 |
| O_matched_secondary | hc2 | OPD | 22.264 | 26.081 | +3.816 [+0.346, +7.162] | 4 |
| O_matched_secondary | hc2 | Direct | 20.730 | 23.899 | +3.169 [-0.106, +6.401] | 4 |
| O_matched_secondary | hc2 | Capped_Direct | 22.495 | 26.135 | +3.640 [+0.254, +6.927] | 4 |
| O_matched_secondary | vidstg | Native | 12.742 | 14.377 | +1.634 [+0.165, +3.402] | 1 |
| O_matched_secondary | vidstg | OPD | 16.579 | 18.635 | +2.056 [+0.250, +4.261] | 1 |
| O_matched_secondary | vidstg | Direct | 16.729 | 18.892 | +2.162 [+0.364, +4.382] | 1 |
| O_matched_secondary | vidstg | Capped_Direct | 17.254 | 19.405 | +2.151 [+0.334, +4.383] | 1 |

## Primary and secondary paired contrasts

| Panel | Target | Contrast | Difference (pp), 95% CI | >5 pp harm parents |
|---|---|---|---|---:|
| P6_primary | hc2 | delta_T_OPD_v | +3.823 [+0.336, +7.169] | 4 |
| P6_primary | hc2 | interaction_OPD_Native_v | +0.068 [-0.794, +0.905] | 1 |
| P6_primary | hc2 | OPD_minus_Capped_Direct_T1_v | +0.397 [-0.574, +1.404] | 1 |
| P6_primary | hc2 | OPD_minus_Direct_T1_v | +2.924 [+0.426, +6.054] | 1 |
| P6_primary | vidstg | delta_T_OPD_v | +2.079 [+0.269, +4.259] | 1 |
| P6_primary | vidstg | interaction_OPD_Native_v | +0.445 [-0.663, +1.479] | 1 |
| P6_primary | vidstg | OPD_minus_Capped_Direct_T1_v | -0.472 [-1.148, +0.201] | 1 |
| P6_primary | vidstg | OPD_minus_Direct_T1_v | +0.156 [-0.869, +1.334] | 1 |
| O_matched_secondary | hc2 | delta_T_OPD_v | +3.816 [+0.346, +7.162] | 4 |
| O_matched_secondary | hc2 | interaction_OPD_Native_v | +0.062 [-0.800, +0.900] | 1 |
| O_matched_secondary | hc2 | OPD_minus_Capped_Direct_T1_v | -0.054 [-1.236, +0.992] | 3 |
| O_matched_secondary | hc2 | OPD_minus_Direct_T1_v | +2.182 [-0.186, +5.103] | 2 |
| O_matched_secondary | vidstg | delta_T_OPD_v | +2.056 [+0.250, +4.261] | 1 |
| O_matched_secondary | vidstg | interaction_OPD_Native_v | +0.422 [-0.723, +1.475] | 1 |
| O_matched_secondary | vidstg | OPD_minus_Capped_Direct_T1_v | -0.771 [-1.680, -0.020] | 2 |
| O_matched_secondary | vidstg | OPD_minus_Direct_T1_v | -0.257 [-0.871, +0.360] | 1 |

## Coverage and scoring boundary

Official dense scoring and an independent NumPy readback agree. Interpolation stays within the original full decoder frame grid; frames outside it receive zero IoU. There is no GT box completion, extrapolation or branch choice. Fixed-GT-event sIoU is identical under T0 and T1 for each spatial arm; tIoU is identical for all four spatial arms under each time choice. Complete mvIoU, mtIoU, msIoU, strict R@0.3/R@0.5 and all positive/negative parent rows are in ROWS and SUMMARY.

| Target | Mean event coverage T1−T0 (pp), CI | Total newly included annotated frames outside Native | Total lost event frames | Gained frames outside decoder support |
|---|---|---:|---:|---:|
| hc2 | +19.147 [+12.381, +26.906] | 1087 | 38 | 0 |
| vidstg | +5.315 [+1.112, +9.458] | 1275 | 36 | 0 |

## Q0 signed-weight cancellation and Jensen gap

All 64 corrected-Native episodes, 192 direction-group readouts, 2,808 candidates and 1,308 signed pairs were read. Exact softmax, asymmetric-pair decomposition, float32 stored displacement and actual parameter-mean versus weighted-box decode were checked. The old full-video RMS budget qualification is not part of this scope. The current pooled Native budget and original target tau remain fixed.

For actual unequal amplitudes, with d_mid=(delta+ + delta−)/2, d_half=(delta+ − delta−)/2 and pair mass m, the contribution is m[d_mid+tanh((R+−R−)/(2τ))d_half]. The simple equal-amplitude tanh formula alone would omit a real midpoint term. Zero-denominator ratios remain undefined; nine Vid groups per direction family have no nonzero candidate displacement. There is no reweighting, candidate reselection or temperature sweep.

| Target | Group | Pairs | Unequal amplitudes | Median abs reward margin/τ | Median tanh retention | Median mean/top1 displacement | Median native Jensen RMS | Median true-mean native RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hc2 | Q_random | 256 | 23 | 0.00925 | 0.00462 | 0.00293 | 0.0000425 | 0.0000015 |
| hc2 | Q_observable | 256 | 49 | 0.02056 | 0.01028 | 0.01364 | 0.0004144 | 0.0000482 |
| hc2 | LN_observable | 256 | 168 | 0.33987 | 0.16832 | 0.12775 | 0.0009046 | 0.0012511 |
| vidstg | Q_random | 180 | 44 | 0.00774 | 0.00387 | 0.00360 | 0.0001343 | 0.0000034 |
| vidstg | Q_observable | 180 | 64 | 0.01162 | 0.00581 | 0.04214 | 0.0016112 | 0.0000459 |
| vidstg | LN_observable | 180 | 132 | 0.08810 | 0.04402 | 0.04236 | 0.0007662 | 0.0002353 |

Observable Query pair margins have medians 0.02056 (HC) and 0.01162 (Vid) relative to tau; corresponding tanh retention is 1.028% and 0.581%. Actual mean/top1 displacement medians are 1.364% and 4.214%, with unequal amplitudes and direction summation accounting for differences from a one-pair illustration. Native Jensen RMS medians are 0.000414 and 0.001611; full-decoder medians are 0.004918 and 0.002424. True mean Native RMS medians are 0.0000482 and 0.0000459. This supports substantial shrinkage/nonlinear readout differences in these saved episodes. It does not identify their unique causal contribution to vIoU or prove tau must be changed.

## Evidence and paper decision

Retain Full unchanged as the reproducible research baseline and retain the P1 system improvements of HC +2.566 and Vid +2.968 pp. The common-state O controls do not establish stable Full-over-Direct superiority; correct Q0 does not establish stable Query-over-LN mean benefit. The new factorial study supports useful temporal coverage recovery, with no established OPD-specific interaction. Do not promote the query residual as indispensable or describe descriptive inherited-state decomposition as the causal effect of writeback. Expanded state, LN-only and full deployment timing still need the original S/C/N queues.

The anonymous full draft has an additive evidence revision; the original CVPR project is preserved. A standalone claims/evidence source is supplied for built-in LaTeX preview. The manuscript is a working draft, not a submission-ready claim that all calibration/latency/state experiments have finished.

All cohorts are historically exposed. Each target contributes 32 parents, the secondary panel repeats those parents, bootstrap uses 10,000 paired parent resamples with seed 20261012, and intervals are unadjusted for multiplicity. Positive-control P6, primary interaction and secondary negative results must be reported together. This finite cached study does not establish a full-population joint method, exact output-norm equivalence of the cap, or general impossibility of OPD/Query adaptation.

ROOT_READBACK records complete source/readout/independent projection/dense/statistical and Q0 pair verification. Root readiness was initially dispatched before CPU_COMPLETION and correctly rejected; the original capture is preserved, then the same root source ran after scoring. Actual PDF pixel view, GitHub remote verification and archive closing have separate receipts.
