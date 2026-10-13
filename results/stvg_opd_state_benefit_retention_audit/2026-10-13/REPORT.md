# State-Benefit Retention Audit

The audit is complete for all 64 historically exposed parents, all 192 sealed fits, all 4,800 saved updates and 4,992 arm-boundary readouts. It used no new model/teacher/forward/backward/optimizer call. Source pixels, evidence, LN initialization, zero Query Residual, fresh original Adam and noise were rebound to the genuine controlled intervention.

**The strongest retention/convergence evidence is in HC. VidSTG has real adverse cases but does not establish the same domain-wide output-contraction explanation.** This is current-query benefit erosion, not a previous-query catastrophic-forgetting experiment or a deployable state-selection method.

## Full endpoint advantage and timing

Q is parent-equal spatial IoU on genuine GT-event decoder frames excluding all physical Omega0 visits. The complete primary result includes empty, neutral and negative parents. Values below are percentage points; paired unadjusted parent-bootstrap 95% intervals.

| Target | Initialization | Before advantage | After advantage | Advantage change | First quarter | Later updates |
|---|---|---|---|---|---|---|
| hc2 | Inherited_order1 | +3.536 [+2.523, +4.641] | +0.385 [-0.464, +1.275] | -3.151 [-4.694, -1.821] | -2.278 [-3.438, -1.137] | -0.873 [-1.971, +0.068] |
| hc2 | Inherited_order2 | +4.201 [+2.699, +5.589] | +0.347 [-0.580, +1.337] | -3.855 [-5.488, -2.046] | -2.832 [-4.273, -1.315] | -1.022 [-2.354, +0.172] |
| vidstg | Inherited_order1 | +4.736 [+2.439, +7.156] | +0.971 [-1.387, +3.163] | -3.765 [-6.822, -0.727] | -1.553 [-4.425, +1.412] | -2.212 [-4.366, -0.120] |
| vidstg | Inherited_order2 | +3.889 [+1.732, +6.139] | -0.386 [-3.784, +2.591] | -4.275 [-7.796, -0.987] | -2.730 [-6.507, +0.644] | -1.545 [-4.261, +1.112] |

HC loses most of its average inherited advantage within the first ten configured updates. Vid timing is mixed; the I1 later-update interval is negative, while its early interval crosses zero. Every round is preserved and no best step is selected.

## Actual output convergence

RMS is mean per-frame normalized cxcywh coordinate distance between the inherited and Source outputs, then equal-weighted across parents. It is not a task-score difference, pixel distance or parameter distance.

| Target | Initialization | RMS Before | RMS After | RMS change, 95% CI | Contracting / expanding / neutral parents |
|---|---|---|---|---|---|
| hc2 | Inherited_order1 | 0.03842 | 0.02167 | -0.017 [-0.024, -0.010] | 24 / 7 / 1 |
| hc2 | Inherited_order2 | 0.04245 | 0.02143 | -0.021 [-0.028, -0.013] | 26 / 5 / 1 |
| vidstg | Inherited_order1 | 0.04805 | 0.04016 | -0.008 [-0.017, +0.002] | 19 / 6 / 7 |
| vidstg | Inherited_order2 | 0.04766 | 0.04247 | -0.005 [-0.015, +0.005] | 19 / 6 / 7 |

Both HC paired RMS-change intervals are below zero. Both Vid intervals include zero. Parameter-state distance increases on average for all four comparisons, despite HC output contraction; this rejects treating output contraction as convergence to the same parameter state.

## Original good unobserved frames can degrade while evidence agreement improves

The predeclared offline stratum requires inherited Before IoU >=0.5 and an initial advantage over Source >=1pp. Persistent loss requires inherited IoU below its own Before at every boundary of the final quarter, not just Source catching up. Frame fractions below are descriptive micro-counts; parent-equal conditional estimates are reported separately. Frames are not independent samples.

| Target | Initialization | Persistent loss frames / stratum frames | Parent-equal stratum gain, pp [CI] | DINO agreement gain, pp [CI] | Joint parents true / false / unknown |
|---|---|---|---|---|---|
| hc2 | Inherited_order1 | 175/288 (60.8%) | -9.438 [-18.202, -2.329] | +24.166 [+16.469, +33.123] | 20 / 6 / 6 |
| hc2 | Inherited_order2 | 204/315 (64.8%) | -9.338 [-17.328, -2.592] | +22.229 [+15.215, +30.312] | 20 / 6 / 6 |
| vidstg | Inherited_order1 | 78/530 (14.7%) | +1.575 [-7.044, +9.297] | +24.026 [+18.329, +30.051] | 7 / 12 / 13 |
| vidstg | Inherited_order2 | 96/455 (21.1%) | +2.763 [-7.276, +11.182] | +24.317 [+18.925, +30.249] | 7 / 10 / 15 |

The joint parent event requires an increase in admitted DINO agreement, a decrease in overall inherited advantage, and at least one persistently degraded frame in the initially correct/advantaged stratum. It is a descriptive conjunction, not an online gate or a significance test. Agreement is unknown in seven Vid empty-evidence parents. No stratum in a parent means unknown, not no damage.

HC stratum mean absolute gain is negative with intervals below zero, so its loss cannot be reduced to Source merely catching up. Vid stratum means are positive with intervals spanning zero: harmful subsets coexist with useful adaptation. Whole-panel inherited After quality can still exceed Source, especially Vid I1; this audit does not select Source reset as a new method.

## Negative tails and concentration

| Target | Initialization | Advantage erosion +/-/neutral | Remove largest absolute 1 parent | Remove largest absolute 3 parents | Leave-one-out mean range, pp |
|---|---|---|---|---|---|
| hc2 | Inherited_order1 | 5/26/1 | -2.691 [-3.870, -1.585] | -2.217 [-3.258, -1.220] | [-3.375, -2.691] |
| hc2 | Inherited_order2 | 4/27/1 | -4.463 [-5.822, -3.143] | -4.003 [-5.251, -2.775] | [-4.463, -3.605] |
| vidstg | Inherited_order1 | 8/16/8 | -3.141 [-6.054, -0.244] | -1.987 [-4.574, +0.669] | [-4.372, -3.141] |
| vidstg | Inherited_order2 | 8/16/8 | -3.178 [-5.961, -0.404] | -2.064 [-4.543, +0.447] | [-4.940, -3.178] |

HC remains negative after removal of its largest one or three absolute-change parents. Vid remains negative after one removal, but both three-removal intervals include zero. These are diagnostics; complete 32-parent estimates remain primary and the removed IDs and all rows are preserved.

## Saved gradients, state separation and headroom alternatives

| Target | Initialization | Initial joint gradient cosine [CI] | Joint state RMS Before -> After | Starting-quality LOO R2 |
|---|---|---|---|
| hc2 | Inherited_order1 | +0.736 [+0.607, +0.850] | 0.05873 -> 0.07477 | 0.498 |
| hc2 | Inherited_order2 | +0.608 [+0.437, +0.755] | 0.06284 -> 0.08127 | 0.620 |
| vidstg | Inherited_order1 | +0.682 [+0.546, +0.807] | 0.10919 -> 0.13035 | 0.313 |
| vidstg | Inherited_order2 | +0.631 [+0.480, +0.771] | 0.10764 -> 0.12807 | 0.049 |

Every saved outgoing gradient, applied step, QR/LN norm, accumulated step length, moment/state identity, pair cosine and incoming output movement is retained for every boundary. No full decoder Jacobian was replayed; the original actual native-head VJP qualification is reused only within its recorded scope. Zero gradients have unknown cosine.

The leave-one-parent-out linear model uses Source and inherited Before quality to predict advantage change. Its association is compatible with headroom and is also mechanically coupled to the outcome through the Before subtraction. It cannot establish that headroom causes or fully explains the intervention effect. Previous-query retention, teacher-independent output sensitivity and gradient interventions were not measured.

## Research decision and completion boundary

HC supports a concrete current-query preservation problem: sparse pseudo-supervision improves teacher agreement while eroding a substantial part of originally useful unvisited-frame localization, with actual output contraction and robustness to the largest few cases. Vid supports state-dependent utility and individual retention failures, but not a uniform forgetting/convergence mechanism. Keep both findings; do not pool domains into a universal claim.

Do not call this catastrophic forgetting or a validated Cross-Query Knowledge Preservation algorithm. No gate, new loss, teacher, hyperparameter search, production method or full main experiment was started. P7-A remains frozen. The cancelled formal baselines retain their accepted prefix and original T standby identity. Future method and main protocol remain separate work.

Complete anonymous files: PARENT_ROWS.jsonl.gz (64 parents, all three pair contrasts), ROUND_ROWS.jsonl.gz (1,664 parent boundaries), FRAME_ROUND_ROWS.jsonl.gz (every genuine decoder-frame boundary, including GT-unknown frames), DENSE_ENDPOINT_ROWS.jsonl.gz (all original three-arm dense Before/After metrics), ALL_PARENT_CONTRASTS.csv (192 contrasts), SUMMARY.json (all masks/arms/rounds/strata/tails/unknowns). Raw boxes, state/gradient vectors, physical frame IDs, annotations, media and text are excluded from publication.

Independent source/math/state/noise/frame/dense/scalar/bootstrap readback passed before figure authoring. All four PDF pages passed actual root pixel review. Remote public-content verification and archive closure are bound by the separate current-scope closing receipts.
