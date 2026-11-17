# R135 — intact initialization and live complete readout repair

## Decision

Engineering restoration: accepted. Full source-prepared A is reproduced for all1798 inputs, and every repaired no-B endpoint returns complete Frozen-A. Only pointUp4896 changes during adaptation; source-trained gates, RGB adapters, match and competition weights remain intact.

TTA development candidate retained: **True**. Primary was fixed as tta_A before new predictions or scoring. Alternative F output is diagnostic and is not promoted by selecting the best target score. Registered working version: R135_intact_live_A_R134_TTA_development.

## Fixed scope and validity

Same1798 exposed R131 inputs, all six source→target directions, original A256 boxes and original registered B. No source retraining, new expert, new B selection, new loss family, or FINAL. Every prediction was sealed and independently CPU audited before cached fixed-box IoUs. Existing target exposure means this is controlled development, not independent validation.

{
  "n": 1798,
  "has_B": 764,
  "no_B": 1034,
  "active": 749,
  "partial_active": 596,
  "pair_active": 478,
  "pairs": 3619,
  "content_groups": 384,
  "frame_paths": 851,
  "descriptions": 899
}

The source-trained adapter Down/Up, gate_P/gate_I/late, RGB path and full competition are restored exactly. Runtime uses live hidden/text/local/own/relationship reads, live late RGB delta and live competition residual. A route is regenerated from intact frozen A then fixed across B/F and updates; no query identity is transferred across descriptions. The source adapted-point output, not raw P, anchors feature keep so zero-step keep is exactly0.

## Factorial zero-update decomposition

| state | reader | A_mIoU | FAB_mIoU |
|---|---|---|---|
| intact | live | 13.394463 | 13.580861 |
| intact | fixed | 13.394463 | 13.515403 |
| reset | live | 12.611255 | 12.566675 |
| reset | fixed | 12.710810 | 12.630518 |

reset means source Up weight/bias and gate_P cleared as in the legacy initializer. fixed means legacy fixed source-A late/R residual with live new h/u; live means complete recomputed source-prepared readout. This2×2×2 separates initialization, readout and A/B fusion rather than assigning their joint effect to one component.

| effect | gain_pp |
|---|---|
| reset_only_A | -0.783208 |
| fixed_only_A | 0.000000 |
| fixed_after_reset_A | 0.099555 |
| reset_under_fixed_A | -0.683653 |
| AB_fusion_intact_live | 0.186398 |
| AB_fusion_reset_fixed | -0.080292 |
| joint_reset_fixed_A | -0.683653 |

Reset×fixed-reader interaction on A: 0.099555 pp. Main effects should not be summed without this interaction.

## Overall task results

| arm | weighted_mIoU | raw_mIoU | weighted_Acc25 | weighted_Acc50 |
|---|---|---|---|---|
| original_source | 12.392140 | 12.430660 | 24.608060 | 4.595003 |
| legacy_zero_F | 12.630518 | 12.695263 | 25.405773 | 5.054341 |
| legacy_R134 | 12.873988 | 12.911249 | 25.646258 | 5.170967 |
| intact_live_A | 13.394463 | 13.403041 | 26.263306 | 4.599297 |
| intact_live_F | 13.580861 | 13.605854 | 26.099350 | 4.688056 |
| AB_mean | 13.447193 | 13.476794 | 25.914692 | 4.595879 |
| direct_A | 13.354206 | 13.363969 | 26.209250 | 4.599297 |
| tta_A | 13.498989 | 13.506393 | 26.600013 | 4.644834 |
| wrong_A | 13.412728 | 13.407191 | 26.457141 | 4.599297 |
| direct_F | 13.544418 | 13.579433 | 26.069366 | 4.688056 |
| tta_F | 13.606614 | 13.625244 | 26.393334 | 4.859257 |
| wrong_F | 13.569486 | 13.577392 | 26.420022 | 4.733042 |

Design-weighted is the equal-six-direction mean after original within-direction content-group sampling weights. Raw is the equal-six-direction mean without those sampling weights. Acc thresholds use strict >.25 and >.5.

| method | control | gain_pp | better | worse | query_changed | largest_positive_contribution_pp |
|---|---|---|---|---|---|---|
| tta_A | intact_live_A | 0.104526 | 29 | 17 | 85 | 0.028355 |
| tta_A | direct_A | 0.144783 | 31 | 18 | 96 | 0.034101 |
| tta_A | wrong_A | 0.086261 | 25 | 26 | 87 | 0.028018 |
| tta_A | legacy_R134 | 0.625001 | 349 | 318 | 1186 | 0.076153 |
| tta_F | intact_live_F | 0.025752 | 25 | 26 | 98 | 0.053304 |
| tta_F | intact_live_A | 0.212150 | 85 | 83 | 285 | 0.053304 |
| tta_F | direct_F | 0.062195 | 27 | 28 | 109 | 0.053304 |

## Six-direction results

| direction | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| drone→quad | intact_live_A | 13.219501 | 19.524869 | 1.935711 |
| drone→quad | intact_live_F | 13.473304 | 20.083064 | 1.935711 |
| drone→quad | direct_A | 13.277863 | 19.524869 | 1.935711 |
| drone→quad | tta_A | 13.403254 | 19.806534 | 1.935711 |
| drone→quad | tta_F | 13.532223 | 20.646396 | 1.935711 |
| drone→quad | wrong_A | 13.184374 | 19.524869 | 1.935711 |
| drone→quad | legacy_R134 | 12.953906 | 20.014590 | 2.217376 |
| drone→waymo | intact_live_A | 18.693482 | 41.475469 | 7.165491 |
| drone→waymo | intact_live_F | 18.352576 | 39.580893 | 7.165491 |
| drone→waymo | direct_A | 18.398932 | 40.839908 | 7.165491 |
| drone→waymo | tta_A | 18.569062 | 41.566263 | 7.165491 |
| drone→waymo | tta_F | 18.193522 | 39.036127 | 7.560446 |
| drone→waymo | wrong_A | 18.740042 | 42.365254 | 7.165491 |
| drone→waymo | legacy_R134 | 16.850602 | 37.948411 | 6.523575 |
| quad→drone | intact_live_A | 8.971539 | 16.120219 | 0.546448 |
| quad→drone | intact_live_F | 9.253800 | 15.027322 | 0.546448 |
| quad→drone | direct_A | 9.149500 | 16.666667 | 0.546448 |
| quad→drone | tta_A | 9.269590 | 16.666667 | 0.546448 |
| quad→drone | tta_F | 9.283365 | 15.300546 | 0.546448 |
| quad→drone | wrong_A | 9.175754 | 16.120219 | 0.546448 |
| quad→drone | legacy_R134 | 8.423716 | 15.027322 | 0.546448 |
| quad→waymo | intact_live_A | 11.555492 | 25.169407 | 4.530033 |
| quad→waymo | intact_live_F | 11.451578 | 24.999924 | 5.086905 |
| quad→waymo | direct_A | 11.327612 | 24.660959 | 4.530033 |
| quad→waymo | tta_A | 11.555492 | 25.169407 | 4.530033 |
| quad→waymo | tta_F | 11.859262 | 25.635485 | 5.722466 |
| quad→waymo | wrong_A | 11.588745 | 25.169407 | 4.530033 |
| quad→waymo | legacy_R134 | 10.622049 | 22.855664 | 6.140120 |
| waymo→drone | intact_live_A | 10.731475 | 22.131148 | 3.825137 |
| waymo→drone | intact_live_F | 10.948505 | 22.950820 | 3.005464 |
| waymo→drone | direct_A | 10.833124 | 22.404372 | 3.825137 |
| waymo→drone | tta_A | 10.964382 | 22.950820 | 4.098361 |
| waymo→drone | tta_F | 10.993484 | 23.224044 | 3.278689 |
| waymo→drone | wrong_A | 10.883631 | 22.404372 | 3.825137 |
| waymo→drone | legacy_R134 | 10.645770 | 22.950820 | 4.918033 |
| waymo→quad | intact_live_A | 17.195291 | 33.158724 | 9.592961 |
| waymo→quad | intact_live_F | 18.005404 | 33.954078 | 10.388314 |
| waymo→quad | direct_A | 17.138206 | 33.158724 | 9.592961 |
| waymo→quad | tta_A | 17.232155 | 33.440390 | 9.592961 |
| waymo→quad | tta_F | 17.777826 | 34.517409 | 10.111784 |
| waymo→quad | wrong_A | 16.903823 | 33.158724 | 9.592961 |
| waymo→quad | legacy_R134 | 17.747886 | 35.080740 | 10.680251 |

## Reconnecting R134 without conflating it with the repair

Original rho0/3steps/Armijo9 and weights1 remain. Log-gap kappa2.742430260548657 is held fixed. Supported groups and registration thresholds are unchanged. Teacher probabilities and the resulting C/pair evidence are recomputed through the restored full source model, frozen before fitting. This necessarily changes labels compared with legacy R134 and is disclosed as a complete-path port, not an isolated parameter-reset effect. The inherited kappa was calibrated under the old foundation reader and pooled source-own labels; no fresh reader-specific calibration is claimed.

The original R134 pair mechanism detects AB-agreed order reversed by F_AB, and that same frozen pair set is shared by A and F student arms. In A it can start satisfied and act during later partial-label changes; no alternative A-derived pair scheme is substituted. Partial+exact group-score constrained projection is the no-update direct control. Wrong-pair controls preserve S,C, multiplicity bins, graph and margins, but not initial hinge or semantic correctness; with no effective permuted pairs the correct result is reused. They are not a full wrong-partial-label control.

No B means no adaptation or fusion: return complete source-prepared Frozen-A. Rows with B but no qualified task retain their own complete zero-step output; F is only an alternative diagnostic. S/C supervise the loss and are not hard final-output filters.

## Good/bad cases and component limits

| method | control | changed | same_geometric_group | higher_candidate_outside_C | good_C_but_wrong_final |
|---|---|---|---|---|---|
| tta_A | intact_live_A | 46 | 1 | 9 | 283 |
| tta_F | intact_live_F | 51 | 3 | 6 | 285 |

All positive and negative task changes are exported with anonymous frame/content/scene clusters, compared with own frozen, same-information direct/wrong and legacy R134. Higher-IoU is a box-quality proxy, not human-verified instance identity. Same geometric group may contain near-duplicate boxes; different groups do not certify different objects. No raw images/GT were opened for semantic relabeling. Whole-denominator no-B/B contributions and per-scene results are included; repeated physical observations are not independent scenes, and the single Drone recording does not support a scene-level population CI.

## Audit and cost

{
  "status": "PASS",
  "n": 1798,
  "fit_count": 2454,
  "no_B_complete_frozen_fallback": 1034,
  "max_probability_error": 1.8787070850123655e-06,
  "max_state_error": 0.0,
  "max_objective_error": 4.656617313969491e-10,
  "max_adapter_error": 1.9073486328125e-06,
  "max_projection_violation_or_KKT": 4.141140212213327e-15,
  "GT_read": false,
  "FINAL_access": false,
  "full_decoder_gradients_not_independently_CPU_recomputed": true
}

| source | target | n | prepare_seconds | all_four_fit_seconds |
|---|---|---|---|---|
| drone | quad | 357 | 103.3041032597539 | 379.72786350303795 |
| drone | waymo | 176 | 58.00441136735026 | 119.44447895820485 |
| quad | drone | 366 | 104.66640473547159 | 333.8522100811242 |
| quad | waymo | 176 | 47.083283719897736 | 90.8891625488177 |
| waymo | drone | 366 | 120.01071001228411 | 324.69200232450385 |
| waymo | quad | 357 | 122.14197942410829 | 355.94443639868405 |

CPU audit independently rebuilds native matching and full competition from saved live features and frozen source weights, checks source A equivalence, source-adapter+delta feature reconstruction, three-step/Armijo traces, objectives, frozen labels, exact direct projection KKT and all no-B fallbacks. Full decoder gradients are not independently CPU regenerated; engineering finite differences test their direction. Failed engineering/bookkeeping attempts and the premature absent-seal CPU audit are retained, with no target GT read and no result selection.

Public payload excludes original descriptions, scene IDs, frame IDs, media, coordinates, features, weights and third-party code. Source checkpoints/base fingerprints are unchanged. No automatic sweep or further experiment is queued.

## Post-score interpretation: stronger same-information baseline and metric tradeoff

The registered A-protocol gate and DECISION.json are preserved exactly. Passing
that gate supports A-output development only; it does **not** establish global
superiority over every same-information readout. Complete no-update F_AB is
13.580861%, above A-output TTA13.498989% by0.081872pp. Complete F direct readout
13.544418% also exceeds A-output TTA. These stronger controls were predeclared,
run and retained, and must not be hidden by the narrower A-protocol gate.

F-output TTA13.606614% is0.025752pp above its own complete zero and0.062195pp
above its direct readout. Its25improvements/26harms and largest positive case
contribution0.053304pp (larger than net gain) make this a fragile secondary
signal. It was not promoted over the registered A primary by target-score argmax.

Only522of749 qualified A fits actually changed parameters;85changed Top1 and
46changed IoU (29better/17worse). F has749nonzero updates,98Top1 changes and
51IoU changes (25better/26worse). Active-task count is not actual-update count,
and actual-update count is not effective correction count.

The factorial result identifies reset as the dominant A-only starting loss:
reset alone with live head costs0.783208pp. Fixed source-A context alone costs
0 at intact zero-A, while after reset it offsets0.099555pp of damage. Thus
initialization and readout losses are not additive. Under intact source state,
full live F gains0.186398pp over A; stale fixed context loses0.065458pp versus
live F. The old reset+fixed path instead loses0.080292pp when adding F.

Old zero12.630518→intact A13.394463 restores0.763945pp. Old R134 already added
0.243470pp to that weakened zero; new A TTA adds0.104526pp to the intact zero.
Consequently final A improves0.625001pp over R134, and **that entire0.625001pp
must not be attributed to stronger TTA**. Against old R134, A's Acc50 decreases
from5.170967% to4.644834% (-0.526133pp), despite mIoU/Acc25 gains. All signed
metric/direction/scene comparisons remain in the public tables.

Disposition: retain the intact initialization/live-reader/no-B engineering
repair. Keep A TTA only as the predeclared A-protocol development candidate;
retain complete frozen-F and the secondary F-update evidence as separate
controls. This exposed-cohort experiment does not establish a stable globally
best TTA method. No prediction, threshold, step, readout or registered gate was
changed after scoring.
