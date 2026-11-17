# R134 — Retained-candidate ordinal repair of LOG partial-label TTA

**Finite repair attempt completed. Original primary retained: True; pre-score task-linked ablation retained: True. No robust or independent-validation adoption. Current working version: R134_partial_pair3_development_candidate.**

## Hypothesis and actual change

R132/R133 proved that source-calibrated LOG sets activate useful partial-label adaptation, but most updates do not change the final selection and additional fitting largely plateaus. The partial task specifies probability mass in C, not which retained group should win. R134 adds only AB-supported order that the initial full FAB readout reverses. This is a task-specific prototype, not a published baseline or a guarantee of target identity.

For frozen retained groups i,j, both full A/B group-max log scores must prefer i while the initial full FAB group-max prefers j. The frozen target margin is min(gapA,gapB,kappa), kappa2.742430260548657 unchanged. Groups are the original geometric duplicates. No positive unique target, no all-other negatives, no new B selection, no source retraining, no external model or FINAL access.

The new ordinal loss is mean max(0, margin - max_query_logscore(i) + max_query_logscore(j)). The primary uses original partial loss + ordinal loss + original feature keep, fixed unit weights. Three normalized steps, original source-derived per-step radius, original9-level Armijo and Up4896. The original partial task retains mean-group probability aggregation; the added order term uses maximum score because the final A256 decision selects the highest individual query. All signals freeze before adaptation.

Unlike R127 full-distribution KL, this does not transfer every teacher probability ratio. Unknown/disagreed orders remain unconstrained. A/B agreement is still not correctness. No new weight, margin threshold or endpoint is selected from target scores.

## Fixed evidence, controls and execution

All1798 historically exposed R131 observations remain:384 point-content groups,851 alias paths,899 descriptions,84 recordings (Waymo80/Drone1/Quad3). This is neither a fresh holdout nor new-scene validation; prior pooled three-source source-own AB kappa calibration is inherited, not strict direction-isolated single-source label selection.

| Evidence property | Count |
|---|---|
| n | 1798 |
| pair_active | 381 |
| pair_edges | 7157 |
| new_pair_tasks_on_old_inactive | 114 |
| original_partial_active | 646 |
| identity_wrong_pair_inputs | 0 |
| engineering_n | 11 |

Only12 inputs have a frozen reversed pair directly against initial final top group. The other pair evidence may change internal representation/competition but is not pre-certified to correct the current error.114 pair-active inputs previously had C=S and therefore no partial task; the rest267 already had a partial task. All1417 no-pair inputs reuse original partial3 for primary/wrong arms and zero for pair-only. Original zero/direct/partial3 states and predictions are inherited bit-exact; only381 inputs require three new three-step fits.

Controls: pair-only drops the partial term; wrong-pair preserves S/C, pair graph/margin multiset and duplicate-group multiplicities but permutes endpoints inside C. Initial losses and actual accepted budgets are reported; they are not claimed matched semantic mass or gradient scale. Zero-update direct projection minimizes group-log-score displacement under the exact same pair-margin inequalities. It adds common offsets within groups and uses actual query scores. Applying it after original partial3 is a strong same-three-step-budget hybrid. This hard feasible direct control is stronger than merely showing that a soft surrogate decreased.

Engineering11 spans six directions and both cached/reconstructed original-B paths. Zero FAB readout bridge is exact, directional derivative checks pass, source/prepared parameters reset after every input. All predictions/states/controls are sealed before cached fixed-A256 IoUs are opened. No new media/GT geometry or A backbone inference is needed; missing B descriptors reconstruct the same original registered B.

## Pre-score task-relevance extension

Before any R134 task scores, an additional eligibility ablation was locked (experiment/TASK_RELEVANCE_EXTENSION_LOCK.json). Only when a frozen AB pair opposes the initial final top group does tasklinked_pair_F use the new fitted output; otherwise it reuses oldpartial3. Exactly12 inputs qualify by frozen teacher evidence. Matched wrong/direct controls use the same gate. No new gradients or target-GT conditions. Original primary/comparisons remain unchanged; the tasklinked candidate has its separate predeclared comparison against oldpartial3, zero, gateddirect, gatedpartial3+direct and gatedwrong in both macros.

## Overall results

| Arm | Design-weighted mIoU | Acc25 | Acc50 | Raw macro mIoU |
|---|---|---|---|---|
| zero_F | 12.630518 | 25.405773 | 5.054341 | 12.695263 |
| partial_direct_F | 12.644153 | 25.407180 | 5.054341 | 12.707285 |
| partial3_F | 12.793107 | 25.694609 | 5.170967 | 12.828305 |
| pair_direct_F | 12.637549 | 25.361642 | 5.054341 | 12.700672 |
| partial3_pair_direct_F | 12.786602 | 25.649072 | 5.170967 | 12.821797 |
| wrongpair_direct_F | 12.623557 | 25.314698 | 5.008804 | 12.694787 |
| pair_only_F | 12.693362 | 25.311884 | 5.052934 | 12.759228 |
| partial_pair_F | 12.873988 | 25.646258 | 5.170967 | 12.911249 |
| partial_wrongpair_F | 12.790198 | 25.482658 | 5.125430 | 12.824139 |
| tasklinked_pair_F | 12.794077 | 25.602127 | 5.170967 | 12.829223 |
| tasklinked_wrongpair_F | 12.780682 | 25.647665 | 5.170967 | 12.815949 |
| tasklinked_direct_F | 12.636960 | 25.361642 | 5.054341 | 12.700087 |
| tasklinked_partial3_direct_F | 12.785433 | 25.649072 | 5.170967 | 12.820628 |

| Aggregation | Primary minus control | Gain pp |
|---|---|---|
| raw | zero_F | +0.215987 |
| raw | partial3_F | +0.082944 |
| raw | pair_direct_F | +0.210577 |
| raw | partial3_pair_direct_F | +0.089452 |
| raw | partial_wrongpair_F | +0.087111 |
| design_weighted | zero_F | +0.243470 |
| design_weighted | partial3_F | +0.080881 |
| design_weighted | pair_direct_F | +0.236439 |
| design_weighted | partial3_pair_direct_F | +0.087387 |
| design_weighted | partial_wrongpair_F | +0.083790 |

The predeclared comparison requires aggregate primary advantage over zero, oldpartial3, zero-update pair direct, same-budget partial3+pair direct and wrong-pair update in raw/design-weighted macro. Direction/scene negatives never trigger per-direction switching. No strongest endpoint is relabeled primary after scoring. These are internal mechanism controls, not external-paper baselines.

## Six directions and recording dependence

| Direction | zero_F | partial_direct_F | partial3_F | pair_direct_F | partial3_pair_direct_F | wrongpair_direct_F | pair_only_F | partial_pair_F | partial_wrongpair_F | tasklinked_pair_F | tasklinked_wrongpair_F | tasklinked_direct_F | tasklinked_partial3_direct_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| waymo→drone | 10.563851 | 10.563851 | 10.826849 | 10.563851 | 10.835208 | 10.544145 | 10.750907 | 10.645770 | 10.531257 | 10.826849 | 10.826849 | 10.563851 | 10.826849 |
| waymo→quad | 17.872754 | 17.873595 | 17.529129 | 17.878556 | 17.534090 | 17.869263 | 17.715585 | 17.747886 | 17.512608 | 17.534090 | 17.529129 | 17.878556 | 17.534090 |
| drone→waymo | 16.547162 | 16.514637 | 16.864618 | 16.514637 | 16.864618 | 16.514637 | 16.547162 | 16.850602 | 17.005511 | 16.864618 | 16.864618 | 16.514637 | 16.864618 |
| drone→quad | 12.707409 | 12.754368 | 12.826492 | 12.759723 | 12.824076 | 12.757256 | 12.771993 | 12.953906 | 12.667612 | 12.877284 | 12.751943 | 12.756189 | 12.825425 |
| quad→waymo | 9.927004 | 10.003150 | 10.674359 | 10.003150 | 10.674359 | 10.019835 | 9.980405 | 10.622049 | 10.903723 | 10.674359 | 10.674359 | 10.003150 | 10.674359 |
| quad→drone | 8.164930 | 8.155315 | 8.037195 | 8.105380 | 7.987260 | 8.036208 | 8.394119 | 8.423716 | 8.120478 | 7.987260 | 8.037195 | 8.105380 | 7.987260 |

All raw and design-weighted target/recording rows are included. Same physical frames/descriptions across directions are not independent scene observations. One remaining Drone recording cannot support population-level scene CIs.

## Task changes, negative tails and auxiliary fit

| Method | Control | Better | Worse | Same | Content groups | Recordings | Max positive pp | Max negative pp |
|---|---|---|---|---|---|---|---|---|
| partial_pair_F | pair_direct_F | 54 | 34 | 1710 | 66 | 10 | +0.065409 | -0.030236 |
| partial_pair_F | partial3_F | 27 | 25 | 1746 | 42 | 7 | +0.021102 | -0.017684 |
| partial_pair_F | partial3_pair_direct_F | 25 | 23 | 1750 | 40 | 7 | +0.021102 | -0.017684 |
| partial_pair_F | partial_wrongpair_F | 35 | 30 | 1733 | 49 | 8 | +0.023049 | -0.033445 |
| partial_pair_F | tasklinked_direct_F | 54 | 34 | 1710 | 66 | 10 | +0.065409 | -0.030236 |
| partial_pair_F | tasklinked_partial3_direct_F | 25 | 24 | 1749 | 41 | 7 | +0.021102 | -0.017684 |
| partial_pair_F | tasklinked_wrongpair_F | 28 | 24 | 1746 | 42 | 7 | +0.021102 | -0.017684 |
| partial_pair_F | zero_F | 54 | 31 | 1713 | 61 | 8 | +0.065409 | -0.030236 |
| tasklinked_pair_F | pair_direct_F | 52 | 29 | 1717 | 58 | 11 | +0.065409 | -0.030236 |
| tasklinked_pair_F | partial3_F | 4 | 2 | 1792 | 5 | 2 | +0.009910 | -0.008995 |
| tasklinked_pair_F | partial3_pair_direct_F | 3 | 3 | 1792 | 6 | 2 | +0.009910 | -0.001872 |
| tasklinked_pair_F | partial_wrongpair_F | 35 | 24 | 1739 | 48 | 10 | +0.023049 | -0.033445 |
| tasklinked_pair_F | tasklinked_direct_F | 52 | 28 | 1718 | 58 | 11 | +0.065409 | -0.030236 |
| tasklinked_pair_F | tasklinked_partial3_direct_F | 2 | 1 | 1795 | 3 | 1 | +0.009910 | -0.001872 |
| tasklinked_pair_F | tasklinked_wrongpair_F | 5 | 1 | 1792 | 5 | 2 | +0.010553 | -0.008995 |
| tasklinked_pair_F | zero_F | 52 | 28 | 1718 | 56 | 9 | +0.065409 | -0.030236 |

| New fit | Inputs | Backward | Reader calls | Accepted steps | Mean parameter L2 | Mean path | Initial ordinal | Final ordinal |
|---|---|---|---|---|---|---|---|---|
| pair_only | 381 | 1143 | 3696 | 874 | 0.05850441 | 0.07524168 | 0.18814540 | 0.01450985 |
| partial_pair | 381 | 1143 | 3974 | 1123 | 0.07151887 | 0.09770083 | 0.18814540 | 0.01716444 |
| partial_wrongpair | 381 | 1143 | 3896 | 1134 | 0.09304993 | 0.10205235 | 0.49335618 | 0.18347406 |

Operation counts describe newly executed fits and omit common frozen prefix/B/teacher construction. Other no-pair inputs reuse priorpartial3; that reuse is not zero deployment cost. Same per-step/three-step nominal budget does not guarantee equal accepted displacement. Loss decrease, pair satisfaction, nonzero gradients and more selection switches do not define success.

## What remains inside the508 previous unchanged updates?

This is post-seal descriptive GT analysis, never a training eligibility or output selector. Acc25-positive is the official threshold metric, not a perfect continuous-IoU result. No-good-in-A256 refers to that threshold, not all possible IoU improvement.

| Category | Count | With new pair evidence | With direct top reversal |
|---|---|---|---|
| recoverable_Acc25_candidate_inside_C | 146 | 63 | 3 |
| Acc25_candidate_outside_C | 116 | 36 | 1 |
| already_Acc25_positive | 145 | 49 | 1 |
| no_Acc25_candidate_in_A256 | 101 | 46 | 2 |

Among these508 previous unchanged-query inputs, the primary now changes 24 query selections. This count alone is not a task gain.

## Supervision quality and previous good/bad cases

All old49 improvements/27 harms are retained, together with every new improvement/harm/no-op. No identity is inferred solely from IoU or duplicate groups. For frozen pairs, post-seal relative IoU uses each group representative selected by initial FAB (not the oracle). It is a box-quality proxy; tied zero-IoU pairs remain ambiguous, not certified correct/incorrect identities.

```json
{
  "n": 381,
  "pairs": 7157,
  "old_top_challenged_inputs": 12,
  "recoverable_zero_error_inputs": 222,
  "recoverable_insideC_with_top_challenge": 7,
  "pair_quality_counts_box_proxy_only": {
    "pair_quality_better": 1097,
    "pair_quality_worse": 819,
    "pair_quality_tied": 5241,
    "top_pair_quality_better": 4,
    "top_pair_quality_worse": 2
  },
  "mean_initial_ordinal_loss": 0.18814539851845738,
  "mean_final_ordinal_loss": 0.017164441704497293,
  "identity_not_certified": true,
  "GT_only_postseal_descriptive": true
}
```

Previous76-record audit:

```json
{
  "n": 76,
  "better": 49,
  "worse": 27,
  "with_new_pair_evidence": 44,
  "harms_with_new_pair_evidence": 17
}
```

## Audit and disposition

```json
{
  "status": "PASS_FROZEN_PAIRS_CONVEX_DIRECT_KKT_NATIVE_READOUT_STATES_STEPS_OBJECTIVES_AND_REFERENCE_REUSE",
  "n": 1798,
  "pair_active": 381,
  "arms": 13,
  "task_relevant_policy_inputs": 12,
  "max_CPU_probability_error": 1.659247963692323e-06,
  "max_state_error": 0.0,
  "max_objective_error": 9.313225746154785e-10,
  "max_point_adapter_feature_error": 9.5367431640625e-07,
  "max_direct_constraint_violation": 3.375077994860476e-14,
  "max_direct_KKT_residual": 9.225783518027132e-15,
  "full_decoder_gradients_not_independently_CPU_recomputed": true,
  "engineering_directional_derivative_checked": true,
  "GT_read": false,
  "FINAL_access": false
}
```

Independent CPU head replay, exact normalized state arithmetic, reconstructed point Adapter outputs, final objective recomputation, pair geometry and direct convex projection KKT conditions all checked. Decoder gradients are not independently rerun on CPU for every input; engineering finite-difference checks cover the new path. Every23374 scalar and paired/direction/recording weighted result is independently audited.

```json
{
  "primary": "partial_pair_F",
  "target_GT_configuration_selection": false,
  "FINAL_access": false,
  "comparisons": {
    "raw": {
      "mIoU_percent": 12.911249468788144,
      "gains_pp": {
        "zero_F": 0.21598678449539577,
        "partial3_F": 0.08294439155034894,
        "pair_direct_F": 0.21057726867999982,
        "partial3_pair_direct_F": 0.08945198607384341,
        "partial_wrongpair_F": 0.08711066605994944
      }
    },
    "design_weighted": {
      "mIoU_percent": 12.873988195727632,
      "gains_pp": {
        "zero_F": 0.2434697673507884,
        "partial3_F": 0.0808812928216085,
        "pair_direct_F": 0.23643894793037212,
        "partial3_pair_direct_F": 0.08738654809093305,
        "partial_wrongpair_F": 0.08379008048089531
      }
    }
  },
  "recipe_retained": true,
  "tasklinked_comparisons": {
    "raw": {
      "mIoU_percent": 12.829223439894363,
      "gains_pp": {
        "zero_F": 0.13396075560161513,
        "partial3_F": 0.0009183626565683056,
        "tasklinked_direct_F": 0.1291368632478722,
        "tasklinked_partial3_direct_F": 0.008595482648063424,
        "tasklinked_wrongpair_F": 0.013274648186326843
      }
    },
    "design_weighted": {
      "mIoU_percent": 12.794076518866378,
      "gains_pp": {
        "zero_F": 0.16355809048953418,
        "partial3_F": 0.0009696159603542753,
        "tasklinked_direct_F": 0.15711614263491924,
        "tasklinked_partial3_direct_F": 0.008643156665048224,
        "tasklinked_wrongpair_F": 0.013394434438218639
      }
    }
  },
  "tasklinked_recipe_retained": true,
  "robust_method_adopted": false,
  "independent_validation": false,
  "current_working_version": "R134_partial_pair3_development_candidate",
  "scoring_utility_provisional_working_version": "R134_tasklinked_pair3_development_candidate",
  "registration_rule": "original predeclared primary precedes additional ablation when both gates pass; no maximum-target-score selection"
}
```

If the primary fails its fixed comparisons, retain R132LOG3 and stop this ordinal recipe. A negative result narrows whether existing AB order can supply the missing identity information; it does not prove all self-supervised adaptation impossible. No hidden sweep, per-case patch, new source preparation or FINAL evaluation follows. See LITERATURE.md for exact paper access and differences; no theoretical PLL guarantee or RKD reproduction is claimed.

## Administrative registration note

{
  "status": "ADMINISTRATIVE_PRIMARY_PRECEDENCE_CORRECTED",
  "as_scored_utility_value": "R134_tasklinked_pair3_development_candidate",
  "final_registered_value": "R134_partial_pair3_development_candidate",
  "reason": "original predeclared primary precedence; scoring utility nested ternary prioritized added ablation when both pass",
  "prediction_metric_gate_or_method_changes": false,
  "target_argmax_configuration_selection": false,
  "original_decision_sha256": "13844ce96175ca7036eb8f8c01b3444f54a8df4109c351050f3977a5a8386a21",
  "utc": "2026-11-17T11:18:06.404092+00:00"
}

All original scored rows and gates are unchanged. The original decision is retained locally in attempts/DECISION_AS_SCORED.json, and the provisional utility field is preserved in DECISION.json. Final registration follows original-primary precedence, not target-score argmax.
