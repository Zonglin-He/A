# R132 — Skip diagnosis, fixed top-three B, and source-calibrated log-gap sets

## Scope

R131 is the reference, not rerun or relabeled. Same fixed1,798 exposed direction-inputs:384 point-content groups,851 alias paths,899 descriptions,84 recording clusters. All aliases/descriptions/sources remain together; existing group sampling weights and six-direction aggregation remain unchanged. No new source training, external expert, boxes, architecture, step/radius grid, loss, EMA or FINAL access. Source-prepared A context and complete F_AB readout remain. Every UNKNOWN/no-update/negative row stays in the denominator.

The two factors are tested separately: B3 changes sensor-B acquisition while preserving the original tau; LOG changes the candidate-set rule at the original B. There is no B3+LOG result. Each factor has its own same-information zero/direct/correct-update/wrong-update controls. These are internal controlled experiments, not published baselines.

## Cached diagnosis: what actually failed before optimization?

No Backbone or new task scoring was needed for this diagnosis. Reason/decision rows were sealed before cached R131 task scalars were attached.

| Exclusive reason | Direction-inputs |
|---|---|
| no_usable_B | 1034 |
| C_equals_S | 600 |
| actual_update | 164 |

| C=S subtype | Inputs |
|---|---|
| both_views_all | 496 |
| A_only_all | 45 |
| B_only_all | 59 |
| union_fills_complementary_exclusions | 0 |

All600 C=S cases have at least one view whose maximum conditional probability is already <=tau. This supports the absolute-probability scale explanation on these caches; it is not evidence that forcing nonempty negatives is safe. Parameters changed on 164 inputs, 492 steps were accepted and 8 were backtracked smaller. Query changed on 17 inputs, exact box changed on 17, IoU on 8. The fixed geometric grouping marks 17 between-group switches and 0 within-group switches, not true semantic instance IDs.

| Original q0 versus strongest other group, before any GT | Active inputs |
|---|---|
| widening | 129 |
| shrinking | 35 |

These margins use a competitor fixed by zero-step model score, never the GT-best candidate. Shared-parameter coupling can move other scores; shrinking or widening this margin does not itself certify correctness. Saved probability differences and parameter/gradient/step magnitudes are all in the anonymous diagnosis table.

## Registration failure detail

| Stage and first failed check | Unique A-frame records |
|---|---|
| forward__insufficient_fit_correspondences | 47 |
| forward__cloud_median_residual | 82 |
| forward__insufficient_spatial_holdout_correspondences | 23 |
| registered__geometry_cycle_supported | 365 |
| forward__RGB_inlier_fraction | 79 |
| forward__RGB_median_reprojection | 163 |
| reverse__cloud_median_residual | 14 |
| reverse__RGB_median_reprojection | 20 |
| no_candidate_B__no_candidate_B | 6 |
| forward__PnP_no_RANSAC_consensus | 11 |
| reverse__RGB_inlier_fraction | 41 |

Checks can fail jointly. This is the prespecified first-failure hierarchy, not exclusive physical causes. Legacy insufficient-correspondence logs do not fully separate RGB feature matching from depth binding, so that distinction is not invented. New attempts add visible-depth and RGB-match counters without altering original PnP/cycle math. The backend accepted flag also includes unavailable foreground-mask conditions; this experiment continues the exact R131 geometry-only numeric wrapper rather than silently changing to that flag.

## B3: evidence acquisition, no relaxed acceptance

Same clean ordinary candidate pool as R131, same recording/camera suffix and counter ordering. First candidate is exactly R131. Preselect at most3 distinct point-content/RGB candidates, then take the first to pass the unchanged forward/reverse/cycle gates. No source score, task label or model confidence chooses a B, and no later candidate is tried after a pass. All arms use that same selected B.

| Selected candidate rank | A-frame count |
|---|---|
| None | 403 |
| 1 | 365 |
| 2 | 72 |
| 3 | 11 |

The original845 B1 attempts are reused exactly. New target registration attempts: 875, summed pair CPU seconds 838.943. Four-worker registration wall time includes the separate source-calibration registration collection. Registration coverage is not target identity certification. B3 versus R131 uses more acquisition computation; within-B3 zero/direct/update comparisons share the selected evidence.

## LOG: one source-own real-AB calibration

The fixed log-ratio rule is C=union_v{g:logmax p_v-log p_v(g)<=kappa}. Same supported groups, n>=4 and B coverage>=.5; uniform/insufficient evidence remains unknown, with no forced exclusion or intersection. The original B is preserved, and no B3 evidence enters this arm.

Calibration reuses source-VAL caches from each source-own model, excluding every known protected-content group and every R131 A/B content group from source A/B. No source weights are trained. All313 source predictions were sealed and CPU audited before their existing source labels were opened. There are77 eligible registered supported-good inputs/154 view gaps across all three sources. One pooled90th-percentile log gap to the best Acc25-good supported group gives kappa=2.742430260549. No target outcomes, update rate or direction-specific rule selects it.

| Source-only diagnostic | Value |
|---|---|
| Old absolute-tau good-group retention | 0.961039 |
| LOG good-group retention | 0.935065 |
| Old mean retained group fraction | 0.537216 |
| LOG mean retained group fraction | 0.147224 |

Calibration pools source-own caches from all three source models, following the existing R129 shared-calibration protocol. No R131 evaluation labels enter it. Because platforms also act as targets in other directions, this is pooled cross-source development, not a claim of strict direction-by-direction single-source label isolation. The new calibration matches real AB conditions; the old tau was source-A-only. Consequently the tested factor is a source-recalibrated log-set policy, not an isolated unit change under identical old calibration observations. No claim that log units alone caused a task gain is made. This is an exposed-source quantile heuristic, not a formal target coverage guarantee.

Before new target task scoring, LOG supplied646 nonzero partial tasks versus164 in R131, adding482 previously C=S inputs.118 usable-B inputs remain C=S. This process increase is reported regardless of task performance.

## Complete task results

A first score-write attempt failed on a NumPy boolean JSON type; the empty output directory, original code and log were retained. Explicit boolean conversion and actual parameter-change diagnostic fields were added without altering any prediction, threshold or metric definition. All task values are indexed from R131’s sealed all-A256 IoU vectors after every new prediction/state is sealed. No new raw target GT or geometric scoring, and original4 arms are bit-identical. Main metric is the fixed design-weighted six-direction macro; raw macro, platforms and all recordings remain visible.

| Arm | Weighted mIoU | Weighted Acc25 | Weighted Acc50 | Raw mIoU |
|---|---|---|---|---|
| zero_F | 12.630518 | 25.405773 | 5.054341 | 12.695263 |
| direct_partial_F | 12.631040 | 25.405773 | 5.054341 | 12.695781 |
| partial_F | 12.676120 | 25.570750 | 5.125430 | 12.728039 |
| wrong_base_F | 12.567540 | 25.222216 | 5.008804 | 12.638584 |
| B3_zero | 12.638022 | 25.494351 | 5.008253 | 12.686853 |
| B3_direct | 12.624834 | 25.447407 | 5.008253 | 12.673738 |
| B3_partial | 12.636028 | 25.557907 | 5.079341 | 12.655461 |
| B3_wrong | 12.519651 | 25.209374 | 4.962716 | 12.558248 |
| LOG_zero | 12.630518 | 25.405773 | 5.054341 | 12.695263 |
| LOG_direct | 12.644153 | 25.407180 | 5.054341 | 12.707285 |
| LOG_partial | 12.793107 | 25.694609 | 5.170967 | 12.828305 |
| LOG_wrong | 12.539317 | 24.916941 | 5.101286 | 12.605654 |

| Factor | Aggregation | Control | Update gain pp |
|---|---|---|---|
| B3 | raw | B3_zero | -0.031392 |
| B3 | raw | B3_direct | -0.018277 |
| B3 | raw | B3_wrong | +0.097213 |
| B3 | raw | partial_F | -0.072578 |
| B3 | design_weighted | B3_zero | -0.001994 |
| B3 | design_weighted | B3_direct | +0.011194 |
| B3 | design_weighted | B3_wrong | +0.116377 |
| B3 | design_weighted | partial_F | -0.040092 |
| LOG | raw | LOG_zero | +0.133042 |
| LOG | raw | LOG_direct | +0.121020 |
| LOG | raw | LOG_wrong | +0.222652 |
| LOG | raw | partial_F | +0.100266 |
| LOG | design_weighted | LOG_zero | +0.162588 |
| LOG | design_weighted | LOG_direct | +0.148954 |
| LOG | design_weighted | LOG_wrong | +0.253790 |
| LOG | design_weighted | partial_F | +0.116987 |

| Direction | R131 update | B3 zero | B3 direct | B3 update | LOG direct | LOG update |
|---|---|---|---|---|---|---|
| waymo→drone | 10.563851 | 10.528734 | 10.528734 | 10.528734 | 10.563851 | 10.826849 |
| waymo→quad | 17.675194 | 17.772107 | 17.772107 | 17.574546 | 17.873595 | 17.529129 |
| drone→waymo | 16.547162 | 16.746623 | 16.746623 | 16.610820 | 16.514637 | 16.864618 |
| drone→quad | 12.788485 | 12.622634 | 12.543507 | 12.553942 | 12.754368 | 12.826492 |
| quad→waymo | 10.319461 | 9.803345 | 9.803345 | 10.195802 | 10.003150 | 10.674359 |
| quad→drone | 8.162565 | 8.354689 | 8.354689 | 8.352324 | 8.155315 | 8.037195 |

| Target | Arm | Weighted mIoU |
|---|---|---|
| waymo | partial_F | 13.433312 |
| waymo | B3_partial | 13.403311 |
| waymo | LOG_partial | 13.769488 |
| drone | partial_F | 9.363208 |
| drone | B3_partial | 9.440529 |
| drone | LOG_partial | 9.432022 |
| quad | partial_F | 15.231840 |
| quad | B3_partial | 15.064244 |
| quad | LOG_partial | 15.177811 |

## Coverage, effective updates and task competition

| Direction | Factor | Usable B | Active | Weights changed | Effective wrong | Queries changed | IoUs changed | Better/worse vs own zero | Accepted steps |
|---|---|---|---|---|---|---|---|---|---|
| drone→quad | B3 | 211 | 50 | 50 | 49 | 9 | 6 | 3/3 | 150 |
| drone→quad | LOG | 178 | 159 | 159 | 158 | 48 | 31 | 18/13 | 477 |
| drone→waymo | B3 | 62 | 13 | 13 | 13 | 1 | 1 | 0/1 | 39 |
| drone→waymo | LOG | 50 | 39 | 39 | 39 | 8 | 3 | 3/0 | 117 |
| quad→drone | B3 | 192 | 25 | 25 | 25 | 3 | 1 | 0/1 | 75 |
| quad→drone | LOG | 154 | 85 | 85 | 85 | 17 | 7 | 4/3 | 255 |
| quad→waymo | B3 | 62 | 12 | 12 | 12 | 3 | 1 | 1/0 | 36 |
| quad→waymo | LOG | 50 | 33 | 33 | 33 | 7 | 5 | 5/0 | 99 |
| waymo→drone | B3 | 192 | 57 | 57 | 56 | 3 | 0 | 0/0 | 171 |
| waymo→drone | LOG | 154 | 154 | 154 | 154 | 32 | 13 | 9/4 | 462 |
| waymo→quad | B3 | 211 | 47 | 47 | 47 | 3 | 2 | 0/2 | 141 |
| waymo→quad | LOG | 178 | 176 | 176 | 176 | 26 | 17 | 10/7 | 528 |

| Direction | Factor | Old-zero recoverable | Good in S | Good in C | S-good lost by C | Active q0 excluded C | Correct/wrong initial mass |
|---|---|---|---|---|---|---|---|
| drone→quad | B3 | 201 | 85 | 72 | 13 | 1 | 0.84799/0.07171 |
| drone→quad | LOG | 201 | 72 | 61 | 11 | 0 | 0.87528/0.17449 |
| drone→waymo | B3 | 80 | 20 | 17 | 3 | 0 | 0.85344/0.09939 |
| drone→waymo | LOG | 80 | 17 | 10 | 7 | 0 | 0.78217/0.19139 |
| quad→drone | B3 | 275 | 142 | 128 | 14 | 0 | 0.81130/0.01953 |
| quad→drone | LOG | 275 | 114 | 88 | 26 | 0 | 0.85856/0.16592 |
| quad→waymo | B3 | 114 | 33 | 28 | 5 | 0 | 0.66508/0.00646 |
| quad→waymo | LOG | 114 | 28 | 19 | 9 | 0 | 0.83202/0.14039 |
| waymo→drone | B3 | 180 | 86 | 63 | 23 | 1 | 0.80047/0.02241 |
| waymo→drone | LOG | 180 | 70 | 33 | 37 | 1 | 0.86912/0.21608 |
| waymo→quad | B3 | 146 | 68 | 57 | 11 | 0 | 0.78896/0.00386 |
| waymo→quad | LOG | 146 | 54 | 48 | 6 | 0 | 0.85413/0.19353 |

The wrong labels preserve S, group cardinality, group multiplicities and update budget. They do not preserve initial semantic mass/q0 inclusion; beating them alone does not prove target identity supervision. Full and effective-only diagnostics retain all failures and identity cases. New-B acquisition and previously C=S subgroup results include their contributions to the unchanged full denominator, not only favorable new successes.

## All good and bad changes

| Factor | Control | IoU better | Worse | Same | Changed content groups | Changed recordings | Largest positive pp | Largest negative pp |
|---|---|---|---|---|---|---|---|---|
| B3 | B3_zero | 4 | 7 | 1787 | 10 | 4 | +0.065409 | -0.030236 |
| B3 | partial_F | 16 | 16 | 1766 | 24 | 6 | +0.033243 | -0.027009 |
| LOG | LOG_zero | 49 | 27 | 1722 | 54 | 9 | +0.065409 | -0.030236 |
| LOG | partial_F | 47 | 25 | 1726 | 53 | 8 | +0.046091 | -0.018580 |

Every improving and worsening record relative to own zero AND R131 update is exported. Different control rows can refer to the same observation. Query/group/IoU changes are separate, and no numerical box change is automatically called true identity correction. Group/frame/recording anonymous IDs expose dependence. These historically exposed inputs are development diagnostics, not a new blind test or new-scene validation; one remaining Drone recording cannot support population-level scene CIs.

## Verification and locked decisions

Independent CPU readout max probability error 1.63e-06. Audits bind input/source/calibration/registration/model/evaluation seals; verify first-passing B policy, all original scores, both set rules, cardinality-matched shuffles, normalized parameter writebacks and complete native reader. Independent scalar checks cover12 arms×1798 task values, all directions, recordings and weights.

| Factor | Prelocked working-factor criterion passed |
|---|---|
| B3 | False |
| LOG | True |

The prelocked criterion requires weighted update gains over own zero, own direct, own wrong and original R131 update, without demanding every direction improve. Regardless of that flag, no robust method/deployment adoption follows. No B3+LOG combination, radius sweep, protection loss, source training or next experiment is silently launched. All resource receipts include source calibration, extra registration, engineering and actual model work. Cached context construction and original R131 costs are not presented as new standalone latency.
