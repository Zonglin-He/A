# R131 — Fixed partial-label recipe on content-grouped ordinary expansion

**Decision: retain as a working candidate for further development.** No configuration is selected after viewing these task scores. No robust deployment adoption, new source training or automatic follow-up experiment is claimed.

## Fixed method

Exact R129/R130 unguarded partial_F: same three source-only epoch100/native-foundation and R91-prepared states; original A256 boxes and six decoder positions; registered A/B; complete F_AB readout; 4,896 Point Adapter Up parameters; original normalized-step radius, three steps, feature keep weight1 and nine-trial Armijo backtracking. No unknown-output hard KL, EMA, new teacher, architecture change, extra loss or parameter search.

Initial-A RGB-late and competition context are reused from the unchanged frozen R98 source-prepared cache. A bridge to the R129 provider is exact on all six checks, with source/prepared files pinned. B is re-registered and its point backbone features are computed on this cohort. Unsupported seeds retain A. No C observation is generated, encoded or used, and this study makes no unfitted-C transfer claim.

Four arms: same-model/same-observation zero_F; same partial-set direct readout without updates; correct partial_F updates; and wrong_base_F updates that keep S, |C|, equal-size group label counts and update budget fixed. Identity shuffles are explicitly counted. These are internal controls, not published external baselines.

## Data selection and failed attempt

The first path-only selection was insufficient: 239 selected directory paths had raw point bytes identical to old DEV/EXP observations. Its model run was cancelled before any new task GT scoring; partial outputs, registration results and logs are retained under cancelled attempt001. They do not contribute to the main result. This repair used content hashes and grouping metadata, not target outcomes.

The corrected input lock groups by platform plus exact raw point-file SHA. Entire groups touching old DEV/EXP A/B/C attempts or known protected aliases are excluded. From remaining ordinary groups, choose128 per target: at least one per recording, remaining allocation proportional to group population, deterministic hash selection within recording. Keep every selected alias path, every description and both source directions together.

| Target | Remaining groups | Selected groups | Frame paths | Expressions | Recording clusters |
|---|---|---|---|---|---|
| waymo | 2000 | 128 | 128 | 176 | 80 |
| drone | 205 | 128 | 366 | 366 | 1 |
| quad | 236 | 128 | 357 | 357 | 3 |

Total:384 content groups,851 directory frames,899 expressions and1,798 directed inputs. Content-disjoint from the specified old development A/B/C pool; this is not proof of different physical scenes, and point-file hashing does not establish that visually similar neighboring observations are independent. Older full-VAL scores from other algorithms were already exposed. Therefore this is fixed-configuration expansion, not a blind test or new-scene confirmation. FINAL receives no new access.

The primary aggregation is six equal directions, each using inverse content-group inclusion weights within the prespecified recording strata. All aliases/descriptions remain observations in that expression-level estimand; their dependence is shown with anonymous content/frame/recording clusters. This estimates the clean remaining ordinary population, not original full VAL. Raw means are also reported. Drone has one remaining recording, so no scene-population confidence interval is asserted.

## Registration and coverage

B is the first nearest distinct-image, distinct-point-content ordinary candidate within recording/suffix, selected before SIFT/depth-PnP/reverse/cycle checks. No replacement after a gate failure. 365/845 fixed B pairs pass the original gates. Missing/failed registration, no supported S, no effective labels and no update remain in the full denominator.

## Main results

mIoU, Acc@25 and Acc@50 are percentages; gains are percentage points. Main and raw comparisons are both complete.

| Aggregation | Arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| design_weighted | zero_F | 12.630518 | 25.405773 | 5.054341 |
| design_weighted | direct_partial_F | 12.631040 | 25.405773 | 5.054341 |
| design_weighted | partial_F | 12.676120 | 25.570750 | 5.125430 |
| design_weighted | wrong_base_F | 12.567540 | 25.222216 | 5.008804 |
| raw | zero_F | 12.695263 | 25.325849 | 5.378833 |
| raw | direct_partial_F | 12.695781 | 25.325849 | 5.378833 |
| raw | partial_F | 12.728039 | 25.467232 | 5.426845 |
| raw | wrong_base_F | 12.638584 | 25.142552 | 5.333296 |

| Aggregation | Control | partial_F gain pp |
|---|---|---|
| raw | zero_F | +0.032777 |
| raw | direct_partial_F | +0.032258 |
| raw | wrong_base_F | +0.089455 |
| design_weighted | zero_F | +0.045601 |
| design_weighted | direct_partial_F | +0.045080 |
| design_weighted | wrong_base_F | +0.108580 |

## Six directions and target-platform results

| Direction | n | Zero weighted | Direct weighted | Partial weighted | Wrong weighted | Partial-minus-zero pp | Partial better/worse |
|---|---|---|---|---|---|---|---|
| waymo→drone | 366 | 10.563851 | 10.563851 | 10.563851 | 10.373858 | +0.000000 | 0/0 |
| waymo→quad | 357 | 17.872754 | 17.872754 | 17.675194 | 17.773612 | -0.197560 | 0/2 |
| drone→waymo | 176 | 16.547162 | 16.547162 | 16.547162 | 16.597301 | +0.000000 | 0/0 |
| drone→quad | 357 | 12.707409 | 12.710535 | 12.788485 | 12.780288 | +0.081076 | 3/1 |
| quad→waymo | 176 | 9.927004 | 9.927004 | 10.319461 | 9.943690 | +0.392457 | 1/0 |
| quad→drone | 366 | 8.164930 | 8.164930 | 8.162565 | 7.936490 | -0.002365 | 0/1 |

| Target | Arm | Weighted mIoU | Raw mIoU |
|---|---|---|---|
| waymo | zero_F | 13.237083 | 13.337247 |
| waymo | direct_partial_F | 13.237083 | 13.337247 |
| waymo | partial_F | 13.433312 | 13.494680 |
| waymo | wrong_base_F | 13.270496 | 13.389488 |
| drone | zero_F | 9.364390 | 9.364390 |
| drone | direct_partial_F | 9.364390 | 9.364390 |
| drone | partial_F | 9.363208 | 9.363208 |
| drone | wrong_base_F | 9.155174 | 9.155174 |
| quad | zero_F | 15.290082 | 15.384150 |
| quad | direct_partial_F | 15.291645 | 15.385705 |
| quad | partial_F | 15.231840 | 15.326230 |
| quad | wrong_base_F | 15.276950 | 15.371091 |

All recording results and raw/weighted Acc25/50 are provided in results/ALL_SCENE_ROWS.csv. No every-direction-positive criterion or best-platform output selection is applied.

## Correct versus shuffled candidate sets

| Direction | Active | Correct retained initial mass | Wrong retained initial mass | Correct retains q0 | Wrong retains q0 |
|---|---|---|---|---|---|
| drone→quad | 39 | 0.870027 | 0.090774 | 22 | 2 |
| drone→waymo | 9 | 0.850850 | 0.001442 | 8 | 0 |
| quad→drone | 23 | 0.807806 | 0.001930 | 18 | 0 |
| quad→waymo | 10 | 0.714042 | 0.007121 | 4 | 0 |
| waymo→drone | 43 | 0.800370 | 0.027366 | 40 | 1 |
| waymo→quad | 40 | 0.799018 | 0.002468 | 37 | 0 |

Cardinality/support matching does not match initial semantic probability mass or q0 inclusion. A win over arbitrary retained identities supports plausible-set usefulness, not by itself correct RGB–LiDAR target identity. Effective-only and no-B comparisons are retained in diagnostics/SUBSET_COMPARISONS.csv.

## Coverage and internal ordering

| Direction | n | B usable | Active | Effective wrong | Recoverable zero errors | Good in S | Good in C | q0 in S but excluded C | C teacher prefers better / eligible |
|---|---|---|---|---|---|---|---|---|---|
| drone→quad | 357 | 178 | 39 | 38 | 201 | 72 | 62 | 1 | 1/42 |
| drone→waymo | 176 | 50 | 9 | 9 | 80 | 17 | 15 | 0 | 1/8 |
| quad→drone | 366 | 154 | 23 | 23 | 275 | 114 | 101 | 0 | 2/59 |
| quad→waymo | 176 | 50 | 10 | 10 | 114 | 28 | 24 | 0 | 1/8 |
| waymo→drone | 366 | 154 | 43 | 42 | 180 | 70 | 50 | 1 | 0/47 |
| waymo→quad | 357 | 178 | 40 | 40 | 146 | 54 | 47 | 0 | 0/40 |

The within-C diagnostic uses GT only after all states/predictions were sealed to find the best-IoU retained candidate, then asks whether initial A/B group consensus already favored its group over q0. This is descriptive and does not select a target, loss, radius, case or output. IoU and geometric group identity do not certify true instance identity. A missing good candidate in S, exclusion by C and failure to rank within C are kept distinct.

## Execution cost

| Direction | Joint four-arm seconds/input | Partial accepted steps | Mean active parameter L2 |
|---|---|---|---|
| drone→quad | 0.548809 | 117 | 0.07850203 |
| drone→waymo | 0.475948 | 27 | 0.09357116 |
| quad→drone | 0.494533 | 69 | 0.08970120 |
| quad→waymo | 0.486844 | 30 | 0.09423798 |
| waymo→drone | 0.541285 | 129 | 0.10988043 |
| waymo→quad | 0.548309 | 120 | 0.11575938 |

Timing is the joint four-arm per-input source-prefix/association/readout/update execution, not isolated per-arm or complete cold-start latency. Source/prepared model loading and sensor registration are separate. Initial-A late/residual and calibrated geometry caches were reused; their original construction cost is not included. All bounded job receipts, including the cancelled first attempt, are retained. No claim of a standalone real-time system is made.

## All improved and worsened cases

Relative to zero, 4 direction-inputs improve and 4 worsen; 1790 have identical selected-box IoU. Changed rows span 7 point-content groups and 3 recording clusters. These are not independent scene replicates.

| Better-candidate location in changed rows | Count |
|---|---|
| better_candidate_inside_C | 6 |
| better_candidate_in_S_excluded_C | 1 |
| zero_already_A256_IoU_oracle | 1 |

Largest individual signed contribution to primary six-direction gain: positive +0.065409pp, negative -0.030236pp.

Every good and bad row is included in diagnostics/CHANGED_CASES.csv, with zero/direct/correct/wrong IoU and post-seal coverage/rank fields. All1,798 rows, including no-change, UNKNOWN and no-update cases, are in tasks/. No case is removed and no new rule is developed from this tail analysis. New semantic visual identity annotations are not collected; numerical changes are not automatically labeled identity corrections.

## Verification and disposition

Independent CPU replay of the native matching head has max probability disagreement 1.88e-06. The audit independently verifies original fixed boxes, S/C counts, source-context bridge, shuffled cardinalities, complete direct readout, saved normalized update steps and prediction seals. Batch-yaw IoU agrees with independent oriented geometry within 5.55e-16; an independent scalar audit recomputes all means, weights, paired changes and aggregate metrics.

The prelocked decision uses signs of partial_F minus zero, direct and wrong in the design-weighted six-direction macro. Passing retains a research working candidate only; failure does not authorize a new loss or retuning in this run. Raw and per-target signs remain visible regardless of this flag. The original R129 protected-primary failure and R130 strict-gate failure are not rewritten. No blanket conclusion about all TTA is inferred from this recipe.
