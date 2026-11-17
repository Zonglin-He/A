# R133 — Fixed LOG partial-label TTA at one, three and five steps

**Three steps remains the predeclared primary. One/five are complete exposed-data sensitivity endpoints; no target-GT step selection or new step configuration adoption occurs in this study.**

## Fixed scope

Same R132 LOG_partial working architecture, three source/prepared states, original registered B, initial A context, A256, complete F_AB reader and4,896 Point Adapter Up parameters. Same frozen S/C/wrong-C, kappa2.742430260548657, original per-step rho, feature keep1 and nine-trial Armijo. No B3, new source training, new loss, hard unknown-output protection, recalibration, EMA, new view or module update. All1,798 rows remain:384 point-content groups,851 alias frame paths,899 descriptions and84 recording clusters.

The dataset is historically exposed development. This is neither fresh validation nor new-scene confirmation. Existing R129/R132 pooled cross-source source-own calibration is inherited; no new source labels or target labels choose a parameter. This does not retroactively become strict direction-isolated source-label selection. FINAL has no new access.

## Reuse and trajectory equivalence

All R132 three-step predictions, zero/direct readouts and parameter endpoints are reused unchanged. The first-step endpoint is reconstructed from saved first gradients, GPU-reported gradient norms and accepted line-search fractions. Reassembling all original646×2 three-step states is bit-exact, maximum error0. The five-step endpoint starts from the actual original step3 state and executes only steps4/5, with unchanged objective and line search. Manual normalized-gradient optimization has no additional optimizer state to restore.

Twelve engineering inputs cover all six directions and both B-provider paths. The native step3 readouts are reproduced exactly for proper and wrong states.588 active inputs reuse already-saved B descriptors;58 reconstruct the identical original B descriptors. No original A backbone, source calibration or full three-step fitting leaderboard is rerun. The646 active and1,152 inactive inputs remain fixed across step counts. Correct/wrong updates share the same nominal per-step budget; impossible identity shuffles remain explicitly identical.

## Main results

Primary aggregation: six equal directions, each with the original inverse content-group sampling weights within recording. Raw means, targets, all recordings and all negative/no-op rows are also retained. Percentages are task scores; differences are percentage points.

| Arm | Weighted mIoU | Weighted Acc25 | Weighted Acc50 | Raw mIoU |
|---|---|---|---|---|
| zero_F | 12.630518 | 25.405773 | 5.054341 | 12.695263 |
| direct_F | 12.644153 | 25.407180 | 5.054341 | 12.707285 |
| partial1_F | 12.720862 | 25.452166 | 5.052934 | 12.779621 |
| partial3_F | 12.793107 | 25.694609 | 5.170967 | 12.828305 |
| partial5_F | 12.794152 | 25.759933 | 5.263144 | 12.821481 |
| wrong1_F | 12.673933 | 25.495991 | 5.101286 | 12.748505 |
| wrong3_F | 12.539317 | 24.916941 | 5.101286 | 12.605654 |
| wrong5_F | 12.314279 | 24.567417 | 5.013103 | 12.392969 |
| R131_partial_F | 12.676120 | 25.570750 | 5.125430 | 12.728039 |

| Aggregation | Steps | vs zero pp | vs direct pp | vs same-step wrong pp | vs primary3 pp |
|---|---|---|---|---|---|
| design_weighted | 1 | +0.090344 | +0.076710 | +0.046930 | -0.072245 |
| design_weighted | 3 | +0.162588 | +0.148954 | +0.253790 | +0.000000 |
| design_weighted | 5 | +0.163633 | +0.149999 | +0.479873 | +0.001045 |
| raw | 1 | +0.084358 | +0.072336 | +0.031116 | -0.048684 |
| raw | 3 | +0.133042 | +0.121020 | +0.222652 | +0.000000 |
| raw | 5 | +0.126218 | +0.114196 | +0.428512 | -0.006824 |

The original3-step row must remain identical even if another endpoint has higher target mIoU. No per-case, per-direction, per-scene or global best-endpoint output is selected. The zero/direct controls use precisely the same information as each update; wrong labels retain the same frozen S/C cardinalities and group-size-bin counts, not the same initial semantic mass. These are internal controls, not published external baselines.

## Directions and target-platform results

| Direction | Zero | Direct | Step1 | Step3 primary | Step5 | Wrong1 | Wrong3 | Wrong5 |
|---|---|---|---|---|---|---|---|---|
| waymo→drone | 10.563851 | 10.563851 | 10.878250 | 10.826849 | 10.723310 | 10.668796 | 10.900802 | 11.154843 |
| waymo→quad | 17.872754 | 17.873595 | 17.675445 | 17.529129 | 17.763223 | 17.848554 | 17.404207 | 16.582525 |
| drone→waymo | 16.547162 | 16.514637 | 16.561178 | 16.864618 | 16.827838 | 16.547162 | 16.617132 | 16.273684 |
| drone→quad | 12.707409 | 12.754368 | 12.911359 | 12.826492 | 12.911186 | 12.783116 | 12.364054 | 12.331278 |
| quad→waymo | 9.927004 | 10.003150 | 10.119458 | 10.674359 | 10.361841 | 10.002782 | 10.082348 | 9.966039 |
| quad→drone | 8.164930 | 8.155315 | 8.179485 | 8.037195 | 8.177513 | 8.193185 | 7.867356 | 7.577306 |

| Target | Step1 | Step3 | Step5 |
|---|---|---|---|
| waymo | 13.340318 | 13.769488 | 13.594839 |
| drone | 9.528867 | 9.432022 | 9.450411 |
| quad | 15.293402 | 15.177811 | 15.337204 |

All recording-level tables use anonymous clusters. Single remaining Drone recording cannot justify scene-population confidence intervals; descriptions and source directions sharing frames are not independent scene evidence.

## Actual displacement and cumulative budget

Per-step proposal radius is unchanged. More steps therefore allow a larger cumulative update budget: this experiment does not isolate iteration count at a fixed total displacement. Norms below are actual parameter displacements/path lengths on646 active inputs, not inferred from step count.

| Label | Steps | Mean parameter L2 | Mean actual path length | Mean accepted-radius sum | Mean nominal T*rho | Mean task after | Mean keep after |
|---|---|---|---|---|---|---|---|
| partial | 1 | 0.035723510 | 0.035723510 | 0.035723511 | 0.036052794 | 0.133998458 | 0.000471365 |
| partial | 3 | 0.098699912 | 0.105895682 | 0.105895684 | 0.108158381 | 0.081221586 | 0.003483321 |
| partial | 5 | 0.152873037 | 0.175025346 | 0.175025348 | 0.180263968 | 0.053763405 | 0.007618247 |
| wrong | 1 | 0.035882851 | 0.035882851 | 0.035882851 | 0.036052794 | 3.338439659 | 0.000448218 |
| wrong | 3 | 0.103498689 | 0.107476669 | 0.107476670 | 0.108158381 | 2.767241841 | 0.003672453 |
| wrong | 5 | 0.166146375 | 0.178849346 | 0.178849349 | 0.180263968 | 2.352060951 | 0.009247065 |

| Label | Steps | Active | Nonzero states | Accepted steps | Backward calls if standalone | Reader calls if standalone |
|---|---|---|---|---|---|---|
| partial | 1 | 646 | 646 | 646 | 646 | 2599 |
| partial | 3 | 646 | 646 | 1938 | 1938 | 6565 |
| partial | 5 | 646 | 646 | 3230 | 3230 | 10579 |
| wrong | 1 | 646 | 646 | 646 | 646 | 2592 |
| wrong | 3 | 646 | 646 | 1938 | 1938 | 6496 |
| wrong | 5 | 646 | 646 | 3230 | 3230 | 10413 |

Standalone suffix operation counts reconstruct each trajectory: one backward per attempted active step, two objective forwards plus each line-search trial per step, plus one endpoint readout. They omit common frozen teacher/prefix/geometry construction. Identity wrong controls can be deduplicated. They are operation counts, not measured cold-start latency.

Measured incremental new work in this study reuses existing steps1/3 gradients and caches; it must not be interpreted as complete method latency or evidence that cached3 is cheaper than1.

| Measured component on active inputs | Mean seconds |
|---|---|
| prefix_and_B_seconds | 0.044189 |
| correct_decode1_seconds | 0.019751 |
| correct_extra_steps4and5_seconds | 0.343455 |
| correct_decode5_seconds | 0.020791 |
| joint_new_seconds | 0.945769 |

All direction-specific displacement, objective and cost rows are provided in diagnostics/COST_SUMMARY.csv. No endpoint is chosen by lower loss or greater displacement.

## Decision changes and all good/bad outcomes

| Steps | Control | Better IoU | Worse | Same | Changed content groups | Recordings | Largest positive pp | Largest negative pp |
|---|---|---|---|---|---|---|---|---|
| 1 | zero_F | 27 | 10 | 1761 | 34 | 6 | +0.025582 | -0.030236 |
| 1 | partial3_F | 19 | 24 | 1755 | 34 | 7 | +0.018580 | -0.065409 |
| 3 | zero_F | 49 | 27 | 1722 | 54 | 9 | +0.065409 | -0.030236 |
| 5 | zero_F | 62 | 39 | 1697 | 72 | 9 | +0.065409 | -0.032702 |
| 5 | partial3_F | 23 | 21 | 1754 | 42 | 7 | +0.039961 | -0.046091 |

| Direction | Steps | Active | Query changed | Group changed | Within-group switch | IoU changed | Better/worse |
|---|---|---|---|---|---|---|---|
| drone→quad | 1 | 159 | 26 | 25 | 1 | 17 | 11/6 |
| drone→quad | 3 | 159 | 48 | 46 | 2 | 31 | 18/13 |
| drone→quad | 5 | 159 | 54 | 52 | 2 | 34 | 21/13 |
| drone→waymo | 1 | 39 | 5 | 5 | 0 | 1 | 1/0 |
| drone→waymo | 3 | 39 | 8 | 8 | 0 | 3 | 3/0 |
| drone→waymo | 5 | 39 | 10 | 10 | 0 | 3 | 3/0 |
| quad→drone | 1 | 85 | 6 | 6 | 0 | 2 | 1/1 |
| quad→drone | 3 | 85 | 17 | 17 | 0 | 7 | 4/3 |
| quad→drone | 5 | 85 | 29 | 29 | 0 | 14 | 9/5 |
| quad→waymo | 1 | 33 | 2 | 2 | 0 | 2 | 2/0 |
| quad→waymo | 3 | 33 | 7 | 7 | 0 | 5 | 5/0 |
| quad→waymo | 5 | 33 | 10 | 10 | 0 | 5 | 4/1 |
| waymo→drone | 1 | 154 | 19 | 18 | 1 | 7 | 6/1 |
| waymo→drone | 3 | 154 | 32 | 31 | 1 | 13 | 9/4 |
| waymo→drone | 5 | 154 | 39 | 38 | 1 | 16 | 9/7 |
| waymo→quad | 1 | 176 | 13 | 13 | 0 | 8 | 6/2 |
| waymo→quad | 3 | 176 | 26 | 26 | 0 | 17 | 10/7 |
| waymo→quad | 5 | 176 | 40 | 40 | 0 | 29 | 16/13 |

## Revisit all R13249 improved and27 worsened records

The exact76 previously changed records are revisited without selecting a favorable subset or using their labels in updates. Both continuous IoU and Acc25 context are retained. Their fixed group/retained-set categories are:

| Previous change | Candidate-competition category | Records |
|---|---|---|
| better | both_retained_groups_reordered | 27 |
| better | unsupported_group_competition | 20 |
| better | same_geometric_group_query_switch | 2 |
| worse | unsupported_group_competition | 8 |
| worse | both_retained_groups_reordered | 19 |

Among the27 previous harms, 8 had a zero-step Acc25-positive selected box; 0 had its previous selected supported group excluded by C. These categories are descriptive, not deployment rules or identity certification.

The table R132_ALL76_CASES.csv records zero/direct/1/3/5 IoUs, membership of old/new selected groups in S/C, maximum possible supported/retained IoUs, and whether zero was already the fixed A256 oracle. This distinguishes retaining/excluding a better candidate from changes inside C or unsupported competition. A geometric duplicate group is a proxy, not a verified object identity; ambiguous identity remains unknown. No sample-specific patches or oracle-best-step predictions are used. All new sensitivity positive/negative cases remain in CHANGED_CASES.csv and full task tables.

## Verification and disposition

Independent CPU replay max probability error 1.7e-06; reconstructed parameter max error 0; objective discontinuity from original step3 to first continued evaluation 0. Audits verify every old reference, every normalized step and actual cumulative path, fixed original boxes, scope-only parameter writes and all scalar/cluster aggregates.

All new endpoints are sealed before reading R131 saved all-A256 IoU vectors. There is no new raw target GT geometry, source-label step tuning, target-GT step selection or FINAL access. Three steps remains the working recipe by predeclaration; one/five describe sensitivity only. Existing R132 adoption and limitations remain unchanged. No new radius, kappa, protection, B3 combination or model training follows automatically.
