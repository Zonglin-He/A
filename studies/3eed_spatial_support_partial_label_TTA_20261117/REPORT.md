# R129 — Spatial-support-conditioned partial-label adaptation

Status: completed; all predictions sealed before task scoring. Exact method and adoption gate were locked before target GT. No target-dependent retuning or arm switching occurred.

**Decision: does not pass the prelocked development comparison gate; do not adopt this recipe or change the registered main method.**

The experiment keeps the R127 native-foundation zero-up/zero-point-gate starting state, three epoch100 source-only checkpoints and fixed source-A R91 late context. Only 4,896 point Adapter Up parameters change. All A256 boxes/decoder positions remain fixed. No new source training, external expert, raw-media forward or FINAL access.

DEV768 and EXP1536 are historically exposed six-direction development inputs. Each direction has 128 DEV and 256 EXP rows. All 2,304 direction-inputs remain in the denominator, including 154 protected aliases that reuse the archived no-update output without new feature/media/GT indexing. Different source directions can reuse physical observations; these are not 2,304 independent scenes.

## Main task results

mIoU is a six-direction equal-weight mean, in percent. All methods choose the same cached A256. The zero_F reader is this experiment’s own F_AB zero-update baseline, not the ordinary frozen R91 model. original_source and prepared_A are additional archived references.

| Method | DEV mIoU | EXP mIoU | DEV Acc@25 | EXP Acc@25 | DEV Acc@50 | EXP Acc@50 |
|---|---|---|---|---|---|---|
| original_source | 16.811383 | 16.144619 | 30.989583 | 30.598958 | 14.973958 | 12.434896 |
| prepared_A | 16.228652 | 16.251067 | 29.947917 | 30.598958 | 12.630208 | 11.783854 |
| zero_F | 17.279671 | 16.418423 | 32.161458 | 30.989583 | 14.322917 | 12.044271 |
| direct_global_F | 17.192255 | 16.255326 | 32.031250 | 30.729167 | 14.322917 | 11.848958 |
| R127_global_F | 16.954268 | 16.370022 | 31.510417 | 30.924479 | 14.062500 | 12.109375 |
| global_scope_F | 16.954268 | 16.370022 | 31.510417 | 30.924479 | 14.062500 | 12.109375 |
| direct_conditional_F | 17.229421 | 16.252912 | 31.901042 | 30.664062 | 14.453125 | 11.848958 |
| conditional_F | 16.962684 | 16.373575 | 31.250000 | 30.924479 | 14.322917 | 12.109375 |
| direct_partial_F | 17.298035 | 16.436529 | 32.161458 | 31.054688 | 14.453125 | 12.109375 |
| partial_F | 17.394852 | 16.498101 | 32.291667 | 31.184896 | 14.583333 | 12.044271 |
| partial_guard_F | 17.279671 | 16.416507 | 32.161458 | 30.989583 | 14.322917 | 12.044271 |
| wrong_guard_F | 17.274008 | 16.417346 | 32.161458 | 30.924479 | 14.322917 | 12.109375 |

The R127 global loss and same-scope global control are reused sealed states; no completed R127 experiment was rerun. global_scope qualifies the same 785 inputs as conditional, making eligibility effects visible separately from the full archived global result. The conditional arm also changes duplicate-group normalization, so it is not a pure visibility-only causal contrast.

## Primary paired effects and zero-update controls

| Cohort | Control | Primary gain (pp) | Scene-poststratified gain (pp) |
|---|---|---|---|
| DEV | zero_F | +0.000000 | +0.000000 |
| DEV | direct_conditional_F | +0.050250 | +0.103814 |
| DEV | direct_partial_F | -0.018364 | -0.006234 |
| DEV | global_scope_F | +0.325403 | +0.192018 |
| DEV | wrong_guard_F | +0.005663 | +0.001922 |
| EXP | zero_F | -0.001917 | -0.001213 |
| EXP | direct_conditional_F | +0.163595 | +0.202862 |
| EXP | direct_partial_F | -0.020022 | -0.012659 |
| EXP | global_scope_F | +0.046485 | +0.069101 |
| EXP | wrong_guard_F | -0.000839 | +0.020682 |

Direct conditional and direct partial readouts use the same frozen candidate sets and A/B teacher information, preserve unknown query probabilities and S total mass, and require no parameter update. The partial direct readout removes excluded S\C probability and redistributes it inside C; it does not invent an ordering inside C. These are matched information controls, not external published baselines.

| Cohort | Method | Control | Improved | Worsened | Unchanged | Query changes | Mean gain (pp) |
|---|---|---|---|---|---|---|---|
| DEV | global_scope_F | zero_F | 5 | 12 | 751 | 32 | -0.325403 |
| DEV | conditional_F | zero_F | 4 | 13 | 751 | 30 | -0.316987 |
| DEV | partial_F | zero_F | 5 | 3 | 760 | 10 | +0.115181 |
| DEV | partial_guard_F | zero_F | 0 | 0 | 768 | 0 | +0.000000 |
| DEV | partial_guard_F | partial_F | 3 | 5 | 760 | 10 | -0.115181 |
| DEV | partial_guard_F | wrong_guard_F | 1 | 0 | 767 | 3 | +0.005663 |
| DEV | partial_guard_F | direct_partial_F | 0 | 1 | 767 | 1 | -0.018364 |
| DEV | partial_guard_C | zero_C | 2 | 0 | 766 | 2 | +0.017642 |
| EXP | global_scope_F | zero_F | 13 | 17 | 1506 | 53 | -0.048402 |
| EXP | conditional_F | zero_F | 8 | 12 | 1516 | 37 | -0.044848 |
| EXP | partial_F | zero_F | 6 | 4 | 1526 | 19 | +0.079678 |
| EXP | partial_guard_F | zero_F | 0 | 1 | 1535 | 5 | -0.001917 |
| EXP | partial_guard_F | partial_F | 3 | 6 | 1527 | 14 | -0.081594 |
| EXP | partial_guard_F | wrong_guard_F | 1 | 2 | 1533 | 6 | -0.000839 |
| EXP | partial_guard_F | direct_partial_F | 1 | 2 | 1533 | 5 | -0.020022 |
| EXP | partial_guard_C | zero_C | 0 | 0 | 1536 | 2 | +0.000000 |

## All six directions

| Cohort | Direction | Zero F | Global same scope | Conditional | Partial | Partial+bound | Direct partial | Wrong binding |
|---|---|---|---|---|---|---|---|---|
| DEV | waymo→drone | 18.193613 | 17.513606 | 17.513606 | 18.193613 | 18.193613 | 18.193613 | 18.193613 |
| DEV | waymo→quad | 32.566884 | 32.543149 | 33.016085 | 33.019131 | 32.566884 | 32.566884 | 32.566884 |
| DEV | drone→waymo | 14.711135 | 14.368402 | 14.077677 | 14.711135 | 14.711135 | 14.711135 | 14.711135 |
| DEV | drone→quad | 16.631282 | 16.047930 | 15.861297 | 16.739598 | 16.631282 | 16.741469 | 16.597304 |
| DEV | quad→waymo | 9.725965 | 9.607077 | 9.661497 | 9.725965 | 9.725965 | 9.725965 | 9.725965 |
| DEV | quad→drone | 11.849145 | 11.645445 | 11.645943 | 11.979668 | 11.849145 | 11.849145 | 11.849145 |
| EXP | waymo→drone | 15.090863 | 14.652305 | 14.752748 | 15.090863 | 15.090863 | 15.090863 | 14.986405 |
| EXP | waymo→quad | 30.897831 | 30.850023 | 30.893166 | 30.897831 | 30.897831 | 30.897831 | 30.897831 |
| EXP | drone→waymo | 14.718295 | 14.332351 | 14.586210 | 14.706796 | 14.706796 | 14.718295 | 14.718295 |
| EXP | drone→quad | 16.943861 | 17.215145 | 16.910342 | 17.669434 | 16.943861 | 17.052497 | 17.041856 |
| EXP | quad→waymo | 11.330441 | 11.732305 | 11.732305 | 11.100500 | 11.330441 | 11.330441 | 11.330441 |
| EXP | quad→drone | 9.529250 | 9.438000 | 9.366682 | 9.523182 | 9.529250 | 9.529250 | 9.529250 |

All 25 readouts, Acc@25/50, target-platform means, paired differences and scene clusters are in the public scalar tables. Scene poststratification uses existing full-cohort scene counts only, with no new full-cohort predictions. It is an exposed-cohort sensitivity analysis, not full evaluation or a new-scene claim. Covered full-cohort mass minimum is 0.998149. No single-scene observation bootstrap is reported as a multi-scene confidence interval.

## Teacher qualification and label-set quality

Source-only shared calibration: n_A/n_B >= 4, B fraction >= 0.5; tau=0.335611596211; cumulative unknown-output KL budget=0.000789698105. Tau is a pooled source-A 90th-percentile gap to a source-GT Acc@25-capable group; the budget is the median nonzero collapsed KL of 23 saved source-GT task-improving one-step states. This source-only offline development does not update source weights. The .5 registration majority is a predeclared convention, not a source-B calibrated guarantee.

| Cohort | All | Qualified S | Nontrivial C | Recoverable zero errors in partial scope | Those with Acc25 candidate in C | Original q0 in S | q0 excluded | Excluded q0 originally Acc25-correct | C retains S-best |
|---|---|---|---|---|---|---|---|---|---|
| DEV | 768 | 258 | 81 | 28 | 8 | 50 | 0 | 0 | 40 |
| EXP | 1536 | 527 | 176 | 70 | 17 | 123 | 1 | 0 | 100 |

S is observation qualification, not a true-target set. C expresses non-exclusion by either observation, not a verified identity. Supports are unique A seed locations, not independent physical returns or instance masks. A/B correspondence can reuse B neighbors. The procedure cannot prove reference relations, unseen surfaces or dynamic-object identity. All GT-based coverage diagnostics above are computed only after prediction sealing and never screen an update.

## Actual updates and endpoint constraint

| Cohort | Loss arm | Active | Nonzero | Accepted steps | Unknown-bound rejected trials | Armijo rejected trials | Mean active parameter L2 | Maximum unknown KL |
|---|---|---|---|---|---|---|---|---|
| DEV | conditional | 258 | 258 | 774 | 0 | 1211 | 0.03290914 | 1.07721039 |
| DEV | partial | 81 | 81 | 243 | 0 | 28 | 0.09358309 | 1.41301354 |
| DEV | partial_guard | 81 | 81 | 235 | 877 | 21 | 0.02313842 | 0.00078895 |
| DEV | wrong_guard | 81 | 81 | 236 | 779 | 0 | 0.02981184 | 0.00078969 |
| EXP | conditional | 527 | 526 | 1577 | 0 | 2473 | 0.03369353 | 0.34343342 |
| EXP | partial | 176 | 176 | 528 | 0 | 60 | 0.09250120 | 0.97917758 |
| EXP | partial_guard | 176 | 176 | 509 | 1772 | 36 | 0.02628377 | 0.00078962 |
| EXP | wrong_guard | 176 | 176 | 509 | 1652 | 5 | 0.03191280 | 0.00078947 |

The partial loss has exactly zero direct gradient outside S. Shared Up parameters can nevertheless alter unknown candidates, so the endpoint guard constrains the actual complete deployed distribution. All endpoints are checked relative to the initial F distribution, not only to the preceding step. C=S causes zero task update; no thresholds were tightened to manufacture coverage. The wrong-binding control preserves group/query cardinality within equal group-size bins; some sets can remain identical.

## Error types and unfitted-observation check

| Cohort | Subset | Method | Control | n | Better | Worse | Gain (pp) |
|---|---|---|---|---|---|---|---|
| DEV | teacher_worse | global_scope_F | zero_F | 10 | 0 | 6 | -8.892798 |
| DEV | teacher_worse | conditional_F | zero_F | 10 | 0 | 7 | -11.917513 |
| DEV | teacher_worse | partial_guard_F | zero_F | 10 | 0 | 0 | +0.000000 |
| DEV | teacher_kept_source_query | global_scope_F | zero_F | 691 | 1 | 6 | -0.325627 |
| DEV | teacher_kept_source_query | conditional_F | zero_F | 691 | 1 | 6 | -0.293921 |
| DEV | teacher_kept_source_query | partial_guard_F | zero_F | 691 | 0 | 0 | +0.000000 |
| EXP | teacher_worse | global_scope_F | zero_F | 24 | 0 | 14 | -16.339449 |
| EXP | teacher_worse | conditional_F | zero_F | 24 | 0 | 8 | -7.493717 |
| EXP | teacher_worse | partial_guard_F | zero_F | 24 | 0 | 1 | -0.122660 |
| EXP | teacher_kept_source_query | global_scope_F | zero_F | 1360 | 6 | 3 | +0.055094 |
| EXP | teacher_kept_source_query | conditional_F | zero_F | 1360 | 3 | 4 | -0.045643 |
| EXP | teacher_kept_source_query | partial_guard_F | zero_F | 1360 | 0 | 0 | +0.000000 |

| Cohort | Unfitted C method | mIoU | Delta vs zero C (pp) |
|---|---|---|---|
| DEV | zero_C | 16.492047 | +0.000000 |
| DEV | R127_global_C | 16.164069 | -0.327978 |
| DEV | conditional_C | 16.262892 | -0.229155 |
| DEV | partial_C | 16.455844 | -0.036202 |
| DEV | partial_guard_C | 16.509689 | +0.017642 |
| DEV | wrong_guard_C | 16.513087 | +0.021040 |
| EXP | zero_C | 16.204012 | +0.000000 |
| EXP | R127_global_C | 16.143559 | -0.060453 |
| EXP | conditional_C | 16.253279 | +0.049266 |
| EXP | partial_C | 16.332358 | +0.128345 |
| EXP | partial_guard_C | 16.204012 | +0.000000 |
| EXP | wrong_guard_C | 16.218971 | +0.014959 |

C is decoded only after all proposed parameter states are fixed; it participates in neither fitting nor acceptance nor method selection. It is a registered observation, not independent-scene confirmation. Every task change is retained in CHANGED_CASES.csv, including all negative changes. IoU changes and same-geometry-group flags do not alone certify whether the physical instance changed.

## Measurement and conclusion

Independent CPU full matching/readout replay: maximum probability error 2.14e-06; endpoint violations 0; all 785 global reference states bit-identical to archived R127. Independent oriented-IoU versus batch geometry error 6.66e-16. All scalar means, paired counts, poststratified weights and anonymous exports are independently recalculated.

The prelocked primary is partial_guard_F. Passing requires positive gains against own zero, both same-information direct controls, same-scope global KL and wrong binding in both cohorts, before and after scene weighting. C is descriptive only. A loss decrease or reduction in update coverage does not satisfy that gate. See DECISION.json for the exact outcome. No additional step, tau, epsilon, support threshold, primary arm, source training or model-size search follows this scoring. The registered main method remains unchanged.

See LITERATURE_AND_PROTOCOL.md for PASLE/Latte provenance, exact adaptation boundaries, source calibration and reproduction details. This result bounds this recipe on exposed cohorts; it does not establish that every possible partial-label or spatial TTA method succeeds or fails.
