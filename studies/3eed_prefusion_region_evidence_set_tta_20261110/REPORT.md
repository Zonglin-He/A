# R66 — Pre-fusion regional evidence and conditional-set matching adaptation

Completed on historically exposed DEV336: 168 Drone and 168 Quad observations in two Drone and three Quad scene clusters. All results are development measurements, not independent confirmation. No FINAL384 access, full5912 evaluation, source training, external predictor or external-model feedback cache. Only the primary regional-contrast path was executed; RobOT correspondence and 3DSSD sampling changes were not combined.

**Result:** after one disclosed control-validity repair, a real one-proposal matching-head update plus the unchanged R59 fusion reaches **22.818641%**, **+0.207525 pp** above R59. Both platform mIoUs improve. Its direct regional-evidence readout remains below R59, ordinary ABC updating also declines, and heldout-sampling C declines **0.117246 pp** from its own frozen source. Thus the R59-specific development gain is retained, but the predeclared joint supervision/transfer criterion is not met. Do not adopt this recipe as robust TTA, and do not describe all arms as failures.

## Fixed intervention and supervision

Waymo-only epoch100, original 256 final boxes, source backbone and candidate geometry fixed. Capture source A/B before point–language fusion. A region is a hypothesis group's seed support. Remove its visual features before fusion, re-zero masked residuals after each cross-encoder layer, and block visual self-attention and language-to-visual keys. Re-gather decoder queries from the intervened fused features. Persistently mask decoder visual keys and keys of queries anchored on removed seeds. Preserve original spatial anchors and six decoder geometry inputs; do not silently transplant an unrelated forward's query identities.

This is **pre-fusion seed-feature ablation**, not physical-object deletion: PointNet exterior receptive fields and fixed geometry priors can retain information. Engineering checks on real samples and full independent replays show zero-intervention equality, zero blocked attention and no influence from replacing blocked features with a large constant. These establish implementation validity, not object identity or supervision quality.

Computational duplicates require box IoU >=0.5 AND point-support Jaccard >=0.5 with at least four shared points. Use four highest A/B source group-mass hypotheses, not GT-selected recoverable candidates. These groups are not instance truth. A/B correspondence uses the existing transport formula on A/B point support only. Category uses the existing native first-lexicon-noun proxy from the actual description, without external parsing; it is not independently verified subject-class annotation. Category fusion is recomputed at the same full-description spatial hypotheses and positions. It is a conditional spatial probe, not unconstrained category grounding.

For group g, let D(t,g,r)=log p(g|t)-log p(g|t,ablate r), using native candidate matching group mass. Evidence is E_g=[D(full,g,region)-D(full,g,control)]-[D(category,g,region)-D(category,g,control)]. Stable positive signs in both A/B at epsilon 1e-6 define partial positive set S. Stable positive/negative groups define tested set Cset; other candidates stay UNKNOWN. Both signs are required for informative exclusion. UNKNOWN is never promoted to negative. No single local preference becomes a full-candidate positive label.

Direct use reweights only tested-pool mass and leaves every untested candidate's probability unchanged, invariant to common evidence offsets. A second no-update control assigns the tested-pool mass to S. Full-only, category-only, control-only and unchanged R59 are retained.

In parallel, run a real episodic update of the existing 184,960-parameter matching MLP: one Adam proposal, lr 0.0005, gradient clip 1, conditional set-mass loss plus source full-output KL with weight 1. Use the previously fixed nine half-scales for loss-only acceptance; no GT step selection. No layer/strength/loss grid. A/B evidence and all fits are sealed before C features are opened. C is a heldout sampling of the same original observation, not a new physical viewpoint or disjoint measurement. The A256 geometry and the existing C transport/UNKNOWN conventions remain fixed.

## Initial v1 and the one permitted repair

V1 matched the number of removed seeds but did not match removed decoder query keys. Only **14/2174** paired valid view-groups matched query-key removal; aggregate counts were **36231 vs 3983**. Thus regional and control ablations had unequal computational strength. V1 task scores remain real, but do not cleanly identify a semantic regional effect.

After v1 task/control diagnostics were exposed, lock exactly one validity revision before any v2 task score: choose pool-external control seeds in query-anchor/nonanchor strata to match both counts. Reuse all sealed source/region responses and regenerate only controls. No E formula, sign epsilon, optimizer, loss, source, pool, update site or step-budget changes. No third revision or positive-support threshold run.

V2 matches both counts for **2174/2174 valid paired view-groups**, aggregate **36231 vs 36231**. Independent audit also verifies all 2240 supported individual view-groups. Both-view valid groups remain 1087, so this is not simply a reduction in structural coverage. The control is outside compared regions, not certified semantically irrelevant; it may still contain useful contextual information. Same-v2-tested-pool v1-evidence direct controls separate coverage from changed evidence. This repair is disclosed development iteration, not a fresh preregistered confirmation.

## Complete v1 task table

Units: percent or percentage points. Improvement/worsening counts compare IoU against R59, not instance identity.

| method | Drone_mIoU | Quad_mIoU | balanced_mIoU | Acc25 | Acc50 | improved_vs_R59 | worsened_vs_R59 |
|---|---|---|---|---|---|---|---|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 | 14 | 21 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 | 8 | 10 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 | 0 | 0 |
| C_source | 16.085318 | 29.342198 | 22.713758 | 43.154762 | 21.726190 | 15 | 15 |
| ABC_E_direct | 15.851330 | 28.712359 | 22.281845 | 40.476190 | 21.130952 | 15 | 23 |
| R59_E_direct | 15.781232 | 28.343194 | 22.062213 | 40.476190 | 21.130952 | 9 | 18 |
| R59_set_direct | 15.731721 | 28.463217 | 22.097469 | 40.773810 | 21.428571 | 4 | 12 |
| R59_full_only | 15.607214 | 28.813017 | 22.210116 | 41.369048 | 21.726190 | 5 | 12 |
| R59_category_only | 15.633696 | 28.823649 | 22.228672 | 42.261905 | 21.428571 | 9 | 14 |
| R59_control_only | 16.348260 | 28.564777 | 22.456518 | 42.857143 | 21.726190 | 4 | 6 |
| C_E_direct | 15.663507 | 28.805268 | 22.234388 | 40.178571 | 21.130952 | 21 | 26 |
| C_set_direct | 15.408777 | 29.126368 | 22.267573 | 41.071429 | 21.726190 | 16 | 27 |
| ABC_tta1 | 15.480302 | 28.508367 | 21.994334 | 41.071429 | 21.428571 | 8 | 15 |
| R59_tta1 | 16.637110 | 28.490336 | 22.563723 | 42.261905 | 22.023810 | 4 | 7 |
| C_tta1 | 15.897453 | 29.175427 | 22.536440 | 42.261905 | 21.726190 | 15 | 20 |

## Complete matched-control v2 task table

| method | Drone_mIoU | Quad_mIoU | balanced_mIoU | Acc25 | Acc50 | improved_vs_R59 | worsened_vs_R59 |
|---|---|---|---|---|---|---|---|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 | 14 | 21 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 | 8 | 10 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 | 0 | 0 |
| C_source | 16.085318 | 29.342198 | 22.713758 | 43.154762 | 21.726190 | 15 | 15 |
| ABC_E_direct | 16.213596 | 28.070473 | 22.142035 | 40.476190 | 20.833333 | 17 | 24 |
| R59_E_direct | 16.077133 | 28.302323 | 22.189728 | 40.178571 | 20.833333 | 14 | 20 |
| R59_set_direct | 15.695480 | 28.836664 | 22.266072 | 41.964286 | 21.130952 | 1 | 7 |
| R59_full_only | 16.246535 | 28.285971 | 22.266253 | 41.369048 | 21.428571 | 8 | 13 |
| R59_category_only | 15.772910 | 28.640766 | 22.206838 | 41.071429 | 21.726190 | 8 | 12 |
| R59_control_only | 16.192675 | 28.680220 | 22.436447 | 42.261905 | 21.428571 | 12 | 10 |
| C_E_direct | 16.374832 | 28.283777 | 22.329305 | 40.773810 | 21.130952 | 22 | 27 |
| C_set_direct | 15.412696 | 29.342198 | 22.377447 | 41.964286 | 21.428571 | 14 | 22 |
| ABC_tta1 | 15.585822 | 28.841525 | 22.213673 | 41.964286 | 21.428571 | 11 | 11 |
| R59_tta1 | 16.657288 | 28.979993 | 22.818641 | 42.857143 | 22.023810 | 5 | 2 |
| C_tta1 | 15.850826 | 29.342198 | 22.596512 | 42.559524 | 21.726190 | 15 | 18 |
| R59_v1_E_same_v2_pool | 15.058290 | 28.294053 | 21.676172 | 39.583333 | 20.238095 | 12 | 21 |
| C_v1_E_same_v2_pool | 15.381839 | 29.047344 | 22.214591 | 40.178571 | 20.833333 | 24 | 26 |

The primary v2 R59-updated readout improves five inputs and worsens two versus R59, with 12 query changes. Acc@25 changes from 43.154762% to 42.857143%; Acc@50 changes from 21.726190% to 22.023810%. Hence even the positive mIoU result does not improve every task measure. Ordinary updated ABC is 22.213673% versus frozen ABC 22.302460%. Updated C is 22.596512% versus C source 22.713758%. R59 uses the same unchanged representation-fusion formula, recomputed with the updated matching head. The positive result is readout-dependent; it is not evidence that arbitrary matching updates or future observations improve.

V2 R59 E-direct is 22.189728%, above v1 evidence on the identical v2 tested pool (21.676172%), yet still below frozen R59. This demonstrates that matching computational intervention strength changes evidence use, without establishing that the new evidence correctly identifies the referent.

## Coverage, reliability and accepted updates

| diagnostic | v1 | v2 |
|---|---|---|
| inputs | 336 | 336 |
| A256_oracle_good | 268 | 268 |
| pool_oracle_good | 196 | 196 |
| tested_oracle_good | 148 | 147 |
| positive_set_oracle_good | 130 | 121 |
| R59_failure_A256_recoverable | 123 | 123 |
| R59_failure_pool_recoverable | 51 | 51 |
| R59_failure_tested_recoverable | 34 | 35 |
| R59_failure_positive_set_has_good | 29 | 28 |
| eligible_inputs | 85 | 47 |
| eligible_S_max_above_N | 31 | 14 |
| eligible_S_max_below_N | 17 | 13 |
| eligible_S_max_equal_N | 37 | 20 |
| eligible_tested_has_good_but_S_excludes_all_good | 10 | 7 |
| eligible_original_R59_good_but_in_negative | 8 | 7 |
| valid_groups | 1087 | 1087 |
| valid_full_region_drop_positive | 242 | 242 |
| valid_category_region_drop_positive | 137 | 137 |
| valid_good_groups | 305 | 305 |
| valid_good_groups_positive_E | 169 | 170 |
| valid_bad_groups | 782 | 782 |
| valid_bad_groups_positive_E | 343 | 375 |
| positive_E_groups | 512 | 545 |
| positive_E_both_raw_full_drops_positive | 142 | 125 |
| positive_E_both_corrected_full_drops_positive | 132 | 39 |
| positive_E_mean_full_support_nonpositive | 350 | 479 |
| valid_view_group_controls | 2174 | 2174 |
| matched_query_key_counts | 14 | 2174 |
| region_query_keys_total | 36231 | 36231 |
| control_query_keys_total | 3983 | 36231 |

All task-proxy diagnostics above use saved candidate IoUs only after predictions are sealed. A group's best IoU >0.25 is a localization proxy, **not independent instance-identity truth**. They never enter candidate generation, evidence signs, acceptance or optimization.

A256 contains a potentially good box for 123 R59 failures, but the four-group pool covers only 51. V2 tests 35 and retains a good box in its positive set for 28. Thus 72/123 recoverable errors are already outside the compared pool; later evidence cannot address them. Among the 47 eligible v2 inputs, positive-set best IoU exceeds negative-set best IoU in 14, is lower in 13 and equal in 20. Seven eligible inputs exclude every tested good candidate from S; seven place an originally good R59 selection in the negative set. This is substantive erroneous supervision alongside limited coverage, not just an optimizer issue.

A second defect is interpretive: **479/545 positive-E groups have nonpositive mean control-corrected full-description support**. Relative full-minus-category evidence can become positive because the category term is even more negative; it need not indicate that the full sentence actually relies on the region. Only 39 positive-E groups have positive control-corrected full support in both views. Positive sign stability therefore does not establish positive referent evidence. A positive-support filter was not tested and must not be advertised as a solution.

V1: 85 eligible, 84 accepted nonzero updates, one loss-only rollback, 251 with no feedback. V2: 47 eligible, 47 accepted, 289 without feedback. No-feedback, UNKNOWN, rejection and fallback observations remain in the full336 denominator. Independent functional gradients, accepted parameters and all direct/C readouts match exactly; scores are not explained by zero gradients or a missing parameter update.

## Paired influence of the control repair

The following is post-seal saved-scalar stratification, not an evaluated deployment filter or a GT-selected method. Contributions use denominator336, not only the active subset.

| v1_update_eligible | v2_update_eligible | n | v1_R59_full336_contribution_pp | v2_R59_full336_contribution_pp | v1_C_full336_contribution_pp | v2_C_full336_contribution_pp |
|---|---|---|---|---|---|---|
| False | False | 243 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| False | True | 8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| True | False | 46 | -0.173189 | 0.000000 | -0.055299 | 0.000000 |
| True | True | 39 | 0.125796 | 0.207525 | -0.122019 | -0.117246 |

The 46 inputs eligible only in v1 contributed -0.173189 pp to R59 and are inactive in v2. The 39 shared-eligible inputs contribute +0.125796 pp with v1 versus +0.207525 pp with v2. Eight newly eligible inputs produce no task change. Thus v2 is helped both by avoiding harmful v1 activity and by changing updates on shared inputs. This does not identify the exact causal role of correct semantic evidence: no shuffled-positive-set update control was run, and both versions' C contributions on the shared inputs remain negative.

## Five scene clusters, all retained

No frame-independent significance claim or pooled CI is made from the five exposed scene clusters. All arms for both versions are exported; the main v2 comparisons are shown below.

| method | platform | scene_alias | n | mIoU | gain_vs_R59_pp |
|---|---|---|---|---|---|
| R59 | drone | D_S01 | 106 | 11.761612 | 0.000000 |
| R59 | drone | D_S02 | 62 | 24.291041 | 0.000000 |
| R59 | quad | Q_S01 | 62 | 9.632116 | 0.000000 |
| R59 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| R59 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| C_source | drone | D_S01 | 106 | 11.328022 | -0.433590 |
| C_source | drone | D_S02 | 62 | 24.218760 | -0.072281 |
| C_source | quad | Q_S01 | 62 | 10.117087 | 0.484971 |
| C_source | quad | Q_S02 | 49 | 36.238137 | 0.170556 |
| C_source | quad | Q_S03 | 57 | 44.325635 | 0.815865 |
| R59_E_direct | drone | D_S01 | 106 | 11.712996 | -0.048616 |
| R59_E_direct | drone | D_S02 | 62 | 23.538400 | -0.752641 |
| R59_E_direct | quad | Q_S01 | 62 | 9.987700 | 0.355584 |
| R59_E_direct | quad | Q_S02 | 49 | 35.070901 | -0.996679 |
| R59_E_direct | quad | Q_S03 | 57 | 42.404889 | -1.104881 |
| R59_tta1 | drone | D_S01 | 106 | 12.536308 | 0.774696 |
| R59_tta1 | drone | D_S02 | 62 | 23.702835 | -0.588206 |
| R59_tta1 | quad | Q_S01 | 62 | 10.020492 | 0.388376 |
| R59_tta1 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| R59_tta1 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| C_tta1 | drone | D_S01 | 106 | 11.307484 | -0.454128 |
| C_tta1 | drone | D_S02 | 62 | 23.618477 | -0.672565 |
| C_tta1 | quad | Q_S01 | 62 | 10.117087 | 0.484971 |
| C_tta1 | quad | Q_S02 | 49 | 36.238137 | 0.170556 |
| C_tta1 | quad | Q_S03 | 57 | 44.325635 | 0.815865 |

The scene table's delta column always references R59. For heldout-sampling transfer, compare C_tta1 against C_source within the same scene instead; these are different baselines. All original negative comparisons (8254 within-run ordered losses and 71 v2-minus-v1 losses) are preserved as anonymous scalar rows. No adverse input is deleted from scoring.

## Original-method audit and limits

[CRG paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10172.pdf) and [official code](https://github.com/meetdavidwan/crg) combine original and blacked-out generation logits as (1+alpha)original-alpha*blackout. Here we test native grounding probabilities and a fixed spatial feature-ablation interface, not sentence generation likelihood or a CRG reproduction.

[DeYO official implementation](https://github.com/Jhyun17/DeYO/blob/main/methods/deyo.py) checks prediction confidence change under perturbation, with positive PLPD selection and entropy filtering/reweighting. Our A/B stable relative contrast does not reproduce its full reliability recipe. In particular positive E does not imply positive full-description support. These results cannot be generalized to complete DeYO.

[PASLE official implementation](https://github.com/palm-ml/PASLE/blob/main/code/adapt_algorithm.py) includes negative log probability mass over a partial candidate set, alongside its own selection/buffer/prototype and temperature machinery. Our denominator conditions on tested groups and adds source-output KL to protect untested predictions. It is PASLE-inspired partial-set supervision, not original PASLE. Different text inputs have different candidate identities; no universal query-class prototypes are assumed.

Readout, correspondence, grouping, native subject proxy and intervention validity remain competing explanations. Same-category neighbors, bad source supports, contextual control regions and unchanged geometry priors can all affect the contrast. The experiment does not prove that internal regional evidence, all TTA, or the 3EED direction is intrinsically impossible.

## Decision and reproducibility

The locked G criterion fails in both versions. V2 R59+TTA exceeds R59 and every registered same-information direct control, so its R59-specific gate passes. The independently heldout C gate fails, hence the joint TTA gate fails. **Stop this fixed regional-evidence/set-update recipe; retain v2 as a positive R59-specific development observation, not a new selected method.** No further threshold tuning, source training, transport/sampling mixing or protected evaluation was used to rescue it.

Engineering/provenance, independent encoder/decoder replay, independent gradient/readout, saved-scalar and derived audits are exported. All source/backbone inference and auxiliary replay counts are separately recorded in costs/COLLECTION_COUNTS_TIMES.json. The one validity repair and two saved-analysis implementation failures are retained in history; neither changed sealed predictions retroactively. Source artifacts and exact local provenance remain private. Public own_code includes only our programs, not upstream snapshots, raw media/descriptions/coordinates/IDs, features, weights or secrets. Remote publication uses immutable-byte verification and preservation of all prior study/root entries before moving main.
