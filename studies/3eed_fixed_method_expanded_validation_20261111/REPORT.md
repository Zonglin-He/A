# R69 — fixed-method expanded validation, without new algorithms or tuning

This is the user-authorized validation of the original R68-selected configurations on more target observations and additional target recording sequences. It is not R69 method development: gamma=1, eta=1, update module, loss, acceptance rule, candidate transport, UNKNOWN handling, seed prefixes and sampling quota are unchanged. No source retraining, external visual model or new RGB loss. VAL1280 and TRAIN1024 are reported separately, with no favorable-cohort selection and no method changes after new task labels.

## Data and exposure boundary

- VAL1280: 640 expressions/platform, outside DEV336 and FINAL384 by identifier, point-media hash and image-media hash; 1,033 distinct media pairs (428 Drone,605 Quad), at most two distinct descriptions/media. These are the same two Drone and three Quad official recording sequences as existing VAL, not five new scenes. Full VAL historically received source/other-method evaluation: this is expanded fixed-method evidence outside the main development/protected-media subsets, not never-exposed target data.
- TRAIN1024: 512 expressions/platform,1,024 unique media pairs, one expression/media; two additional official recording sequences/platform and distinct nominal place-name families. Three TRAIN sequence families resembling VAL place names were excluded before model execution. Every selected TRAIN point/image hash is disjoint from all target VAL media across both platforms. The source checkpoint remains Waymo-only; target TRAIN labels are used solely for this postseal evaluation, never source training, G construction, fit acceptance or method selection.
- Origin headers contain empty sequence_name and local integer timestamps. Equal local timestamps from different sequences do not prove repeated frames. Byte hashes and official sequence identity establish the stated exclusions; global geographical non-revisit and absolute-wallclock separation cannot be certified from available metadata. Do not call TRAIN a certified geographic holdout.
- FINAL384 was historically scored in R46. This round reads its manifest identities and hashes its media bytes only to exclude overlaps; it does not decode those media for model input, generate predictions, read numeric GT, reuse historical FINAL predictions, or score FINAL again. This is not a claim of zero FINAL byte access or a pristine historical holdout.
- Selection is stratified across sequences and eight local-time bins, deterministic hash ordering, media deduplication and a fixed per-media description cap. Target TRAIN indexes all caption records with the source text-length interface constraint, without GT class/box/correctness filtering. All selected captions fit the fixed interface. Paths, captions, hashes and raw sequence names remain private.

An initial metadata-only selection incorrectly treated empty-sequence/local-timestamp pairs as global identity. It was repaired before formal model execution or task scoring; the failed selection is preserved. It did not change an algorithm, use task outcomes, or remove negative predictions. Independent data audit verifies all2,304 joins, exclusion hashes and cohort counts.

## Fixed algorithms and execution

Source: strict Waymo-only epoch100 with pinned weights and fingerprint. All11 task arms select from the same input's original256 final A boxes; no new box proposals, geometry refinement or regression update. A,B,C are different samplings of the same physical point cloud, not new camera views or independent scenes. All groups/fits use A/B only; every fitted matching-head snapshot is sealed before any C inference. Source model weights remain unchanged.

R59 uses original representation fusion gamma1 with the original R57 correspondence, fixed transport mass/UNKNOWN, temperature and readout. R66 is exact v2 region-vs-matched-control full-description-minus-category contrast, matching both seed and masked query-key counts. Initial group/pool/stable-sign/partial-set rules remain fixed. An eligible example must have both positive and negative groups; the existing matching MLP performs one original Adam .0005 step with clip1 and nine label-free loss-backtracking scales. Accepted eta1 snapshots are loaded exactly. No GT-dependent trigger, early stop, sample exclusion, update-layer change or strength search. Noneligible/rejected examples remain in every denominator.

Random SA3 retains256 native D-FPS seeds plus256 random remaining seeds at512 total; original source SA2, query anchors and six decoder geometry inputs remain fixed. Three seed prefixes are the original R67/R68 rules; they are three repeated policies, not an output ensemble, bestseed selection, or extra information for R66. Each policy receives the same three A/B/C point samplings. Source/ABC/R59 use no regional perturbation forwards; R66 has additional evidence-construction cost. E-direct and set-direct are retained same-evidence/no-update diagnostics, not new optimized alternatives. C distributions are transported to the same canonical A256 candidates using the fixed correspondence; they do not score a new physical scene or new box set. C source versus C adapted is a transfer diagnostic and not a posthoc veto of the deployment-readout comparison.

Predictions and statuses were sealed before numeric task GT was released. One official-convention GT pass computes all original256 candidate IoUs/input, then looks up every11 selected index. The two cohort tables derive from this one pass, not two extra scores. Independent geometry uses rectangle-polygon intersection and vertical overlap rather than the producer convex-polyhedron implementation. All UNKNOWN, fallback, no-update and negative comparisons are exported.

## Main fixed-method results

Units: percent for IoU/accuracy, percentage points for differences. Both cohorts have equal platform counts, so expression mean equals the predeclared platform-equal mIoU. Random mean is over three policy scores and has no single associated selected-query output.


### VAL1280

| method | Drone_mIoU | Quad_mIoU | balanced_mIoU | Acc25 | Acc50 | gain_vs_R59_pp | improved_vs_R59 | worsened_vs_R59 |
|---|---|---|---|---|---|---|---|---|
| source | 10.712182 | 25.584854 | 18.148518 | 33.125000 | 16.484375 | -0.851764 | 49 | 91 |
| frozen_ABC | 11.255613 | 25.965582 | 18.610598 | 34.375000 | 17.031250 | -0.389684 | 36 | 58 |
| R59 | 11.483505 | 26.517059 | 19.000282 | 34.687500 | 17.500000 | 0.000000 | 0 | 0 |
| R66_R59 | 11.644900 | 26.527055 | 19.085978 | 34.687500 | 17.656250 | 0.085696 | 10 | 11 |
| R66_E_direct | 11.038612 | 26.239414 | 18.639013 | 33.203125 | 17.031250 | -0.361269 | 79 | 91 |
| R66_set_direct | 11.416057 | 26.607498 | 19.011778 | 34.453125 | 17.343750 | 0.011496 | 30 | 28 |
| random_s0 | 10.981586 | 26.502302 | 18.741944 | 34.062500 | 17.187500 | -0.258338 | 22 | 39 |
| random_s1 | 11.302088 | 26.215467 | 18.758778 | 33.750000 | 17.187500 | -0.241504 | 20 | 37 |
| random_s2 | 11.354384 | 26.563240 | 18.958812 | 34.296875 | 17.343750 | -0.041470 | 31 | 37 |
| C_source | 10.738388 | 26.299112 | 18.518750 | 33.515625 | 16.875000 | -0.481532 | 70 | 96 |
| C_R66 | 10.903166 | 26.297632 | 18.600399 | 33.593750 | 16.953125 | -0.399883 | 74 | 97 |

### TRAIN1024

| method | Drone_mIoU | Quad_mIoU | balanced_mIoU | Acc25 | Acc50 | gain_vs_R59_pp | improved_vs_R59 | worsened_vs_R59 |
|---|---|---|---|---|---|---|---|---|
| source | 16.504735 | 19.482304 | 17.993519 | 34.863281 | 16.894531 | -0.006311 | 32 | 39 |
| frozen_ABC | 16.673721 | 19.468151 | 18.070936 | 35.449219 | 16.894531 | 0.071106 | 30 | 26 |
| R59 | 16.830404 | 19.169256 | 17.999830 | 34.863281 | 16.699219 | 0.000000 | 0 | 0 |
| R66_R59 | 16.268579 | 19.028274 | 17.648427 | 34.179688 | 16.601562 | -0.351404 | 7 | 14 |
| R66_E_direct | 15.478904 | 18.838183 | 17.158543 | 33.300781 | 16.210938 | -0.841287 | 35 | 78 |
| R66_set_direct | 15.630498 | 19.101431 | 17.365964 | 33.593750 | 16.503906 | -0.633866 | 16 | 37 |
| random_s0 | 17.036622 | 19.230846 | 18.133734 | 34.863281 | 16.699219 | 0.133904 | 23 | 17 |
| random_s1 | 16.495883 | 19.796425 | 18.146154 | 35.058594 | 16.406250 | 0.146324 | 33 | 17 |
| random_s2 | 16.576861 | 19.893385 | 18.235123 | 35.449219 | 17.089844 | 0.235293 | 26 | 23 |
| C_source | 16.899864 | 19.037633 | 17.968749 | 35.058594 | 16.601562 | -0.031082 | 65 | 64 |
| C_R66 | 16.421850 | 18.930599 | 17.676225 | 34.277344 | 16.503906 | -0.323606 | 62 | 69 |

## Three prespecified random policies

| cohort | seed0_mIoU | seed1_mIoU | seed2_mIoU | repeat_mean_mIoU | repeat_min_mIoU | repeat_max_mIoU | repeat_std_population | mean_gain_vs_R59_pp | all_three_over_R59 | R66_minus_random_policy_mean_pp | not_ensemble_or_best_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VAL1280 | 18.741944 | 18.758778 | 18.958812 | 18.819845 | 18.741944 | 18.958812 | 0.098505 | -0.180437 | False | 0.266133 | True |
| TRAIN1024 | 18.133734 | 18.146154 | 18.235123 | 18.171670 | 18.133734 | 18.235123 | 0.045153 | 0.171840 | True | -0.523244 | True |

The repeat mean/range/std measure seed-policy variability, not an independent-scene confidence interval. All three outputs and ordered negatives are preserved. The best repeat is not selected for deployment or substituted for the policy mean.

## Update coverage, C transfer and source-restoration attribution

| cohort | n | unique_media_frames | update_eligible | accepted_nonzero | coverage_percent | R66_gain_vs_R59_pp | R66_gain_vs_source_pp | R66_rescue_source_failure25 | R66_break_source_success25 | R66_C_gain_vs_C_source_pp | R59_frozenABC_gain_pp | A256_oracle_good25 | R59_failure_recoverable25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VAL1280 | 1280 | 1033 | 188 | 188 | 14.687500 | 0.085696 | 0.937459 | 38 | 18 | 0.081649 | 0.389684 | 1010 | 566 |
| TRAIN1024 | 1024 | 1024 | 159 | 158 | 15.527344 | -0.351404 | -0.345092 | 16 | 23 | -0.292524 | -0.071106 | 796 | 439 |

| cohort | category | n | contribution_pp |
|---|---|---|---|
| VAL1280 | exact_source_query_restored | 0 | 0.000000 |
| VAL1280 | source_level_IoU_other_query | 0 | 0.000000 |
| VAL1280 | beyond_source_IoU | 10 | 0.190779 |
| VAL1280 | improved_R59_but_below_source | 0 | 0.000000 |
| TRAIN1024 | exact_source_query_restored | 0 | 0.000000 |
| TRAIN1024 | source_level_IoU_other_query | 0 | 0.000000 |
| TRAIN1024 | beyond_source_IoU | 7 | 0.011018 |
| TRAIN1024 | improved_R59_but_below_source | 0 | 0.000000 |

Restoration bins describe R66 improvements relative to R59. Exact source query restoration and equal-source-IoU alternative queries are separate from improvements beyond source; IoU equality alone is not independent instance-identity certification. These are postseal diagnostics, not deployment gates. Negative contributions are not hidden by the positive-only attribution bins; the full net and all harmed rows remain in the next table and exported scalars. Largest single-case contributions and the remainder retain the full original cohort denominator.

| cohort | method | gain_vs_R59_pp | improved_vs_R59 | worsened_vs_R59 | query_changes_vs_R59 | largest_positive_full_denominator_pp | rest_after_largest_full_denominator_pp |
|---|---|---|---|---|---|---|---|
| VAL1280 | R66_R59 | 0.085696 | 10 | 11 | 37 | 0.041093 | 0.044602 |
| VAL1280 | random_s0 | -0.258338 | 22 | 39 | 141 | 0.056432 | -0.314770 |
| VAL1280 | random_s1 | -0.241504 | 20 | 37 | 125 | 0.058683 | -0.300187 |
| VAL1280 | random_s2 | -0.041470 | 31 | 37 | 168 | 0.058683 | -0.100153 |
| TRAIN1024 | R66_R59 | -0.351404 | 7 | 14 | 40 | 0.005054 | -0.356457 |
| TRAIN1024 | random_s0 | 0.133904 | 23 | 17 | 94 | 0.067184 | 0.066720 |
| TRAIN1024 | random_s1 | 0.146324 | 33 | 17 | 104 | 0.060212 | 0.086112 |
| TRAIN1024 | random_s2 | 0.235293 | 26 | 23 | 106 | 0.071143 | 0.164150 |

## Every recording-sequence cluster

| cohort | method | platform | scene_alias | n | mIoU | gain_vs_R59_pp |
|---|---|---|---|---|---|---|
| VAL1280 | source | drone | D_S01 | 543 | 8.309535 | -0.836310 |
| VAL1280 | source | drone | D_S02 | 97 | 24.162053 | -0.407531 |
| VAL1280 | source | quad | Q_S01 | 338 | 12.320371 | -0.634071 |
| VAL1280 | source | quad | Q_S02 | 115 | 38.940218 | -1.903217 |
| VAL1280 | source | quad | Q_S03 | 187 | 41.347039 | -0.873928 |
| VAL1280 | frozen_ABC | drone | D_S01 | 543 | 8.943964 | -0.201881 |
| VAL1280 | frozen_ABC | drone | D_S02 | 97 | 24.196077 | -0.373506 |
| VAL1280 | frozen_ABC | quad | Q_S01 | 338 | 12.727560 | -0.226883 |
| VAL1280 | frozen_ABC | quad | Q_S02 | 115 | 39.313988 | -1.529447 |
| VAL1280 | frozen_ABC | quad | Q_S03 | 187 | 41.684218 | -0.536749 |
| VAL1280 | R59 | drone | D_S01 | 543 | 9.145845 | 0.000000 |
| VAL1280 | R59 | drone | D_S02 | 97 | 24.569584 | 0.000000 |
| VAL1280 | R59 | quad | Q_S01 | 338 | 12.954443 | 0.000000 |
| VAL1280 | R59 | quad | Q_S02 | 115 | 40.843435 | 0.000000 |
| VAL1280 | R59 | quad | Q_S03 | 187 | 42.220967 | 0.000000 |
| VAL1280 | R66_R59 | drone | D_S01 | 543 | 9.313927 | 0.168082 |
| VAL1280 | R66_R59 | drone | D_S02 | 97 | 24.693544 | 0.123960 |
| VAL1280 | R66_R59 | quad | Q_S01 | 338 | 12.973371 | 0.018928 |
| VAL1280 | R66_R59 | quad | Q_S02 | 115 | 40.843435 | 0.000000 |
| VAL1280 | R66_R59 | quad | Q_S03 | 187 | 42.220967 | 0.000000 |
| VAL1280 | R66_E_direct | drone | D_S01 | 543 | 8.712625 | -0.433219 |
| VAL1280 | R66_E_direct | drone | D_S02 | 97 | 24.059342 | -0.510242 |
| VAL1280 | R66_E_direct | quad | Q_S01 | 338 | 12.365027 | -0.589416 |
| VAL1280 | R66_E_direct | quad | Q_S02 | 115 | 41.057759 | 0.214323 |
| VAL1280 | R66_E_direct | quad | Q_S03 | 187 | 42.204299 | -0.016668 |
| VAL1280 | R66_set_direct | drone | D_S01 | 543 | 9.165533 | 0.019688 |
| VAL1280 | R66_set_direct | drone | D_S02 | 97 | 24.014351 | -0.555233 |
| VAL1280 | R66_set_direct | quad | Q_S01 | 338 | 13.125689 | 0.171246 |
| VAL1280 | R66_set_direct | quad | Q_S02 | 115 | 40.843435 | 0.000000 |
| VAL1280 | R66_set_direct | quad | Q_S03 | 187 | 42.220967 | 0.000000 |
| VAL1280 | random_s0 | drone | D_S01 | 543 | 8.569512 | -0.576333 |
| VAL1280 | random_s0 | drone | D_S02 | 97 | 24.484231 | -0.085353 |
| VAL1280 | random_s0 | quad | Q_S01 | 338 | 12.948630 | -0.005813 |
| VAL1280 | random_s0 | quad | Q_S02 | 115 | 40.296393 | -0.547043 |
| VAL1280 | random_s0 | quad | Q_S03 | 187 | 42.517386 | 0.296419 |
| VAL1280 | random_s1 | drone | D_S01 | 543 | 8.950381 | -0.195464 |
| VAL1280 | random_s1 | drone | D_S02 | 97 | 24.466801 | -0.102783 |
| VAL1280 | random_s1 | quad | Q_S01 | 338 | 12.552168 | -0.402274 |
| VAL1280 | random_s1 | quad | Q_S02 | 115 | 39.824534 | -1.018901 |
| VAL1280 | random_s1 | quad | Q_S03 | 187 | 42.542485 | 0.321518 |
| VAL1280 | random_s2 | drone | D_S01 | 543 | 9.016002 | -0.129842 |
| VAL1280 | random_s2 | drone | D_S02 | 97 | 24.444499 | -0.125084 |
| VAL1280 | random_s2 | quad | Q_S01 | 338 | 12.761555 | -0.192887 |
| VAL1280 | random_s2 | quad | Q_S02 | 115 | 40.716662 | -0.126774 |
| VAL1280 | random_s2 | quad | Q_S03 | 187 | 42.805624 | 0.584656 |
| VAL1280 | C_source | drone | D_S01 | 543 | 8.353443 | -0.792402 |
| VAL1280 | C_source | drone | D_S02 | 97 | 24.089166 | -0.480418 |
| VAL1280 | C_source | quad | Q_S01 | 338 | 12.795582 | -0.158860 |
| VAL1280 | C_source | quad | Q_S02 | 115 | 41.415305 | 0.571870 |
| VAL1280 | C_source | quad | Q_S03 | 187 | 41.410506 | -0.810461 |
| VAL1280 | C_R66 | drone | D_S01 | 543 | 8.540535 | -0.605310 |
| VAL1280 | C_R66 | drone | D_S02 | 97 | 24.129029 | -0.440555 |
| VAL1280 | C_R66 | quad | Q_S01 | 338 | 12.792780 | -0.161663 |
| VAL1280 | C_R66 | quad | Q_S02 | 115 | 41.415305 | 0.571870 |
| VAL1280 | C_R66 | quad | Q_S03 | 187 | 41.410506 | -0.810461 |
| TRAIN1024 | source | drone | D_S01 | 360 | 14.572133 | -0.617384 |
| TRAIN1024 | source | drone | D_S02 | 152 | 21.081949 | 0.365233 |
| TRAIN1024 | source | quad | Q_S01 | 299 | 23.665393 | 0.523748 |
| TRAIN1024 | source | quad | Q_S02 | 213 | 13.610267 | 0.017274 |
| TRAIN1024 | frozen_ABC | drone | D_S01 | 360 | 14.781191 | -0.408326 |
| TRAIN1024 | frozen_ABC | drone | D_S02 | 152 | 21.156030 | 0.439314 |
| TRAIN1024 | frozen_ABC | quad | Q_S01 | 299 | 23.176847 | 0.035203 |
| TRAIN1024 | frozen_ABC | quad | Q_S02 | 213 | 14.262047 | 0.669055 |
| TRAIN1024 | R59 | drone | D_S01 | 360 | 15.189517 | 0.000000 |
| TRAIN1024 | R59 | drone | D_S02 | 152 | 20.716715 | 0.000000 |
| TRAIN1024 | R59 | quad | Q_S01 | 299 | 23.141645 | 0.000000 |
| TRAIN1024 | R59 | quad | Q_S02 | 213 | 13.592993 | 0.000000 |
| TRAIN1024 | R66_R59 | drone | D_S01 | 360 | 14.514916 | -0.674601 |
| TRAIN1024 | R66_R59 | drone | D_S02 | 152 | 20.421991 | -0.294724 |
| TRAIN1024 | R66_R59 | quad | Q_S01 | 299 | 23.141645 | 0.000000 |
| TRAIN1024 | R66_R59 | quad | Q_S02 | 213 | 13.254106 | -0.338887 |
| TRAIN1024 | R66_E_direct | drone | D_S01 | 360 | 13.528544 | -1.660973 |
| TRAIN1024 | R66_E_direct | drone | D_S02 | 152 | 20.098176 | -0.618539 |
| TRAIN1024 | R66_E_direct | quad | Q_S01 | 299 | 23.329441 | 0.187797 |
| TRAIN1024 | R66_E_direct | quad | Q_S02 | 213 | 12.533552 | -1.059441 |
| TRAIN1024 | R66_set_direct | drone | D_S01 | 360 | 13.609380 | -1.580138 |
| TRAIN1024 | R66_set_direct | drone | D_S02 | 152 | 20.417356 | -0.299359 |
| TRAIN1024 | R66_set_direct | quad | Q_S01 | 299 | 23.076059 | -0.065586 |
| TRAIN1024 | R66_set_direct | quad | Q_S02 | 213 | 13.522025 | -0.070968 |
| TRAIN1024 | random_s0 | drone | D_S01 | 360 | 15.426837 | 0.237320 |
| TRAIN1024 | random_s0 | drone | D_S02 | 152 | 20.849270 | 0.132555 |
| TRAIN1024 | random_s0 | quad | Q_S01 | 299 | 23.056219 | -0.085426 |
| TRAIN1024 | random_s0 | quad | Q_S02 | 213 | 13.860956 | 0.267964 |
| TRAIN1024 | random_s1 | drone | D_S01 | 360 | 14.687352 | -0.502165 |
| TRAIN1024 | random_s1 | drone | D_S02 | 152 | 20.779246 | 0.062531 |
| TRAIN1024 | random_s1 | quad | Q_S01 | 299 | 23.687746 | 0.546101 |
| TRAIN1024 | random_s1 | quad | Q_S02 | 213 | 14.333960 | 0.740967 |
| TRAIN1024 | random_s2 | drone | D_S01 | 360 | 14.774966 | -0.414551 |
| TRAIN1024 | random_s2 | drone | D_S02 | 152 | 20.844508 | 0.127793 |
| TRAIN1024 | random_s2 | quad | Q_S01 | 299 | 23.544493 | 0.402849 |
| TRAIN1024 | random_s2 | quad | Q_S02 | 213 | 14.768120 | 1.175127 |
| TRAIN1024 | C_source | drone | D_S01 | 360 | 15.126949 | -0.062568 |
| TRAIN1024 | C_source | drone | D_S02 | 152 | 21.098872 | 0.382157 |
| TRAIN1024 | C_source | quad | Q_S01 | 299 | 22.646593 | -0.495052 |
| TRAIN1024 | C_source | quad | Q_S02 | 213 | 13.971535 | 0.378542 |
| TRAIN1024 | C_R66 | drone | D_S01 | 360 | 14.644203 | -0.545314 |
| TRAIN1024 | C_R66 | drone | D_S02 | 152 | 20.632066 | -0.084649 |
| TRAIN1024 | C_R66 | quad | Q_S01 | 299 | 22.646593 | -0.495052 |
| TRAIN1024 | C_R66 | quad | Q_S02 | 213 | 13.714251 | 0.121258 |

Expression-level observations within a recording, and repeated descriptions of the same media, are not independent scenes. The complete per-frame-cluster table and this equal-frame/platform sensitivity readout are supplied; primary metrics remain expression-level. With only2/3 VAL sequences and2/2 TRAIN sequences, no frame-iid confidence interval or broad cross-city generalization claim is made. A single harmed recording is not an automatic rejection gate, but platform/sequence asymmetry remains a material limit.

| cohort | method | Drone_unique_frames | Quad_unique_frames | equal_frame_then_platform_mIoU |
|---|---|---|---|---|
| VAL1280 | source | 428 | 605 | 18.739422 |
| VAL1280 | frozen_ABC | 428 | 605 | 19.365008 |
| VAL1280 | R59 | 428 | 605 | 19.704155 |
| VAL1280 | R66_R59 | 428 | 605 | 19.793724 |
| VAL1280 | R66_E_direct | 428 | 605 | 19.258929 |
| VAL1280 | R66_set_direct | 428 | 605 | 19.647767 |
| VAL1280 | random_s0 | 428 | 605 | 19.384969 |
| VAL1280 | random_s1 | 428 | 605 | 19.436479 |
| VAL1280 | random_s2 | 428 | 605 | 19.616437 |
| VAL1280 | C_source | 428 | 605 | 19.195150 |
| VAL1280 | C_R66 | 428 | 605 | 19.295455 |
| TRAIN1024 | source | 512 | 512 | 17.993519 |
| TRAIN1024 | frozen_ABC | 512 | 512 | 18.070936 |
| TRAIN1024 | R59 | 512 | 512 | 17.999830 |
| TRAIN1024 | R66_R59 | 512 | 512 | 17.648427 |
| TRAIN1024 | R66_E_direct | 512 | 512 | 17.158543 |
| TRAIN1024 | R66_set_direct | 512 | 512 | 17.365964 |
| TRAIN1024 | random_s0 | 512 | 512 | 18.133734 |
| TRAIN1024 | random_s1 | 512 | 512 | 18.146154 |
| TRAIN1024 | random_s2 | 512 | 512 | 18.235123 |
| TRAIN1024 | C_source | 512 | 512 | 17.968749 |
| TRAIN1024 | C_R66 | 512 | 512 | 17.676225 |

## Fixed-validation decision

- state: R69_FIXED_VALIDATION_COMPLETE_NO_RESELECT_OR_RETRAIN
- R66_gain_VAL_pp: 0.08569554206328789
- R66_gain_TRAIN_pp: -0.35140351878567344
- R66_exceeds_R59_both_cohorts: False
- R66_exceeds_random_mean_both_cohorts: False
- new_method_or_hyperparameter_selected: False
- C_diagnostic_not_veto: True
- source_training: False
- FINAL_model_scoring: False
- VAL_historical_fullVAL_exposure_disclosed: True
- TRAIN_scene_claim: new_recording_sequences_and_nominal_place_families_not_certified_global_geographic_holdout

VAL1280: R66 minus R59 **+0.085696pp**, R66 minus three-repeat random policy mean **+0.266133pp**, C adapted minus C source **+0.081649pp**. Update eligibility is 188/1280; accepted nonzero snapshots 188.

TRAIN1024: R66 minus R59 **-0.351404pp**, R66 minus three-repeat random policy mean **-0.523244pp**, C adapted minus C source **-0.292524pp**. Update eligibility is 159/1024; accepted nonzero snapshots 158.


No algorithm is promoted merely because loss falls, parameters change, a single expression is restored, or source is exceeded. Evidence for R66's incremental task value requires comparison with R59 and the cheaper fixed random policy under the reported information/cost differences. Consistent positive direction on these cohorts would support expanded fixed-method validation, not automatically establish general semantic supervision; absent/reversed gains do not justify another eta/gamma/layer/threshold sweep on these labels. This round terminates after reporting the fixed comparisons. No new method is developed using these outcomes and no follow-on run is queued.

## Independent verification, resources and reproducibility

All2,304 examples receive independent CPU reconstruction of AB groups/control counts/evidence algebra, gradient/Adam/backtracking, every saved parameter, ABC candidate transport/UNKNOWN and all11 readouts. A pre-model locked72-input subset (eight per cohort/platform/recording) receives independent full source, regional encoder/decoder and random-suffix GPU replay. It is a stratified mechanism replay subset, not a claim of repeating every full GPU computation. Runtime zero-mask h/query/token equality, unchanged source fingerprints and sampler/anchor checks additionally run throughout the full producer. All589,824 candidate IoUs, task/status, scene, negative, repeat, source-restoration and frame-cluster summaries are independently verified. Public package has an independent per-file private-token/secret/schema scan and immutable remote-byte verification before main compare-and-swap; no original media/descriptions/IDs/coordinates/features/weights/secrets or third-party source snapshots are published.

Collection timing includes input decoding, cache I/O, fits and audits. Regional replays/suffixes are not full-backbone forwards. Random repeat means are not three-repeat deployment ensembles. Per-policy deployment latency is not inferred by dividing total time across heterogeneous computation. Engineering four-input checks are disclosed separately and not counted as extra evaluation observations. Source loader's unused RoBERTa pooler-initialization warning does not change the strict loaded grounding checkpoint or matching-state equality.
