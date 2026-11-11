# R67 — fixed-anchor internal SA3 mixed sampling

Actual test-time representation adjustment with frozen Waymo-only epoch100 on all historically exposed DEV336 (168/platform, two Drone and three Quad scenes). Not parameter TTA. Original A256 final boxes, ABC input samples, source weights, native first-noun proxy readout, R57 transport/UNKNOWN and R59 fusion held fixed. No source training/external model/cache feedback, R66 regional supervision, new loss/gradient, FINAL384 or full5912 access. One location and one fixed ratio; no task-score-driven revision.

## Results

| method | Drone_mIoU | Quad_mIoU | balanced_mIoU | Acc25 | Acc50 | gain_vs_R59_pp | improved | worsened | query_changes |
|---|---|---|---|---|---|---|---|---|---|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 | -0.780785 | 14 | 21 | 67 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 | -0.308656 | 8 | 10 | 38 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 | 0.000000 | 0 | 0 | 0 |
| R66_R59_tta1 | 16.657288 | 28.979993 | 22.818641 | 42.857143 | 22.023810 | 0.207525 | 5 | 2 | 12 |
| native_R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 | 0.000000 | 0 | 0 | 0 |
| random_R59 | 16.834981 | 28.722168 | 22.778575 | 44.047619 | 21.726190 | 0.167459 | 10 | 7 | 41 |
| feature_R59 | 16.398310 | 28.491894 | 22.445102 | 42.559524 | 21.428571 | -0.166014 | 4 | 7 | 30 |

Feature minus originalR59: **-0.166014 pp**; feature minus same-budget random: **-0.333473 pp**; feature minus historicalR66+R59: **-0.373538 pp**. Prelocked overall criterion passed: **False**. All rows/negative outcomes retained; per-platform unanimity or zero harm is not required. R66 is a historical development reference with a different evidence/computation budget, not an isolated sampling comparator.

## What actually changes

SA1=2048 and SA2=1024 original points/features recaptured with original source samples. Only SA3 1024->512 centers changed:256 geometryFPS indices plus256 novel indices from featureFPS or a fixed hash-seeded random permutation. FeatureFPS uses unnormalized squared distance on concatenated XYZ and native128-dimensional SA2 features, matching the official3DSSD distance construction. Explicit indices bypass the misleading native FS branch. Deduplicate preserving traversal order; if insufficient novel secondary indices, fill with geometry order then ascending unused index. Both modified arms share D256 and exactly512 unique indices. Coordinate uniqueness is separately recorded; physical repeated points are not magically new evidence.

All downstream SA3 grouping/MLPs, SA4, FP1/FP2 and language-point fusion genuinely recomputed with source weights. Final1024 seedXYZ stay bitexact because FP2 returns to SA2. Original per-view queryseedindices and all six decoder geometry inputs remain fixed, with queryfeatures re-gathered from new fused features. The final model still selects original A256 boxes; no proposal geometry gain is mixed in. R57 transport/masses remain unchanged because the original hypotheses remain the carriers. Native suffix replay reproduces source features and h/query/tokens bitexact on every ABC input.

Thus this tests coarse-layer aggregation under frozen D-FPS-trained weights. It does not increase final seed count or recover targets already absent at SA2. No parameter updates occurred for any new policy; inherited R66 result retains its actual historical update-status metadata.

[3DSSD official code](https://github.com/JIA-Lab-research/3DSSD/blob/master/lib/utils/layers_util.py) concatenates XYZ/features for F-FPS. [3EED backbone](https://github.com/worldbench/3EED/blob/main/models/point_backbone_module_v2.py) propagates back to SA2 XYZ. [Native sampling code](https://github.com/worldbench/3EED/blob/main/ops/teed_pointnet/pointnet2_batch/pointnet2_modules.py) uses XYZ in FS when features are present; this experiment does not simply rename its config. This is a sampling-mechanism transfer, not complete3DSSD reproduction; that detector was trained with its sampling scheme, while this checkpoint was not. Unnormalized feature/geometry scales and frozen feature-distribution mismatch remain explicit limitations, not evidence that all featureFPS fails.

## Scene results

All five exposed clusters are reported; no independent-frame CI or new-confirmation claim.

| method | platform | scene_alias | n | mIoU | gain_vs_R59_pp |
|---|---|---|---|---|---|
| source | drone | D_S01 | 106 | 10.584991 | -1.176621 |
| source | drone | D_S02 | 62 | 24.011174 | -0.279868 |
| source | quad | Q_S01 | 62 | 10.117522 | 0.485406 |
| source | quad | Q_S02 | 49 | 32.998900 | -3.068681 |
| source | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| frozen_ABC | drone | D_S01 | 106 | 10.711738 | -1.049873 |
| frozen_ABC | drone | D_S02 | 62 | 24.326767 | 0.035725 |
| frozen_ABC | quad | Q_S01 | 62 | 10.506319 | 0.874203 |
| frozen_ABC | quad | Q_S02 | 49 | 35.070901 | -0.996679 |
| frozen_ABC | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| R59 | drone | D_S01 | 106 | 11.761612 | 0.000000 |
| R59 | drone | D_S02 | 62 | 24.291041 | 0.000000 |
| R59 | quad | Q_S01 | 62 | 9.632116 | 0.000000 |
| R59 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| R59 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| R66_R59_tta1 | drone | D_S01 | 106 | 12.536308 | 0.774696 |
| R66_R59_tta1 | drone | D_S02 | 62 | 23.702835 | -0.588206 |
| R66_R59_tta1 | quad | Q_S01 | 62 | 10.020492 | 0.388376 |
| R66_R59_tta1 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| R66_R59_tta1 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| native_R59 | drone | D_S01 | 106 | 11.761612 | 0.000000 |
| native_R59 | drone | D_S02 | 62 | 24.291041 | 0.000000 |
| native_R59 | quad | Q_S01 | 62 | 9.632116 | 0.000000 |
| native_R59 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| native_R59 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| random_R59 | drone | D_S01 | 106 | 11.914572 | 0.152960 |
| random_R59 | drone | D_S02 | 62 | 25.247293 | 0.956252 |
| random_R59 | quad | Q_S01 | 62 | 10.025807 | 0.393691 |
| random_R59 | quad | Q_S02 | 49 | 34.845945 | -1.221636 |
| random_R59 | quad | Q_S03 | 57 | 43.794263 | 0.284493 |
| feature_R59 | drone | D_S01 | 106 | 11.779256 | 0.017644 |
| feature_R59 | drone | D_S02 | 62 | 24.295403 | 0.004362 |
| feature_R59 | quad | Q_S01 | 62 | 9.884918 | 0.252802 |
| feature_R59 | quad | Q_S02 | 49 | 35.237931 | -0.829650 |
| feature_R59 | quad | Q_S03 | 57 | 42.931836 | -0.577934 |

## Sampling process and postseal support proxies

Support below is measured inside the best existing A256 candidate with saved IoU>0.25, only after all predictions are sealed. It is an offline approximate object-box proxy, not exact target segmentation or independent identity truth. No GT is used to choose points, scales, location, policies, readouts or acceptance. Fixed final seed support cannot increase by construction; these statistics examine changed SA3 coarse support, not a claim of new fp2 seeds.

| platform | view | policy | n | mean_unique_SA3_coordinates | mean_changed_indices_vs_native | mean_XYZ_pair_squared_distance | mean_feature_pair_squared_distance | fallback_indices_total | postseal_good_box_with_SA2_support_n | postseal_mean_unique_SA3_proxy_support | postseal_zero_SA3_proxy_support |
|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | A | native | 168 | 512.000000 | 0.000000 | 406.183208 | 67.831325 | 0 | 125 | 6.792000 | 0 |
| drone | A | random | 168 | 512.000000 | 170.904762 | 406.183208 | 67.831325 | 0 | 125 | 7.032000 | 0 |
| drone | A | feature | 168 | 512.000000 | 153.482143 | 406.183208 | 67.831325 | 0 | 125 | 10.760000 | 0 |
| drone | B | native | 168 | 512.000000 | 0.000000 | 405.893899 | 68.077308 | 0 | 125 | 6.704000 | 0 |
| drone | B | random | 168 | 512.000000 | 170.029762 | 405.893899 | 68.077308 | 0 | 125 | 7.032000 | 0 |
| drone | B | feature | 168 | 512.000000 | 153.398810 | 405.893899 | 68.077308 | 0 | 125 | 10.576000 | 0 |
| drone | C | native | 168 | 512.000000 | 0.000000 | 406.926276 | 68.204069 | 0 | 125 | 6.760000 | 0 |
| drone | C | random | 168 | 512.000000 | 171.488095 | 406.926276 | 68.204069 | 0 | 125 | 6.664000 | 0 |
| drone | C | feature | 168 | 512.000000 | 152.720238 | 406.926276 | 68.204069 | 0 | 125 | 10.760000 | 0 |
| quad | A | native | 168 | 512.000000 | 0.000000 | 432.950629 | 70.294749 | 0 | 140 | 2.421429 | 1 |
| quad | A | random | 168 | 512.000000 | 170.940476 | 432.950629 | 70.294749 | 0 | 140 | 2.721429 | 6 |
| quad | A | feature | 168 | 512.000000 | 150.011905 | 432.950629 | 70.294749 | 0 | 140 | 3.921429 | 1 |
| quad | B | native | 168 | 512.000000 | 0.000000 | 431.805219 | 70.438161 | 0 | 140 | 2.371429 | 1 |
| quad | B | random | 168 | 512.000000 | 170.267857 | 431.805219 | 70.438161 | 0 | 140 | 2.692857 | 3 |
| quad | B | feature | 168 | 512.000000 | 149.785714 | 431.805219 | 70.438161 | 0 | 140 | 3.800000 | 0 |
| quad | C | native | 168 | 512.000000 | 0.000000 | 434.830327 | 70.316945 | 0 | 140 | 2.421429 | 2 |
| quad | C | random | 168 | 512.000000 | 171.130952 | 434.830327 | 70.316945 | 0 | 140 | 2.657143 | 7 |
| quad | C | feature | 168 | 512.000000 | 151.065476 | 434.830327 | 70.316945 | 0 | 140 | 3.850000 | 0 |

The intervention materially changes representative points: feature sampling replaces about153.20/150.29 SA3 indices per Drone/Quad view relative to native. Averaged over ABC, good-candidate-box proxy support rises from6.752 to10.699 in Drone and2.405 to3.857 in Quad; random gives6.909/2.690. These are postseal approximate supports, not exact ground-truth object points. More proxy-region coarse support therefore did not translate into superior candidate ranking under this fixed representation/readout. This is evidence against the simple explanation that the intervention never changed sampling; it does not uniquely identify why the frozen model fails to exploit the new aggregation. Feature-policy comparison is negative even though random policy has a small positive overall development result(+0.167459ppR59,10improved7worsened). Preserve that positive control without attributing it to semantic feature selection.

## Validity, budget and decision

Two real engineering inputs/six original source forwards and18 suffixes passed. Formal1008 full-source captures plus3024 native/random/feature suffixes. Independent1008 full-source recaptures establish SA2 feature/coordinate provenance and source h/query/tokens/indices, then independently reconstruct explicit sampling and manually execute SA3 group/MLP/max aggregation plus native SA4/FP/encoder/decoder for3024 suffixes. All arrays/indices bitexact. Source fingerprint remains unchanged. Collection times include private cache writes and audit overhead, not isolated per-policy deployment latency; F-FPS additionally computes a joint distance matrix despite equal selected-point/forward budgets. Independent CPU readouts and saved-IoU/scene/negative audit run before reporting. All2352 task values reuse already saved original-candidate IoUs; no new raw GT geometry score or annotation-media access for scoring.

Decision: **R67_COMPLETED_CURRENT_SA3_MIXED_SAMPLING_NOT_ADOPTED**. This decision is bounded to one SA3/256+256/source-scale implementation on exposedDEV336. Do not adopt this fixed sampling recipe; do not rescue it with postscore scale/ratio/site searches or regional supervision. This does not reject all internal sampling or the overall3EED direction.

All negative ordered comparisons, status rows and five-scene results are public. Original data, descriptions, raw IDs/coordinates, model features/weights/secrets and upstream snapshots remain private. Publication requires independent privacy audit, immutable remote-byte verification and preservation of previous root/study trees before CASmain.

Local public-export attempt1 stopped on a stale guard-basename allowlist. It was preserved and repaired only in publication metadata; no scientific predictions, task values, method choices or tolerances changed. Accepted export attempt2 is separately audited.
