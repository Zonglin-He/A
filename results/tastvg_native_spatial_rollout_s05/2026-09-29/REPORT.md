# S0.5 — Native Spatial Rollout Support Test

Completed / audited. Teacher-independent parameter rollouts produce limited support at this one fixed radius. Do not advance this configuration to S1 on the measured headroom; no OPD or online learning was run.

| Group | Native sIoU % | Six-layer oracle gain pp | 9-rollout oracle gain pp | Six-layer + rollout union gain pp | S0 union gain pp | New union − S0 union pp (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| corruption | 45.3509 | 0.6965 | 0.8119 | 1.2441 | 1.3886 | -0.1444 [-0.4291, 0.1573] |
| clean | 45.8489 | 0.7704 | 0.8101 | 1.2838 | 1.4280 | -0.1442 [-0.4406, 0.1591] |
| frame_drop_5 | 44.9906 | 0.7146 | 0.8196 | 1.3140 | 1.5306 | -0.2166 [-0.5233, 0.0935] |
| frame_freeze_5 | 45.7540 | 0.7401 | 0.8103 | 1.2323 | 1.3109 | -0.0785 [-0.3765, 0.2563] |
| motion_blur_5 | 45.5072 | 0.6952 | 0.8067 | 1.2439 | 1.3988 | -0.1549 [-0.4467, 0.1497] |
| occlusion_5 | 45.2275 | 0.6896 | 0.8319 | 1.2691 | 1.3728 | -0.1038 [-0.3866, 0.1897] |
| exposure_5 | 45.2751 | 0.6430 | 0.7910 | 1.1614 | 1.3297 | -0.1683 [-0.4473, 0.1255] |

## What was actually tested

Same16 original C3 historically exposed VidSTG parent sources, one query/source, clean and five existing source-hash seed0 transient5% corruptions:96cells. Frozen source VidSTG TA-STVG checkpoint, exact native H caches. Parameter interface1792: second-pass spatial query residual256 and final spatial block5 norm1/norm3/norm4 weight+bias1536. LN changes apply on both native calls; every candidate restores the source model. H, text, motion and other parameters are not updated. Original dynamic suffix is recomputed.

Four orthonormal Gaussian-QR directions, seed20260929, fixed across every source/condition; plus/minus pairs and native center yield9 candidates. One radius: .05 times the initial full parameter norm 31.53160385, giving absolute L2 radius 1.57658019. This parameter radius is unrelated to old .004 H steps. No radius/direction-count sweep. Source parameters determine the scale; neither teacher outputs nor GT do. All96 cells have9 distinct tube boxes.

The worker rejects expert caches, propagation outputs and GT files.864 complete candidate tubes and parameter states are sealed before GT scoring. The existing interface can hold persistent parameters, but this support screen uses the same unlearned central state on every arrival. This does not establish online transfer. No expert ranking, optimizer, backward pass or pseudo-box regression enters generation.

The oracle selects one whole candidate tube using sIoU on all GT-valid sampled frames, never a frame-wise mixture. Native is retained, so nonnegative oracle gain is structural. Six-layer union contains15 entries including duplicated native (at most14 distinct); new rollouts are9 distinct within each cell. Compare union to the same-cohort S0 union1.3886pp; do not compare9-vs6 candidate counts as a controlled mechanism isolation. Five corruption conditions average within source, then source macro;10000 source bootstrap draws seed20260929. Descriptive exposed-development evidence, not independent confirmation.

## Secondary transfer and fixed-arm behavior

corruption: whole-tube oracle gain 0.8119pp CI[0.5383, 1.1255]; union gain 1.2441pp CI[0.7766, 1.7489]. Unobserved positions gain at whole-tube oracle: matched12 0.8535pp CI[0.5228, 1.2363]; all16 0.8016pp. Same deterministic five uniform positions as S0, without reading masks. Observed gain 1.0096pp (12sources). Mean per-source minimum candidate self-sIoU 93.6581%.
clean: whole-tube oracle gain 0.8101pp CI[0.5402, 1.1239]; union gain 1.2838pp CI[0.7699, 1.8510]. Unobserved positions gain at whole-tube oracle: matched12 0.8520pp CI[0.5289, 1.2297]; all16 0.8019pp. Same deterministic five uniform positions as S0, without reading masks. Observed gain 0.9993pp (12sources). Mean per-source minimum candidate self-sIoU 93.4125%.

| Fixed arm | Corruption Δs pp | 95% CI pp | Sources better / worse (>0.1pp) | Sources worse >5pp |
|---|---:|---|---|---:|
| 0 | 0.0000 | [0.0000, 0.0000] | 0 / 0 | 0 |
| 1 | 0.1000 | [-0.0734, 0.2947] | 8 / 4 | 0 |
| 2 | -0.1221 | [-0.3060, 0.0506] | 5 / 8 | 0 |
| 3 | 0.1636 | [0.0437, 0.2904] | 9 / 4 | 0 |
| 4 | -0.1625 | [-0.2846, -0.0433] | 1 / 9 | 0 |
| 5 | 0.4434 | [0.1211, 0.8170] | 9 / 3 | 0 |
| 6 | -0.4207 | [-0.7551, -0.1266] | 3 / 9 | 0 |
| 7 | -0.4258 | [-0.6686, -0.1892] | 2 / 13 | 0 |
| 8 | 0.4479 | [0.2173, 0.6920] | 13 / 2 | 0 |

| Source | Rollout oracle gain pp | Union gain pp | Union change vs S0 pp |
|---|---:|---:|---:|
| Q01 | 0.0000 | 0.0000 | 0.0000 |
| Q02 | 0.4534 | 0.6318 | 0.0000 |
| Q03 | 0.5951 | 2.9365 | 0.0000 |
| Q04 | 0.5071 | 0.5071 | -0.3137 |
| Q05 | 0.5532 | 0.5532 | 0.2733 |
| Q06 | 1.7970 | 1.7970 | 1.2967 |
| Q07 | 0.6701 | 0.6701 | -0.7683 |
| Q08 | 0.1287 | 0.2368 | -0.7065 |
| Q09 | 2.3194 | 2.3194 | -0.6937 |
| Q10 | 0.5455 | 0.5455 | -0.5288 |
| Q11 | 0.3109 | 2.2683 | 0.0000 |
| Q12 | 0.9198 | 0.9198 | -1.2955 |
| Q13 | 1.5912 | 1.5912 | 0.7562 |
| Q14 | 1.0100 | 1.0208 | 0.0308 |
| Q15 | 1.1501 | 3.4698 | 0.0000 |
| Q16 | 0.4388 | 0.4388 | -0.3611 |

## Decision and boundaries

The current9-candidate/.05-radius parameter neighborhood does not materially open the old1.39pp union support: its union is1.2441pp, paired difference−.1444pp with CI spanning zero. This is well below the user’s illustrative3–5pp scale. Do not start S1 critic/Reverse-KL from this screen, and do not automatically widen radius or add a sweep. Preserve the positive rollout cases and unobserved gains. This bounded result does not prove that spatial OPD, larger/different neighborhoods, or persistent adaptation are impossible.

Earlier threshold propagation v1 and probability/moment v2 completed before the latest route change; both remain separately marked superseded. Their respective corruption union gains2.1408/2.0765pp are real development results, but teacher-target-generated candidates do not satisfy the latest student-only support constraint. Do not rewrite them as uniformly failed regression.

Future S1 would freeze student-generated candidates, use Sa2VA only for scalar preferences, and update persistent parameters by Reverse-KL of a finite-support compatibility softmax. This compatibility is a proposed surrogate distribution, not an established native likelihood or measured online-TTA gain. No temporal OPD rescue or production promotion.

## Verification and resources

GPU process 78.65s;864 suffix candidate evaluations and2 nonzero-state full-native reinsertion checks;0 new experts,0 backward,0 H captures,0 GPU failures.96 exact native checks,1440 dual metric calls,3 CPU contracts,independent NumPy parameter support checks and public scalar/CI reconstruction. First summary serialization encountered NumPy int64 counts after metric computation; explicit Python-int conversion fixed it, CPU rescoring matched existing ROWS exactly, original log retained and no GPU rerun. See SCORING_RECOVERY.json.
