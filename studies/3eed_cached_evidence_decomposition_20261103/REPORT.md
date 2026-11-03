# R52: cached color / position / reference evidence decomposition

Decision: do not adopt the current point-carrier rule-evidence G and do not start TTA. Its position term is the largest demonstrated source of damage. Target-color measurement is weaker on the point carrier, but deleting color makes the existing complete G worse. Removing position improves the complete G but does not beat source.

This is a fixed, exposed DEV336 diagnostic (168 Drone + 168 Quad; two Drone and three Quad scene clusters). It is not a new independent test or full5912 result. No FINAL384 artifacts, new models, raw media, training, parameter updates or full-target rescoring were accessed. All original 256 source boxes and the Waymo-only epoch100 initialization are unchanged.

## Frozen design

V is cached target color evidence, P is cached image-position evidence, R is cached reference-relation evidence. Scores are native source score plus 2V + 2P + 2R for included cues, in this order. Existing anchors, clipping, UNKNOWN masks, support weights and shared color factor 0.5 are unchanged. Every subset of the three cues is evaluated; no coefficient search or best-subset deployment is performed.

The new family uses the final R51 whole-proposal point-support medoid carrier. The old family is the R51 appearance-only replacement diagnostic: learned masks, measured target color and old reference proposals. It is not the full original external G. Its reference path and coverage also differ, so old-versus-new R is not an isolated carrier experiment. With R absent, V/P comparisons do not use reference scores, but still bundle carrier, measurement and association differences.

PR removes target appearance only. Reference color reweighting already inside R remains fixed; this is not an all-RGB-off experiment. Missing evidence remains UNKNOWN/zero rather than negative. All cases, fallbacks and negatives stay in the denominator.

## Overall results

| Readout | Drone mIoU % | Quad mIoU % | Balanced mIoU % | Delta vs source pp | Acc25 % | Acc50 % | Improved / worsened | Rescue / break25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 / 0 |
| external_G | 21.020217 | 35.667894 | 28.344056 | +6.513725 | 52.380952 | 23.511905 | 106 / 29 | 55 / 16 |
| new_V | 15.432610 | 27.540804 | 21.486707 | -0.343624 | 37.797619 | 20.535714 | 36 / 44 | 12 / 22 |
| new_P | 12.131125 | 19.041974 | 15.586549 | -6.243781 | 29.166667 | 16.071429 | 13 / 66 | 9 / 48 |
| new_R | 12.099955 | 26.432728 | 19.266342 | -2.563989 | 35.416667 | 19.940476 | 3 / 32 | 1 / 19 |
| new_VP | 14.436510 | 20.894571 | 17.665540 | -4.164790 | 31.250000 | 16.369048 | 46 / 76 | 19 / 51 |
| new_VR | 14.534965 | 26.924666 | 20.729815 | -1.100515 | 36.607143 | 19.642857 | 35 / 50 | 12 / 26 |
| new_PR | 11.259466 | 18.646315 | 14.952890 | -6.877440 | 27.380952 | 16.071429 | 13 / 75 | 8 / 53 |
| new_VPR | 14.512588 | 20.405383 | 17.458986 | -4.371345 | 30.952381 | 16.369048 | 46 / 78 | 19 / 52 |
| old_V | 16.462218 | 28.942610 | 22.702414 | +0.872083 | 41.071429 | 21.428571 | 39 / 24 | 14 / 13 |
| old_P | 14.894162 | 25.459120 | 20.176641 | -1.653690 | 36.011905 | 19.642857 | 34 / 41 | 13 / 29 |
| old_R | 15.857107 | 28.622566 | 22.239837 | +0.409506 | 41.666667 | 21.726190 | 14 / 13 | 9 / 6 |
| old_VP | 16.987843 | 27.362978 | 22.175410 | +0.345079 | 39.583333 | 20.238095 | 57 / 38 | 21 / 25 |
| old_VR | 16.626323 | 30.286258 | 23.456290 | +1.625959 | 43.154762 | 22.023810 | 46 / 25 | 21 / 13 |
| old_PR | 15.823644 | 27.764133 | 21.793889 | -0.036442 | 39.880952 | 21.726190 | 42 / 39 | 20 / 23 |
| old_VPR | 18.155697 | 28.562013 | 23.358855 | +1.528524 | 42.261905 | 21.726190 | 65 / 39 | 28 / 23 |

Improved/worsened uses continuous IoU delta tolerance 1e-10. Rescue/break25 means crossing the official strict IoU>0.25 threshold relative to source. This is an operational localization success measure, not semantic instance-identity gold. Equal platform sizes make micro and balanced means coincide here; the primary criterion remains equal-platform mIoU.

## What changed in the diagnosis

1. Position is not merely a fusion interaction: source+P alone loses 6.243781 pp, with 9 threshold rescues and 48 breaks. It is below source in all five clusters. Adding P to VR loses another 3.270830 pp (21 continuous improvements / 38 deteriorations). Removing it recovers score, but new_VR remains 1.100515 pp below source.

2. Reference relations are not validated by the earlier DINO replacement comparison. Current source+R loses 2.563989 pp, with only 1 threshold rescue and 19 breaks. Adding R to VP changes 25 queries and loses 0.206555 pp. Same-grounder reference localization remains a possible engineering replacement, not a demonstrated independent relation benefit.

3. Color is not the dominant cause of complete-G failure. New V alone is -0.343624 pp (12 rescues / 22 breaks), so it has not established standalone value on this carrier. However VPR beats PR by 2.506095 pp (50 improved / 15 worsened; 18 rescues / 6 breaks). Removing target appearance is harmful in this fixed context.

4. Measured color on the old learned-mask path has a small positive standalone value: old_V is +0.872083 pp (39 improved / 24 worsened; 14 rescues / 13 breaks). This repairs the missing zero-appearance comparison from R51. It does not show equivalence to Qwen: the complete external G is 28.344056%, whereas measured-color old_VPR is 23.358855%.

5. The current evidence is not a hidden positive component defeated only by three-way fusion: every new-carrier single cue and every new-carrier combination is below source. Fixed native scoring therefore gives no basis to train the student on these preferences. The old-path results show that the measurement/association context matters; they do not identify a unique bad clustering algorithm.

## All factorial additions

| Family | Add | Before | After | Balanced delta pp | Improve / worsen | Rescue / break25 |
|---|---|---|---|---:|---:|---:|
| new | V | source | new_V | -0.343624 | 36 / 44 | 12 / 22 |
| new | V | new_P | new_VP | +2.078991 | 52 / 22 | 19 / 12 |
| new | V | new_R | new_VR | +1.463474 | 41 / 21 | 14 / 10 |
| new | V | new_PR | new_VPR | +2.506095 | 50 / 15 | 18 / 6 |
| new | P | source | new_P | -6.243781 | 13 / 66 | 9 / 48 |
| new | P | new_V | new_VP | -3.821167 | 18 / 41 | 7 / 29 |
| new | P | new_R | new_PR | -4.313451 | 14 / 47 | 9 / 36 |
| new | P | new_VR | new_VPR | -3.270830 | 21 / 38 | 8 / 27 |
| new | R | source | new_R | -2.563989 | 3 / 32 | 1 / 19 |
| new | R | new_V | new_VR | -0.756892 | 2 / 10 | 1 / 5 |
| new | R | new_P | new_PR | -0.633659 | 7 / 13 | 4 / 10 |
| new | R | new_VP | new_VPR | -0.206555 | 6 / 9 | 4 / 5 |
| old | V | source | old_V | +0.872083 | 39 / 24 | 14 / 13 |
| old | V | old_P | old_VP | +1.998770 | 37 / 15 | 20 / 8 |
| old | V | old_R | old_VR | +1.216453 | 38 / 20 | 16 / 11 |
| old | V | old_PR | old_VPR | +1.564967 | 32 / 12 | 15 / 7 |
| old | P | source | old_P | -1.653690 | 34 / 41 | 13 / 29 |
| old | P | old_V | old_VP | -0.527004 | 30 / 28 | 11 / 16 |
| old | P | old_R | old_PR | -0.445948 | 30 / 28 | 12 / 18 |
| old | P | old_VR | old_VPR | -0.097435 | 32 / 27 | 12 / 15 |
| old | R | source | old_R | +0.409506 | 14 / 13 | 9 / 6 |
| old | R | old_V | old_VR | +0.753876 | 10 / 5 | 9 / 2 |
| old | R | old_P | old_PR | +1.617248 | 15 / 4 | 14 / 1 |
| old | R | old_VP | old_VPR | +1.183445 | 15 / 6 | 13 / 4 |

These are fixed-context cue interventions on cached scores. They are not a unique causal attribution or Shapley contribution: information inside R and cross-cue redundancy remain.

## Scene boundary

| Scene | n | Source | new V | new P | new R | new VR | new VPR | old V | old P | old R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D_S01 | 106 | 10.584991 | 8.860426 | 6.764892 | 6.951389 | 7.887888 | 8.709346 | 10.762316 | 10.436261 | 11.453750 |
| D_S02 | 62 | 24.011174 | 26.668925 | 21.305653 | 20.902343 | 25.899321 | 24.434260 | 26.207212 | 22.515733 | 23.385429 |
| Q_S01 | 62 | 10.117522 | 7.327243 | 4.187142 | 6.452888 | 6.567123 | 5.160805 | 9.835359 | 6.051722 | 7.380393 |
| Q_S02 | 49 | 32.998900 | 33.995579 | 26.806883 | 31.848213 | 32.844892 | 27.939941 | 33.995579 | 34.713705 | 36.012089 |
| Q_S03 | 57 | 43.509770 | 43.978642 | 28.524764 | 43.509770 | 43.978642 | 30.510129 | 45.382156 | 38.613224 | 45.375692 |

No frame-independence confidence interval or broad-domain significance claim is made from five exposed clusters. All split-level results and all scene/arm values are in the public scalar tables.

## Scale and measurement interpretation

Median native score range is 0.895950; median source Top1/Top2 gap is 0.096973. The median range of the weighted new-P term is 3.964205. Source probabilities were verified in [0,1].

The inherited common multiplier is therefore not an equal influence across cues. This supports a scale-dominance risk, but does not prove that rescaling would recover accuracy: P is derived from a partial-visible carrier and a fixed image-coordinate predicate, and P alone already harms all five scenes. The data do not separate wrong spatial interpretation, wrong target support and numerical dominance. No learning-rate, coefficient or gate search was performed.

The carrier uses visible points, not complete-object boundaries or a certified semantic instance. The negative findings are compatible with a source-box-to-background feedback loop, but this diagnostic does not independently certify instance identity, camera calibration or the complete-object center. Do not state that any one of these is proven as the sole cause.

## Decision and next discriminating change

Stop the current fixed point-carrier V/P/R rule recipe; do not connect its feedback to MLP or Q/V-LoRA TTA. Preserve the R50 external G as the stronger direct-reranking research baseline. Do not promote new_VR or old_VR as a final method just because they rank higher among these development diagnostics.

If a subsequent repair is authorized, target the position/association interface first: use the same measured color and relation machinery, compare the actual current position readout with an explicit description-frame/instance-position measurement and a prespecified scale control, retaining a no-position control. It must establish overall net value before any parameter update. Do not retrain source or add TTA regularizers to compensate for currently harmful supervision.

For reference localization, retain multiple source hypotheses and UNKNOWN as in R51, but require a matched direct source+relation comparison to become positive. The prior nearly lossless replacement of DINO did not establish that its relation evidence itself was useful.

## Evidence and accounting

All 5,376 task cells are retained: 1,680 exact reused baseline cells and 3,696 newly scored cells in one post-seal CPU scoring pass. All 10,907 directed negative comparisons are exported. Independent prediction audit checks all 5,376 candidate-score vectors and canonical source-box/readout bytes. Independent saved-scalar aggregation checks all cases, scene/split aggregates, reused identity joins and negatives; its maximal overall error is 7.11e-15 pp.

The decomposition uses existing numeric caches only. Predictions were sealed before target GT scoring. No source training, VLM/segmentation/detector calls, RGB/point decoding, TTA, holdout access, full-target rerun or automation was performed. This report does not add its development evidence to R51 as an independent replication.

Reused reference study: [R51 immutable report](https://github.com/Zonglin-He/A/blob/720adf8c976b846c72ee5fa0b63d94ab4392610c/studies/3eed_single_grounder_evidence_replacement_20261102/REPORT.md). Public tables contain anonymized scalar outcomes only; data, descriptions, coordinates, original identifiers, candidate indices, features, weights and third-party snapshots are excluded.
