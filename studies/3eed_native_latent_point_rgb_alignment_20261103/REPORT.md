# R53: native point-language-candidate latent links for RGB support

Decision: do not adopt this latent-support G as a replacement for the external instance path, and do not start TTA. Native attention raises full-G balanced DEV mIoU from 17.458986% to 17.662414% (+0.203428 pp), but remains -4.167917 pp below source. Pre-fusion feature support gives 17.659093%, essentially the same primary score (+0.003320 pp native-minus-pre). Native-minus-fixed-permutation is +0.646645 pp, a small descriptive correspondence signal rather than demonstrated foreground identity.

This is a newly authorized frozen-model experiment on the same exposed DEV336:168 Drone/168 Quad, two Drone and three Quad scene clusters. Predictions were sealed before one new scoring pass. No FINAL384 access, full5912 evaluation, source training, optimizer steps, new TTA loss, parameter adaptation, image encoder or external vision inference was performed. The R50 external G is retained only as the stronger cached direct-use reference.

## Interfaces actually extracted and verified

One unchanged Waymo-only epoch100 3EED was run for all336 original sampled inputs and full descriptions. The capture retains pre-fusion fp2_features, post-fusion seed_features, text_memory, six decoder outputs, final64D query/token matching features, three encoder token-to-point matrices, six decoder query-to-point and six query-to-token matrices, final-decoder inputs, seed coordinates, sampling provenance and token offsets. Native attention weights are head-averaged outputs from the existing trained Q/K projections; no dimension-based query/point dot-product shortcut is used.

All336 source boxes and native first-noun source probabilities match the prior canonical source byte for byte. Frozen model fingerprint is 0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9. There were 336 full grounding forwards, 137.620s capture time, 679689216 peak Torch allocated bytes, zero updates.

The actual backbone is Point_Backbone_V2. Its fp2_inds takes the first1024 indices from the first sampling stage. The correct mapping for the retained second-stage coordinates is the composition of the first and second sampling indices.

Official indices fail the exact coordinate test on 7/336 observations. Among those observations, median/max of the per-observation maximum coordinate discrepancy is 10.421232/16.886616m. Composed indices recover every fp2_xyz exactly, and every query anchor maps exactly to its seed. The experiment uses fp2_xyz with this verified provenance and geometry-constrained nearest-seed lifting.

This does not explain the earlier R51/R52 geometric-G decline: those carriers use raw point support, and the inference coordinates/source outputs remain unchanged. It is a verified new latent-mapping hazard, not a checkpoint repair or evidence of better grounding accuracy.

For text-to-point binding, only the existing literal subject/reference phrases are used. Separator-only narrowing inside the original span bridges 286 entity spans; target masks are available for 336/336 and reference masks for 571/573. The two unavailable reference spans remain UNKNOWN. No role, phrase, attribute or final source readout is reparsed or selected with GT.

Reference phrases retain their own preexisting 256-query sets and original geometric-to-group source probabilities. Those query indices are never assumed to equal the main-description indices. Reference token masks here are taken from the same full-description forward, while original reference proposal mass/top3 remain fixed.

## Frozen association intervention

Every method retains the final R51 group membership, medoid member, geometric visible-point bag, original boxes, rectangles, visibility/depth projection and pure text graph. It changes only the soft point-to-pixel evidence support within that candidate bag. This is a within-carrier association test, not a new instance proposal generator or complete-instance segmentation test.

| Support | Fixed definition |
|---|---|
| Geometry | R51 uniform weights on the original medoid visible points; original readouts copied exactly |
| Pre-fusion feature | Affine cosine between pre-fusion point features and the proposal anchor point feature; nearest seed restricted to the same candidate box |
| Native attention | Geometric mean of final query-to-point attention and the entity-restricted query-to-token-to-point path, lifted from same-box seeds |
| Seed-permutation control | One prespecified hash-seeded permutation of native semantic seed values, retaining the geometry and global marginal values |

The same original HSV/Lab/color prototypes, scene correction, pseudo-count0.5, minimum-eight/reliability rule and spatial predicates are used. Weighted means use Kish effective sample size; uniform weights reproduce every registered R51 color descriptor within1e-11. Target pixel centers and reference UV centers use the same support weights; full object boundaries and physical centers are not inferred from visible surfaces. Missing latent support falls back to the original geometry and is flagged.

Scores remain native_source+2V+2P+2R with shared color factor0.5 and the same anchoring/clipping/UNKNOWN handling. V-only and VR-without-position were fixed before scoring to distinguish the known harmful position term; no coefficient/temperature search or best-subset promotion is performed. This creates no trained shared RGB-point-text embedding: RGB remains a measured observation.

## Overall results

| Readout | Drone mIoU % | Quad mIoU % | Balanced mIoU % | Delta source pp | Acc25 % | Acc50 % | Improve / worsen vs source | Rescue / break25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 / 0 |
| external_G | 21.020217 | 35.667894 | 28.344056 | +6.513725 | 52.380952 | 23.511905 | 106 / 29 | 55 / 16 |
| geometry_full | 14.512588 | 20.405383 | 17.458986 | -4.371345 | 30.952381 | 16.369048 | 46 / 78 | 19 / 52 |
| geometry_V | 15.432610 | 27.540804 | 21.486707 | -0.343624 | 37.797619 | 20.535714 | 36 / 44 | 12 / 22 |
| geometry_VR | 14.534965 | 26.924666 | 20.729815 | -1.100515 | 36.607143 | 19.642857 | 35 / 50 | 12 / 26 |
| pre_full | 14.608374 | 20.709813 | 17.659093 | -4.171238 | 31.845238 | 16.369048 | 45 / 77 | 21 / 51 |
| pre_V | 15.641271 | 27.540804 | 21.591037 | -0.239293 | 38.095238 | 20.535714 | 36 / 44 | 12 / 21 |
| pre_VR | 14.452637 | 26.924666 | 20.688652 | -1.141679 | 36.607143 | 19.642857 | 35 / 50 | 12 / 26 |
| native_full | 14.833918 | 20.490910 | 17.662414 | -4.167917 | 31.547619 | 16.666667 | 47 / 76 | 20 / 51 |
| native_V | 16.118764 | 27.436075 | 21.777419 | -0.052911 | 38.690476 | 20.833333 | 37 / 40 | 15 / 22 |
| native_VR | 14.761669 | 27.155555 | 20.958612 | -0.871719 | 37.202381 | 20.238095 | 35 / 48 | 14 / 26 |
| shuffle_full | 13.916562 | 20.114975 | 17.015769 | -4.814562 | 30.059524 | 16.071429 | 46 / 80 | 18 / 54 |
| shuffle_V | 15.378898 | 27.492023 | 21.435461 | -0.394870 | 38.095238 | 20.535714 | 37 / 44 | 13 / 22 |
| shuffle_VR | 13.972472 | 27.206829 | 20.589651 | -1.240680 | 36.607143 | 19.940476 | 35 / 50 | 13 / 27 |

Continuous improvement/deterioration tolerance is1e-10 IoU. Rescue/break25 means crossing the official strict IoU>0.25 threshold relative to source, not semantic instance-identity truth. All UNKNOWN, fallback and zero-update rows remain in the denominator. No significance claim or frame-independence confidence interval is made from five exposed scene clusters.

## Matched same-information no-update comparisons

| Before | After | Balanced delta pp | Improve / worsen | Query changes |
|---|---|---:|---:|---:|
| geometry_full | pre_full | +0.200108 | 5 / 2 | 14 |
| geometry_full | native_full | +0.203428 | 7 / 9 | 37 |
| pre_full | native_full | +0.003320 | 5 / 10 | 36 |
| shuffle_full | native_full | +0.646645 | 12 / 13 | 58 |
| geometry_V | pre_V | +0.104330 | 3 / 2 | 8 |
| geometry_V | native_V | +0.290712 | 10 / 7 | 32 |
| pre_V | native_V | +0.186382 | 9 / 8 | 33 |
| shuffle_V | native_V | +0.341959 | 12 / 10 | 54 |
| geometry_VR | pre_VR | -0.041164 | 2 / 3 | 9 |
| geometry_VR | native_VR | +0.228796 | 7 / 8 | 35 |
| pre_VR | native_VR | +0.269960 | 6 / 7 | 31 |
| shuffle_VR | native_VR | +0.368961 | 10 / 10 | 55 |

All methods here are direct no-update readouts. No student-update benefit is asserted. A single registered permutation is a diagnostic negative control, not an estimate over random mappings.

## Scene results

| Scene | n | Source | Geometry full | Pre full | Native full | Permuted full | Native V | Native VR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_S01 | 106 | 10.584991 | 8.709346 | 9.048905 | 9.736219 | 8.598415 | 9.978074 | 8.764788 |
| D_S02 | 62 | 24.011174 | 24.434260 | 24.113273 | 23.549339 | 23.008879 | 26.617362 | 25.014401 |
| Q_S01 | 62 | 10.117522 | 5.160805 | 5.985711 | 5.392554 | 5.347568 | 7.043462 | 6.283342 |
| Q_S02 | 49 | 32.998900 | 27.939941 | 27.939941 | 27.939941 | 26.707944 | 33.995579 | 33.995579 |
| Q_S03 | 57 | 43.509770 | 30.510129 | 30.510129 | 30.510129 | 30.510129 | 43.978642 | 43.978642 |

## Why the signal remains limited

The native query-to-point attention normalized entropy median is 0.944342 (uniform=1). After lifting within a fixed visible bag, median effective-pixel-count/original-count is 0.970356. Thus this soft-support rule leaves most of the effective observation support intact; it has not demonstrated foreground/background separation. These statistics are not a mask-purity measurement.

There are 48896 candidate-hypothesis groups, 94 with no seed in their box. Among 46895 groups with at least8 visible pixels, the median number of distinct matched seeds is 8.0; only 1103 use a single distinct seed. The failure is therefore not adequately explained by all objects collapsing to one seed. 7670 proposal anchors lie outside their regressed box, another limit of anchor-prototype association; this is descriptive, not a GT-based filter.

Native V-only reaches 21.777419% (-0.052911 pp source), while native VR reaches 20.958612% (-0.871719 pp). Removing the known harmful P does not establish net source improvement. Native V gains on Drone and loses on Quad, so its near-source average is not stable positive cross-platform evidence.

The plausible remaining explanations include broad/context attention rather than instance ownership, an overly soft local weighting rule, limitations of the unchanged carrier, and the weak existing color/relation/readout interface. This experiment does not isolate a unique cause. The small native-minus-permutation difference does not certify physical identity; the near-tie between native and pre-fusion full G also does not prove that language conditioning adds useful final-task information.

## Audits, scope and disposition

Independent all-case audit checks 400296 latent support vectors, all336 composed index/query-anchor relations, original source boxes/scores and every frozen readout. Four hash-selected observations (two/platform) replay the cross encoder and last decoder: 20 native attention matrices match their explicit trained Q/K formula, max error 0; saved attention and component features replay bitexact. These are 8 component replays and four text-encoder calls, not extra full grounding forwards.

Four separately hash-selected observations remeasure raw RGB and weighted centers using the original color function with an isolated weighted-mean proxy. All12 variant/case checks pass, max cue error 1.42e-15. The already audited R51 spatial operator is reused; this is not an independent calibration-truth or foreground-identity audit.

All 4704 task cells are retained (3024 new scored cells and 1680 exact reused cells), plus all 6058 directed negative comparisons. Independent saved-scalar reaggregation max overall error is 7.11e-15pp. Prior target exposure and overlapping scenes remain development exposure. No new holdout or independent-test claim is made.

The RoBERTa asset loader reports a generic pooler initialization warning; the complete trained checkpoint is then loaded strictly and the full frozen state fingerprint and canonical source outputs are verified. There is no untrained adapter or new random fusion layer in this experiment.

Do not adopt the present latent-weighted G, connect these preferences to TTA, infer a best update layer, or promote one exposed readout as a final method. Preserve the verified extraction/index interfaces and the small descriptive association signal. The current evidence does not distinguish cross-encoder binding, decoder evidence reading and final matching as parameter-update bottlenecks.

A subsequent explicitly authorized experiment should first test whether native entity-conditioned correspondence separates the target from competing/background support, with an unchanged no-update readout and a matched mapping control; any sharper weighting or expanded carrier must be fixed before scoring. The current within-bag result does not test global latent instance proposal generation. Do not rerun this fixed configuration or add a TTA loss to compensate for supervision that remains below source.

Public export excludes raw images, descriptions, coordinates, original IDs, latent/attention tensors, model weights and third-party snapshots. The source-only checkpoint remains unchanged. Earlier comparison: [R52 immutable report](https://github.com/Zonglin-He/A/blob/b5650da763ffa6f2220f075a988fa03e3422914f/studies/3eed_cached_evidence_decomposition_20261103/REPORT.md).
