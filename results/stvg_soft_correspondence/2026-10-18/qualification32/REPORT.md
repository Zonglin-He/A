# Frozen visual-correspondence reward qualification32

Same exposed rollback16 parent sources/domain. Phase1 only: no parameter adaptation and no new DINO call. Original frozen Source appearance projection maps before text fusion, not a trained correspondence encoder. Whole Native interval fixed. Original online Before trajectories center the same M32 antithetic logit-cxcywh samples in every feedback condition. Admitted-anchor original round0 samples replay exactly. This is a reward-ranking test, not a new vIoU result.

## Primary unobserved and far actual decoder-frame results

Rank delta = weighted candidate true IoU minus uniform candidate true IoU. Units below are percentage points. Parent bootstrap resamples query-parent clusters (10,000 draws). Queries without evaluable GT for a support remain in total32, contribute unknown support, and are not filled with zeros. Empty DINO evidence yields uniform weights/no adaptation signal.

| Domain | Support | Parents / query denominator | Known frames | Reward | Δrank pp | Parent CI95 pp |
|---|---|---:|---:|---|---:|---:|
| hc2 | unobserved | 16 / 16 | 301 | real | +0.088934 | [+0.047657, +0.137684] |
| hc2 | unobserved | 16 / 16 | 301 | shuffled | +0.093314 | [+0.051813, +0.147386] |
| hc2 | unobserved | 16 / 16 | 301 | interpolation | +0.204741 | [-0.008392, +0.511636] |
| hc2 | far | 4 / 16 | 29 | real | +0.120902 | [+0.063873, +0.197361] |
| hc2 | far | 4 / 16 | 29 | shuffled | +0.105165 | [+0.055090, +0.169364] |
| hc2 | far | 4 / 16 | 29 | interpolation | +0.097686 | [+0.026530, +0.166634] |
| hc2 | observed | 15 / 16 | 49 | real | +0.084936 | [-0.013174, +0.202582] |
| hc2 | observed | 15 / 16 | 49 | shuffled | +0.084936 | [-0.013174, +0.202582] |
| hc2 | observed | 15 / 16 | 49 | interpolation | +0.332245 | [+0.106220, +0.571418] |
| vidstg | unobserved | 16 / 16 | 515 | real | +0.217409 | [+0.114392, +0.324942] |
| vidstg | unobserved | 16 / 16 | 515 | shuffled | +0.202547 | [+0.098966, +0.315094] |
| vidstg | unobserved | 16 / 16 | 515 | interpolation | +0.348922 | [+0.138648, +0.573681] |
| vidstg | far | 7 / 16 | 157 | real | +0.186771 | [+0.067030, +0.324967] |
| vidstg | far | 7 / 16 | 157 | shuffled | +0.148837 | [+0.031464, +0.304867] |
| vidstg | far | 7 / 16 | 157 | interpolation | +0.138527 | [-0.151962, +0.463584] |
| vidstg | observed | 12 / 16 | 30 | real | +0.202949 | [+0.065064, +0.366680] |
| vidstg | observed | 12 / 16 | 30 | shuffled | +0.202949 | [+0.065064, +0.366680] |
| vidstg | observed | 12 / 16 | 30 | interpolation | +0.542801 | [+0.183239, +0.950222] |

## Same-sample paired controls

| Domain | Support | Real minus control | Δrank pp | Parent CI95 pp |
|---|---|---|---:|---:|
| hc2 | unobserved | real_minus_shuffled | -0.004380 | [-0.026282, +0.015800] |
| hc2 | unobserved | real_minus_interpolation | -0.115806 | [-0.382852, +0.071746] |
| hc2 | far | real_minus_shuffled | +0.015737 | [-0.003947, +0.035421] |
| hc2 | far | real_minus_interpolation | +0.023216 | [-0.042164, +0.088596] |
| vidstg | unobserved | real_minus_shuffled | +0.014862 | [-0.016600, +0.050637] |
| vidstg | unobserved | real_minus_interpolation | -0.131513 | [-0.269007, -0.005816] |
| vidstg | far | real_minus_shuffled | +0.037934 | [-0.013228, +0.106555] |
| vidstg | far | real_minus_interpolation | +0.048244 | [-0.181627, +0.314791] |

## Fixed interface and validity

For each anchor, cosine affinities between L2-normalized frozen local appearance vectors use target-to-anchor row softmax at fixed0.07. All valid patches may correspond; there is no GT/Native proximity window. DINO regions are fractional box-cell masks, not segmentation. Soft-IoU includes original candidate area and outside-map area, preventing reward from ignoring exterior box area. Keep separate anchor maps/rewards and average scores with equal admitted-anchor weights; no unique teacher trajectory or coordinate averaging is constructed. These maps are not calibrated identity probabilities.

The feature grid is coarse (approximately seven vertical cells). Padding/validity and actual shapes are retained privately. Actual target frames supply their own visual features. Shuffle cyclically deranges only non-planned frame features and keeps anchor features fixed. Interpolation uses the same admitted boxes on physical frame IDs, constant nearest-box outside their span. Reward temperature remains original domain tau; matching temperature is a single prespecified interface, not tuned on these outcomes. Reward dynamic-range differences are part of this interface test, so failure does not isolate calibration from correspondence quality.

All32 samples/maps/three rewards/weights were SHA-sealed before GT evaluation. GT candidate quality uses only actual decoder frames with known annotation, preserving historical HC negative-coordinate clipping for evaluation only. Candidate generation/rewards are unchanged by GT. Scalar weighted ranking is independently summed per candidate/frame. Missing annotation outside the event is unknown, not background or failure. Far means physical distance to nearest planned DINO slot at least25% of input span, set before scoring. It is not hand-picked from GT.

Observed includes all planned slots, even rejected references; unobserved excludes all four planned slots. Empty-reference queries remain. Target area and true-time normalized motion use the previous full-cohort common bins; query form is parsed metadata. Old WI/CI/CP/MX/UN labels are reused only for offline sensitivity; they are model-assisted actual-pixel labels, not independent human annotations. Per-anchor rank diagnostics retain wrong-reference and unknown labels without routing or filtering.

## Prelocked continuation gate

- hc2: `{"at_least8_unobserved_parents": true, "at_least8_far_parents": false, "real_unobserved_positive_CI": true, "real_beats_shuffle_CI": false, "real_beats_interpolation_CI": false, "real_far_positive": true, "real_far_beats_both": true}`; passed=False.
- vidstg: `{"at_least8_unobserved_parents": true, "at_least8_far_parents": false, "real_unobserved_positive_CI": true, "real_beats_shuffle_CI": false, "real_beats_interpolation_CI": false, "real_far_positive": true, "real_far_beats_both": true}`; passed=False.

**Stop this frozen-feature feedback version. No new OPD or direct optimization arm is run.** The prelocked two-domain unobserved/far and matched-control conjunction failed. No training, new feature extractor, temperature sweep, expert refresh or rollout expansion follows. This negative result does not rule out trained correspondence or all dense feedback. Original full online DINO OPD remains unchanged.

## Actual cost and exclusions

32 frozen Source captures, 0 new DINO calls, 0 Student updates, 0 new training. GPU lease 185.486940s; summed Source feature forward capture 101.908940s; summed correspondence/sampling/reward 26.689718s. Lease includes loading, CPU transfers and verification, not a pure kernel number or full first-download deployment latency. Features and maps are cached once per actual video/query frame grid in this finite panel; the loop does not rerun experts per candidate.

Code, fixed protocol, domain configuration, all signed query/frame ranking diagnostics, size/motion/query strata, controls, negative/neutral/unknown and gate results are preserved. Raw media, query text, private parent IDs, GT coordinates, weights and tensors are excluded. No production registry is changed.

Contrastive Random Walk learns representations with a correspondence objective (https://proceedings.neurips.cc/paper/2020/hash/e2ef524fbf3d9fe611d5a8e90fefdc9c-Abstract.html). This audit only borrows the correspondence principle; it does not reproduce that training or claim its guarantees for frozen STVG features.

## Engineering recovery and actual view

A CPU scorer path-join typo was caught by the pre-GT feedback SHA check and corrected before any GT score was produced. No Source/feature/map/reward/sample was rerun or edited; original failure and scorer are preserved privately. Independent NumPy rectangle integration and Gaussian/weight/derangement/aggregate checks are in ROOT_INDEPENDENT_AUDIT.json. Two fixed first-domain cases were actually viewed in24 private RGB cells showing per-anchor maps and real/shuffle/interpolation. Engineering inspection only, zero new independent identity labels.
