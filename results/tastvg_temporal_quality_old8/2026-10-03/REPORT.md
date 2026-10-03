# Fixed Old8 temporal localization-quality review

The fixed-support experiment is complete and independently audited. The one new signal is cached PE pooled-query inner-minus-outer temporal contrast (outer ratio .25). A remains the baseline; this result does not promote a new deployment method.

## Configuration and data boundary

VidSTG and HC-STVG-v2 each use the same 32 development + 16 source-disjoint confirmation sources within the current batch, one query/source, two orders, clean + five fixed 5% transient corruptions, 25% scheduled experts. All have historical exposure. There are 1,152 readouts, 288 expert positions (240 corruption + 48 clean), and 864 positions that remain exactly A. The comparison is against A, not Frozen. No formula was chosen on confirmation.

A boxes, pre/post persistent spatial states, Uniform Rank-RKL writes, Old8 intervals and their order, corruption pixels, checkpoints and expert schedule were identical. Vid K1 and HC K8 use their sealed best learning rates and teacher temperatures. No model forward, backward, new expert observation, new temporal view, target training, or source quality-head training occurred.

## Signal and its limits

The original critic takes the maximum proposal-confidence-weighted interval overlap. The new critic uses cosine of each cached projected PE video feature with text column zero (the projected pooled query), and ranks the **same Old8** by mean similarity inside minus mean similarity in two adjacent bands, each .25 times the interval length. Fractional bin overlap maps original phase-zero 2-Hz features to the unchanged physical observed window. There is no proposal confidence multiplier or consensus. A full-window candidate has no observed outer context and receives neutral zero contrast; it remains in the pool. Constant evidence falls back to A; numerical score ties follow native-first original candidate order.

The contrast geometry is inspired by [AutoLoc, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.pdf), which uses trained class activations and video-level action supervision. Here frozen query similarities are only an inference proxy. This is not AutoLoc training or calibrated interval IoU. The [official PE interface](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md) and the cached poolandtoken extraction contract were inspected before the rule was locked. Persistent objects/background similarity and incomplete action semantics can make this proxy prefer a wrong or too-short segment.

## Actual corruption readouts

Equal-source means and paired 95% source-cluster bootstrap intervals; vIoU/tIoU differences are percentage points. All-flow gains are scored directly, not expert means multiplied by .25.

| Panel | A vIoU | New vIoU | All-flow ΔvIoU [95% CI] | Expert ΔvIoU [95% CI] | All-flow ΔtIoU [95% CI] |
|---|---:|---:|---:|---:|---:|
| vidstg search | 17.4529 | 17.0444 | -0.4085 [-1.0458, +0.0404] | -1.6339 [-4.0547, +0.1425] | -1.7596 [-3.3405, -0.3953] |
| hc2 search | 30.9637 | 30.7475 | -0.2161 [-0.8022, +0.2926] | -1.1319 [-3.6109, +1.1526] | -0.7545 [-2.0554, +0.2773] |
| vidstg confirm | 32.6465 | 32.5923 | -0.0542 [-0.4990, +0.3754] | -0.2166 [-1.9276, +1.5515] | +0.0996 [-0.7654, +0.9842] |
| hc2 confirm | 26.7630 | 26.4352 | -0.3278 [-0.9475, +0.2046] | -1.3830 [-4.0075, +1.0347] | -1.0384 [-3.0777, +0.5022] |

## Same-support oracle regret

The oracle is GT-best selection from the unchanged Old8 under A boxes. It is privileged offline evaluation, not an attainable unlabelled selector. Regret reduction is algebraically the same selected-vIoU improvement on a fixed pool; it is not independent confirmation. Negative recovery fractions mean the new selector increases remaining selection error.

| Corruption expert panel | Cells / independent sources | Old regret pp | New regret pp | Regret reduction pp [95% CI] | Fraction recovered |
|---|---:|---:|---:|---:|---:|
| vidstg search | 80 / 16 | 2.1197 | 3.7536 | -1.6339 [-4.0547, +0.1425] | -77.08% |
| hc2 search | 80 / 14 | 5.5354 | 6.6673 | -1.1319 [-3.6109, +1.1526] | -20.45% |
| vidstg confirm | 40 / 8 | 3.1790 | 3.3956 | -0.2166 [-1.9276, +1.5515] | -6.81% |
| hc2 confirm | 40 / 7 | 3.5431 | 4.9260 | -1.3830 [-4.0075, +1.0347] | -39.03% |

## Harm, original good results, and clean controls

| Corruption expert panel | Better / worse / equal | >5pp harm | Old positive rerank harmed / available | Correct→wrong at .3 / .5 | Wrong→correct at .3 / .5 |
|---|---:|---:|---:|---:|---:|
| vidstg search | 11 / 32 / 37 | 9 | 23 / 35 | 0 / 0 | 0 / 0 |
| hc2 search | 19 / 29 / 32 | 9 | 17 / 37 | 2 / 3 | 0 / 2 |
| vidstg confirm | 12 / 13 / 15 | 1 | 2 / 11 | 0 / 0 | 0 / 0 |
| hc2 confirm | 12 / 13 / 15 | 8 | 12 / 24 | 3 / 4 | 1 / 1 |

| Panel | Clean all-flow ΔvIoU pp [95% CI] | Corrupt all-flow order values pp | Leave-one-source-out range pp |
|---|---:|---:|---:|
| vidstg search | -0.4903 [-1.6527, +0.4505] | -0.1828, -0.6341 | -0.4685, -0.1621 |
| hc2 search | +0.1337 [-0.3379, +0.6657] | -0.4078, -0.0245 | -0.3483, -0.0047 |
| vidstg confirm | -0.0326 [-0.5606, +0.5363] | +0.1229, -0.2312 | -0.1784, +0.0729 |
| hc2 confirm | -0.4247 [-1.4048, +0.3549] | +0.1035, -0.7592 | -0.4568, -0.1354 |

## Ranking evidence and concrete successes/failures

Every harmed position still contains the old A choice in its unchanged pool, so support deletion cannot explain this new harm. There are no missing/constant embedding fallbacks in the actual 288 expert positions; only one candidate has no outer observation. Pair accuracy is diagnostic rather than the primary endpoint: better ordering of some pairs does not ensure that the top-ranked interval has higher tube quality.

| Corruption expert panel | Strict-pair cells / sources | Old pair accuracy % | New pair accuracy % | Difference pp [95% CI] |
|---|---:|---:|---:|---:|
| vidstg search | 57 / 12 | 76.450 | 55.747 | -20.7025 [-36.0726, -6.1449] |
| hc2 search | 80 / 14 | 62.339 | 52.600 | -9.7392 [-22.3623, +2.2831] |
| vidstg confirm | 40 / 8 | 53.927 | 57.897 | +3.9699 [-10.9821, +21.7583] |
| hc2 confirm | 40 / 7 | 70.499 | 69.061 | -1.4387 [-20.4427, +17.9649] |

| Confirmation example | Anonymous source / condition / order | ΔvIoU pp | Old→new tIoU | Contrast of A choice→new choice |
|---|---|---:|---:|---:|
| vidstg harm | 36 / frame_drop_5 / order1 | -5.4593 | 0.5248→0.4504 | 0.011962→0.015387 |
| vidstg gain | 41 / exposure_5 / order1 | +5.5478 | 0.8057→0.9514 | -0.002619→0.003123 |
| hc2 harm | 43 / exposure_5 / order2 | -11.7804 | 0.8750→0.5882 | 0.001099→0.009487 |
| hc2 gain | 47 / frame_freeze_5 / order1 | +8.1310 | 0.3586→0.4638 | 0.025289→0.033909 |

For HC confirmation source43/exposure/order2, the contrast increases from .001099 to .009487, but tIoU falls from .8750 to .5882 and vIoU falls by 11.7804 pp. The chosen interval extends farther to the right; its lower average outer similarity outweighs its slightly lower inner similarity. This is a direct counterexample to treating higher PE contrast as higher localization quality. It does not establish that this mechanism explains every failure. HC also has a positive source47/frame-freeze case (+8.1310 pp), and both types are preserved rather than retuning the formula.

## Verification and costs

All 5633 private input files were hash-checked before and after. Every A box/state and original critic score/selection was checked; all 288 candidate pools and their old oracle values match the predecessor. The new activation was independently recomputed from cached projected features. Official dense metrics and anonymous arithmetic were independently checked, including group means, paired bootstrap, counts and decisions. Root maximum official error: 1.11e-15. Both datasets and panels were sealed before this new offline GT scoring. No GT entered the candidate scorer or decoder.

CPU generation/readout wall time 7.498s; score wall time 36.438s; root audit wall time 38.672s. There are 270 distinct reused feature inputs. These timings include cache IO and verification, and are not GPU kernel time or an uncached deployment estimate. New model/expert calls and backwards are all zero.

A syntax typo at pre-lock startup was preserved in recovery/startup_syntax_001 and repaired before any predictions or label read. All subsequent fixes, if any, are enumerated in ENGINEERING_HISTORY; no outcome-dependent science revision is permitted.

## Decision scope

Evidence label: `not_positive_in_both_confirmation_panels`. Preserve A, the previous New8 negative result, and all positive and negative examples from this test. A single unlabelled semantic contrast cannot establish that every localization-quality signal works or fails. Fixed-Old8 scoring cannot repair the large Grid−Old8 candidate coverage deficit. Any next mechanism or expansion is a new decision; no further experiment, spatial change, memory or method promotion starts here.

Full anonymous cell values, curves, eight candidate scores/utilities, cases, source summaries and order sensitivity are supplied alongside the report. Predicted interval indices and normalised observed-window coordinates are included; target annotations, GT coordinates, captions, raw features, boxes, states and weights are not.
