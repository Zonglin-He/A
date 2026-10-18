# Frozen source24: sparse observation utilization

No new training. Repaired HC epoch30 / Vid epoch20, the same 24 source-validation queries and four epsilon paths, sigma .213 to .03 / 16 steps. These sources are held out only for the auxiliary prior, not established unseen to the pretrained Student. No target inference, Student adaptation, new DINO calls, or method promotion.

## Conditions and controls

Five conditions are evaluated at both clocks. none has no observation; DINO_all uses all original admitted boxes; DINO_common and GT_common share precisely the same positions with exact source GT, changing only box values; GT_planned uses available source GT at the original four planned anchors, including otherwise missing DINO positions. No exterior GT is fabricated. GT_common/GT_planned are source-only oracle controls, and the common-support DINO arm is also GT-assisted diagnosis. DINO_all versus GT_planned is not a pure reference-quality intervention. All 24 queries and empty-observation cases remain.

HC has 43 actual DINO / 33 common / 37 planned GT observations; Vid has 41 / 21 / 23. GT_planned is nonempty in 12 HC and 9 Vid queries, common support in 11 HC and 7 Vid; empty cases remain no-update guidance controls.

Fixed clock pins Native temporal state at initialization, every clean estimate, and every proposal; its observation membership is the exact physical integer Native interval. Free clock preserves the old event-relative floating hard mask. The latter can disable observations as the interval changes. This is still the original event-relative 32-knot representation, not a fixed-video identity model.

Guidance uses the clean denoised estimate and its complete derivative to noisy state. The learned association MLP is not executed; fixed .5 inlier/outlier mixture, original source-trained variance, .02*sigma^2 step and .2 component cap are retained. No tuning for the oracle. Individual posterior paths are decoded separately, never averaged as coordinates or selected by GT.

## First-path results

Percent; 12 queries per domain. Full sIoU uses original benchmark support; fixed_time_v uses common Native temporal output.

| Clock / condition | HC vIoU | HC tIoU | HC sIoU | HC fixed-time vIoU | Vid vIoU | Vid tIoU | Vid sIoU | Vid fixed-time vIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed / Native | 43.280488 | 67.739714 | 64.016598 | 43.280488 | 18.195709 | 35.521063 | 50.654046 | 18.195709 |
| fixed / Native_codec | 43.367793 | 67.739714 | 64.120651 | 43.367793 | 18.224082 | 35.521063 | 50.788860 | 18.224082 |
| fixed / none | 42.642370 | 67.739714 | 63.241881 | 42.642370 | 18.228638 | 35.521063 | 49.757293 | 18.228638 |
| fixed / DINO_all | 42.659869 | 67.739714 | 63.267357 | 42.659869 | 18.235282 | 35.521063 | 49.773074 | 18.235282 |
| fixed / DINO_common | 42.665881 | 67.739714 | 63.283224 | 42.665881 | 18.236790 | 35.521063 | 49.797526 | 18.236790 |
| fixed / GT_common | 42.676052 | 67.739714 | 63.291766 | 42.676052 | 18.243352 | 35.521063 | 49.802827 | 18.243352 |
| fixed / GT_planned | 42.680230 | 67.739714 | 63.296300 | 42.680230 | 18.251415 | 35.521063 | 49.840633 | 18.251415 |
| free / Native | 43.280488 | 67.739714 | 64.016598 | 43.280488 | 18.195709 | 35.521063 | 50.654046 | 18.195709 |
| free / Native_codec | 43.367793 | 67.739714 | 64.120651 | 43.367793 | 18.224082 | 35.521063 | 50.788860 | 18.224082 |
| free / none | 38.329163 | 65.918976 | 59.296228 | 39.764529 | 17.679277 | 39.501725 | 48.631961 | 17.812547 |
| free / DINO_all | 38.202796 | 65.781940 | 59.306812 | 39.668586 | 17.701927 | 39.495265 | 48.993449 | 17.837906 |
| free / DINO_common | 38.099812 | 65.473344 | 59.396081 | 39.771042 | 17.730912 | 39.474776 | 49.688071 | 17.881123 |
| free / GT_common | 38.244440 | 65.569146 | 59.524791 | 39.939665 | 17.786567 | 39.517922 | 49.718673 | 17.904343 |
| free / GT_planned | 38.690105 | 65.540052 | 60.018678 | 40.418678 | 17.794943 | 39.517922 | 49.758413 | 17.912769 |

## Fixed-support spatial completion

The physical support is held identical for every condition and clock: annotated source frames inside the original Native interval and original decoder input range, excluding the four originally planned anchors for the two unobserved supports. Native decoder frames and benchmark interpolation positions are disjoint. Null supports are not converted to zero; their contributing query and frame counts are reported. These geometry metrics do not annotate identity correctness.

| Clock / condition | HC observed sIoU (n) | HC unobserved decoder sIoU (n) | HC unobserved interpolated sIoU (n) | Vid observed sIoU (n) | Vid unobserved decoder sIoU (n) | Vid unobserved interpolated sIoU (n) |
|---|---:|---:|---:|---:|---:|---:|
| fixed / Native_codec | 63.584821 (12) | 64.105503 (12) | 64.023480 (12) | 51.829289 (9) | 52.989321 (10) | 53.007050 (10) |
| fixed / none | 59.926552 (12) | 62.659819 (12) | 63.161393 (12) | 53.257764 (9) | 50.976952 (10) | 51.763895 (10) |
| fixed / DINO_all | 60.073832 (12) | 62.677765 (12) | 63.191185 (12) | 53.317656 (9) | 50.989101 (10) | 51.783886 (10) |
| fixed / DINO_common | 60.158169 (12) | 62.692082 (12) | 63.212011 (12) | 53.366008 (9) | 51.006809 (10) | 51.813963 (10) |
| fixed / GT_common | 60.207883 (12) | 62.688179 (12) | 63.222341 (12) | 53.440965 (9) | 51.011130 (10) | 51.824318 (10) |
| fixed / GT_planned | 60.231122 (12) | 62.688512 (12) | 63.226967 (12) | 53.695371 (9) | 51.051541 (10) | 51.879448 (10) |
| free / Native_codec | 63.584821 (12) | 64.105503 (12) | 64.023480 (12) | 51.829289 (9) | 52.989321 (10) | 53.007050 (10) |
| free / none | 60.204738 (12) | 58.857256 (12) | 59.681416 (12) | 51.292572 (9) | 48.802802 (10) | 49.615011 (10) |
| free / DINO_all | 59.714314 (12) | 58.729457 (12) | 59.525473 (12) | 51.701360 (9) | 49.491827 (10) | 50.006003 (10) |
| free / DINO_common | 60.650987 (12) | 58.784860 (12) | 59.661646 (12) | 52.950337 (9) | 50.100941 (10) | 50.850926 (10) |
| free / GT_common | 63.054050 (12) | 58.524011 (12) | 59.960857 (12) | 53.150740 (9) | 50.158145 (10) | 50.916028 (10) |
| free / GT_planned | 63.507824 (12) | 58.873439 (12) | 60.490108 (12) | 53.460733 (9) | 50.199799 (10) | 50.974086 (10) |

## Paired differences

Source-query paired Student-t 95% intervals in percentage points, with positive/negative/neutral counts. All four per-path outcomes and per-query mean-of-four outcomes are included in RESULTS.json; four paths are not four independent sources. No checkpoint or path selection.

| Clock / comparison / metric | HC mean [CI95], +/−/= | Vid mean [CI95], +/−/= |
|---|---|---|
| fixed / DINO_all - none / v | +0.017499 [+0.008436, +0.026561], 10/1/1 | +0.006644 [-0.002546, +0.015834], 5/4/3 |
| fixed / DINO_all - none / unobserved_native_s | +0.017946 [+0.004684, +0.031208], 10/1/1 | +0.012149 [-0.003308, +0.027605], 6/3/1 |
| fixed / DINO_all - none / unobserved_interpolated_s | +0.029791 [+0.011968, +0.047615], 10/1/1 | +0.019991 [-0.004699, +0.044681], 5/4/1 |
| fixed / GT_common - DINO_common / v | +0.010171 [-0.005249, +0.025591], 8/3/1 | +0.006562 [-0.003138, +0.016262], 6/1/5 |
| fixed / GT_common - DINO_common / unobserved_native_s | -0.003903 [-0.024830, +0.017024], 7/4/1 | +0.004321 [-0.017079, +0.025721], 5/2/3 |
| fixed / GT_common - DINO_common / unobserved_interpolated_s | +0.010330 [-0.029071, +0.049731], 9/2/1 | +0.010355 [-0.011324, +0.032034], 6/1/3 |
| fixed / GT_planned - none / v | +0.037860 [+0.026740, +0.048979], 12/0/0 | +0.022777 [+0.008046, +0.037507], 8/0/4 |
| fixed / GT_planned - none / unobserved_native_s | +0.028694 [+0.009340, +0.048047], 12/0/0 | +0.074589 [-0.010646, +0.159824], 9/0/1 |
| fixed / GT_planned - none / unobserved_interpolated_s | +0.065573 [+0.038782, +0.092365], 12/0/0 | +0.115553 [+0.000637, +0.230469], 8/1/1 |
| free / DINO_all - none / v | -0.126367 [-0.416237, +0.163502], 4/7/1 | +0.022650 [-0.018310, +0.063610], 5/5/2 |
| free / DINO_all - none / unobserved_native_s | -0.127799 [-0.341609, +0.086011], 5/6/1 | +0.689024 [-0.864959, +2.243007], 6/3/1 |
| free / DINO_all - none / unobserved_interpolated_s | -0.155943 [-0.711474, +0.399588], 7/4/1 | +0.390991 [-0.487312, +1.269294], 5/4/1 |
| free / GT_common - DINO_common / v | +0.144628 [-0.165981, +0.455237], 6/5/1 | +0.055655 [-0.057632, +0.168943], 6/1/5 |
| free / GT_common - DINO_common / unobserved_native_s | -0.260849 [-0.661452, +0.139755], 2/9/1 | +0.057204 [+0.003298, +0.111111], 6/1/3 |
| free / GT_common - DINO_common / unobserved_interpolated_s | +0.299211 [-0.330891, +0.929312], 5/6/1 | +0.065102 [-0.040544, +0.170747], 5/2/3 |
| free / GT_planned - none / v | +0.360942 [-0.663815, +1.385699], 5/7/0 | +0.115666 [-0.029216, +0.260547], 8/0/4 |
| free / GT_planned - none / unobserved_native_s | +0.016183 [-0.884154, +0.916520], 5/7/0 | +1.396996 [-1.589495, +4.383487], 8/1/1 |
| free / GT_planned - none / unobserved_interpolated_s | +0.808691 [-0.370648, +1.988031], 10/2/0 | +1.359075 [-1.426906, +4.145055], 7/2/1 |

## Influence and interpretation limits

Per-query traces expose active reference count, mixture responsibility, full-chain gradient, actual bounded guidance displacement, clipping, and resulting state difference from none. Empty or identical observation sets are explicitly reused, and free none is exact previous sigma.213 reuse. These traces distinguish an ineffective observation interface from a claim that all conditional completion is impossible. Sparse GT may be geometrically correct while receiving almost no inlier responsibility when the current predicted trajectory is far away. Such near-zero influence does not establish a failure of properly trained conditional imputation.

This fixed small source-development diagnosis tests the frozen prior and current observation likelihood only. It does not establish cross-domain identity association, reliable dense rewards, OPD superiority, or a new TTA result. No automatic training/target/OPD continuation. Pixel panels inspect actual inputs and decoded outputs only and add no semantic identity labels.

## Cost and verification

Independent scalar/support/seal checks: 4498. GPU lease 11.853135s; cumulative 49.940007s under original14400s cap. Includes two old no-observation interface replays; excludes upstream cached video/expert/feature work, CPU audit and publication. It is not online end-to-end latency.

## Four-path sensitivity and guidance strength

Mean of four separately decoded paths, vIoU percent; 12 query denominator per domain.

| Clock / condition | HC | Vid |
|---|---:|---:|
| fixed / none | 43.089996 | 18.005705 |
| fixed / DINO_all | 43.104821 | 18.014140 |
| fixed / GT_common | 43.122213 | 18.022142 |
| fixed / GT_planned | 43.125533 | 18.029621 |
| free / none | 37.509759 | 18.120326 |
| free / DINO_all | 37.516541 | 18.121130 |
| free / GT_common | 37.667345 | 18.144474 |
| free / GT_planned | 37.926198 | 18.152497 |

Guidance influence over nonempty queries (mixture statistics average the four paths; step norms refer to their joint four-path tensor). Latent changes are not pixel units, and are not direct Student parameter updates.

| Clock / condition | HC mean responsibility / summed step norm | Vid mean responsibility / summed step norm |
|---|---:|---:|
| fixed / DINO_all | 0.800686 / 0.022240 | 0.615877 / 0.020453 |
| fixed / DINO_common | 0.767451 / 0.026129 | 0.614250 / 0.024404 |
| fixed / GT_common | 0.993787 / 0.023896 | 0.752167 / 0.023816 |
| fixed / GT_planned | 0.992367 / 0.023724 | 0.791337 / 0.030389 |
| free / DINO_all | 0.757456 / 0.059372 | 0.570077 / 0.052774 |
| free / DINO_common | 0.733843 / 0.063451 | 0.604722 / 0.057801 |
| free / GT_common | 0.986775 / 0.055327 | 0.713931 / 0.060492 |
| free / GT_planned | 0.975429 / 0.060552 | 0.758034 / 0.059755 |

On the fixed clock, GT_planned first-path gains over none are only +0.037860/+0.022777 pp vIoU (HC/Vid), and unobserved decoder gains are +0.028694/+0.074589 pp. The common-support GT-minus-DINO difference is +0.010171/+0.006562 pp vIoU; this is the reference-quality comparison without a coverage change. This does not establish useful correction, and GT observation quality alone does not solve this frozen interface.

GT influence is not generally zero: fixed-clock GT_planned mean mixture responsibility is approximately .992 HC / .791 Vid, so widespread rejection of correct observations is not the general explanation. Some Vid oracle observations do receive essentially zero responsibility; the mixture can still suppress distant correct evidence. Actual capped steps are small under the unchanged .02*sigma^2 rule. This is a measured weak-response configuration, not evidence that a trained sparse-conditional completion model cannot work.

Free-clock DINO_all changes first-path vIoU by -.126367/+.022650 pp, while GT_planned changes it by +.360942/+.115666 pp. HC GT_planned tIoU drops .378925 pp; geometric observations therefore do not establish event-boundary evidence. The original moving interval mask and event-relative nodes are still confounded with geometry at this clock. Even oracle free-clock outcomes remain below raw Native overall.

The zero-guidance numerical interface replay uses the two prelocked first sources, four paths each at both clocks, with autograd enabled but its guidance exactly zero. It quantifies floating-point differences between inference and autograd execution kernels, not a new scored method. The probe leaves all scientific predictions unchanged; near-zero state differences should not be overinterpreted. Detailed maximum errors and actual-guidance displacements are in NUMERIC_INTERFACE_COMPLETION.json.

## Decision

The current frozen observation-likelihood interface has not demonstrated practically useful unobserved-frame completion, even with correct sparse source GT. Do not proceed to target adaptation, dense OPD, identity-model training, or solver/noise sweeps from this result. The result localizes a representation/conditioning/response limitation in this implementation without resolving their individual causal contributions. A redesigned sparse-conditional training task or fixed-video identity representation would be a separate proposal requiring its own bounded protocol; neither was run here.

Additional numerical-probe GPU reservation is 0.974941s; total current diagnosis 12.828076s, cumulative 50.914948s under original14400s. No training.

Nonempty-reference subsets (descriptive; do not replace full denominators):

| Domain / clock | DINO_all n / ΔvIoU pp | GT_common n / ΔvIoU pp | GT_planned n / ΔvIoU pp |
|---|---:|---:|---:|
| hc2 / fixed | 11 / +0.019089 | 11 / +0.036744 | 12 / +0.037860 |
| hc2 / free | 11 / -0.137855 | 11 / -0.092425 | 12 / +0.360942 |
| vidstg / fixed | 11 / +0.007248 | 7 / +0.025224 | 9 / +0.030369 |
| vidstg / free | 11 / +0.024709 | 7 / +0.183926 | 9 / +0.154221 |
