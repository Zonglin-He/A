# R60 — internal-state detector, actual channel weight patch, native replay

**Decision: neither dynamic patch is adopted.** Noise-dominance editing leaves every final selection unchanged versus frozenABC. Candidate-common-response editing yields only +0.002229pp; its fixed and wrong-input controls give exactly the same final selections. No input-specific editing advantage is established. Do not enlarge attenuation merely to force a result.

## Scope and actual mechanism

Fixed Waymo-only epoch100, historically exposedDEV336(168per platform,five scene clusters), originalA256boxes and R57ABC sampling/soft transport/source-neutral UNKNOWN. Source resets for each input. Use audited source hidden states after the second ReLU and original text masks. No new external model, source training, rawRGB semantic encoder, target GT in detector, gradient, least-squares, pseudo-label fitting or per-input patch search. Development is AI-assisted; no separate Astra process was invoked, and no inference-stage external AI is used.

Only final64x288weight columns change; bias and earlier parameters stay fixed. At most8columns attenuated, alpha=.25 fixed before task scores, no hyperparameter grid. Formula Wx=W0 diag(1-a(x)),0<=a<=.25. Source restored after native final normalization/token-logsumexp matching/temperature/.07 and originalABC readout. There are two distinct diagnostic hypotheses, not assertions of discovered harmful channels:

1. Noise: normalize additive hidden-column contributions by source output norm, project to a unit subject-versus-other-token mean direction. Use fixed soft correspondence to compare within-sampling fluctuation with between-candidate variation. Require B/C support, select at most8positive excess channels, attenuation proportional to excess/(noise+signal). No hard identity claim is made from correspondence or token direction. This statistic is not raw activation variance and is not the final score.
2. Common response: candidate-common activation fraction above.5, normalized projection energy and low entity-direction alignment identify possible angle-compressing channels. Select at most8. These channels can still encode useful category semantics; the intervention must establish task benefit.

Both detectors triggered on336/336inputs,median8columns,median233supported anchors. Absence of evidence would give a zero patch, but no such input occurred in this cohort. No sample IDs, scene names, GT or caption-specific exceptions enter the detector. Hashes are only for reproducible negative-control randomization, not deployment logic.

## Controls and results

Shared fixed patch: top8mean dynamic attenuation over all336unlabeled internal states, shared across platforms. It is a target-transductively calibrated repair control, not source-only; it does not have identical norm to every dynamic input. Wrong-input control permutes patches within platform, preserving the exact patch multiset and no fixed points. Random control excludes selected channels, matching channel count, individual column edit magnitudes and total Frobenius norm; its safety attenuation cap.5 is registered. Exact activation-gating counterparts verify the weight-scaling equivalence. R58online is an inherited system reference with different history/supervision; R59directAll is the stronger same-internal-information no-edit reference.

| Arm | Drone mIoU% | Quad mIoU% | Balanced mIoU% | Delta vs ABC pp | Acc25% | Acc50% |
|---|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | -0.472129 | 40.773810 | 21.130952 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 0.000000 | 42.261905 | 21.428571 |
| r56_one | 15.908954 | 28.086002 | 21.997478 | -0.304982 | 41.369048 | 21.130952 |
| r58_online_ema_teacher | 15.924842 | 28.841619 | 22.383230 | 0.080770 | 42.559524 | 21.428571 |
| r58_online_ema_student | 15.868511 | 29.104569 | 22.486540 | 0.184079 | 42.857143 | 21.726190 |
| r59_direct_all | 16.385568 | 28.836664 | 22.611116 | 0.308656 | 43.154762 | 21.726190 |
| noise_dynamic | 15.736332 | 28.868588 | 22.302460 | 0.000000 | 42.261905 | 21.428571 |
| noise_fixed | 15.740789 | 28.868588 | 22.304689 | 0.002229 | 42.261905 | 21.428571 |
| noise_shuffled | 15.902723 | 28.868588 | 22.385656 | 0.083196 | 42.559524 | 21.428571 |
| noise_random | 15.736332 | 28.868588 | 22.302460 | 0.000000 | 42.261905 | 21.428571 |
| noise_activation | 15.736332 | 28.868588 | 22.302460 | 0.000000 | 42.261905 | 21.428571 |
| common_dynamic | 15.740789 | 28.868588 | 22.304689 | 0.002229 | 42.261905 | 21.428571 |
| common_fixed | 15.740789 | 28.868588 | 22.304689 | 0.002229 | 42.261905 | 21.428571 |
| common_shuffled | 15.740789 | 28.868588 | 22.304689 | 0.002229 | 42.261905 | 21.428571 |
| common_random | 15.736332 | 28.823825 | 22.280078 | -0.022382 | 41.964286 | 21.428571 |
| common_activation | 15.740789 | 28.868588 | 22.304689 | 0.002229 | 42.261905 | 21.428571 |

## All primary pairwise comparisons

| Dynamic arm | Reference | Delta pp | Improved | Worsened | Query changes |
|---|---|---:|---:|---:|---:|
| noise_dynamic | frozen_ABC | +0.000000 | 0 | 0 | 0 |
| noise_dynamic | noise_fixed | -0.002229 | 0 | 1 | 1 |
| noise_dynamic | noise_shuffled | -0.083196 | 0 | 1 | 1 |
| noise_dynamic | noise_random | +0.000000 | 0 | 0 | 0 |
| noise_dynamic | r58_online_ema_teacher | -0.080770 | 7 | 7 | 27 |
| noise_dynamic | r58_online_ema_student | -0.184079 | 11 | 12 | 56 |
| noise_dynamic | r59_direct_all | -0.308656 | 8 | 10 | 38 |
| common_dynamic | frozen_ABC | +0.002229 | 1 | 0 | 2 |
| common_dynamic | common_fixed | +0.000000 | 0 | 0 | 0 |
| common_dynamic | common_shuffled | +0.000000 | 0 | 0 | 0 |
| common_dynamic | common_random | +0.024610 | 2 | 0 | 3 |
| common_dynamic | r58_online_ema_teacher | -0.078542 | 8 | 7 | 28 |
| common_dynamic | r58_online_ema_student | -0.181851 | 11 | 11 | 55 |
| common_dynamic | r59_direct_all | -0.306427 | 9 | 10 | 40 |

Noise dynamic changes no final query. Common dynamic changes2queries, one tinyIoUimprovement and one unchangedIoU; fixed and shuffled pick identical queries on all336. No accuracy-threshold gain from either dynamic arm. Wrong-input noise patch has +0.083196pp from one improved observation, not evidence to adopt the wrong-input control. All16arms and1424directed negative comparisons retained; these are comparison rows, not1424unique failed observations. Zero unknown/fallback/no-update cases are removed from denominators.

## Process changes do not establish task repair

| Arm | Actual edits n | Median edit norm/source % | Median disagreement change % | Median separation change % |
|---|---:|---:|---:|---:|
| noise_dynamic | 336 | 1.4697 | -0.015964 | +0.018176 |
| noise_fixed | 336 | 0.3285 | +0.155012 | +0.259064 |
| noise_shuffled | 336 | 1.4697 | +0.001073 | +0.088744 |
| noise_random | 336 | 1.4697 | -0.000685 | +0.004126 |
| noise_activation | 0 | 1.4697 | -0.015966 | +0.018176 |
| common_dynamic | 336 | 3.7828 | +5.020997 | +5.931597 |
| common_fixed | 336 | 3.3693 | +3.942599 | +5.055489 |
| common_shuffled | 336 | 3.7828 | +4.272124 | +5.764317 |
| common_random | 336 | 3.7828 | -0.360002 | -0.479659 |
| common_activation | 0 | 3.7828 | +5.021000 | +5.931599 |

Noise dynamic changes native probabilities on335inputs but the median true log-odds disagreement decreases only0.015964%,with zero query correction. The diagnostic proxy thus finds some small score fluctuations, not an established cause of grounding failures. Common dynamic increases candidate separation about5.9316%but also disagreement about5.0210%; greater score separation does not mean better instance discrimination. Its selected channel-set overlap with fixed control has median Jaccard.77778, compared with.06667for noise. This aligns with its near-static behavior, but is not an independent causal proof. Inputwise dynamic patches have not outperformed fixed/wrong-input controls. Do not interpret harmlessness or nonzero parameters as a method success.

Weight scaling versus equivalent activation gating has max probability discrepancy 9.67e-08, from FP32 operation ordering, and every final query matches. Storage as weights is not an additional information source.

## Five exposed scene clusters

| Platform | Anonymous cluster | n | Arm | mIoU% |
|---|---:|---:|---|---:|
| drone | 1 | 106 | frozen_ABC | 10.711738 |
| drone | 1 | 106 | r58_online_ema_teacher | 10.592784 |
| drone | 1 | 106 | r59_direct_all | 11.761612 |
| drone | 1 | 106 | noise_dynamic | 10.711738 |
| drone | 1 | 106 | common_dynamic | 10.711738 |
| drone | 2 | 62 | frozen_ABC | 24.326767 |
| drone | 2 | 62 | r58_online_ema_teacher | 25.040940 |
| drone | 2 | 62 | r59_direct_all | 24.291041 |
| drone | 2 | 62 | noise_dynamic | 24.326767 |
| drone | 2 | 62 | common_dynamic | 24.338844 |
| quad | 1 | 62 | frozen_ABC | 10.506319 |
| quad | 1 | 62 | r58_online_ema_teacher | 10.433242 |
| quad | 1 | 62 | r59_direct_all | 9.632116 |
| quad | 1 | 62 | noise_dynamic | 10.506319 |
| quad | 1 | 62 | common_dynamic | 10.506319 |
| quad | 2 | 49 | frozen_ABC | 35.070901 |
| quad | 2 | 49 | r58_online_ema_teacher | 35.070901 |
| quad | 2 | 49 | r59_direct_all | 36.067581 |
| quad | 2 | 49 | noise_dynamic | 35.070901 |
| quad | 2 | 49 | common_dynamic | 35.070901 |
| quad | 3 | 57 | frozen_ABC | 43.509770 |
| quad | 3 | 57 | r58_online_ema_teacher | 43.509770 |
| quad | 3 | 57 | r59_direct_all | 43.509770 |
| quad | 3 | 57 | noise_dynamic | 43.509770 |
| quad | 3 | 57 | common_dynamic | 43.509770 |

## Independent checks and failed attempt

Reconstructed1008prefix states, both detector formulas, fixed patch, donor derangements, randomized count/norm controls and3360noninherited native readouts. Max detector error 4.86e-17,native score error0. Every actual edit restores source, all boxes/reference outputs fixed. Independent task aggregate max error 7.11e-15pp. Independently checked3360process rows and original final log-odds stability/separation. First auditor attempt failed because a local seal dictionary shadowed builtin abs; preserved failed code/plan/receipt, renamed that auditor variable, changed no predictions/tolerance/method, and passed attempt2before scoring.

One saved-candidate-IoU evaluation after sealed predictions and independent audit:5376values,zero new rawGT/geometry calls. DEV336was already exposed, so no independent-confirmation or frame-independent-CI claim. FINAL384and ALL5912 untouched. Main replay takes10.15s plus detector5.23s on CPU from frozen caches; no GPU. This is cached-head timing, not full deployment latency.

## Paper and claim boundaries

DICE uses contribution-directed sparsity for OOD detection, and ASH post-hoc activation shaping. We checked their original abstract/official implementation. Neither guarantees 3Dgrounding accuracy, neither is reproduced here. [DICE official code](https://github.com/deeplearning-wisc/dice/blob/master/models/route.py), [DICE paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4405_ECCV_2022_paper.php), [ASH official code](https://github.com/andrijazz/ash/blob/main/ash.py), [ASH paper](https://arxiv.org/abs/2209.09858).

## Adoption and next boundary

Do not adopt either detector/column-edit recipe, and do not use the wrong-input control as a method. Two of maximum3mechanisms tested,zero revisions; no evidence-based revision is compelled by these near-zero overall effects. This stops this bounded column-damping recipe, not all internal editing or TTA. Small, nuisance-ranked columns were not shown to cause the task failures; distributed effects, stronger justified interventions and correspondence quality remain unresolved. Preserve R58and R59direct internal fusion as exposed-development references. No new model/loss/EMA combination/source preparation/full-target/holdout run is scheduled. Deployment interface detect_patch_replay is supplied for reproducibility, not production adoption.

Publication preparation initially retained a prior-round exporter filename; registration failed before an export authorization was produced, and an attempted export therefore could not start. Preserved the draft/plan and repaired only metadata filenames. Scientific predictions, scores and tolerances are unchanged.
