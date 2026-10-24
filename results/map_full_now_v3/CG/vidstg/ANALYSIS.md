# CG-STVG VidSTG10303 — complete final-MAP and recomputed DP

All **10,303 queries / 732 parents** are sealed, independently scored and exactly replayed. Frozen Vid gamma=.5/u=.5/link=.25/beta=1/kappa=64/context=1, original420FP32 cross-source EMA and same eight ReferDINO observations are unchanged. Final q1 uses legal joint physical-interval MAP with C-order first-argmax tie handling; spatial DP is actually recomputed at the new MAP interval. No Source, prefix, expert, RGB, GPU or optimizer work was added.

| Arm | tIoU (%) | vIoU (%) |
|---|---:|---:|
| Original Source: native MAP | 17.318935 | 7.532313 |
| Source+our MBR | 23.481297 | 9.965719 |
| SourceJointMAP: q0 MAP with Source boxes | 17.318935 | 7.532313 |
| FullMBR: original Full control | 27.166958 | 11.494698 |
| MAP interval + old Full boxes | 25.172061 | 10.792985 |
| FullMAP: MAP interval + recomputed DP | 25.172061 | 10.792929 |

| Paired contrast | tIoU delta (pp), parent-cluster 95% CI | vIoU delta (pp), parent-cluster 95% CI |
|---|---|---|
| FullMAP-Source | +7.853126 [7.170431, 8.567708] | +3.260615 [2.848851, 3.690977] |
| FullMAP-SourceJointMAP | +7.853126 [7.170431, 8.567708] | +3.260615 [2.848851, 3.690977] |
| FullMAP-SourcePlusMBR | +1.690764 [1.156960, 2.268653] | +0.827210 [0.470307, 1.193177] |
| FullMAP-FullMBR | -1.994898 [-2.277387, -1.717930] | -0.701769 [-0.821914, -0.582801] |
| FullMAP-FullMAPFixedBoxes | +0.000000 [0.000000, 0.000000] | -0.000057 [-0.008746, 0.008405] |

**MAP underperforms FullMBR in both tIoU and vIoU, with both CIs entirely negative.** It still improves over original Source/SourceJointMAP and over Source+MBR. SourceJointMAP and original Source metrics coincide here; TA uses a different native per-offset-MAP/hull readout, so this equality must not be generalized across backbones.

The new spatial solve versus fixed-box MAP changes vIoU by only **−0.000056618pp**, CI **[−0.008745564,0.008405186]**. No mean benefit is established for this final re-solve comparison. It is a real recomputation, not reuse of fixed boxes masquerading as full MAP. The FullMAP-versus-MBR mean loss is predominantly associated with the changed interval in this matched comparison. It does not establish a general claim that all spatial DP is unnecessary.

## Severe tails, rescues and duration groups

| Reference for FullMAP | severe t | severe v | union once | rescue >5pp union |
|---|---:|---:|---:|---:|
| Source | 675 | 941 | 1267 | 3860 |
| SourceJointMAP | 675 | 941 | 1267 | 3860 |
| SourcePlusMBR | 1725 | 1490 | 2435 | 2830 |
| FullMBR | 1674 | 650 | 1677 | 754 |
| FullMAPFixedBoxes | 0 | 9 | 9 | 6 |

Severe means a decline greater than 5 percentage points; temporal/spatial union is counted once per query. Relative to FullMBR, vIoU improves on2,309 queries, declines on3,614 and ties on4,380. Compared with original Source, FullMAP has1,267 severe-union cases versus FullMBR1,375; fewer tails under this reference coexist with the lower MAP mean. These baseline comparisons are not interchangeable. Re-solving DP versus fixed-box MAP has9severe-v cases and6rescue-union cases despite a near-zero mean difference.

Short events (<=5s), 6060 queries. FullMAP-FullMBR: t -0.722842 CI[-0.9747340834101192, -0.47311667573367155]; v -0.210376 CI[-0.30965679204382124, -0.11262197261917478] FullMAP-Source: t +4.345167 CI[3.6422344618149354, 5.085866474342154]; v +1.345571 CI[0.9248081820879949, 1.7851924638955974] FullMAP-SourcePlusMBR: t +1.396869 CI[0.9346029905872253, 1.883336603126649]; v +0.228741 CI[-0.1446795048080573, 0.6148739985096772]
Long events (>5s), 4243 queries. FullMAP-FullMBR: t -3.811692 CI[-4.3216313908974024, -3.305516609882227]; v -1.403594 CI[-1.635421550687196, -1.1742304408568667] FullMAP-Source: t +12.863315 CI[11.778325407842926, 13.937100959296883]; v +5.995748 CI[5.316178999251929, 6.68599270017225] FullMAP-SourcePlusMBR: t +2.110515 CI[1.0715316120482523, 3.204891146960842]; v +1.681964 CI[1.086999833765323, 2.318398534807984]

Against original Source, fixed-support unobserved sIoU changes **-2.723290 [-3.279424, -2.181002]pp**, CI entirely negative; observed sIoU **+0.670148 [-0.087162, 1.412578]pp** crosses zero. Spatial damage is not hidden by improved t/v means. Relative to FullMBR, observed/unobserved sIoU differences both have CIs crossing zero.

## Verification, exposure and cost

**61,818 official/dense checks**, maximum discrepancy **2.6645352591003757e-15**. Complete MAP seal and originalCG-Vid seal/score preceded labels. FullMBR/SourcePlusMBR metrics are exactly equal to parent Full/Source+MBR. Source originalnativeMAP is preserved. Postseal audit verified **51,515 parent/new prediction/receipt/capture SHA bindings**, recalculated every MAP/MBR array exactly and recomputed every mean. Score/report processes exited0; the independent domain exited normally. Original frozen source/plan/hash validation passed; no equality tolerance or hash waiver was introduced.

All full official sets have historical TA exposure. Vid64CG development queries selected the inherited Vid profile. HC227parents were all development sources. No sample, duration-specific decoder, new hyperparameter or threshold is selected from this full comparison. CIs are pointwise parent bootstrap intervals without familywise correction. Source calibration-transfer and historical gamma-selection provenance limits remain.

Prediction worker3080615/start239068054: sum cached per-query CPU-stage conditional wall **776.180153251s** across10,303queries. Domain prediction wall **37284.568536s** includes waiting for original GPU receipts, setup and outer work; neither quantity is uncached deployment latency. Inherited originalSource/teacher paid acquisition costs remain in parent receipts, not zero-cost deployment. Shared-host pauses/resource interventions do not establish a causal speedup. GPUcalls0; Sourcecalls0; expertcalls0 for this MAP study.

[All means and paired CIs](report.json), [all signed metrics](all_signed.tsv), [severe/timegroups/rescues](all_signed_severe_timegroups.tsv), [matched Source+MBR signed rows](all_signed_against_SourcePlusMBR.tsv), [complete paired-tail summary](paired_tail_summary.json), [score receipt](score_receipt.json), [replay audit receipt](reporting_receipt.json). Media/captions/coordinates/raw grids/masks/checkpoints remain private.

All four TA/CG MAP domains are now complete. The original CG fulls and current-MAP TA components/final-DP/budget studies are also complete. Old strict deletion remains superseded and unexecuted. No new CPU/GPU follow-up or parameter selection follows these results.
