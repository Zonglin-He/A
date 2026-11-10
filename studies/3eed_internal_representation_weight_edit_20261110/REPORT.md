# R59 — internal representation-driven local matching-weight editing

**Measured decision:** the fixed local weight-edit recipe is not adopted. Direct all-view representation fusion reaches 22.611116% balanced mIoU; closed-form editing reaches 22.337610% (-0.273506 pp versus fusion). Gradient256 produces identical final selections to the closed-form edit, and an unchanged-model functional readout reproduces the written edit bit-for-bit. Internal representation evidence has a weak development signal, but writing parameters has no established extra value in this episodic experiment.

## Fixed experiment and mathematical target

Waymo-only epoch100; exposed DEV336,168inputs per platform and five scene clusters; original A256 boxes and R57 A/B/C samplings/transport fixed. Reuse original frozen head features and token projections. No new encoder, source training, external visual model, protected holdout or full-target evaluation. Frozen source A/B projections construct the target; R58 adapted predictions are inherited references only and do not enter target generation.

The existing last query-matching linear layer alone is editable:64x288weights plus64biases,18496scalars. Earlier matching layers remain frozen. Source hidden features after the second ReLU are augmented with1 for the bias. Target directions use raw-affinity-weighted A/B internal agreement; existing soft transport maps them to C without sharing query indices. Preserve C source pre-normalization norm and weight each training row by its maximum existing geometry affinity. Unsupported rows get zero training weight; no forced nearest match or GT-based selection. Raw geometric support is not an independently verified instance label.

Weighted objective: ||H_w DeltaM^T-E||_F^2 + lambda||DeltaM||_F^2. G=H_w^T H_w,B=E^T H_w; DeltaM=B(G+lambdaI)^-1. The base ridge is max(trace(G)/289,1e-12). Increase it by the registered dual bisection only when necessary to limit ||DeltaM||_F to5%of source affine-layer norm. Both gradient and closed-form arms receive the identical target, effective ridge and bias handling. FP64 solves/optimization are cast to the original FP32 last layer for native normalized matching.

Autograd SGD uses safe quadratic step1/[2(lambda_max(G)+lambda)] and prespecified1/3/16/256checkpoints. No learning-rate/ridge/radius selection follows task scores. One source reset per input, frozen teacher and no online edit state. R58 online EMA/backtracking differs in objective, parameter scope and retained history; it is a system reference, not a clean single-factor algorithm comparison. Only gradient versus ridge shares the exact representation problem.

## All registered task results

| Arm | Drone mIoU % | Quad mIoU % | Balanced mIoU % | vs frozen ABC pp | vs R58 slow pp | Acc25 % | Acc50 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | -0.472129 | -0.552900 | 40.773810 | 21.130952 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 0.000000 | -0.080770 | 42.261905 | 21.428571 |
| r56_one | 15.908954 | 28.086002 | 21.997478 | -0.304982 | -0.385753 | 41.369048 | 21.130952 |
| r58_online_ema_teacher | 15.924842 | 28.841619 | 22.383230 | 0.080770 | 0.000000 | 42.559524 | 21.428571 |
| r58_online_ema_student | 15.868511 | 29.104569 | 22.486540 | 0.184079 | 0.103309 | 42.857143 | 21.726190 |
| zero_edit | 15.736332 | 28.868588 | 22.302460 | 0.000000 | -0.080770 | 42.261905 | 21.428571 |
| direct_C | 15.771826 | 28.967502 | 22.369664 | 0.067204 | -0.013566 | 42.261905 | 21.428571 |
| direct_all | 16.385568 | 28.836664 | 22.611116 | 0.308656 | 0.227885 | 43.154762 | 21.726190 |
| grad1 | 15.766200 | 28.874403 | 22.320301 | 0.017841 | -0.062929 | 42.559524 | 21.428571 |
| grad3 | 15.523561 | 28.823825 | 22.173693 | -0.128767 | -0.209537 | 41.964286 | 21.428571 |
| grad16 | 15.530475 | 28.625176 | 22.077825 | -0.224635 | -0.305405 | 42.261905 | 21.428571 |
| grad256 | 16.050044 | 28.625176 | 22.337610 | 0.035149 | -0.045621 | 42.559524 | 21.726190 |
| ridge_edit | 16.050044 | 28.625176 | 22.337610 | 0.035149 | -0.045621 | 42.559524 | 21.726190 |
| functional_edit | 16.050044 | 28.625176 | 22.337610 | 0.035149 | -0.045621 | 42.559524 | 21.726190 |
| shuffled_edit | 15.539129 | 28.523207 | 22.031168 | -0.271292 | -0.352063 | 41.369048 | 21.428571 |

Direct C fusion changes only C raw projections with soft support weighting. Direct all-view fusion applies the frozen internal target to all three view representations before the original normalized matching and A-candidate transport. It is a preregistered same-information no-update arm, not a score-selected replacement for the registered weight-edit primary.

## Edit versus controls and all negative results

| Reference | Edit delta pp | Improved | Worsened | Query changes |
|---|---:|---:|---:|---:|
| frozen_ABC | +0.035149 | 6 | 4 | 24 |
| direct_C | -0.032054 | 5 | 7 | 28 |
| direct_all | -0.273506 | 9 | 9 | 47 |
| grad1 | +0.017308 | 5 | 5 | 23 |
| grad3 | +0.163917 | 7 | 5 | 24 |
| grad16 | +0.259784 | 5 | 1 | 13 |
| grad256 | +0.000000 | 0 | 0 | 0 |
| functional_edit | +0.000000 | 0 | 0 | 0 |
| shuffled_edit | +0.306442 | 8 | 7 | 31 |
| r58_online_ema_teacher | -0.045621 | 9 | 9 | 36 |
| r58_online_ema_student | -0.148930 | 13 | 13 | 61 |

All15arms, every no-update/unknown/fallback case and every directed negative comparison are retained. IoU changes are not instance-identity annotations. Equal-platform overall mIoU determines the bounded result; no case-specific correction or zero-negative requirement is imposed.

## Optimization validity and what it does not establish

All336inputs change the actual last linear parameters; earlier layers and original source restoration pass. The5%radius is active on330/336inputs. Median objective falls from0.246245794to0.178142321; gradient256median is0.178157342. Its median/max parameter distance to the exact ridge result is0.00367079/0.0485765. It is not parameter-identical convergence, but all336final candidate selections coincide. No best gradient checkpoint is selected after evaluation.

The functional control forms the same effective FP32 linear operator without mutating model parameters. It exactly matches physical writeback on all336inputs. This expected algebraic equivalence identifies a change of current prediction function, not extra supervision supplied by storage in parameters. It does not test potential future-input reuse or online state-compression benefits. Likewise, restriction to observed feature directions does not guarantee specificity to real unrelated observations: other inputs can share that feature subspace.

The correct-correspondence edit beats norm-matched shuffled targets, but remains below direct internal fusion. This supports some useful content in the association/representation signal under this test, not guaranteed target identity. No source-covariance preservation, source-stage editor preparation or persistent online editing was tested; this fixed recipe cannot rule out those different mechanisms.

## Independent audit, data boundaries and cost

Reconstructed1008frozen prefix feature sets and all internal targets, re-solved normal equations independently, and checked autograd updates against spectral GD solutions. Maxridge solve error6.15e-16; maxspectral gradient error9.31e-16; maxnormal residual4.92e-15. All3360noninherited native readouts match exactly, with zero probability/Top1error. Bias and normalization are included. Functional/writeback outputs are bitexact on all336inputs.

Predictions sealed before one saved-IoU evaluation. All5040taskvalues reuse the old fixed256candidateIoUs after exact cohort/source/box checks, including exact R56one and R58reference joins; no new rawGT/geometry scoring. Independent task/scene/negative reaggregation maxerror7.11e-15pp. Five dependent historically exposed scene clusters only; no independent confirmation or frame-independent confidence intervals.

Entire target construction,86016SGDsteps,edits and private snapshots took81.05s on CPU, peakRSS0.621GiB; independent prediction audit18.97s. Frozen caches make this head-only computation valid. These are aggregate research-stage timings, not per-algorithm fresh deployment latency or a measured closed-form speedup.

## Literature boundary

MEMIT motivates residual/key-based linear map updates and explicit preservation terms; its official code uses a covariance term and double-precision solve before float writeback. It receives requested factual targets and edits language-model memory, unlike this source-internal unlabeled grounding target. This is not MEMIT reproduction and no language understanding of weight semantics is attributed to3EED. [Official implementation](https://github.com/kmeng01/memit/blob/main/memit/memit_main.py), [author project](https://memit.baulab.info/).

## Decision

Registered local weight-edit primary criterion passed: False. Do not adopt this weight-edit recipe or retune it on these scores. Preserve direct internal fusion as a weak exposed-development control/candidate; it improves overall average but has platform/scene regressions. Keep R58 online candidate and all historical results intact. No source retraining, holdout/full-target evaluation, new loss or next experiment starts automatically.
