# R57 — same-source EMA and sampling-ensemble self-training

**Measured outcome:** primary online EMA slow-model result is 22.138070% balanced mIoU; +0.307739 pp versus source and -0.164391 pp versus the same-information frozen ABC ensemble. No robust adoption or independent confirmation is claimed.

## Fixed scope and mechanism

Waymo-only epoch100, exposed DEV336 (168 each platform; 2 Drone and 3 Quad scene clusters), original A256 boxes fixed. No external vision model, RGB feedback, source retraining, FINAL or ALL5912 evaluation. A/B are existing real independently sampled source forwards, C is a new independently sampled source forward for every input. Sampling does not rotate or flip the scene or alter descriptions. These are point-sampling views of one observation, not separate camera views or temporally independent observations. All methods choose from the same original A boxes.

The existing 184960-parameter query matching MLP is the only adapted module. Backbone, decoder, token projection and normalization state are frozen. Head-only replay on cached frozen features is exact for this update scope; it does not test adapting feature extraction. Full source fingerprints and canonical source scores/boxes are checked. Teacher initialization is the same trained source, with no additional pretrained knowledge.

Cross-forward candidate transport uses shared raw-point support and box IoU, at least 4 shared points, and column-substochastic transport; no query-ID equivalence or forced nearest match. Unassigned mass remains source-neutral, fixed before fitting so adaptation cannot lower loss merely by discarding more support. It is a geometric approximation, not verified instance identity. Teacher A or AB average is detached; student uses independent C. Output is ABC equal averaging except explicitly named static arms. This tests a real third-view gradient rather than identical-input zero-gradient self-copying.

Adam 0.0005, clipping 1, EMA 0.99, no new conflict/rule/entropy loss, no sharpening or score-selected hyperparameters. Episodic student/teacher/optimizer reset per input; checkpoints 1,3,16. Online streams reset at platform boundary, 1 step per input with parameter and Adam history retained; fixed input-hash order is not chronological deployment. Candidate distributions and identities do not carry across inputs. Online pre/post readouts are separate. All 24 registered outputs and all negative cases are retained.

At momentum 0.99, initial teacher coefficient after 3/16/168 updates is 0.970299/0.851458/0.184805; episodic short-step teacher movement is intrinsically small, whereas the online experiment genuinely contains history.

## All registered overall results

| Arm | Drone mIoU % | Quad mIoU % | Balanced mIoU % | vs source pp | vs frozen ABC pp | Acc25 % | Acc50 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 0.000000 | -0.472129 | 40.773810 | 21.130952 |
| frozen_AB | 15.195535 | 28.593071 | 21.894303 | 0.063972 | -0.408157 | 41.666667 | 21.130952 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 0.472129 | 0.000000 | 42.261905 | 21.428571 |
| epi_fixed_student_1 | 15.433634 | 28.486923 | 21.960279 | 0.129948 | -0.342181 | 41.666667 | 21.726190 |
| epi_fixed_student_3 | 15.149891 | 28.264121 | 21.707006 | -0.123325 | -0.595454 | 40.773810 | 21.428571 |
| epi_fixed_student_16 | 16.300632 | 27.237674 | 21.769153 | -0.061178 | -0.533307 | 41.071429 | 21.130952 |
| epi_ema_student_1 | 15.433634 | 28.486923 | 21.960279 | 0.129948 | -0.342181 | 41.666667 | 21.726190 |
| epi_ema_teacher_1 | 15.902723 | 28.868588 | 22.385656 | 0.555325 | 0.083196 | 42.559524 | 21.428571 |
| epi_ema_student_3 | 15.149891 | 28.264121 | 21.707006 | -0.123325 | -0.595454 | 40.773810 | 21.428571 |
| epi_ema_teacher_3 | 15.885884 | 28.868588 | 22.377236 | 0.546905 | 0.074776 | 42.559524 | 21.428571 |
| epi_ema_student_16 | 16.133713 | 27.519012 | 21.826362 | -0.003968 | -0.476098 | 41.369048 | 21.428571 |
| epi_ema_teacher_16 | 16.035465 | 28.587256 | 22.311360 | 0.481029 | 0.008900 | 42.559524 | 21.428571 |
| online_fixed_A_student_pre | 14.567129 | 28.407106 | 21.487118 | -0.343213 | -0.815342 | 41.369048 | 20.238095 |
| online_fixed_A_student_post | 14.557620 | 28.382258 | 21.469939 | -0.360392 | -0.832521 | 40.773810 | 20.238095 |
| online_fixed_AB_student_pre | 15.179524 | 28.377811 | 21.778667 | -0.051663 | -0.523793 | 40.773810 | 21.428571 |
| online_fixed_AB_student_post | 14.982844 | 28.422574 | 21.702709 | -0.127622 | -0.599751 | 41.071429 | 21.130952 |
| online_ema_A_student_pre | 14.052484 | 28.022064 | 21.037274 | -0.793056 | -1.265186 | 39.583333 | 20.238095 |
| online_ema_A_teacher_pre | 15.277259 | 28.718446 | 21.997852 | 0.167521 | -0.304608 | 41.666667 | 21.428571 |
| online_ema_A_student_post | 14.048555 | 27.848700 | 20.948627 | -0.881704 | -1.353833 | 39.285714 | 20.238095 |
| online_ema_A_teacher_post | 15.277259 | 28.718446 | 21.997852 | 0.167521 | -0.304608 | 41.666667 | 21.428571 |
| online_ema_AB_student_pre | 15.367573 | 28.637072 | 22.002322 | 0.171992 | -0.300138 | 41.369048 | 21.428571 |
| online_ema_AB_teacher_pre | 15.434440 | 28.841619 | 22.138029 | 0.307699 | -0.164431 | 41.666667 | 21.428571 |
| online_ema_AB_student_post | 15.588718 | 28.681836 | 22.135277 | 0.304946 | -0.167183 | 41.964286 | 21.428571 |
| online_ema_AB_teacher_post | 15.434520 | 28.841619 | 22.138070 | 0.307739 | -0.164391 | 41.666667 | 21.428571 |

## EMA versus augmentation, timing and averaging

| Contrast | Teacher views | Timing | Balanced delta pp |
|---|---|---|---:|
| EMA_student_minus_fixed_student | A | pre | -0.449843 |
| EMA_student_minus_fixed_student | A | post | -0.521312 |
| EMA_teacher_minus_fixed_student | A | pre | +0.510734 |
| EMA_teacher_minus_fixed_student | A | post | +0.527913 |
| EMA_student_minus_fixed_student | AB | pre | +0.223655 |
| EMA_student_minus_fixed_student | AB | post | +0.432568 |
| EMA_teacher_minus_fixed_student | AB | pre | +0.359362 |
| EMA_teacher_minus_fixed_student | AB | post | +0.435361 |
| AB_minus_A_fixed_student | AB_minus_A | pre | +0.291550 |
| AB_minus_A_fixed_student | AB_minus_A | post | +0.232770 |
| AB_minus_A_ema_student | AB_minus_A | pre | +0.965048 |
| AB_minus_A_ema_student | AB_minus_A | post | +1.186650 |
| AB_minus_A_ema_teacher | AB_minus_A | pre | +0.140177 |
| AB_minus_A_ema_teacher | AB_minus_A | post | +0.140218 |

Student-to-student fixed/EMA comparisons hold inference averaging constant and isolate the effect of changing the target-generating state under this protocol. Slow-teacher versus fast-student comparisons also change parameter averaging, so they cannot by themselves prove better feedback quality. A separately averaged fixed-teacher student was not registered; complete separation of these mechanisms remains outside this round. No result is advertised as a new tuned best method.

## Positive and negative changes, including scene clustering

| Arm | IoU improved / worsened | Query changes | Scene mean improved / worsened |
|---|---:|---:|---:|
| source | 0 / 0 | 0 | 0 / 0 |
| frozen_AB | 10 / 7 | 29 | 1 / 3 |
| frozen_ABC | 13 / 7 | 37 | 4 / 0 |
| epi_fixed_student_1 | 13 / 12 | 49 | 1 / 3 |
| epi_fixed_student_3 | 12 / 17 | 64 | 2 / 2 |
| epi_fixed_student_16 | 17 / 21 | 79 | 3 / 2 |
| epi_ema_student_1 | 13 / 12 | 49 | 1 / 3 |
| epi_ema_teacher_1 | 13 / 6 | 36 | 4 / 0 |
| epi_ema_student_3 | 12 / 17 | 64 | 2 / 2 |
| epi_ema_teacher_3 | 13 / 6 | 36 | 4 / 0 |
| epi_ema_student_16 | 19 / 23 | 81 | 3 / 2 |
| epi_ema_teacher_16 | 14 / 9 | 40 | 3 / 1 |
| online_fixed_A_student_pre | 16 / 21 | 78 | 1 / 4 |
| online_fixed_A_student_post | 15 / 22 | 78 | 1 / 4 |
| online_fixed_AB_student_pre | 18 / 20 | 78 | 1 / 3 |
| online_fixed_AB_student_post | 18 / 21 | 78 | 1 / 3 |
| online_ema_A_student_pre | 17 / 27 | 94 | 1 / 4 |
| online_ema_A_teacher_pre | 19 / 12 | 65 | 2 / 2 |
| online_ema_A_student_post | 17 / 29 | 97 | 1 / 4 |
| online_ema_A_teacher_post | 19 / 12 | 65 | 2 / 2 |
| online_ema_AB_student_pre | 22 / 18 | 77 | 1 / 3 |
| online_ema_AB_teacher_pre | 18 / 9 | 56 | 3 / 1 |
| online_ema_AB_student_post | 24 / 19 | 80 | 2 / 2 |
| online_ema_AB_teacher_post | 18 / 9 | 59 | 3 / 1 |

All task/status rows, directed negative comparisons, and per-scene results are exported. Same scenes and multiple descriptions are dependent; no frame-independent confidence interval is reported. Previously exposed development labels are used once after sealing for this round, with no hyperparameter changes afterwards. This does not turn DEV336 into an independent test.

## Implementation and independent audit

Completed 12096 real optimizer steps; independent replay of 7728 non-source readouts on CPU, 0 Top1 disagreements, maximum probability error 6.97e-07. Verified 6048 tensor-level EMA recurrences, exact raw-point association probes for 8 matrices, platform resets and online parameter continuity. Initial C-view gradients are nonzero on 336/336; identical-A controls have gradient below 1e-6. Teacher state changes are retained in fit traces. Raw model/weight snapshots remain private.

All 8064 task values reuse the previously sealed 256-candidate IoUs after cohort, original-box, source and query-index checks; no new raw GT/geometry calls. Independent scalar/scene/negative-table audit passed, maximum aggregate error 7.11e-15 pp. Predictions were sealed and independently audited before this round's task evaluation.

Costs: C full source capture 19.60s, association 123.24s, all fitting plus private snapshot export 209.14s, independent CPU prediction audit 70.45s. Peak recorded GPU allocation: capture 0.632 GiB; head fitting 0.026 GiB. Existing 672 source forwards were reused; these timings are cached research-stage costs, not complete fresh end-to-end deployment latency.

## Literature fidelity and limits

MM-TTA uses fast/slow BN states, within-modality prediction aggregation and cross-modal pseudo-label refinement, with the slow model at inference. This checkpoint lacks a separate RGB prediction branch and the adapted head has no BN; this round therefore tests same-source EMA-inspired grounding, not MM-TTA reproduction. [Original paper](https://arxiv.org/html/2204.12667v1).

CoTTA official code verifies source initialization, detached soft targets, EMA after student update, and teacher predictions returned before the current update. This round records both timings and isolates sampling averaging/EMA. Its 32 image augmentations, confidence switch, stochastic restoration and full-network/BN adaptation are omitted; this is not a complete CoTTA baseline. [Official implementation](https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py).

## Decision

Do not replace the current reference with an EMA configuration on these development results alone. Preserve the all-arm evidence and stop this fixed recipe after its registered comparison. Results concern fixed epoch100, approximate candidate transport, matching-head-only soft self-training, a single deterministic stream order and five exposed scene clusters. They do not establish that all EMA methods, full MM-TTA/CoTTA, other update sites or properly prepared source models are ineffective. No source training or further experiment is started automatically.

## Descriptive mechanism check, no retuning

Against frozen ABC, episodic EMA teacher at 1 step changes only one selected query: one IoU improvement, zero worsening, one scene. The +0.083196 pp increment is therefore a single development observation, not a broad extra adaptation gain. At 3 steps there is one improvement and one worsening; at 16 steps there are two improvements and four worsenings. Online EMA AB teacher versus frozen ABC has 10 improvements and 10 worsenings; their magnitudes produce a negative net effect. These are localization-IoU changes, not independently verified instance identity labels.

Optimization is real but not uniformly successful. The first Adam step reduces the same detached-target KL on only 156/336 cases; its median increases from 0.162987 to 0.171102. Fixed-teacher median KL after 16 steps is about 0.128093, so longer fitting does lower the auxiliary objective overall while failing to beat frozen ABC. A fixed borrowed learning rate and approximate transport remain competing explanations; the result cannot be reduced to either "optimizer did nothing" or "EMA supervision is intrinsically impossible". No learning-rate search or new loss was run after these results.

The clearest supported statement is: sampling aggregation gives the strongest broad gain in this registered comparison; EMA stabilizes some fast-model self-training, but parameter adaptation has not demonstrated a stable advantage over the same-information frozen ensemble.
