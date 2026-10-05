# DeCoTA real-input transformation P0 review

Two matched P0s are complete. All new predictions and unlabeled consistency scores were globally sealed before GT. The only new inputs are known temporal padding/cropping and spatial flip/brightness transforms; no new expert, optimization, parameter write, or old online-stream rerun. Native time and Uniform4/admitted Top1 remain the research working point until qualification; CURRENT is not promoted.

## Setting and actual controls

Official same-domain TA-STVG EMA; each dataset32 development +16 source-disjoint but historically exposed confirmation sources, one query/source, two orders, clean +five5% deployment corruptions with identical original pixels. There are576 unique original inputs and1152 logical arrivals. The spatial correction is the sealed Uniform4/single DINO/admitted Top1 energy, joint1792 parameters, Adam.03/10 steps/first own-loss minimum. Query state resets; online100 uses the old actual LN1/16 carryover chain. This P0 has no new persistent state.

Temporal view1 prepends ceil(N/8) repeated first frames on an exactly transported virtual frame grid. View2 retains half the available sampled context on each side of the Native interval. Every nonidentity view runs the encoder anew, then both official offsets and the original envelope readout; inverse intervals are clipped to original support and invalid ones fall back. The sole deployable P0 readout is the endpoint median of Native/shift/crop. Individual shift/crop arms are controls, not a pool from which a winner can be selected.

Spatial views are a real horizontal RGB flip and deterministic brightness.95. Both run the encoder anew. Only within that same new input do we reuse its frozen prefix to decode saved prearrival and selected states. Flip boxes are inverse-mapped; primary consistency is mean cross-view IoU inside the fixed Native interval, full-clip IoU is secondary. Episodic uses source Native prestate, while online100 uses its own inherited prearrival prestate.

## Temporal P0: consensus minus Native time

Fixed episodic corrected spatial boxes; corruption source-macro means and paired95% CIs in pp. Native-spatial and actual online100 corrected-box controls, clean, both orders and all five conditions are in temporal/SUMMARY.json.

| Dataset/panel | ΔtIoU pp | ΔvIoU pp | >5/>20 pp harm |
|---|---:|---:|---:|
| vidstg/search | +3.2407 [-0.0979, +7.8201] | +0.6870 [-0.3404, +1.9801] | 10/0 |
| vidstg/confirm | +0.3257 [-0.7530, +1.8453] | +0.3747 [-0.2341, +1.4105] | 2/0 |
| hc2/search | +0.0939 [-0.3540, +0.5102] | +0.1169 [-0.1225, +0.3546] | 0/0 |
| hc2/confirm | +0.0742 [-0.8343, +0.9847] | +0.1890 [-0.2817, +0.8230] | 2/0 |

Temporal qualification: **False**. Per-dataset locked gates: {'vidstg': False, 'hc2': False}. Conditional514-head stage: skipped_P0_not_qualified.

## Spatial P0: does stability predict current help/harm?

Fixed score ΔC, no training/fitted gate. Source-disjoint confirmation is evaluated directly. Pearson correlation keeps zero utility; ROC help/harm excludes exact zero utility and reports denominator. Source bootstrap resamples each source with all its orders and conditions together.

| Dataset/panel/state | corr(ΔC,ΔvIoU) | help/harm AUC | help/harm/zero cells |
|---|---:|---:|---:|
| vidstg/search/episodic | +0.5958 [+0.2108, +0.7832] | +0.7542 [+0.5324, +0.9119] | 104/82/134 |
| vidstg/search/online100 | +0.6372 [+0.2921, +0.8082] | +0.7829 [+0.6149, +0.9125] | 98/87/135 |
| vidstg/confirm/episodic | -0.1301 [-0.4751, +0.3057] | +0.4898 [+0.2222, +1.0000] | 98/12/50 |
| vidstg/confirm/online100 | -0.0835 [-0.3470, +0.2049] | +0.4754 [+0.1322, +0.9811] | 96/14/50 |
| hc2/search/episodic | +0.1516 [-0.1476, +0.4763] | +0.6322 [+0.3724, +0.8750] | 222/68/30 |
| hc2/search/online100 | +0.2039 [-0.0695, +0.5204] | +0.6929 [+0.5429, +0.8218] | 182/106/32 |
| hc2/confirm/episodic | +0.2079 [-0.0705, +0.4509] | +0.8118 [+0.6368, +0.9239] | 124/18/18 |
| hc2/confirm/online100 | +0.2452 [-0.0229, +0.5151] | +0.5480 [+0.1343, +0.9278] | 110/32/18 |

Spatial qualification: **False**. Locked dataset gates: {'vidstg': False, 'hc2': False}. Conditional acceptance stage: skipped_P0_not_qualified.

No adapted state or optimizer step is accepted/rejected using these scores in P0. An association would authorize a separate matched acceptance trial; it would not establish an online gain. Wide intervals or a failed qualification do not prove that every input-stability signal is useless. The full-clip and non-directional panels cannot override the locked primary decision.

## Task preservation and coverage

A known coordinate transport law is exact; correct localization and semantic invariance are not guaranteed. Cropping is chosen by the predicted Native interval, so it can remove an unobserved true event. Repeated padding changes motion context. The same directional text can refer to a different entity after horizontal flip. All affected cells remain in the main panel; non-directional results are separately reported. Neither GT nor a good-case list chooses the transform.

| Dataset | unique inputs | identity crops | clipped shift predictions | invalid shift fallback | directional inputs |
|---|---:|---:|---:|---:|---:|
| vidstg | 288 | 5 | 81 | 0 | 0 |
| hc2 | 288 | 0 | 44 | 0 | 42 |

Post-seal GT-only support audit (does not change crops or filter the primary results):

| Dataset/panel | GT event not fully retained | newly cut by crop | already incomplete original grid | total inputs |
|---|---:|---:|---:|---:|
| vidstg/search | 96 | 72 | 24 | 192 |
| vidstg/confirm | 54 | 30 | 24 | 96 |
| hc2/search | 53 | 53 | 0 | 192 |
| hc2/confirm | 27 | 27 | 0 | 96 |

## Compute and verification

| Dataset | full two-offset view passes | individual offset forwards | state suffix replays | wall seconds | peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| vidstg | 1147 | 2294 | 2784 | 1302.00 | 22.66 |
| hc2 | 1152 | 2304 | 3280 | 1035.15 | 5.41 |

Smoke adds the separately recorded first-two clean native/full-corrected parity forwards. Cost is measured worker wall time, not isolated GPU kernel time. New encoder H is not persisted; only predictions, input hashes, view geometry and old-state receipts remain private. No DINO weights are loaded for this P0 and new backward/expert counts are zero.

ROOT_AUDIT independently verifies crop/shift inverse maps, scalar box IoU, saved parameter hashes and every official dense row. PUBLIC_AUDIT independently recomputes source bootstrap, source-pair-kernel AUC, means, tails and the fixed gates from anonymous rows. It cannot re-run private RGB or GT. All positive/negative/zero rows are retained. A status-only smoke completion error was saved and repaired without recomputing or changing completed predictions; see ENGINEERING_RECOVERY.json.

## Research decision

If neither P0 qualifies, stop these two variants and retain Native WHEN +Uniform4 +single DINO admitted Top1 current correction +LN1/16 as the research working point. This is not a production registry change and not a claim of universal failure. No full official-query benchmark, baseline, new memory/scorer/gate, third transform, or old paused queue is automatically started.

## Representative cases

| Experiment/dataset/panel | source/condition/order | change pp | stability ΔC |
|---|---|---:|---:|
| temporal/vidstg/search/worst | 5/frame_drop_5/order1 | -5.6985 | +0.0000 |
| temporal/vidstg/search/best | 12/frame_freeze_5/order1 | +20.1555 | +0.0000 |
| temporal/vidstg/confirm/worst | 36/exposure_5/order1 | -6.1059 | +0.0000 |
| temporal/vidstg/confirm/best | 46/frame_drop_5/order1 | +7.5693 | +0.0000 |
| temporal/hc2/search/worst | 31/motion_blur_5/order1 | -4.3112 | +0.0000 |
| temporal/hc2/search/best | 14/exposure_5/order1 | +3.3608 | +0.0000 |
| temporal/hc2/confirm/worst | 44/occlusion_5/order1 | -6.0937 | +0.0000 |
| temporal/hc2/confirm/best | 38/exposure_5/order1 | +6.8429 | +0.0000 |
| spatial/vidstg/search/worst | 28/exposure_5/order1 | -28.5078 | -0.1364 |
| spatial/vidstg/search/best | 17/exposure_5/order1 | +23.8493 | +0.2846 |
| spatial/vidstg/confirm/worst | 36/occlusion_5/order1 | -1.3649 | -0.2202 |
| spatial/vidstg/confirm/best | 37/frame_freeze_5/order1 | +36.0567 | -0.0311 |
| spatial/hc2/search/worst | 27/frame_freeze_5/order1 | -36.3237 | -0.0979 |
| spatial/hc2/search/best | 19/frame_drop_5/order1 | +61.5182 | -0.1400 |
| spatial/hc2/confirm/worst | 47/exposure_5/order1 | -15.0136 | -0.0320 |
| spatial/hc2/confirm/best | 33/motion_blur_5/order1 | +22.2155 | +0.0369 |
| online_spatial/vidstg/search/worst | 28/exposure_5/order1 | -32.6848 | -0.1916 |
| online_spatial/vidstg/search/best | 17/exposure_5/order1 | +24.8151 | +0.3786 |
| online_spatial/vidstg/confirm/worst | 36/occlusion_5/order1 | -1.2902 | -0.2168 |
| online_spatial/vidstg/confirm/best | 37/exposure_5/order1 | +34.2559 | -0.0353 |
| online_spatial/hc2/search/worst | 7/occlusion_5/order1 | -39.3578 | -0.3652 |
| online_spatial/hc2/search/best | 19/frame_drop_5/order1 | +60.8056 | -0.1670 |
| online_spatial/hc2/confirm/worst | 47/occlusion_5/order1 | -14.8759 | -0.1540 |
| online_spatial/hc2/confirm/best | 33/motion_blur_5/order1 | +22.2155 | +0.0369 |

## Root interpretation of the measured outcome

Neither P0 qualifies. Temporal confirmation corruption gains are Vid +0.3257 pp tIoU / +0.3747 pp vIoU, HC +0.0742 / +0.1890 pp; every paired95% interval crosses zero. Consensus actually changes only48/160 Vid and54/160 HC logical corruption intervals. This fixed readout adds no >20 pp harms, but its small positive points do not establish a usable temporal teacher. Individual-view results cannot be promoted post hoc.

Spatial development and confirmation disagree. Vid episodic development has r=.5958 [.2108,.7832], AUC=.7542 [.5324,.9119], but confirmation has r=-.1301 [-.4751,.3057], AUC=.4898 [.2222,1.0000]. HC episodic confirmation AUC=.8118 [.6368,.9239] is a specific positive result worth preserving, but its correlation interval crosses zero and actual online100 AUC=.5480 [.1343,.9278] does not qualify. Excluding directional text does not rescue the locked joint decision. AUC valid bootstrap draws are explicitly reported; Vid confirmation has8791/10000 finite draws because some resamples contain no help or no harm. The broad interval must not be hidden.

Correctness need not improve with stability. Vid confirmation source37/frame_freeze/order1 gains36.06 pp vIoU while consistency falls.0311; HC online source47/exposure/order1 loses10.80 pp while consistency rises.0100. These descriptive cases establish counterexamples to a monotonic sign rule, not that every possible stability estimator is useless. Complete signed counterexamples and interval-change counts are in ROOT_INTERPRETATION.json.

Task preservation also has a measured limitation. Confirmation across all six conditions has30/96 Vid and27/96 HC inputs where cropping newly removes some true-event support that the original sampled grid retained. Vid also has24/96 inputs whose original sampled support was already incomplete. GT enters only this post-seal audit and never alters the transform or primary membership. Exact coordinate transport therefore cannot be presented as guaranteed semantic equivariance. These counts do not isolate the causal contribution of cropping to the final consensus result.

The binary ROC first assigns each cell a base weight of1/(all eligible cells of its source), then excludes exact-zero utility. It is the corresponding conditional source-weighted help/harm population; source bootstrap retains both orders and all conditions together. Source-disjoint confirmation here is historically exposed, not a fresh benchmark.

Both conditional stages are explicitly skipped. The current research working point remains Native temporal readout +Uniform4 +single frozen DINO/admitted Top1 energy +joint1792 Adam.03 current-query correction +fixed LN delta1/16. methods/CURRENT_METHOD.json remains its separately registered deployed version. No production promotion, acceptance gate, third view, memory, new teacher or benchmark starts in this task.
