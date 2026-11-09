# R58 — label-free backtracking with R57 supervision and candidate transport fixed

**Primary measured result:** online backtracking EMA slow model 22.383230% balanced mIoU; +0.245161 pp versus matched CPU original Adam, +0.080770 pp versus same-information frozen ABC, and +0.385753 pp versus historical R56 one step. Registered primary task criterion passed: True. Exposed-development evidence only; no robust adoption or independent confirmation.

## Intervention and fixed conditions

Same trained Waymo-only epoch100, original A256 boxes, R57 A/B teacher and independent C student point samplings, exact transport matrices and fixed missing mass, full matching MLP184960 parameters, base Adam learning rate0.0005 and clip1. No source retraining, new full-model forwards, RGB rule, new loss, external model, FINAL or full-target evaluation. R56 one-step and R57 frozen ABC outputs are retained exactly. Historical R56 has a different supervision/information budget and is a reference rather than the matched causal optimizer control.

The initial Adam proposal is computed once using the current detached teacher target. Try displacement scales1,1/2,...,1/256. Accept the first scale whose same-target KL is lower by at least max(1e-12,1e-6*abs(KL_before)); keep the one proposed Adam moment/counter update. If none passes, restore parameters and all optimizer moments/counters, and hold the teacher. EMA0.99 updates exactly once after an accepted student update. It is not advanced on rejection. All trials share the identical target; no GT participates in acceptance, stopping or step selection.

Fixed teachers give the same target across original and backtracking runs. EMA target-generation rules are fixed, but accepted histories change subsequent teacher parameters and hence later targets. This coupling is part of the tested update policy; the experiment does not pretend all subsequent EMA target distributions remain identical. Monotonicity is guaranteed only against the current frozen target, not across changing EMA targets or grounding accuracy.

Episodic: source/student/teacher/Adam reset for each input,16proposal attempts with1/3/16checkpoints. Online:168observations per platform, one proposal per input, parameters/optimizer carry within platform, platform-boundary resets and exact R57 fixed nonchronological order. Rerun only unconditional matching-head Adam as a CPU backend control. The GPU was occupied by another STVG project when this run began and became available after208episodic inputs; the full comparison remains CPU-only for uniformity. Frozen backbone caches are valid for this matching-head-only update scope.

## All overall task results

| Arm | Drone mIoU % | Quad mIoU % | Balanced mIoU % | vs source pp | vs frozen ABC pp | vs R56 one pp | Acc25 % | Acc50 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 0.000000 | -0.472129 | -0.167147 | 40.773810 | 21.130952 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 0.472129 | 0.000000 | 0.304982 | 42.261905 | 21.428571 |
| cpu_frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 0.472129 | 0.000000 | 0.304982 | 42.261905 | 21.428571 |
| r56_one | 15.908954 | 28.086002 | 21.997478 | 0.167147 | -0.304982 | 0.000000 | 41.369048 | 21.130952 |
| original_epi_fixed_student_1 | 15.433634 | 28.486923 | 21.960279 | 0.129948 | -0.342181 | -0.037199 | 41.666667 | 21.726190 |
| original_epi_fixed_student_3 | 15.149891 | 28.264121 | 21.707006 | -0.123325 | -0.595454 | -0.290472 | 40.773810 | 21.428571 |
| original_epi_fixed_student_16 | 16.300632 | 27.237674 | 21.769153 | -0.061178 | -0.533307 | -0.228325 | 41.071429 | 21.130952 |
| original_epi_ema_student_1 | 15.433634 | 28.486923 | 21.960279 | 0.129948 | -0.342181 | -0.037199 | 41.666667 | 21.726190 |
| original_epi_ema_teacher_1 | 15.902723 | 28.868588 | 22.385656 | 0.555325 | 0.083196 | 0.388178 | 42.559524 | 21.428571 |
| original_epi_ema_student_3 | 15.149891 | 28.264121 | 21.707006 | -0.123325 | -0.595454 | -0.290472 | 40.773810 | 21.428571 |
| original_epi_ema_teacher_3 | 15.885884 | 28.868588 | 22.377236 | 0.546905 | 0.074776 | 0.379758 | 42.559524 | 21.428571 |
| original_epi_ema_student_16 | 16.133713 | 27.519012 | 21.826362 | -0.003968 | -0.476098 | -0.171115 | 41.369048 | 21.428571 |
| original_epi_ema_teacher_16 | 16.035465 | 28.587256 | 22.311360 | 0.481029 | 0.008900 | 0.313883 | 42.559524 | 21.428571 |
| original_online_fixed_student_pre | 15.179524 | 28.377811 | 21.778667 | -0.051663 | -0.523793 | -0.218810 | 40.773810 | 21.428571 |
| original_online_fixed_student_post | 14.982844 | 28.422574 | 21.702709 | -0.127622 | -0.599751 | -0.294769 | 41.071429 | 21.130952 |
| original_online_ema_student_pre | 15.334464 | 28.637072 | 21.985768 | 0.155437 | -0.316692 | -0.011710 | 41.369048 | 21.428571 |
| original_online_ema_teacher_pre | 15.434520 | 28.841619 | 22.138070 | 0.307739 | -0.164391 | 0.140592 | 41.666667 | 21.428571 |
| original_online_ema_student_post | 15.601991 | 28.681836 | 22.141913 | 0.311583 | -0.160547 | 0.144436 | 41.964286 | 21.428571 |
| original_online_ema_teacher_post | 15.434520 | 28.841619 | 22.138070 | 0.307739 | -0.164391 | 0.140592 | 41.666667 | 21.428571 |
| backtrack_epi_fixed_student_1 | 15.748041 | 28.486923 | 22.117482 | 0.287152 | -0.184978 | 0.120005 | 41.964286 | 21.726190 |
| backtrack_epi_fixed_student_3 | 15.496811 | 28.535590 | 22.016200 | 0.185869 | -0.286260 | 0.018722 | 41.666667 | 21.726190 |
| backtrack_epi_fixed_student_16 | 15.663569 | 27.592679 | 21.628124 | -0.202207 | -0.674336 | -0.369354 | 41.071429 | 21.428571 |
| backtrack_epi_ema_student_1 | 15.748041 | 28.486923 | 22.117482 | 0.287152 | -0.184978 | 0.120005 | 41.964286 | 21.726190 |
| backtrack_epi_ema_teacher_1 | 15.902723 | 28.868588 | 22.385656 | 0.555325 | 0.083196 | 0.388178 | 42.559524 | 21.428571 |
| backtrack_epi_ema_student_3 | 15.496811 | 28.535590 | 22.016200 | 0.185869 | -0.286260 | 0.018722 | 41.666667 | 21.726190 |
| backtrack_epi_ema_teacher_3 | 15.885884 | 28.868588 | 22.377236 | 0.546905 | 0.074776 | 0.379758 | 42.559524 | 21.428571 |
| backtrack_epi_ema_student_16 | 15.609860 | 27.102554 | 21.356207 | -0.474124 | -0.946254 | -0.641271 | 40.178571 | 20.833333 |
| backtrack_epi_ema_teacher_16 | 16.035465 | 28.587256 | 22.311360 | 0.481029 | 0.008900 | 0.313883 | 42.559524 | 21.428571 |
| backtrack_online_fixed_student_pre | 15.622024 | 28.940895 | 22.281459 | 0.451128 | -0.021001 | 0.283981 | 42.559524 | 21.428571 |
| backtrack_online_fixed_student_post | 15.635756 | 28.985658 | 22.310707 | 0.480376 | 0.008247 | 0.313229 | 42.857143 | 21.428571 |
| backtrack_online_ema_student_pre | 15.689613 | 28.806263 | 22.247938 | 0.417607 | -0.054522 | 0.250460 | 42.261905 | 21.428571 |
| backtrack_online_ema_teacher_pre | 15.924842 | 28.841619 | 22.383230 | 0.552900 | 0.080770 | 0.385753 | 42.559524 | 21.428571 |
| backtrack_online_ema_student_post | 15.868511 | 29.104569 | 22.486540 | 0.656209 | 0.184079 | 0.489062 | 42.857143 | 21.726190 |
| backtrack_online_ema_teacher_post | 15.924842 | 28.841619 | 22.383230 | 0.552900 | 0.080770 | 0.385753 | 42.559524 | 21.428571 |

## First-step process diagnosis

| Rule | Teacher | Attempts | Accepted | Rejected | Nonzero changes | Loss increases | Median KL before | Median KL after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | fixed | 336 | 336 | 0 | 336 | 180 | 0.16298705 | 0.17110174 |
| original | ema | 336 | 336 | 0 | 336 | 180 | 0.16298705 | 0.17110174 |
| backtrack | fixed | 336 | 334 | 2 | 334 | 0 | 0.16298705 | 0.15061753 |
| backtrack | ema | 336 | 334 | 2 | 334 | 0 | 0.16298705 | 0.15061753 |

A zero accepted-loss-increase rate is an enforced property, so it alone is not scientific progress. Acceptance coverage, nonzero displacements and final task increments determine whether the procedure does more than reject everything. All first-step scales and rejected observations are retained.

## Full optimization costs and acceptance

| Protocol | Rule | Teacher | Proposals | Accepted | Rejected | Nonzero accepted | Loss increases | Trial evaluations |
|---|---|---|---:|---:|---:|---:|---:|---:|
| episodic | original | fixed | 5376 | 5376 | 0 | 5376 | 1951 | 5376 |
| episodic | original | ema | 5376 | 5376 | 0 | 5376 | 1939 | 5376 |
| episodic | backtrack | fixed | 5376 | 3382 | 1994 | 3382 | 0 | 23106 |
| episodic | backtrack | ema | 5376 | 3417 | 1959 | 3417 | 0 | 22898 |
| online | original | fixed | 336 | 336 | 0 | 336 | 68 | 336 |
| online | original | ema | 336 | 336 | 0 | 336 | 72 | 336 |
| online | backtrack | fixed | 336 | 278 | 58 | 278 | 0 | 808 |
| online | backtrack | ema | 336 | 274 | 62 | 274 | 0 | 845 |

Backtracking uses additional same-information head objective evaluations, reported above; it is not equal wall-clock cost to one unconditional Adam proposal. No new input or teacher knowledge is supplied. Rejection restores momentum and counters, so accepted-step counts can be smaller than proposal counts. The CPU baseline bridges the backend difference; it does not retest the full source network.

## Online teacher history

| Platform | Rule | Teacher | Inputs | Accepted updates | Initial teacher recurrence coefficient |
|---|---|---|---:|---:|---:|
| drone | original | fixed | 168 | 168 | 1.000000 |
| drone | original | ema | 168 | 168 | 0.184805 |
| drone | backtrack | fixed | 168 | 141 | 1.000000 |
| drone | backtrack | ema | 168 | 145 | 0.232864 |
| quad | original | fixed | 168 | 168 | 1.000000 |
| quad | original | ema | 168 | 168 | 0.184805 |
| quad | backtrack | fixed | 168 | 137 | 1.000000 |
| quad | backtrack | ema | 168 | 129 | 0.273489 |

The last column is the explicit initial-teacher coefficient in the EMA recurrence, not the complete retained source-information fraction. Students also start from source. Different accepted-update histories are disclosed rather than counted as168EMA events automatically.

## Matched task changes and all negative observations

| Backtracking arm | Reference | Balanced delta pp | Improved | Worsened | Query changes |
|---|---|---:|---:|---:|---:|
| backtrack_epi_fixed_student_1 | original_epi_fixed_student_1 | +0.157204 | 4 | 0 | 15 |
| backtrack_epi_fixed_student_1 | frozen_ABC | -0.184978 | 5 | 7 | 23 |
| backtrack_epi_fixed_student_1 | r56_one | +0.120005 | 15 | 11 | 59 |
| backtrack_epi_fixed_student_3 | original_epi_fixed_student_3 | +0.309194 | 5 | 2 | 21 |
| backtrack_epi_fixed_student_3 | frozen_ABC | -0.286260 | 5 | 13 | 45 |
| backtrack_epi_fixed_student_3 | r56_one | +0.018722 | 15 | 17 | 75 |
| backtrack_epi_fixed_student_16 | original_epi_fixed_student_16 | -0.141029 | 4 | 11 | 28 |
| backtrack_epi_fixed_student_16 | frozen_ABC | -0.674336 | 10 | 27 | 75 |
| backtrack_epi_fixed_student_16 | r56_one | -0.369354 | 14 | 25 | 91 |
| backtrack_epi_ema_student_1 | original_epi_ema_student_1 | +0.157204 | 4 | 0 | 15 |
| backtrack_epi_ema_student_1 | frozen_ABC | -0.184978 | 5 | 7 | 23 |
| backtrack_epi_ema_student_1 | r56_one | +0.120005 | 15 | 11 | 59 |
| backtrack_epi_ema_teacher_1 | original_epi_ema_teacher_1 | +0.000000 | 0 | 0 | 0 |
| backtrack_epi_ema_teacher_1 | frozen_ABC | +0.083196 | 1 | 0 | 1 |
| backtrack_epi_ema_teacher_1 | r56_one | +0.388178 | 15 | 9 | 54 |
| backtrack_epi_ema_student_3 | original_epi_ema_student_3 | +0.309194 | 5 | 2 | 21 |
| backtrack_epi_ema_student_3 | frozen_ABC | -0.286260 | 5 | 13 | 45 |
| backtrack_epi_ema_student_3 | r56_one | +0.018722 | 15 | 17 | 75 |
| backtrack_epi_ema_teacher_3 | original_epi_ema_teacher_3 | +0.000000 | 0 | 0 | 0 |
| backtrack_epi_ema_teacher_3 | frozen_ABC | +0.074776 | 1 | 1 | 2 |
| backtrack_epi_ema_teacher_3 | r56_one | +0.379758 | 15 | 9 | 54 |
| backtrack_epi_ema_student_16 | original_epi_ema_student_16 | -0.470156 | 4 | 14 | 31 |
| backtrack_epi_ema_student_16 | frozen_ABC | -0.946254 | 12 | 31 | 85 |
| backtrack_epi_ema_student_16 | r56_one | -0.641271 | 16 | 30 | 100 |
| backtrack_epi_ema_teacher_16 | original_epi_ema_teacher_16 | +0.000000 | 0 | 0 | 2 |
| backtrack_epi_ema_teacher_16 | frozen_ABC | +0.008900 | 2 | 4 | 10 |
| backtrack_epi_ema_teacher_16 | r56_one | +0.313883 | 16 | 12 | 58 |
| backtrack_online_fixed_student_pre | original_online_fixed_student_pre | +0.502792 | 11 | 5 | 31 |
| backtrack_online_fixed_student_pre | frozen_ABC | -0.021001 | 12 | 13 | 51 |
| backtrack_online_fixed_student_pre | r56_one | +0.283981 | 19 | 16 | 77 |
| backtrack_online_fixed_student_post | original_online_fixed_student_post | +0.607998 | 14 | 5 | 33 |
| backtrack_online_fixed_student_post | frozen_ABC | +0.008247 | 13 | 12 | 50 |
| backtrack_online_fixed_student_post | r56_one | +0.313229 | 20 | 16 | 76 |
| backtrack_online_ema_student_pre | original_online_ema_student_pre | +0.262170 | 11 | 9 | 40 |
| backtrack_online_ema_student_pre | frozen_ABC | -0.054522 | 12 | 14 | 57 |
| backtrack_online_ema_student_pre | r56_one | +0.250460 | 19 | 16 | 80 |
| backtrack_online_ema_teacher_pre | original_online_ema_teacher_pre | +0.245161 | 5 | 5 | 24 |
| backtrack_online_ema_teacher_pre | frozen_ABC | +0.080770 | 7 | 7 | 27 |
| backtrack_online_ema_teacher_pre | r56_one | +0.385753 | 18 | 12 | 66 |
| backtrack_online_ema_student_post | original_online_ema_student_post | +0.344626 | 11 | 6 | 36 |
| backtrack_online_ema_student_post | frozen_ABC | +0.184079 | 12 | 11 | 56 |
| backtrack_online_ema_student_post | r56_one | +0.489062 | 21 | 14 | 80 |
| backtrack_online_ema_teacher_post | original_online_ema_teacher_post | +0.245161 | 5 | 5 | 27 |
| backtrack_online_ema_teacher_post | frozen_ABC | +0.080770 | 7 | 7 | 27 |
| backtrack_online_ema_teacher_post | r56_one | +0.385753 | 18 | 12 | 66 |

All34arms, fallback/rejection cases, task scalars and every directed negative task comparison are exported. IoU improvements are not independent instance-identity labels. Overall mean is the selection metric; individual regressions are retained and not used to add case-specific rules. Only five scene clusters (two Drone,three Quad) are present, with historical target exposure; no independent-frame confidence interval or new confirmation claim is made.

## Independent audit and source bridge

Independent functional CPU readout replay: 10416 outputs, zero Top1 disagreement, maximum score error 0. Replayed 672 first steps, 256 full episodic probe steps and every one of 1344 online proposals with exact parameter, teacher, optimizer-state and target hashes; maximum trial KL error 0. All line-search candidates are checked against the unchanged target and acceptance threshold; rejection preserves complete Adam state. The engineering check exercises rejection after nonempty moments exist.

Historical GPU R57 versus matched unconditional CPU has 7 Top1 differences across the registered bridge endpoints. The within-CPU original/backtracking comparison remains the causal optimizer comparison. All source/candidate/observation bridges pass, and R56one task values match its original saved rows exactly. Existing R55 all256IoUs yield all11424task values, with zero new rawGT/geometry calls, after prediction sealing and independent audit. Independent scalar/scene/negative reaggregation maxerror 7.11e-15pp.

Fitting plus private state export: 416.74s; independent prediction audit 76.32s; peak fitting CPU RSS1.348GiB. Cached research-stage costs are not full fresh end-to-end deployment latency. Complete private optimization traces and weights remain local; public optimization CSVs show every first step and predetermined1/3/16checkpoints plus all online attempts, while aggregate counts cover every proposal.

## Literature and interpretation

Backtracking searches smaller displacements for an objective decrease. Standard Armijo theory assumes descent directions; this finite monotone Adam test does not inherit that guarantee, because Adam momentum may supply a non-descent direction. No gradient-direction fallback was added. [University of Washington original optimization notes](https://sites.math.washington.edu/~burke/crs/516/notes/backtracking.pdf). Complete optimizer rollback is implemented using state snapshots, consistent with the [PyTorch optimizer documentation](https://docs.pytorch.org/docs/stable/optim.html).

This is a minimal optimization intervention on R57 same-source matching-head self-training, not a new MM-TTA/CoTTA reproduction or a three-modal alignment claim. It tests whether controlling auxiliary-loss overshoot helps the same fixed-candidate task. A failure to improve grounding after successful loss acceptance would narrow the explanation toward objective/candidate correspondence limits, without proving all supervision or adaptation routes impossible.

## Decision

Prespecified primary development task criterion passed: True. Robust method adoption: false; no independent confirmation. Keep all secondary endpoints without choosing the best checkpoint after scores. No source training, holdout access, extra loss or next experiment is started automatically.

## Mechanism conclusion at the whole-cohort level

The first-step intervention is not a reject-all result:334/336inputs make nonzero accepted updates,2reject entirely; loss-increasing cases fall from180to0 and median same-target KL changes from0.162987 to0.150618 (unconditional first step instead reaches0.171102). First-step student task mIoU improves by0.157204pp versus unconditional Adam, but remains0.184978pp below frozen ABC.

Online backtracking fixed-teacher student improves by0.607998pp versus its CPU original-Adam control, to22.310707%, approximately tied with frozen ABC (+0.008247pp). Online backtracking EMA student reaches22.486540%, +0.344626pp versus its CPU original control and +0.184079pp versus frozen ABC, with12improved and11worsened observations against ABC. This secondary student endpoint was retained in advance and is not substituted post hoc for the prespecified slow-model primary endpoint.

The primary online EMA slow-model endpoint reaches22.383230%, +0.080770pp over frozen ABC, with7improvements and7worsenings against ABC. It passes the bounded primary development comparison, but the margin is small. Report both teacher and student rather than choosing the better endpoint after scoring.

Preventing auxiliary loss overshoot does not solve the supervision problem. Episodic EMA student16steps drops from21.826362% with unconditional Adam to21.356207% with backtracking; against frozen ABC the latter loses0.946254pp. Episodic slow-model1/3/16task means remain unchanged from the corresponding original-Adam endpoints despite different parameter paths. Lower current-target KL is not a grounding-correctness certificate.

Retain the registered online policy as a weak development candidate and preserve the direct-use benchmark. Do not chase individual negative cases or extend the step budget to fit this objective harder. Original R57 remains intact; this round does not justify source retraining or protected-holdout use automatically.
