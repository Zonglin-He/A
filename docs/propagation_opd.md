# Lightweight propagation policy: HC64 development experiment

This is an isolated research candidate, not the registered method or an equivalent engineering acceleration. The frozen Source plus an output-side matching policy is a changed student. No Source, decoder, encoder, ReferDINO or RGB invocation is needed in its adaptation loop. The registered single-ReferDINO geometric OPD + Strong configuration and paused formal evaluations remain unchanged.

## Question and evidence

Can feedback on states actually visited by a student matching policy improve unobserved-frame localization beyond fixed propagation and fixed teacher-state distillation, with three updates and two paths per update?

All 64 parents are the fixed first half of the previously exposed HC128 development manifest. No new GT is opened until all arms, candidates, paths and receipts have been sealed. Source features, original/ERef tracks, the eight selected-instance ReferDINO masks/boxes, physical frame nodes and Strong interval are reused exactly from existing caches. Source features are query-dependent post-fusion visual tokens, not independently trained tracking descriptors. A target's presence is not proof that the referring event is happening; the original Strong interval remains the sole frozen event prior.

## Candidate interface

There are at most four boxes per physical node: ERef, nearest past mask transport, nearest future mask transport, and an optional separated secondary peak. Past/future masks use the original normalized Source-feature top-20 correspondence and temperature 0.1. For each propagated mask, select x and y grid-edge intervals by maximal projection Dice. An observed node uses that same selected ReferDINO query's native box. Secondary peaks require relative height at least 0.8 and separation at least two cells; their dimensions come from the nearest observation. Candidates are hypotheses, not labeled identities or negative examples.

Region descriptors use fixed fractional-occupancy pooling over the same cached Source grid. Duplicate candidates with IoU at least 0.98 are removed. Outside the frozen event interval, or below correspondence confidence 0.1, only ERef is allowed and memory writes are disabled. This hard fallback applies to every arm, so shared parameters cannot alter unsupported nodes. Confidence is a cycle/agreement heuristic, not calibrated correctness.

## Sequential policy and target

Each query resets six matching weights to (1,0,0,0,0,0) and initializes target memory from the nearest same-query observation. Features for candidate a are memory cosine, anchor cosine, projection Dice, incoming sparse-link consistency with the previous selected region, negative ERef displacement guard, and the ERef indicator. The ERef prior logit is 0; other candidates receive -2. Invalid candidates have zero probability.

Student probabilities are softmax(log q0 + theta dot features / 0.1). Candidate feedback is enumerated at every visited state:

qE(a|s) = softmax(log q0 + [projection + 0.1 anchor_cosine + 0.1 incoming_link - 0.1 guard] / 0.1).

Thus teacher feedback does not copy a potentially drifting student's appearance memory, while its incoming-link term still evaluates the student's actually selected preceding region. Reliable active unobserved steps update normalized memory with equal old/selected descriptor weights. Active observed steps reset memory to the same teacher-instance anchor. Unsupported steps retain memory.

## Fixed comparison

- OPD: two hard student-sampled paths per round; enumerate all valid candidate targets at those states; minimize confidence-weighted forward KL(qE || pi), with detached target feedback.
- FixedKD: sample two teacher paths once before adaptation, retain their states and targets, and optimize the same forward KL for three updates.
- Direct: use the same two Gumbel noise streams on a differentiable soft recurrent path, optimize forward KL with full gradients through memory, incoming links and target feedback. This is a continuous relaxation, not an exact gradient of the hard-path expected objective.
- FixedPropagation: one greedy path from enumerated qE, with no parameter updates.

All adapted arms use Adam learning rate 0.1, three updates, seed 41 and the same six weights. Loss is divided by the number of active nodes, not the sum of confidence. The direct relaxation has temperature 1. No GT-based acceptance, step selection or hyperparameter sweep is performed. Final output is the updated student's own greedy policy path; it is not a best sampled path. Cached ERef + Strong and geometric OPD are preserved as performance references, not matched low-budget optimization controls.

OPD's sampled histories are naturally discrete. Direct receives all available continuous gradients; no artificial teacher or recurrent detach is inserted into it. Differences against Direct can reflect both discrete/relaxed state occupancy and differentiation; only OPD vs FixedKD specifically tests the chosen on-policy state collection against fixed teacher states. Neither comparison is a universal proof that OPD is necessary.

## Evaluation and limits

After complete seal, independently score complete vIoU, tIoU, observed and unobserved fixed-node sIoU, and paired parent confidence intervals. Retain all positive and negative samples and severe regressions. A separate candidate-oracle diagnostic uses GT only after seal; it tests candidate availability and is never used to choose online output. Compare OPD with FixedKD, Direct, FixedPropagation, ERef and cached geometry on the same 64 parents.

Measure once-only propagation/candidate construction, matching adaptation and final path readout separately. Zero heavy-network calls during adaptation does not imply zero deployment cost: original Source/Strong/ReferDINO evidence generation costs remain inherited. All masks, descriptors, trajectories, private identifiers and state traces stay local; publish anonymous metrics, configuration, code and signed failures.

## Sources

[HTR, Temporally Consistent Referring Video Object Segmentation with Hybrid Memory](https://arxiv.org/abs/2403.19407) studies trained hybrid-memory matching and propagation for referring segmentation. Its trained modules/results are not imported here. [GKD, On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) motivates distillation on self-generated states for autoregressive language models. Applying that principle to this cached STVG matching policy is the present unverified transfer hypothesis, not a result of either paper.

## Reproduction

Run the isolated cache-bound experiment with `python -B scripts/propagation_opd/run.py --run`; run its independent scorer only after the complete seal. Paths and asset identities are checked against local private cache receipts. This does not resume formal VidSTG or CGSTVG.

The confidence gate 0.1 was chosen before execution from the already saved unlabelled correspondence summaries: most of the old 12-query prefix had unobserved mean confidence below 0.25. No new GT or candidate-oracle result selected it; confidence still weights the loss without renormalization. This gate is a pilot choice, not a calibrated identity threshold.
