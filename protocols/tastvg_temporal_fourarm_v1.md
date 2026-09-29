# Minimal C3-T: Frozen / Rerank / Hard / OPD

2026-09-29 user scope revision supersedes the C2.5 gate and C0.6 multiseed plan.
Only question: does one-step adaptation improve over simple expert reranking?
The prior registration and partial capture are retained as history; candidate
capture is reused as a prerequisite, with no separate qualification experiment.

Original first16 exposed VidSTG parents, one query each; VidSTG-source TA-STVG
checkpoint 5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83.
Same existing seed0 full-video random burst, independent of query/GT: drop,
freeze, motion blur, occlusion, exposure at 1/5/10 percent. No severity changes,
no extra seeds. 240 transient cells and 16 clean controls; not fresh evaluation.
Unchanged six-layer/final legal-span student candidates (max8), UniversalVTG
best+PE critic, nearest-observed 2fps input, confidence times proposal IoU score,
native-first ties. Expert evidence computed once and cached, clean reused exactly.

Frozen uses sealed native output. Rerank keeps native boxes and replaces temporal
interval with expert-best student candidate. Hard uses that SAME student interval
as native Gaussian endpoint target (native sigma and temporal coefficient).
OPD uses mean softplus(-(ell_i-ell_j)) over all strict expert-ordered pairs;
ell is mean across the two offsets of log start probability plus log end
probability at the candidate endpoints. No temperature, weight or pair filtering.
Physical intervals map to first/last included observed frame per offset; empty
intervals map to nearest start and end-1. Record any mapping collisions.

Hard/OPD each start independently at H0, with one gradient and ONE fixed update:
Delta H_motion = -.004 * ||H_visual||_F * grad_motion/||grad_motion||_F,
where joint norms include both offsets; H_visual includes appearance and motion.
The .004 nominal scale is inherited from the previous Round2 step 2*.02/10,
not tuned here. Zero gradients or no ordered pairs give a numerical no-op.
No backtracking, loss-descent acceptance, radius projection, preservation,
vulnerability/reliability gate, PCGrad or low-rank operation. Fixed candidate
ordering is distilled once; updated native decoding is the final output.
Official detach Jacobian retained. Full suffix dynamically reroutes normally.
Backbone captured once per episode, suffix reused for gradient and final forward.
Extra full forwards only validate replay and selected reinsertion checks.

All predictions sealed before reading authorized prior GT. GT never affects loss,
choice, update or steps. Report s/t/v, paired OPD-Rerank, Hard-Rerank, OPD-Hard,
versus Frozen, negative tails and clean control; average 15 conditions per parent,
then bootstrap 16 parents 10000 times with seed20260929. These intervals describe
this exposed development cohort, not a full benchmark. No new statistical gate.
If reranking is at least as good, retain the simpler method for this recipe;
do not infer all gradient adaptation is impossible. Spatial and other ideas remain
backlog. Production CURRENT unchanged. No follow-on experiment in this run.

Serial GPU, reuse prior caches; 3600s cap each for candidate/critic and adaptation,
8GiB new artifacts and 8GiB free-space floor. Preserve engineering failures.
Verify exact native replay, frozen model hashes, zero non-motion edits, selected
full reinsertion, loss math and independent public scalar aggregation. Archive
and push sanitized code/config/results to Zonglin-He/A; retain media/GT/raw caches
locally. No standalone C2.5 analysis or C0.6 run.
