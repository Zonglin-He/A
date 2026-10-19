# One finite route decision: E-retaining original1792 × Direct/Fixed/Live

User-authorized route comparison after what-condition repair. No more prompts,
observations, local ROI optimization or automatic experiments. Existing exposed
NH16/NV16, all32 parents retained; formal64 excluded. Original Source and E with
finalMBR and original Uniform4 fixed. No teacher calls, training of new model,
backbone replacement, temporal adaptation, writeback or parameter search.

1792 temporary parameters: original second-pass spatial query residual256 plus
last spatial block norm1/norm3/norm4 affine1536, episodic reset and fresh Adam.
Use the exact original cached spatial decoder inputs, all six blocks and all
actual physical output frames. Last spatial norms and residual do not change
frozen temporal decoder/gates; no adapted spatial prefix is cached. Verify zero
native boxes and initial parameters exactly against frozen Source caches.

Fixed output map: E center + (native center - initial native center); E size *
exp(log native size - log initial native size). Zero update is exactly E. E can
exceed normalized1; no clamping. Original Gaussian exploration stays in native
sigmoid-logit coordinates; sampled native boxes are mapped by the same fixed E
transform before frozen reference IoU feedback. Likelihood remains original
native chart, including control-variate and ordinary autograd through decoder. This
is a declared necessary policy extension for E, not an exact historical run.

Real evidence: original E-selected raw DINO boxes, cached original reference
admission/assignment, equal weights; do not use repaired what outputs or GT to
select real support. All three modes use identical references and start state.
Direct: original (5 L1 + 2 (1-GIoU)) sum/4 at final iterate. Fixed: original frozen
first rollout samples/feedback kept throughout, current student likelihood still
updated. Live: native mean from current actual student, re-sample and evaluate
same reference function every round. Original Gaussian32 antithetic, HC40 steps
lr.01 sigma.025 tau.05; Vid10 lr.03 sigma.1 tau.25; Adam(.9,.999),eps1e-8. Do not
change steps or choose best step. Original exactly flat OPD rewards skip Adam.
Fixed/Live round0 and first update must match exactly; log candidates, feedback,
gradients, states and all actual-frame outputs. No-support query is exact E no-op.

Run and seal all real32 ×3 outputs BEFORE preparing/reading GT positive control.
Then diagnostic correct-reference control replaces only reference boxes at E-
retained positions with exact GT where available; missing GT omitted, never
interpolated. Report the differing support counts explicitly: GT-vs-real is not
pure coordinate causality when real anchors have no GT. GT must never enter real
support, admission, state or output selection. Three GT diagnostic modes use
identical matched-known support. They are not deployable results. All diagnostic
outputs seal before final benchmark evaluation. Separate barriers/costs.

Main: full tube vIoU and unobserved ACTUAL-node sIoU, parent-equal means, signed
all-query/tail results and paired10000 bootstrap seed20261018. Report real Live-E,
Live-Direct,Live-Fixed, Direct-E, and separate correct-reference analogues, anchor
fit and zero-reward missed updates. Significant anchor fit alone is insufficient.
Meaningful scale sought: >2-3pp full tube benefit, not tiny positive mean. Real
Live must exceed E/Direct/Fixed and improve unobserved frames before independent
confirmation; uncertainty remains uncertainty. If correct reference works but
real fails, stop student optimization and retain teacher-gap interpretation with
support caveat. If Direct works but Live has no independent advantage, stop this
fixed-box coordinate OPD as main route. If GT control cannot help E materially,
stop this specific interface. No follow-up scan under this task.

Shared nonblocking gpu.lock, RAM/disk guards; total newly held GPU <=900s, own
unit memory high16GiB/max20GiB/swap1GiB; pilot NH001/NV001 one finite original-budget
case each, extrapolate before continuing; charge failed attempts. Never signal
other services. Prefer cached prefix, no new Source captures. Save partials on
cap/failure. Public code/protocol/config/all signed unknown results/audits/cost,
exclude raw references/GT/coordinates/caption/parent IDs/media/weights/caches.
Keep CURRENT/MAIN/PAPER and all paused queues unchanged. Publish and verify
existing GitHub dual branches and maintain archive. Completed negatives close
this round without automatically generating another experiment.
