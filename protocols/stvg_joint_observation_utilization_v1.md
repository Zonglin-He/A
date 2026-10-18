# Source24 observation-utilization diagnosis, frozen models

User-authorized small diagnostic before any conditional-completion retraining.
Freeze repaired HC epoch30 / Vid epoch20, old source-val12 each, cached video/
query conditions and original four epsilon tensors; initial AND schedule sigma
.213→.03,16 VE Euler updates. No model/association/Student training, new expert,
target GT/inference, TopK regeneration, SAM2, Qwen, dense OPD or main promotion.

Run all fixed-Native-clock source24 first, seal its predictions, then run free
clock separately. Fixed clock: project noisy input, clean estimate and proposal
time tokens to Native at EVERY step; temporal delta zero. Original32 event-
relative knots remain (not a fixed-video identity representation). Observation
membership is the exact integer Native interval, held fixed; all current four
planned times are inside it. Sampling coordinates retain the original relative
interpolation/clamp. Free clock: original floating event-relative hard membership
gate retained; no soft-mask patch. Report when changing time disables observations.

Primary three conditions: no observation, original admitted DINO anchors at the
four original planned positions, and source GT at those four planned positions
WHEN precise source annotation exists. NEVER extrapolate GT outside annotated
event or fabricate4 labels. GT is a source-only oracle, not deployable TTA.

Necessary matched-support controls: DINO_common and GT_common use exactly the
same original admitted DINO positions with precise source GT. This availability
subset is GT-assisted diagnosis too, NOT a deployment admission rule. GT_planned
can add observations missing from DINO; do not call DINO_all vs GT_planned a pure
reference-quality effect. DINO_common vs GT_common isolates that effect on the
same support. No IoU-based selection of correct anchors. All24 query denominators
retained, zero-reference outcomes reuse no-observation paths exactly. Also report
nonempty-oracle subset descriptively without replacing the full denominator.

All guided conditions: fixed inlier probability .5 + same original uniform
outlier component, same source-trained variance frozen from old association
checkpoint metadata. DO NOT execute or retrain the learned association head.
Likelihood evaluated on clean estimate with full chain-rule gradient to current
noisy state. Preserve existing .02*sigma² step and per-component .2 cap, no
guidance-strength/variance/noise/solver search. Report active references, mixture
responsibility, gradient/step norms, clipping and actual output movement.

No-observation free-clock reuses prior sigma.213 states, with first source case
each domain replayed for exact interface check. Empty or identical observation
sets reuse exact paths within the same query/clock, with explicit reuse records.
These are inference controls, not new fits or independent sources.

Fixed support for geometry metrics: source GT positions inside Native and model
input range. Observed = original four planned physical frames; unobserved-native
= other actual decoder frame IDs; unobserved-interpolated = benchmark GT frames
between decoder outputs. Same masks across every condition and clock. Null where
no evaluable frames, report contributing query/frame counts. Also full-GT sIoU,
own-time v/t/R03/R05, fixed-Native-time vIoU, reference agreement and per-query
paired contrasts. New time intervals never select evaluation support. Primary
propagation check is unobserved native frames; interpolation is separate.

GT oracle exposes only sparse boxes (and availability) to inference; all other
source GT is used only for postseal scoring. Existing source subset comes from
official train, held out only for auxiliary priors, not established unseen to
pretrained Student. This is exposed source development, not target confirmation.

Before data: CPU double finite-difference clean-guidance/clock projection/
original-mixture-equivalence/empty-guidance contracts. Resource: shared serial
GPU lease≤300s, cumulative previous38.086872592102736s plus new remains≤14400s;
16/20GiB memory/1GiB swap,host6GiB/disk30GiB. CPU report cost separate, no online
E2E claim. Max10 logical conditions/query (5 per clock), all zero-reference
queries preserved. No automatic target or training phase after this diagnostic.

Stop interpretation: GT insufficient influence/unchanged unobserved frames does
not refute all conditional completion; isolate current likelihood/guidance step/
representation limitations. Correct sparse references improving unobserved
positions is necessary but not proof of reliable dense reward or OPD advantage.
Actual DINO failures and source GT successes require matched-support comparison.

Publish code/protocol/all signed/neutral/null results and counts, private actual
pixel engineering view; exclude media/Query/GT coordinate arrays/raw responses/
weights/tensors/credentials. Verify real remote bytes/SHA/blob/mode/tree/both
refs freshCAS and archive check-snapshot-check. Preserve all old results/CURRENT.

Reference principle only: SDA clean-estimate guidance
https://github.com/francois-rozet/sda/blob/master/sda/score.py . CSDI-style masked
conditional training https://arxiv.org/abs/2107.03502 is NOT implemented here.
