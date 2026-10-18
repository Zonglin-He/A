# Sealed query-only E64 update-path diagnosis

Authorized CPU-only readback of all64 fixed-cohort frozen-decoder QR256 paths
from288f2876. No model import/forward/backward/fit, no new candidate or expert,
no new source training, no change of interface, hyperparameters, temporal MBR,
predictions, production registration or selected deployment steps.

All four fit branches and every saved state (initial through prescribed final)
are included. At every step compute full-tube official vIoU/tIoU/sIoU, exact
original-frame GT-known unobserved sIoU, fixed E overlap over all decoder frames,
planned anchors and remaining frames, the branch's feedback support overlap,
Direct's5L1+2GIoU target objective, QR norm and actual output displacement.
Original FP32 E loss targets remain distinct from immutable FP64 E baseline.
Compare all step0/final official metrics with the already published results.
Keep all no-reference/no-update/flat-reward and unknown GT positions.

For saved OPD rollouts independently recompute rewards/weights and local Gaussian
surrogate BEFORE/AFTER the actual stored update with the SAME frozen actions.
Compare gradient_u dot actual_delta_u (first-order parameter prediction), the
negative mean gradient and actual finite logit-output movement, and sampled
weighted reward vs central geometric progress. These are post-hoc local
optimization diagnostics, not a new response filter or global task guarantee.
Sigmoid endpoints use the saved same-call native chart when available; if the
final chart is unavailable and boxes touch endpoints, report unknown instead
of clamping, replaying the model or fabricating a logit.

Separate final E-overlap gain/loss/flat and real vIoU gain/loss/flat with fixed
1e-10 numerical sign tolerance. Report active-only56 alongside all64 (no-case
removal). Count nonmonotone E decreases only among actual updates, and preserve
continuous magnitudes; a single floating decrease is not by itself a scientific
failure criterion. Same-rollout meaningful surrogate rise uses absolute1e-5
and max(1e-5,abs(loss_before)*1e-6) tolerance; report exact changes as well.

User-fixed cases: H008, H128, V038, H119, plus V016 as contrasting teacher/result
example. They supplement all64 summaries and do not select deployment settings.
No GT-best step, checkpoint selection or post-hoc new deployment rule. No identity
labels inferred from geometric metrics. First observed severe negative jump is
only an explanatory event; original final-step predictions stay unchanged.

Input prediction/score/implementation/registry SHA pins and seal prerequisite,
CPU/resource-only guards, independent alternative metric/overlap aggregation,
actual chart view, anonymized all-positive/negative/neutral/unknown export,
verified GitHub two-ref publication and archive closure are required. CPU budget
1h, overall2h; this is cached retrospective analysis, not new TTA performance.
