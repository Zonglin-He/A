# Fixed-A residual-error and GT-event observation diagnosis

Authorized 2026-10-03. This is a GT-assisted diagnosis, not a deployable method.
Predecessor: current_correction_views_v1, verified GitHub 3b3ebd297f621667ce03f1291d130b6fd71236db.
The existing VidSTG and HC-STVG-v2 32 development plus 16 confirmation sources,
one query/source, two orders, clean and five 5% corruptions are reused verbatim.
All sources are historically exposed. A's complete pre/post parameter chain,
official same-domain checkpoint, original observed pixel grid, and 25% expert
positions remain immutable. Vid uses lr .033761698432507946, teacher .34902548789596055,
persistent K1; HC lr .006097133675874025, teacher 1, persistent K8. Both rho .05,
student 1, four directions/nine tubes and 1792 parameters.

## Experiment 1: CPU readout interventions

All 1152 arrivals: A, GT time with the same full A tube, dense GT space with A
time, Joint GT. GT space uses the actual dense annotations, never interpolation
of sparsely sampled GT boxes. Joint GT must score exactly one under the literal
official evaluator; disclose annotation/sampling-support limitations separately.
No GT interval is fed to any decoder. Official dense interpolation, coordinate
conversion, HC lower-coordinate clipping and dataset endpoint conventions are
unchanged. Recheck a vectorized kernel against the official evaluator.

At the 288 existing expert arrivals only, reuse up to eight temporal intervals
and all nine fixed-A probes, including native/center and duplicates. Evaluate
temporal oracle, spatial oracle and all at most 8x9 combinations; no nonexpert
decoder replay. Compare A and existing two-view T. Report paired source-macro
H_temporal=GT_time-A, H_selection=temporal_oracle-A,
H_coverage=GT_time-temporal_oracle and analogous spatial quantities, joint
increments over both singles, raw gains, candidate uniqueness and T's recovered
and remaining selection error. With this official frame-summed numerator and
full GT-span union denominator, GT time is an upper bound for a fixed tube;
verify that property, never silently clamp a violation. Do not call the two
single-branch gaps independent additive causal contributions.

## Experiment 2: one GT-event5 request, four matched observations

Freeze the same 288 pre-arrival A states and their already generated nine tubes.
U, U2 and R masks/rewards/selected tubes and single-step outputs are reused after
state/hash/geometry verification. U's original first-step SGD state was checked
bitwise against the temporary CPU SGD arithmetic for all 288 donors. U2 and R
use the sealed temporary states from round2 and round1 respectively. The old
compact confirmation writer omitted 96 U first-step post-tubes while retaining
exact states/gradients. Reconstruct these outputs by suffix-only replay of those
unchanged saved states, with pre-state parity, zero optimization/experts/backbone;
this completes a missing cached readout, rather than changing U. No Specific
intervention and no new temporal view or loss search.

GT-event5: restrict to original observed frame IDs g<=fid<h (literal saved GT
interval). If fewer than five distinct observed frames exist, record unsupported
and make no request; never fill from outside, duplicate frames, or drop these
arrivals from Experiment 1. Otherwise use nearest integer indices of quantiles
[0,.25,.5,.75,1] over the ordered event-only observed positions. The CPU plan
explicitly reads GT; the model receives the resulting images and original query,
not labels, GT boxes, interval features, or GT-derived rewards.

Budget: at most 288 logical GT-event5 requests, 240 corrupt and 48 clean before
support exclusions. Deduplicate identical prompt/positions/corrupted pixels,
including matching old observation inputs. Use unchanged official Sa2VA-4B
greedy/BF16/eager first-SEG interface. New-call ledger written before inference.
No extra parity expert call: use a requested input that exactly matches an old
receipt, if available; otherwise validate the existing implementation/weight
receipts and the formal outputs. No backbone or complete online stream rerun.

For each eligible donor, compute original rank-RKL with the cached nine detached
targets and a live central suffix replay at its exact A pre-state; preserve the
dataset lr and temperatures. ONE ordinary temporary SGD step, even for HC K8;
discard it immediately and restore the model. Reuse cached U gradient as a live
bitwise control on the same forward graph. Save GT-event gradient, metadata,
temporary state and post tube; no A state writes. Predictions use fixed A time
and fixed GT time only in the subsequent CPU scoring. Seal all eligible donor
outputs before post-intervention GT metric/box-quality analysis.

Primary: event-only GT interval. Secondary: original A interval. Report each
observation's empty evidence/frame rates, valid/scorable event frames, event
expert-box GT IoU (conditional valid mean and all observed event frames with empty
as zero), selected-center and temporary-center vIoU, losses/displacements, gross
gains/losses, .3/.5 correctness transitions and severe harms. Key paired contrasts
are GT-event minus R and minus U2 on the same eligible donors. Also publish
coverage and unsupported full-scheduled no-op sensitivity, rather than silently
conditioning the scientific claim on samples with adequate grid support.

## Inference, statistics and decision

Each dataset and split reported separately, clean/corrupt, order effects and
expert/nonexpert for Experiment 1. 10000 paired source bootstrap, seed 20261003;
equal condition then equal order within source. Sources, not arrivals, are units.
Derived diagnostic rows may be published; captions/media/annotations/weights,
physical GT intervals/boxes, parameter/gradient/H tensors and caches are private.
No parameter or module selection using this diagnostic. Select one principal
next research variable based on remaining coverage/selection/execution gaps and
matched GT-event contrasts, with uncertainty, positive controls and negative
cases. This authorizes a diagnosis and recommendation, not that next experiment.
Development/confirmation are descriptive existing cohorts, not fresh validation.
Complete code/protocol/anonymous results and limitations must be published to
Zonglin-He/A and verified remotely; update RESEARCH_HISTORY check/snapshot/check.
Do not promote a production method, resume old queues or add a third experiment.
