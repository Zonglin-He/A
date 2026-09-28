# Source GT-evidence joint-correction learnability v1

User authorization 2026-09-28 supersedes the unrun finite-component oracle as
the next task. Component finite interventions are deferred ablations, not a
gate. Main candidate: decomposed evidence, joint correction, followed only
conditionally by real evidence qualification and OPD absorption.

## Question and fixed implementation

Can a learned operator convert source GT privileged evidence into improved
free native same-PTD policies on parent-disjoint source validation?
PTD4B, B1 and the audited 256-dimensional union are frozen. B1 common THW stem
and its two query pools supply fixed features. Concatenate those features with
eight evidence/position channels: observed event activity, observed start/end
indicators, fractional box occupancy, box-knownness, physical time, cell y/x.
Missing box frames are unknown, not background. These are box-derived cues,
not segmentation. One 128-channel input projection, depthwise THW3 convolution,
pointwise mixing, and zero-initialized 256-output projection form C_theta.
Only C_theta is trained. There is no new expert, OPD, scalar-mask experiment,
or old full-source-fit continuation.

Output coefficients use tanh and a fixed scale bounded by the previous
two-pass-matched relative norm sqrt((.087687^2+.170316^2)/2). This is an upper
bound, not per-example normalization to that radius; actual norms are logged.
Zero initialization gives exactly B1. Delta lies in the stored audited union,
and the identical FP32 F+Delta enters both B1 passes before its frozen readers.
The query prefill is captured at stock F and held fixed. Native PTD remains
BF16 with the original full-vocabulary spatial decoding and no head patch.

## Data and exposure

Training is all existing 618 queries / 95 Vid source parents. Do not restart
the canceled 85k mixed-source recipe. Existing198/31 validation is developed
and is not used for state selection. A new confirmation roster is selected
from the existing full-roster Vid validation metadata, after excluding every
parent/media hash in the existing816 source rows, PANEL16 and target64 manifest.
Sort eligible parent IDs by SHA256('joint-learnability-v1|'+parent), select31,
retain every available query for those parents. Save membership and counts
before training. Full-roster annotations were previously prepared; this is
an optimization-disjoint, newly locked confirmation panel relative to the
enumerated prior manifests, not a claim of globally unseen/pretraining-unseen
videos. No target pixels/labels. No outcome-based exclusions or replacement.

GT supplies both privileged evidence and training actions. Confirmation GT
also supplies oracle evidence at inference; therefore this is explicitly
privileged source qualification, not label-free TTA. Do not claim GT was first
read after predictions. Metrics are computed only after every prediction is
sealed; training has no confirmation labels, inputs, metrics or selection.

## Locked optimization and endpoints

Two independent mixer seeds20260928/20260929; each exactly one complete618-query
epoch, independently shuffled deterministically. AdamW lr1e-3, wd0, clip1,
accum4 (last2),155 windows, linear warmup8 windows then cosine through155.
No hyperparameter/epoch grid and no best-step/validation selection. These are
first learned-operator feasibility settings, not a sufficiency claim.
At each current model state, generate its free native reference/interval/
anchors, then replay both actual branches on that same support. Equal sum of
mean endpoint CE and mean full-vocabulary coordinate CE. All152775 classes
remain in coordinate denominator; target IDs map through real tokenizer IDs.
No GT decoder prefix. Missing native/action support contributes no invented
loss; log the reason, retain query and all evaluation failures. Accumulation
denominator remains the registered window size, not number of valid samples.
Empty windows produce no Adam update and are disclosed.

Probe first on lexical first training key, same B1: zero-delta native/replay
equality, both branch gradients into mixer, frozen PTD/B1/union, one disposable
Adam step, exact reset. Probe is engineering, not native utility.
Then train both seeds and evaluate exactly Base / seed1-privileged /
seed2-privileged on every locked confirmation query, free native decoding.
Primary: per-parent mean vIoU delta vs Base for each seed; t/s, per-parent CI,
all severe >5pp harms, native-good retention, grammar failures, interval and
reference changes. Two seeds are not independent videos. Descriptive bootstrap
10000,seed20260927. A positive seed mean alone is not reproducible advantage.

Conservative qualification: both seed v means positive with lower paired
parent CI>0, each t/s mean nonnegative, no new >5pp parent v harm and no loss of
native-good v/t. Report all individual gate parts. Failure only withholds this
configuration's qualification; do not infer universal inability. Even passing
does not start expert/OPD/target stages automatically.

## Engineering, persistence, and audit

CPU tests precede registration. Freeze code/data/checkpoint hashes. Probe has
900s allocation; training uses serial3600s allocations, safe complete-window
checkpoints with integer-key Adam/live binding and RNG; resume only same locked
configuration, record every actual allocation and failed/replayed overhead.
Evaluation allocations3600s, write-once completed query triples. No overwrite
of prior evidence. Disk floor8GiB, estimated new artifacts<=12GiB, no deletions.
Cumulative GPU cap=null; before this protocol43107.875212573104s. Per-allocation
limits are engineering protection. Root must audit actual state/scope/counters
and independent scalar/tensor scores. CPU tests are not GPU or task efficacy.

Keep historical independent-correction/negative results, optimizer incident,
production CURRENT and canceled old queues unchanged. Publish checked code,
protocol and anonymous aggregates only to authorized Zonglin-He/A.
