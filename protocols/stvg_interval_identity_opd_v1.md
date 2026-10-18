# Bounded temporal decision and identity-hypothesis OPD study

Supersedes unstarted object-event-feedback training preparation. That source
preflight failed before any model/training; its code/logs/partial data remain.
No new source training, diffusion/video model, Qwen, DINO call or official queue.
CURRENT/MAIN/PAPER unchanged. All cohorts are previously exposed development.

Temporal: original P6 cache32 parents/domain. Strict index s<e maps to physical
[fid_s,fid_e+1), exact legal posterior. Native/projection exact replay;
projection physical length + Native-score location; one hash-fixed Actionness
permutation; exact expected physical tIoU MBR. Each offset and original envelope
reported. Seal64 readouts before existing GT is used for new offline metrics.
No threshold/temperature/length scan, no claim that MBR supplies new semantics.

Spatial: same original rollback/critic32 (16/domain), episodic frozen Source
initial state in ALL arms, original QR/LN1792, original domain Adam LR/rounds/
Gaussian scale/tau/M32, no LN writeback. Historical online OPD is NOT matched.
Four cached DINO positions and retained candidate pools, no new detector call.
Multi-hypothesis branches use top3 valid cached candidates with score>=.35 and
missing choice. Frozen query-conditioned preadaptation ROI features provide
association evidence, not calibrated identity probabilities or action semantics.
Beam8; retain3 nonempty paths plus null, missing unary .35; edge appearance
penalty (1-cos)/2, averaged per observed pair; fixed without GT/search. Retain
all missing and sources as no-op when no informative feedback. A coherent path
is an association hypothesis, NOT proof of actual same-person identity.

Arms: original Top1 per-frame Gaussian OPD; multiple hypotheses+independent
frame candidates; same hypotheses+whole-tube structured mixture Gaussian OPD;
same hypotheses/evidence direct differentiable reward maximization. Whole-path
compatibility accumulates before logsumexp over hypotheses; no per-frame
identity maxima or coordinate averaging between people. Original native legal
posterior membership weights observed positions, with null weight contributing
zero. Outside-event candidates remain in association evidence. Does not claim
object-conditioned action evidence or improved temporal boundaries from DINO.
All arms evaluated at Native and MBR time, with identical frozen temporal input.

Structured policy: full logit tube, mixture means current actual Student mean
plus detached whole-trajectory hypothesis offsets. Common+linear smooth+local
Gaussian residual covariance normalized to original marginal sigma; positive
local diagonal gives invertible density. Component offsets interpolate each
hypothesis's anchor residuals on fixed physical video axis, with constant edge
extensions labeled proposals, not expert labels. Exact mixture log density via
Woodbury. Samples and weights detached.32 joint tube actions/round, reward
softmax original tau; score-function control variate uses (w-1/M), one Adam
update/round, actual decoder rerun. Mean output remains actual Student, never
external candidate or averaged reference. Direct shares reward/evidence,
optimizer and maximum rounds but no stochastic policy; limitations explicit.

First finite GPU pass max3600s lease, 2h wall, 16/20GiB RAM and1GiB swap,
30GiB disk/6GiB host floor. Complete32 all-arm predictions and immutable barrier
before new GT scoring. Report v/t/s, R03/R05, native unobserved geometry, damage
cases and actual cache-inclusive measurement costs; no cached deployment speed
claim. No semantic labels without actual pixel inspection; unknown retained.
All signed/negative results, independent math/root checks, privacy and real
remote bytes/SHA/blob/mode/tree main+research freshCAS/archive before closing.
No automatic fullset, learned association, action module, or further search.
