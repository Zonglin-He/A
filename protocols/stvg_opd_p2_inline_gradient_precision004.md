# Fixed P2: separately pinned in-loop Gaussian precision audit, revision004

This recovery supplements the original mean-gradient assertion in the existing
joint Gaussian fitter. It changes no likelihood, detached rewards, sampling,
autograd derivative, Adam operation, parameter, input, output box, Native WHEN,
source reset, inherited LN, query reset, final-round selection or GT barrier.
Direct L1/GIoU is unchanged. Original code and runtime bytes remain immutable.

The actual HC2 same-domain `frame_drop_5` first-order arrival7 frozen-rollout
fit failed at its second round. The process did not serialize its incomplete
failed fit before exit. One actual complete-prefix replay saved the original
incomplete-fit witness; equality statements refer to this real saved replay,
never to unavailable dead-process memory. The original inline absolute failure
and original mixed-precision failures remain failures.

The new process-local fitter is the AST of the pinned original function with
only the assertion test at line100 replaced by the independent audit call.
The original filename and function name are retained for the already qualified
same-call native-logit chart. The original loss, gradient and original analytic
expression still execute. The measured original error is recorded unchanged.
An AST restoration contract verifies that all other nodes are identical.

For each admitted round, independent CPU float32 autodiff reconstructs the
original Gaussian/control-variate objective using the installed CUDA scalar
reciprocal-multiplication rounding path and retains the original absolute
`2e-5` threshold against the actual GPU derivative. Independent CPU float64
autodiff verifies the closed formula at `2e-5`. Both original float32 derivative
and original float32 analytic expression are bounded against that formula using
unit roundoff `2^-24` and gamma48 times the absolute term scale. This is an
explicit arithmetic bound plus a measured CPU32 discrepancy, not a proof of
all CUDA kernels or a full decoder Jacobian. The unmodified original inline
failure is separately identified. Neither gradient nor error is substituted.

The installed PyTorch source documents scalar CUDA division as reciprocal
followed by multiplication:
[pinned BinaryDivTrueKernel.cu](https://raw.githubusercontent.com/pytorch/pytorch/134179474539648ba7dee1317959529fbd0e7f89/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu).
The local runtime must match this exact installed PyTorch Git version.

Twenty CPU valid contracts and seventeen malformed/wrong-derivative rejection
contracts must pass. The actual serialized original replay must retain its
failed absolute assertion and pass the independent check. Qualification is
predeclared as two real complete 40-round failed-query fits, plus two ordinary
first-order qualification arrivals for each of on-policy, frozen-rollout and
shuffled-feedback under HC2 same-domain frame_drop_5 and VidSTG cross-domain
clean. Ordinary controls compare real original/new complete fits and the prior
stored original qualification. Total: 26 real fits, zero formal acceptance,
zero new DINO and zero GT. Prior full 96-arrival/192-fit P2 physical-condition
qualification remains required and is not reclassified as formal predictions.

Actual root readback independently recomputes every new complete mathematical
dictionary, Adam/state path, 1792 coordinates, chart/readout, query reset and
registered writeback. The first missing formal fit must equal the actual
qualified fit before acceptance. All 286 original formal predictions and their
immutable sidecars must remain byte-identical. New receipts carry this separate
runtime and audit revision; old receipts are dispatched by their original
saved revision. Unknown revisions must fail closed. In-loop CPU audit time is
saved separately and remains included in the original fit wall-time field.

P2 still requires all deployment arms, conditions, orders and both directions
globally sealed before CPU GT scoring. Numerical repair, qualification,
resumption and their public receipts do not close P2 or the paper. P2 actual
root analysis, cases, real visual review, anonymous complete public results and
archive closure must precede P3. EATA and all historical paused queues remain
paused. No new algorithms, tuning, experts, query skipping, frame reduction or
step reduction is authorized by this engineering repair.
