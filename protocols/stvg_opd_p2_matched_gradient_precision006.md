# P2 numerical audit revision006: pinned CUDA broadcast-gradient reduction

This is an authorized bounded engineering repair under the human instructions
“修复接着做” and “原来的实验可以继续了”. It supplements CPU audit arithmetic,
never the fixed P2 algorithm, source weights, inputs, cohorts, configurations,
loss/actions/rewards/weights, gradients, Adam,1792 state, frame budgets, steps,
readout, inherited LN, query reset or Native WHEN. Original sources/runtime,
all prior audit revisions and accepted predictions remain immutable.

The original actual-action005 controller stopped in HC same-domain exposure_5,
order1, arrival26/query1151, frozen-rollout, at an in-fit matched CPU32/GPU32
2e-5 check. The original GPU gradient versus analytic expression passed. The
original dead-process failed fit was not serialized. One unchanged original
GPU replay with complete prefix inheritance saved its actual incomplete witness.
No new prediction, DINO call or GT was accepted during diagnosis.

Installed PyTorch commit134179474539648ba7dee1317959529fbd0e7f89 Reduce.cuh
matches the official pinned source byte-for-byte. For the contiguous(N,32,4)
broadcast-gradient reduction, four float32 accumulators take indices i,i+4,...;
then combine accumulator0+1,+2,+3.32 inputs do not split across warps. The CPU
default sum uses a different order. The independent CPU32 autodiff reconstruction
now uses this exact audit-only reduction backward; all other pointwise scalar
reciprocals follow the prior004 bridge. The unchanged2e-5 matched threshold,
independent CPU64 autodiff/formula check, original analytic32 check and gamma48
error bounds remain. The failed prior004 CPU check is retained, never relabeled
as passing. This does not independently prove a full decoder Jacobian or CUDA
transcendental kernels and does not guarantee all future inputs pass.

Predeclare qualification: two complete40-round fits of the actual failed query,
with bitwise comparison to the saved original replay through all five completed
rounds, initial/current1792 state, original sampled actions and failed derivative;
then two original qualification arrivals for both sources, three Gaussian arms,
an actual original/new full fit in every cell (24 more fits). Source/expert
process hashes, full input hashes, Gaussian/IoU/softmax/gradient/Adam/chart/reset/
writeback and new-repeat equality must pass. Zero qualification predictions or
new DINO/GT. No comparison to unavailable dead-process memory is claimed.

Before formal resumption, root must actually recompute all2410 original accepted
complete fits, saved audit dictionaries with revision-aware dispatch, full1792
inheritance/query-reset/LN-writeback/Native WHEN chains and opaque byte/receipt/
input hashes. The first missing frozen-rollout26 must equal the real qualified
complete fit before acceptance. Unknown revisions/wrong runtime/changed audit
payloads must be rejected. One finite controller continues only the original
missing P2 suffix. All deployment arms/conditions/orders/directions globally
seal before original GT scoring/finalizing and actual phase root/view/public/
archive closure. A bounded repair closure is not P2 or paper completion. Later
P3-P6 obligations, human-paused EATA and old queues remain unchanged.
