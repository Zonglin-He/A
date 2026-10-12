# OPD optimization geometry and common-state control — 2026-10-12

Authorization: the human's `先做这个` and attachment SHA recorded in DESIGN_LOCK.
The authorized immediate experiment is attachment scheme A. Schemes B/C (new
teacher, trajectory reward, reliability gate) are conditional proposals, not
part of this run. CURRENT, scientific sources, old predictions, configurations,
and previously paused queues remain unchanged.

## Question and competing explanations

Does OPD have a correction/safety advantage over Direct L1/GIoU beyond a smaller
output update? Same-direction smaller gradients, different directions, Adam
normalization, and different inherited histories are separate explanations.
The existing independent online Full/Direct histories cannot identify a loss
effect from a parameter-gradient cosine.

## Immediate finite CPU diagnostic

Read all 1792 globally sealed P2 Full cells and their 1792 Direct partners,
complete fits and input bindings from the independently closed mechanism-review
namespace. Use all saved rounds, not a selected best step. Read the already
postseal anonymous quality/propagation rows; no new model, GPU, expert or raw GT
loader. Recheck actual bytes. Report ESS, reward variance, actual weighted target
displacement, OPD output gradient, Direct output gradient evaluated at the same
saved Full boxes, their cosine and norm ratio, and actual Adam parameter/output
movement. Raw parameter gradients from different histories are NOT treated as
common-state gradients. Undefined cosines and ratios remain null with counts.

Output gradients are in the shared inverse-sigmoid box chart. Direct uses the
existing differentiable 5 L1 + 2 GIoU implementation; its output gradient is
mapped by b(1-b). Boundary coordinates are reported separately, not clamped.
Parameter Jacobian attribution requires the GPU common-state experiment.
All round records are descriptive correlated observations. Statistical units
are parent sources: observation/round -> cell -> equal condition/order -> parent.
Use 10000 paired parent bootstrap, seed 20261012, separately by source/direction,
setting/order and primary excluding the existing 32 development parents. Labels
Q>=.5 and positive first reward follow the old fixed diagnostic. Their association
with old Full/Direct outcomes is explicitly noncausal because histories differ.

## One common-state experiment, no parameter search

Use the entire pre-existing 1792-cell P2 cohort (256 historical parents, both
cross-clean orders and five same-domain physical bursts at 5%). Freeze the cell
list from old bindings before any new predictions. Each cell starts all three
arms from the exact saved original Full initial 1792-coordinate state, same input,
Native WHEN, admitted top1 expert, source weights, and freshly reset Adam.
HC: lr .01 / sigma .025 / tau .05 / 40 rounds / writeback 1/16 / M32.
Vid: lr .03 / sigma .1 / tau .25 / 10 rounds / writeback 1/8 / M32.
No new expert or DINO calls; verify cached source/input bytes and native readout.

Arms: original on-policy OPD; original Direct 5 L1 + 2 GIoU; Direct with each
Adam proposal constrained by the original OPD's same-round all-native-frame
normalized cxcywh L2 movement. The latter is a diagnostic control, not a promoted
method. Scale only the proposed parameter displacement, retaining the Direct
Adam moment update. Find a safe fraction by deterministic halving (at most 20),
then eight bracket bisections; zero radius applies zero displacement. Record
proposal/applied movement, fraction, moment states, budget utilization and all
extra forwards. No GT, reward outcome or final score selects that fraction.
If a Direct proposal is already below the cap it is retained (not amplified).
This is an upper-budget control; underfilled budgets prevent claiming exact
norm matching. Report observed/unobserved movements separately.

At common OPD states measure actual parameter gradients for both objectives,
their query/LN and joint cosines, actual Adam displacements and output changes.
The diagnostic probe must be qualified against the original complete OPD fit
without probes. Original Full can be precisely reused as the reference only
after real bitwise GPU qualification. Every condition/order/source first two
cells require real repeats before formal acceptance. Qualification predictions
are never inserted into formal results. All 3584 new Direct/control predictions
and 1792 exact original Full references must be globally sealed before new GT.

These are conditional counterfactual fits at a fixed Full history, not three
new independent online streams. The original independent streams remain intact.
Final comparison uses actual last-round outputs, parent-paired effects and
10000 bootstrap; no best-step readout, temperature sweep or case-based selection.
Preserve benefits, harms, tails, no-admission cells and all negative results.

## Resource and completion

CPU diagnostic runs now. Single GPU remains serial. Do not bypass the existing
GPU lease, mutate a running source, kill/restart a baseline, or consume its live
payload. New common-state GPU work has priority at a safe GPU handoff; record
any human-directed pause separately and preserve the old prefix and runtime.
Current-state scheduling preference is requested separately; elapsed silence
is not permission to kill a process or bypass the lease.

Completion requires actual CPU/GPU qualification, global barrier, independent
math/state/input/dense/statistical root readback, actual report plot view, all
anonymous positive/negative results and code published to Zonglin-He/A with
remote bytes/SHA/Gitblob/tree verification, archive check/snapshot/check, and
real ROOT_CLOSING/FINAL. CPU diagnostic closing alone cannot close the GPU task.

Literature inspiration (verified original pages): GKD ICLR 2024; Video-OPD and
Decomposed OPD ICML 2026. This run studies update geometry, not their token
distributions or training protocol. No independent full-decoder Jacobian proof
or future numerical/OOM guarantee is claimed.
