# Same-state U/R Write Decomposition Audit

User authorization: attachment 58e6d040 (2026-10-02). This is a bounded HC
development diagnostic, not a new complete online method or a deployment promotion.

## Question and matched intervention

Does the extra Routed gradient, relative to Uniform evidence evaluated at exactly
the same state and on exactly the same probes, preferentially help the current
or text-related query while having little or negative generic future utility?

Use the original HC32 search panel, one query per video source, both original
orders, clean plus frame drop/freeze/blur/occlusion/exposure at transient 5%.
Every original expert position 0,4,...,28 is a donor: 96 donor-condition-order
cells, 14 distinct donor sources. No GT-based case selection or new sampling.

The common state is the saved **Routed pre-arrival state** for each donor. This
precommitted choice exactly preserves the native route that produced the cached
Routed5 evidence and permits a bitwise first-step R positive control. It does not
claim the result is independent of which historical state supplies the base.
Uniform5 and Routed5 masks/boxes, corrupted pixels, capture and checkpoint are
receipt-bound. Recompute native routing at this common state and check its five
positions and all nine probe predictions against the original first step.

The nine probes use the original source-state fixed parameter offsets (center,
four antithetic directions), added to the common current state. Do not rescale
them by the norm of the donor state. Official HC2-trained TA-STVG checkpoint;
1792 spatial parameters; rho=.05, D=4, lr=.006097133675874025, teacher/student
temperatures=1; full-clip L1+GIoU compatibility and average-rank reverse KL.
This geometrical compatibility distribution is not a native tube likelihood.

Generate the probe tubes once and detach them. Compute g_U and g_R with the
original rank objective at the common state. Save the ordinary FP32 SGD writes:

- U: psi - lr*g_U;
- R: psi - lr*g_R;
- Specific: psi - lr*(g_R-g_U), with elementwise FP32 gradient subtraction.

No normalization, projection, loss change, learning-rate search or GT gate.
Empty expert evidence means that branch's zero gradient; include such donors in
the primary endpoint and separately report the predeclared both-nonempty subset.
Flat nonempty rewards retain the original rank-loss semantics. This is **one
matched first-step intervention**, not the complete K8 historical arrival write.
Do not subtract the divergent U/R K8 trajectories and call them same-state
gradients. Check R loss, rewards, distributions, gradients, SGD state and self
post-prediction bitwise against the saved R first step, including empty branches.

## GT-free target lock

Reuse the already sealed, frozen HC-checkpoint RoBERTa lexical-mean vectors.
No new encoder, expert or backbone forward. For each donor and original order:

- self: donor query;
- next: first future originally nonexpert query;
- text-near/text-far: maximum/minimum text cosine among **future nonexpert**
  arrivals; ties select earliest;
- broader future: every later originally nonexpert query, equal weight within
  each donor before source/condition/order aggregation.

Unlike the older transfer report, near/far eligibility excludes original expert
positions. Text similarity is a context proxy, not proof of the same event,
object, video or scene. There are 1296 broader donor-target cells, 384 logical
role cells and 1392 unique self/future donor-target cells; aliases remain labeled.
The actual future source is different from the donor. Query ordering is the
original frozen stream, not physical adjacency in a single video.

Replay the exact cached native suffix under the common pre state and each of
the three matched writes, with no new target learning or expert observations.
Primary vIoU fixes the target's native **common pre-state interval**, including
self, to isolate box utility. Also report freely decoded vIoU/tIoU, dense-GT
sIoU, interval changes, correctness at .3/.5, gain/loss and severe negative tails.
No temporal expert rerank in these diagnostic readouts.

## Execution and evidence

Two clean donor smoke checks precede full replay; root accepts no-GT format,
R first-step parity, residual arithmetic, evidence/input binding. Their writes
are reused. Exclusive single-GPU lease. Repeated steps never rerun backbone;
captured H and specialist evidence are private immutable inputs. Model restored
after each replay and after the actor closes. No historical asset modifications.

Seal all 96 write records and all 1392 prediction records globally before any
new GT read. CPU-only scoring reads the existing HC search labels afterward.
Independent reward/rank/probability/loss/parameter arithmetic and dense metric
reconstruction are required. Preserve failures with pinned engineering revisions.

Aggregate by donor-source: average future targets per donor, conditions within
source/order, then orders within source, then sources. Paired 10000 source
bootstrap (seed20261001) for U/R/Specific gains, R-U effect, self/near minus
far/broader and clean controls; additionally target-source clustering as a
sensitivity readout. Orders are repeated observations, not independent datasets.
Report all donors, nonempty coverage, shared target aliases, both orders,
gradient norms/cosines, raw gains and computation costs. Do not infer useful
timescales from cosine alone or treat a non-significant far gain as proved zero.

Slow–Fast stays conditional: only a matched residual pattern with positive
self/near utility and paired support for more local than generic utility can
justify the next small method experiment. Retain contrary/negative results.
Temporal cross-view reliability and functional token attribution remain later
independent work; no new models, cosine projection, latest-reset or full queue.

Publish code, protocol, anonymous complete rows/controls/cases, plots, audit and
limitations to Zonglin-He/A, verify every remote byte/hash, update research archive
check/snapshot/check and completion receipt, then pause the same hourly monitor.
Exclude GT, private captions/media, raw H, weights and parameter/gradient tensors.
