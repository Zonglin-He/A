# Oracle–Mixer Gap Audit v1: frozen 447-query source diagnosis

## Authorization and question

User attachment 3aee4368 (2026-09-28) explicitly requests this single audit.
The completed learned mixer failed qualification: seed1/2 parent delta vIoU
-0.240001/-0.111768 pp, both CIs crossing zero, sIoU declines and incomplete
native-good retention. Learned correction norms approach their fixed cap.
This observation does not establish causal over-correction, insufficient
conditioning, or universal inability. The old analytic Joint capacity evidence
was a GT-assisted development PANEL16 result, not a theoretical upper bound.

Now permanently designate the already scored 447 queries / 31 source parents
as a **diagnosis set**, not a final held-out set for future model selection.
Use exactly their existing order, media/frame/query identities, B1 and both
learned FINAL snapshots. GT is explicitly used for oracle gradients and learned
privileged evidence. No target data, new labels, model training, optimizer,
architecture, expert, OPD, epoch/LR/radius/precision/loss-weight grid.

## Common native state and one fixed analytic correction

Freeze all PTD4B, B1, union256 and both mixer parameters. Common FP32 variable
F is stock merger THW before B1. Query features/prefill stay captured at stock F;
both identity and frozen-reader Jacobians contribute to dL/dF. Same original
BF16 backbone/head/native decoding, deterministic algorithms/CUBLAS strategy.
Capture B1 free native generation, requiring physical inputs and full saved
native geometry/reference/logit readout equality with the sealed prior B1.

Temporal loss: mean CE at the two true native endpoint positions, observed-frame
classes. Spatial loss: valid annotated coordinates at B1's own native semantic
reference/interval/anchors, mean CE over the **full 152775 vocabulary**. No GT
prefix or invented support. Replay must exactly match the captured logits.
These are the current training/native objectives, not source teacher forcing.

Let gT,gS be complete FP32 gradients on the same F. Use the same frozen union U
as both mixers. r=sqrt((.087687^2+.170316^2)/2)=.13545580427763146 is a relative
norm, so the physical perturbation is

    balance = unit(gT) + unit(gS)
    delta_oracle = -r*||F|| * unit(U U^T balance).

Use existing float32 unit/add, FP64 projection then FP32 cast conventions.
This equals the previous pass-matched Joint budget when both directions exist;
not the larger original Joint budget. The identical F+delta goes into both
fresh native passes. One analytic step, no optimizer, no finite-step selection.

CPU baseline-support inventory: 429 queries have both objectives, 7 spatial-only,
8 event-only, 3 neither. Retain all447. Missing branch has explicit reason and
zero gradient for construction (not a fabricated supervised zero loss); unit(0)
is zero. A single available branch still gets the fixed r||F|| budget. If both
are missing or the projected balanced direction is exactly zero, delta is zero.
Undefined cosines are null with explicit denominators, never dropped from native
scoring. This degenerate-support rule is explicit; PANEL16 had no such cases.

## Learned comparison and saved evidence

Recompute each frozen original mixer on exactly the prior GT-evidence fields.
Require its corrected-F SHA and magnitude to reproduce the sealed original
prediction's injection. Reuse each original prediction by verified hard link;
no extra learned-native inference or re-training. Observe unmodified mixer
output logits, reconstruct delta exactly, save tanh coefficients and RMS,
fractions |a|>.99/.999 and mean(1-a^2). The norm/cap = RMS(a) identity is checked
within2e-6 (frozen basis floating-point orthogonality/linear arithmetic).

Save complete gT,gS and delta_oracle/delta_seed1/delta_seed2 in FP32; complete
native endpoint/full-vocabulary coordinate matrices once in the captured trace,
original probe cache tokens/positions/masks, action targets/validity and replay
hashes/CE/cache audits. Save baseline replay and new oracle native predictions
before grammar validation, preserve invalid geometry and any generation failure.
Input pixels, preprocessing, query/grid/time hashes, state hashes, actual pass
calls and all non-updated-parameter gradient checks are retained. No backbone
weights or private raw/labels/media are published.

Four CPU contracts cover old full-support budget, zero/missing/cancelling
directions, observer equality/saturation identity, and signed cross utilities.
An initial registered one-query allocation executes actual inference/gradients
and root raw readback. Its completed episode is retained in the full447 run;
no silent replay. Partial failures stop for explicit isolated recovery.

## Readouts and interpretation

Final comparison: B1, both old learned seeds, and new analytic oracle, total
1788 scored predictions. New GPU work includes447 B1 identity/capture replays,
447 oracle native predictions, and up to894 branch backwards (873 expected
from saved B1 action availability). All predictions seal before this audit's
metric calculation; prior GT/metrics exposure remains disclosed.

Report parent-macro t/s/v, paired-parent CI (10000 draws, seed20260927), every
parent/query negative tail and native v/t>.5 retention. Reproduce old three-arm
scores exactly. Report cos(learned,oracle), -gT dot delta, -gS dot delta, norm/cap,
per-query and parent summaries and B1-good/bad groups. Full raw NumPy norm/dot
check uses atol1e-9+rtol1e-8; analytic field reconstruction uses FP32 conventions
with relative L2<=2e-6. Neither local dot nor group association proves finite
native causality. Nonzero injection is not native success.

Preserve the prior practical qualification: positive oracle delta-v mean and
lower parent CI>0, nonnegative t/s means, no >5pp parent-v harm, full native-good
v/t retention. Report each condition, plus positive-v evidence separately.
Positive oracle benefit with poor learned alignment supports an amortization-gap
hypothesis; alignment with finite harm motivates (not proves) trust-region or
conditioning hypotheses. An oracle failure stops investment in this current
union-mixer/expert/Joint-OPD chain per user direction, without calling one fixed
oracle a universal upper bound. Mixed evidence remains mixed. No downstream
state-aware module or direction distillation starts without a separate protocol.

## Resources and execution

GPU serial, disk floor8GiB, cumulative cap=null; historical settled63593.56801247615s.
First allocation900s/onequery for interface/root check; then complete-episode
3600s allocations with >=120s safety margin, maximum12 full allocations as an
engineering review bound. Failure exits, never auto retries. All measured
loading/failed/replay/finalization/wrapper time is accumulated.
Saved actual grids imply68.528GiB for five complete FP32 fields and11.134GiB
for full native coordinate logits. Lock total new evidence<=110GiB; preserve
8GiB floor, no deletion. Estimated total80–90GiB is not actual usage.
Use the single Luna/max watcher every30min for process/log/receipt/disk only;
root handles failures. End monitor only after registered audit/report completes.
Public code/protocol/anonymous aggregates are pushed after checks; CURRENT,
old queues, failed reports and optimizer corrections stay unchanged.
