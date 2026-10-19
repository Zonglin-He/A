# Candidate-coordinate versus geometric spatial-label OPD, 2026-10-19

Status: predeclared finite exposed-development comparison; not independent confirmation.
Question: with identical Fine-Center student and frozen ordered ROI evidence, does
KL(qK || p_eta K) improve on original KL(q || p_eta)? No new visual evidence,
network, trainable head, backbone change, DINO inference or decoder training.

## Locked design

NH16/NV16 and CH16/CV16: 64 unique previously exposed parent videos, one query
per parent. No FH/FV formal outcomes used for method/grid/configuration selection.
Five outputs: E_MBR, A_Fixed, A_Live, B_Fixed, B_Live. A is the original exact
scoped implementation: candidate mean center state only, width/height fixed,
2 rounds x17 (zero plus8 antithetic pairs), scale .25, ROI temperature .1,
HC step1 and Vid step.5. No calibration, extra steps, mixture loss or sweep.
Reuse exact saved noise and original A fits; recompute only missing A-Fixed HC1
on original NH16. Original blockwise gradient has no video-length normalization.
B uses that same blockwise step convention with the exact spatial-KL gradient.
Equal numerical steps are an initial comparison, not proof of equal gradient strength.

Each Fixed arm samples around eta=0 in both rounds, but recomputes current
student probabilities at updated eta. Each Live arm round2 samples around its
own updated eta and recomputes candidate feedback. Same-input deterministic
teacher results are cached exactly; no artificial repeated expert work.
Keep original E/MBR, interpolation-support-plus-neighbor scope, planned observed
nodes and interpolation. eta=0 restores E exactly; no feedback retains old behavior.
Final output uses student eta, not teacher selection or a heatmap argmax.

## Geometric label representation

Preselected canonical 64x64 grid over the ORIGINAL valid image rectangle [0,1]^2,
independent of padded feature extent. Existing candidate cxcywh coordinates are
already normalized to that rectangle. Each cell intersection is exact fractional
area. Normalize separately for each candidate over its visible image area.
Thus subcell boxes retain mass; no center-in-cell rasterization, padding cells,
learned projection or feature interpolation pretending to add resolution.
Grid is fixed throughout this round, no resolution selection using GT.

User-approved boundary amendment: append ONE explicit outside-image state.
A candidate with exactly zero intersection with the original image maps unit
mass to this state; all other candidates use only area-intersection image cells.
This preserves old teacher validity and candidate probabilities rather than
silently excluding the candidate, clipping its center or changing A. Record all
outside occurrences and separately occurrences in actually updated nodes.
Preflight found one old ROI-valid zero-intersection candidate, NV012 round1
node86 candidate7, outside update scope and uninformative. The amendment was
explicitly authorized before this experiment's predictions or GT evaluation.
The outside state is not image padding or an extra visual observation.

K and q are detached for each round. Q=qK is fixed, P=p_eta K carries gradient
only through current candidate probabilities. Stable float64 logsumexp on the
shared positive support; no epsilon mixture, clipping of positive probabilities,
pathwise coordinate derivatives or teacher gradients. Save separable x/y kernel
factors plus outside flags (exact reconstructible K), Q/P, q/p, candidates,
source eta, masked/unmasked gradients, and final states privately. Raw feature
maps are referenced, not duplicated. Public exports omit raw frames/coordinates,
GT, weights, source IDs, text and caches.

## Checks and interpretation

Check A cached equivalence, first-round A/B and Fixed/Live equality, kernel mass,
Q/P positivity/normalization, q=p zero gradients, overlapping examples giving
A/B different gradients, finite differences with fixed q/K, no-reference noop,
no padding cells, actual B update/readout, genuine Live source-state lineage,
scope/size/time invariance and degenerate-identical kernels. Same first round
is required; forcing same second-round candidates across A/B would invalidate Live.
All64 predictions and trace receipts must seal before fresh offline target GT.
Evaluate all parents including missing references/noops/zero temporal overlap:
complete vIoU, all-known unobserved actual-node sIoU, observed-node and benchmark
interpolated diagnostics, paired parent bootstrap10k seed20261018, signed per-query
changes, serious tails and costs. Panels separately and combined descriptively.
Main contrasts B_Live-A_Live, B_Live-B_Fixed, each arm-E. A/B losses have different
units/geometric contraction; lower spatial KL, entropy or greater motion is not
method success. No hard-readout gate before the complete two-round adaptation.

CPU-only cache experiment, at most900 seconds per finite measurement/scoring/audit
phase; no GPU lease/model forward or new expert. Original resource guards and
protected production/formal files. Failure/partial outputs retained. No automatic
follow-up or production registration. Publish implementation/protocol/all signed
anonymous results including negatives, verify remote refs/bytes/tree, archive.
