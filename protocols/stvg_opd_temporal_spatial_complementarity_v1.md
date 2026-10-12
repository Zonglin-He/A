# Cached temporal × spatial complementarity and Q0 mean-readout diagnosis

Human authorization: execute the supplied next-step review on 2026-10-12.
This is an additive CPU analysis of already completed P6, O and corrected
Native-support Q0. The original Full method, deployment registry, S/C/N queues,
source checkpoints, rosters, optimizer, budget and temperature are unchanged.
There is no new GPU inference, training, expert access or model forward.

## Fixed factorial comparison

First use every original P6 parent: 32 HC-STVG-v2 and 32 VidSTG, one clean
query per historical parent, original order1. T0 is the saved frozen Native
WHEN. T1 is the original P6 Actionness Projection, with the original
target-specific center fraction (HC 1; Vid 0.5), prior weight 0.1 and epsilon
1e-6. Replay its complete two-offset logits/actionness/physical IDs in CPU
float64 and verify its selected physical interval against the saved P6 output.
No new temporal head and no label-conditioned interval selection are allowed.

The four spatial readouts are frozen-source Native, original Full OPD,
common-initial-state Direct and common-initial-state Capped Direct. The latter
three use the complete saved O fits and identical original Full initial state.
Native is the source prediction, not the inherited Before state. Consequently
the Native-versus-adapted factor includes the original state history; OPD
versus Direct/cap is conditional on the common Full state. These are cached
counterfactual readouts, not eight independently trained online streams.
Original P6 SpatialOPD must exactly equal O Full in the primary order1 panel.

Inventory all 1,792 O cells and their 1,536 unique packed inputs before scoring.
Only reuse a P6 temporal capture when target/query identity, actual source-model
state SHA, normalized pixel SHA, corruption, complete frame IDs, Native interval
and frozen boxes all match. Include both original orders of every exact match
as a secondary O panel. Never borrow time signals across source checkpoints or
corruptions. Missing temporal fields are recorded; the unavailable full O
extension remains unexecuted. This does not authorize a new GPU collection.

Seal every eight-arm readout and complete source binding before any new GT
scoring. The source datasets already have historical evaluation exposure;
this is not untouched confirmation. Missing decoder support has zero IoU,
interpolation occurs only between the original physical sampled frames,
HC clipping follows the existing official scorer, and no GT boxes fill gaps.
Use the original official endpoint convention without silently correcting it.

Report mvIoU, mtIoU, fixed-GT-event msIoU, strict R@0.3/R@0.5, actual gained
and lost annotated event-frame coverage, newly included event frames outside
Native, supported versus unsupported frames, and complete positive/negative
parent tails. T0/T1 must have identical fixed-support sIoU for each spatial
arm; each T has identical tIoU across spatial arms. Primary contrasts are
T1−T0 for OPD, its difference from Native T1−T0, and OPD−cap at T1. Also
report all four T1−T0 contrasts and OPD−Direct at both times. Parent means,
10,000 paired resamples, seed 20261012, 95% unadjusted intervals, no selection
of a winner. Secondary repeated orders are clustered by the same parent.
Replay the previous P6 Native and projection scalar metrics as a positive
control; preserve that earlier study's own uncertainty estimate separately.

## Q0 cancellation and Jensen diagnosis

Use all 64 corrected-Native Q0 episodes, all three groups, all signed pairs
and actual true parameter-mean decodes. Keep original tau, calibrated
amplitudes, direction dimension, rewards, candidates and budget fixed. No
parameter sweep or new GT intervention. Check softmax weights and the full
weighted displacement, including float32 serialization rounding.

The equal-amplitude tanh illustration is not imposed on real unequal
amplitudes. For each actual pair, d_mid=(delta+ + delta−)/2 and
d_half=(delta+ − delta−)/2 give the exact pair mean
m*(d_mid+tanh((R+−R−)/(2*tau))*d_half), m=w+ + w−.
Report signed reward margin/tau, absolute tanh retention, amplitude symmetry,
weighted-absolute-displacement cancellation, mean/top1 and mean/max-candidate
displacement ratios, native and complete-decoder Jensen RMS, and true mean
versus weighted-box output movement. Preserve zero denominators as undefined
with counts; do not discard unattained budgets or zero-dimensional groups.
An existing GT oracle candidate can be shown only as a clearly offline
descriptive comparator, never as a selectable action. Algebra or association
alone does not identify a causal explanation for vIoU or justify changing tau.

## Closing and writing

Independent complete source/readout/dense/statistical readback; actual PDF
pixel inspection; anonymous complete positive/negative scalar publication to
Zonglin-He/A with remote bytes/SHA256/Git-blob/tree checks; archive
check/snapshot/check. A bounded-cache FINAL does not certify the unavailable
full O extension or the unfinished S/C/N queues. Update the paper's abstract,
contributions and mechanism discussion: retain measured P1 system gains, do
not claim stable OPD-over-Direct, indispensable query residual, established
state inheritance or joint temporal repair by spatial OPD. No production
promotion, new teacher, new backbone, full Query OPD or temperature tuning.
