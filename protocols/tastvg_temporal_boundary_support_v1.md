# Old8 / Expanded32 temporal coverage and boundary quality

Authorized 2026-10-03 after b7b3790. Only temporal readout is varied. Spatial
support quality C is explicitly removed. There is no spatial scoring signal,
B+D mixture, width search, added view, temporal parameter learning, or new expert.

## Fixed cohort and controls

Same locked VidSTG and HC-STVG-v2 panels, each 32 development +16 confirmation
sources, one query/source, two orders, clean + five 5% transient corruptions,
25% scheduled expert arrivals. 1,152 arrivals, 288 experts (240 corrupt/48 clean),
864 nonexperts exactly experimental A. Confirmation is source-disjoint within
this batch, historically exposed and previously diagnosed, not fresh. No tuning
or post-hoc selection uses confirmation. Old inputs/pixels/boxes/states are sealed.

A means Uniform spatial evidence, original persistent Rank-RKL (1,792 parameters,
Vid K1/HC K8 and their previously sealed learning rates/temperatures), and original
UVTG Fast temporal critic. It is not Frozen or the production CURRENT_METHOD.
Official same-domain checkpoints, original Paper48 sampling and state trajectory
are unchanged. This is a readout intervention, not new future adaptation.

## Exactly two supports

Old8 is reused byte-for-byte in its original order, native first. Expanded32
starts with all eight entries; therefore old A's selected interval is retained.
Enumerate existing merged-grid pairs i<j, physical interval [frame_i,frame_j+1).
Normalize both physical endpoints by the observed window. Append 24 unique pairs
by deterministic farthest-point allocation: maximize minimum squared Euclidean
distance in normalized (start,end) coordinates to all already retained pairs.
Update the minimum distance after each addition. Exact ties use lexicographic
(i,j) grid order. Preserve any original duplicates, do not duplicate additions.
This is a fixed capacity control; no endpoint probabilities, GT, measured outcomes,
confidence or semantic activation guide allocation. No grid/FPS/backbone change.

## Three scorers on each identical support

A: original max(proposal confidence * interval tIoU), unmodified, exact argmax
and original order ties. B: previously locked PE pooled-query cosine inside-minus-
outside contrast, outer ratio .25; exact Old8 scores/selection must reproduce.

D: same cached PE pooled-query cosine activation, no spatial information.
Window w=1.0 second is predeclared (two original phase-zero 2 Hz feature bins),
one choice without a width search. Start transition = mean [s,min(e,s+w)) minus
mean [max(window_start,s-w),s). End transition = mean [max(s,e-w),e) minus mean
[e,min(window_end,e+w)). Q_D is the minimum of these two transitions. Half-open
bins and fractional-overlap integration retain the original feature mapping.
Clip inner bands to the candidate and exterior bands to the observed window.
If an exterior band is absent, its transition is neutral zero; retain the
candidate and record the missing side. A full-window candidate thus has score0,
not positive boundary evidence. Zero-norm embeddings or globally constant curves
(range<=1e-12) fall back to the A selection on the same support. Otherwise B/D
maximize their scores with 1e-12 ties in retained order. No positivity acceptance
gate or score weighting is added. Report short intervals, missing-context sides
and chosen-boundary signs; these are diagnostics, not label-conditioned rules.

[BAM-DETR](https://arxiv.org/html/2312.00083v2) motivates distinguishing matching
from localization quality. Its quality head is trained with IoU labels; this
experiment does not implement that trained head or inherit its guarantees.
D is an unvalidated local transition proxy from frozen PE features, not an IoU
estimator. [Prior B](tastvg_temporal_quality_old8_v1.md) remains unchanged.

## Execution and reporting

Independent CPU-only prepare, generate, global seal, offline GT score, root audit,
public arithmetic audit, report and publication. Both supports and every A/B/D
score/selection for both datasets/panels seal before opening diagnostic GT.
Recompute exact old A/B parity. No backbone forward, backward, new expert/view,
source or target supervised quality-head fitting. Nonexperts remain exactly A.

Official dense scorer on fixed full A boxes. Report O32-O8 on experts; A32-A8;
B32-A32; D32-A32; D8-A8; same-support oracle regrets; all-flow gains relative
A8; top-1 tIoU; strict GT vIoU and tIoU pairwise ordering (ties half credit);
gross gain/loss, >5pp severe harm, strict .3/.5 correctness transitions, original
positive Fast readouts destroyed, clean, expert/nonexpert and order sensitivity.
Source-macro aggregation and paired 10,000 source bootstrap, seed20261003, as
previous locked panels. Oracle capacity is not attainable gain. Regret difference
equals selected vIoU difference on fixed support, not independent corroboration.

Keep positive/negative cases. No automatic promotion, dataset-specific selector
patchwork, full-query job, new module or historical queue restart. Close with
code, actual rules, anonymous per-arrival/candidate results, figures and material
limitations pushed to Zonglin-He/A and independently read back. Private media,
captions, annotations/GT coordinates, weights, raw features/states are excluded.
