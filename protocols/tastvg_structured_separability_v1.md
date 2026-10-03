# Frozen structured P/R failure-separability audit v1

This is a finite CPU diagnostic authorized after a60dcbb, not a selector, probe
fit, online gate, threshold search or method promotion. Reuse the already sealed
information-atlas readouts; never load weights, latents, media or a backbone.
No new GPU/expert/candidate/replay/backward/optimizer calls. A's spatial trajectory,
Old8/Expanded32, L32 proposal, checkpoints, expert positions and pixels stay fixed.

## Prelocked readouts and populations

Primary recipe: existing `candidate/Inside/precision/real` and
`candidate/Endpoint/recall/real`. Separate controls, with no target winner choice:
Inside/Context, Full/Full, Geometry/Geometry, and the corresponding frozen
Inside/Endpoint and Inside/Context training-label-shuffle probes. All selected
alphas and source-only fits are inherited unchanged. Predictions are unbounded
ridge outputs; do not clip, refit, standardize on target or fuse into a new score.

For the fixed A8 anchor A and existing L32 winner W export P_A, R_A, P_W, R_W,
delta_P=P_W-P_A and delta_R=R_W-R_A. A no-op uses W=A. Strict unique L32 proposal
and ties use the existing epsilon 1e-12 and exact predecessor decisions. Retain
duplicates, ties, neutral replacements, clean and no-op counts.

Target: both historical search/confirmation panels, 32+16 planned sources per
dataset, two orders, clean plus five 5% corruptions. Existing readout coverage is
288 expert arrivals (240 corrupt/48 clean), only 45 unique expert sources:
Vid 16 search+8 confirm; HC2 14+7. Do not describe these as all 96 sources or add
the 864 nonexpert cells lacking readouts. These are historically exposed panels.
Source descriptive control: existing 31 Vid/16 HC validation readouts with the
frozen source L32 winner against native candidate 0, not source A8. Validation
previously selected alphas and calibrations; it is not new held-out validation.

## GT-free extraction then diagnostic label join

Hash-bind predecessor results, atlas readout receipts and fit seals, code,
cohorts, checkpoint config and CURRENT_METHOD. Hash-only label-file byte reads
for integrity are allowed and disclosed; parsing diagnostic values occurs only
after all new target feature rows and choices are sealed. The source descriptive
join reuses readouts already frozen by the original atlas barrier, and a
label-bearing source table containing the original L scores; no source labels
enter winner selection or readout construction. This is an ordering check for
this audit, not a claim of first-ever GT blindness. GT-free target features are
source-supervised readouts, not wholly unsupervised evidence.

Use cached official tIoU/vIoU for W and A, without rescoring predictions. Primary
help/harm labels are delta_t > +1e-12 / < -1e-12 among genuine eligible replacements;
neutral and no-op rows are excluded from binary AUC and explicitly counted.
Secondary labels use delta_v with the same epsilon. Severe task harm is delta_v
< -0.05, inherited from previous reports. For target only, derive true candidate
P/R from the already cached span and frame grid, as label arithmetic, not new GT
inference or expert input. Raw spans/frame chronology are never public exports.

## Statistics, orientations and source deletion

Four individual quantities only; orientations are fixed before joining labels:
`-P_A`, `-R_A`, `+delta_P`, `+delta_R` predict helpfulness. Never flip an AUC after
seeing target labels. Report class-source-balanced AUROC: each helpful source has
total weight 1 within the helpful class, each harmful source total weight 1 within
the harmful class. Separately report source-macro within-source AUC, restricted
to sources containing both classes, and all undefined/excluded denominators.
This distinguishes between-source separation from actual within-source ordering.

Distributions use equal source weights within each class: weighted quantiles,
class means and helpful-minus-harmful means. Cluster-bootstrap the union of
informative sources, resampling entire sources with replacement, 10,000 draws,
seed 20261003; retain repeated conditions/orders together. Resamples lacking a
class are undefined, never replaced with 0.5. Include paired delete-one-source
statistics and per-order readback. `LOSO` here is source-deletion influence of
fixed scores/statistics; it neither fits a probe nor gives out-of-fold predictive
validation. The same frozen atlas has no prior LOSO-trained P/R models.

Report prelocked sign quadrants of delta_P/delta_R (>epsilon / <-epsilon / tied)
with helpful/harmful/neutral counts, per-source task deltas and true P/R tradeoffs.
No learned combination, operating threshold, accept/reject simulation or online
gain is added. Correlation/separation is not an anchor-correctness causal test:
delta_t mechanically subtracts A's tIoU. Failure to separate with these probes
does not prove all latent information absent or justify an automatic new expert.

## Verification and closure

Independent audit reconstructs extraction from old sealed vectors, choices,
cached labels and all counts, AUC pair comparisons, distribution moments,
source bootstrap, source deletion and quadrants. Meaningful CPU tests cover
ties, no-op/neutral exclusion, source weights and undefined resamples. Publish
code, protocol, all anonymous rows/statistics including negative results and
plots; no weights, raw GT, media or caches. Verify Zonglin-He/A main and every
changed remote byte, update RESEARCH_HISTORY, check/snapshot/check. Leave current
production method and all historical queues unchanged; no new stage starts.
