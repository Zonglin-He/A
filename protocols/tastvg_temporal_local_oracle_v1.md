# A8 to Expanded32 local-oracle audit

Authorized 2026-10-03. This is a CPU-only post-hoc diagnostic of the sealed
anonymous results of commit 4ecf366dee49c844f1af7aabfcba5517d74c86e4. It does
not generate predictions, fit a selector, change A, or start boundary refinement.
No new model, decoder, expert, video, weight, annotation, or raw-feature access.
The input already contains GT-derived dense vIoU/tIoU; this is labelled
diagnosis, not an unlabelled method or an untouched confirmation experiment.

## Fixed inputs

Four predecessor ROWS/SUMMARY pairs, each dataset's 32 search and 16 confirmation
sources, one query/source, two orders, clean + five 5% transient corruptions,
25% scheduled experts. 1,152 arrivals, 288 complete candidate records
(240 corrupt / 48 clean); 864 nonexperts have no candidate pool and are only
counted, never assigned fabricated oracles. Same official same-domain TA-STVG
checkpoints, original Paper48 sampling, fixed A spatial boxes and Uniform
Rank-RKL trajectory (1,792 parameters; sealed Vid K1 and HC K8 configurations).
All sources have historical exposure. No source selection or parameter change.
Old8 is the first eight entries of Expanded32, and old A8's selected interval
is retained. Preserve candidate duplicates and metric ties.

## Endpoint and geometry diagnostics

A8's original selected interval is the anchor, not necessarily native. O32
means maximum cached **dense vIoU** on the same A spatial trajectory. All
oracle ties within 1e-12 of that maximum are retained. Among ties report
nearest and farthest endpoint-L1 distance to A8, with candidate order breaking
distance ties. Report whether tied candidates have different geometric classes.
These are candidate-oracle displacements, not GT endpoint errors.

For signed ds=s*-sA, de=e*-eA, compute d=|ds|+|de|, centre shift c=(ds+de)/2,
and length change l=de-ds. Verify d=max(2|c|,|l|). The two terms must not be
added as independent contributions. Report endpoint displacements separately
in seconds, fraction of the observed window, and divided by A8 duration.
Cached normalized coordinates are relative to the observed clip window, not
necessarily the entire source video. Report A/oracle tIoU, disjointness,
start-only/end-only moves, expansion, contraction, and co-directional movement.
"Centre-dominant" versus "extent-dominant" compares the two exact terms;
equal terms are mixed/tied. None of these labels certifies event identity.

## Local oracle curves, fixed before this new computation

Endpoint-L1/A8-duration radii: 0, .1, .25, .5, .75, 1, 1.5, 2, 3, 4, 8, infinity.
Endpoint-L1/observed-window radii: 0, .025, .05, .1, .2, .3, .5, 1, 2, infinity.
These are descriptive radii, not searched acceptance thresholds. On each
existing support, keep only candidates within radius of A8 and maximize cached
vIoU, yielding L8(r) and L32(r). A8 is always present, so both are >= A8.
Record support counts and L32-L8; infinity reproduces O8/O32. Also report

    accessible_extra(r) = clip(L32(r)-O8, 0, O32-O8).

Its aggregate divided by aggregate O32-O8 is the share of **global added
capacity** accessible locally beyond the full Old8 oracle. It differs from
L32(r)-L8(r), which compares two locally restricted pools. Report both.
Gain-weighted nearest/farthest O32 distance CDFs quantify where the complete
added opportunity lies. Do not average unstable per-cell gain ratios. Include
gain-positive cell/source counts, zero-gain rows, and gain concentration.
Optional single-boundary restricted oracle uses existing candidates only:
fix A8 start or A8 end exactly (1e-12 numerical tolerance). Report pool counts;
limited support does not prove continuous one-boundary refinement capacity.

## Aggregation and uncertainty

Report search/confirmation separately for each dataset, corrupt experts as
primary and clean experts as control. Per source: average arrivals within
condition/order, then conditions, then orders; average sources. Paired source
bootstrap 10,000 draws, seed 20261003. Gain-share ratios bootstrap numerator
and denominator with identical source draws; zero-denominator draws are
excluded and counted, with positive-denominator intervals explicitly labelled
conditional. Do not equate cell frequency with fraction of total opportunity.
Report nearest/farthest tie sensitivity, both order means, leave-one-source-out,
and anonymous positive/negative examples. Geometry-weighted gain components
sum to O32-O8, but are descriptive attribution, not causal effects.

## Decision scope

Quantify how much opportunity is local at the fixed radii; do not promote a
refinement method from an oracle. Large displacements alone cannot distinguish
wrong event identity from extent error. Candidate-pool pairwise accuracies use
different pair populations; previous improvements do not establish calibration
or an extreme-value cause. That cause remains a hypothesis, untested here.
Keep A; no new PE scorer, Specific, spatial loss, memory, additional expert/view,
new candidates, grid enumeration, full-query run, or historical queue restart.
Publish implementation, rules, all anonymous derived rows, summaries, figures,
negative findings and limitations to Zonglin-He/A with byte readback and audit.
