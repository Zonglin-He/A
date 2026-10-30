# R8 association geometry v1 — pre-real-data contract

This module is prepared without GPU, real samples or target GT. A real smoke with
physical overlays and nonzero update checks is required before any full run lock.
It does not make claims about CLIP understanding all caption spatial relations.

Inputs are source raw6D proposals/text probabilities and original q0, observed XYZ,
externally calibrated uv/depth/image size, and frozen CLIP full-caption cosine for
each returned tight/context RGB crop. No GT boxes/classes/positive_map, learned box
teacher, color gate, query-caption simplification or yaw is accepted. All captions
may be encoded; target-phrase and relation interfaces are optional diagnostics only.

Stage1 uses positive-size proposals with≥8 unique visible support points. Visibility
is pixel-nearest-depth, valid image pixels only, canonical unique XYZ order. Support
is AABB plus0.05×size+0.05m; pool is32 source-text proposals plus16 PC-support-ranked
complement (≤48). Duplicate edge needs≥8 shared points, Jaccard≥.45,
intersection/minsupport≥.65 AND raw AABB IoU≥.10. Complete-link grouping forbids
transitive overlap-chain merging. These are geometric observation groups, not GT
object identities. Group canonical anchor is its original highest puretext-score
proposal (stable query tie). q0 remains original; q1 is always an actual source
query, never relabeled as a new aggregated box.

Before semantics select≤16 physical ROIs by8 best source-text groups then PC-support
fill. Tight pixels are visible group-support extent, min4px per side; context adds
20% per side with min4px. Driver crops actual unresized RGB with these exact boxes.
Encoder supplies[G,2] tight/context cosines for the complete original caption.
Group semantic score=.5tight+.5context; association ranks this plus.02×normalized
log1p(PCsupport). Fixed semantic shuffle rolls the entire tight/context pair by one
physical ROI; PC-only ignores all semantic scores and ranks PCsupport. All grouping,
ROI limits and tie rules are identical across controls. No candidate/ROI is selected
by GT or post-score success.

For the selected group, near-identical geometric proposals collapse to representatives
(max normalized center/size difference>.02 required for a new representative),
preventing duplicate query multiplicity from manufacturing consensus. Need≥2 reps
and observed center spread≥.01m. Member weights=sqrt(visible-support count)×relative
source text score (floor.05). Three fixed Huber-IRLS iterations start at coordinate
weighted medians, Huber delta=max(.1anchor-size,.03m). Consensus uses model proposal
centers, never mean of partial visible points. Shared-model predictions are correlated
and consensus is not an independent geometric truth observation.

Global trust `r_global = margin_conf × PCsupport_conf × multi_rep_conf`:
semantic margin confidence=clip((selected cosine−maxother cosine)/.05,0,1);
PC-only uses relative support gap. A single physical ROI gives zero margin trust.
PCsupport confidence linearly scales8→32 points; rep confidence=(reps−1)/3 clipped.
Axis dispersion=1/(1+(weightedMAD/(.15anchor-size))²). Reject raw-consensus axes with
shift>min(.15anchor-size,.30m). Global trust shrinks the target itself:

`pseudo_center = anchor_center + r_global*(raw_consensus−anchor_center)`

on qualified axes. `w_center` is only relative dispersion/axis eligibility, zero
when r_global=0. It is not falsely described as a common trust amplitude in the R7
normalized weighted loss, because that amplitude would cancel. DirectG and all
parameter/kernel arms receive this exact same confidence-shrunk pseudo box. Driver
LR remains alpha*.1/max(feature-norm²+1), alpha=.25; no repeated global LR multiplier.
Size remains q1's predicted prior/w0; yaw0. Singletons/no margin/low support/rejected
axes permit identity and zero updates. No jitter or forced nonzero fallback.

Output preserves original_source_q, associated_q, source-vs-associated same-group
relation, association_only_box(q1), raw_consensus_center, r_global, pseudo6, w6 and
detailed groups/semantic/margin/dispersion/skip evidence. Original source q0,
association-only q1, directG, parameter1/3 and exactkernel1/3 must be separate arms,
as must semantic-shuffle and PC-only. `dataclasses.replace(cache,query_index=q1)` is
allowed only for the associated arms; source Q0/cache identity is preserved.

Possible failure conditions: CLIP matches object class but ignores relation, mixed
instances share visibility, complete-link fragments physical objects, all correlated
boxes share the same bias, partial point support favors a wrong object, source sizes
are poor, or semantic margins/visibility give no evidence. True nonzero supervision
must be actual proposal disagreement weighted by RGB/text evidence, followed by
nonzero real gradient and parameter changes on real samples. CPU fixtures are only
code gates and do not establish those properties in the task distribution.
