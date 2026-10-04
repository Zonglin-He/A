# Conditional continuation of optimizer/posterior R1

This records the implementation interpretation before any new stage predictions.
It is not a completed experiment or production registration. Original cohorts,
corruptions, checkpoints, observations and historical exposure remain explicit.

R3 uses the exact <=81 product paths in physical observation order. Text term
is sum cached target scores; native term is proposal IoU against the complete
pre-update tube; continuity is adjacent proposal IoU. All coefficients and
temperatures are1. Paths and invalid/empty frames retained as documented;
invalid positive-width/height filtering is unchanged. Continuity is not a
guarantee that the same referent was followed. FrameSum is the matched energy
scale control; Track and TrackAuthority separate path weighting and uncertainty
authority. A GIoU alternative is eligible only after track evidence is useful.
If R1 selects an authority optimizer, FrameSum and raw Track retain the same
frame-level authority as R1; only TrackAuthority changes to path authority.

Development selection uses both datasets, retaining the same arm across them.
R3 promotes a track arm only if its corrupt mean gain is at least the R1 anchor
and FrameSum, with no extra >20pp harms relative to that anchor, on both search
panels. Highest summed gain breaks eligible arm ties. Otherwise frame mean
remains the evidence. Confirmation is reported and never selects the winner.
Track-qualified GIoU is one further matched intervention, with the same
development preservation rule. R4 selects u-only or small-LN only if corrupt
mean and >20pp harms preserve joint on both search panels, then chooses the
largest summed gain; otherwise joint remains. This is a development decision,
not a claim that a small point improvement establishes a transferable method.

R4 fixes the selected optimizer/evidence, comparing joint, u-only and small-LN.
Joint is the inherited1792 interface. u-only current optimizes the256 query
residual for10 steps, selects its own energy minimum; at that state compute one
fresh1536-LN-only optimizer step with the fixed same optimizer/lr, retained for
future1/16 consolidation but not current output. This is an explicit minimal
definition of the attachment's otherwise unspecified slow gradient.
Small-LN keeps the selected full query correction and multiplies the current
LN displacement by the frozen frame/path entropy complement a; query extra
authority is1. Persist only1/16 of the actually used LN displacement.
Neither LN interpolation nor continuity is claimed to ensure task safety.

R5 must run independently evolving selected online LN1/16 chains. Query and
optimizer reset; temporal remains episodic/native according to its earlier
qualification. A preserved configuration with exact existing P1 receipts may
reuse that actual full stream, with a reuse receipt instead of a false new run.

R6 qualifies the frozen combination at25/50/100% query-level expert schedules.
Schedules are nested fixed source hash ranks, identical across corruption and
order, no GT selection. Report all flow and actual expert/nonexpert subsets;
cached logical calls and newly executed detector calls are distinct. Cross-domain
needs its own source checkpoint, capture/evidence hashes and no-GT smoke; a
same-domain H cache cannot be reused under a different checkpoint. No old
cross-domain queue is restarted.
Scope correction005: the preceding resource gate was introduced by the agent,
not required by the user's full R6 authorization. It is preserved in recovery
and the original same-domain decision, but superseded for execution. Run the
new fixed-configuration cross-domain qualification regardless of that gate;
no confirmation-based method reselection. See decota_cross_qualification_v1.md.

Unimplemented or unqualified stages remain explicit in ROUTE status. T1,
occupancy-based new observations and GIoU are conditional, not automatically
expanded to LR/K/beta sweeps. Optional region critic/L-BFGS/long-term duration
state remain deferred. Every executed stage needs its own seal before GT,
independent objective/optimizer/state-chain/dense/source bootstrap checks,
negative-case retention, archive and public GitHub verification.
