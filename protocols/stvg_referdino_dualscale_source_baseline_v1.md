# Cached Source spatial baseline, same MBR

User requested that complete spatial contribution also be measured against no
adaptation. The same-temporal-support Source coordinates are already pinned in
the cached decoder inputs. Seal a separate CPU-only Source_MBR/E readout barrier
before new scoring; no extra model/expert call or student update. All new OPD
predictions still require their original global barrier before ANY new GT score.

Evaluate Source at exactly E's frozen finalMBR time and same actual/interpolated
nodes; report E−Source, Direct−Source, oldLive−Source and frozen selectedLive−Source
with parent/seed semantics unchanged. This is no SPATIAL adaptation, not the
original Source native temporal decode. Exclude these results from policy ranking.
Do not attribute E's already-existing correction gain to OPD. No method/config/
sampling/teacher or selection change. Confirmation remains locked to the policy
selected on development; no confirmation-based switching.
