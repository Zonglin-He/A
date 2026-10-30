# R7 independent post-seal cross-domain evaluation contract

This CPU-only evaluator must not load any target metadata before a stage-specific
`CPU_SCORE_RELEASE=true` document and externally supplied release SHA authorize the
sealed prediction set. The release binds the prediction seal, input manifest,
private scoring index, actual Waymo-source checkpoint and evaluator source SHA.
All prediction NPZ/code/manifest/checkpoint hashes are verified first. A separate
scoring index provides annotation joins; the inference manifest must not provide GT.

Primary selected-box geometry is native source/model6D, yaw0, with the official
predicted-size minimum1e-6 used only for evaluation. The exact raw box and clamp
count remain diagnostic. No PCA, GT yaw replacement, GT candidate selection or GT
class-assisted language readout is introduced. Fixed source query identity is
shared by every arm. Source ranking uses saved pure-text scores and stable argmax;
saved native Top10 is preferred if available and is validated for ties. Top5/10 and
all256 candidate coverage are explicitly GT-assisted source-pool retrieval
statistics, not adapted Top1 or exact physical-instance identification.

For efficiency, source pool coverage uses the previously audited float64 batched
2D-yaw×height geometry implementation. Every unique selected arm box is separately
scored by the independent general3D convex implementation, with exact identity
reuse. Their selected-box agreement is checked. Full256 coverage is not described
as independently covered by general3D convex geometry.

Drone and Quad are separate populations. All requested samples, errors, invalid
sizes, identity skips, accepted-any-axis and actual nonzero changes are counted.
All comparisons report their paired source score on the exact same mask. The
common-eligible mask is the intersection of nonzero G reliability across RGB,
PC-only and shuffled-color modes, not a GT-defined subset. No failure is silently
converted to a success or silently dropped; finite scored count and all-requested
count are both explicit, along with pessimistic all-requested lower summaries if
an output is absent. Negative deltas are retained for every case in private rows.

Scene is the resampling unit. Target VAL has only2Drone+3Quad scenes, not thousands
of independent trials. Equal-scene summaries and paired scene bootstrap intervals
are descriptive with such few clusters. Multiple expressions/frames/arms are not
independent clusters. Source model was trained only on Waymo, but the target VAL
has already been evaluated in R5 and is not a fresh research holdout.

Post-seal GT may diagnose whether a G center displacement points toward the target
center and whether task IoU improves. It may not select colors, G thresholds,
learning rate, step count, query identity, alpha or an adaptation arm. All raw rows,
annotation references and per-scene identifiers remain private; public projections
must use an explicit scalar allowlist and hashed scene IDs.
