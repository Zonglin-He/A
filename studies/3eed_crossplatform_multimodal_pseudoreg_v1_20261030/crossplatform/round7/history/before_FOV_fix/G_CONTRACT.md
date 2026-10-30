# R7 non-learning G contract, version1 (synthetic preparation)

G keeps the pure-text source Top1 query identity. It accepts only XYZ P, RGB image,
text q, source6D candidates, pure-text source scores, fixed query index and external
per-point image uv/camera depth. No GT/class-map/positive_map, learned 2D grounding,
teacher/EMA, yaw, reranking or new candidates are accepted. Camera projection is
root-owned: Waymo inverse suffix then A@inv(E); Drone undo z+1.8 then E; Quad E.
Absent distortion interpreted as zero is a declared projection convention, not a
measured zero-distortion fact. G consumes actual projected pixels/depth, not RGB
features cached from a different frame.

Only an opening phrase `[the|a|an] COLOR NOUN` is supported. COLOR is exactly one of
red/green/blue/yellow/white/black; NOUN is a fixed finite object lexicon. Caption is
cut before the sentinel `. not mentioned`; negation or multiple finite color words
before the first spatial relation makes the phrase unsupported. Relation-object
colors are never target colors. Unsupported language returns source with w=0.

Per-pixel nearest positive-depth XYZ visibility is computed before cropping using
all P. Equal-depth ties use lexicographic XYZ, independent of P order. Read RGB at
rounded nearest pixel. HSV rules are fixed in DEFAULT_CONFIG; shuffled mode uses
fixed cycle red→green→blue→yellow→red and white↔black. PC-only uses identical camera,
language eligibility, crop, connected components and face tests but no RGB filter.

Crop is source AABB plus0.25m. Color-supported unique XYZ components use0.35m Euclidean
connectivity; require12points and largest/second ratio≥1.5. A nonduplicate candidate
(AABB IoU<0.8 with selected source box) with support≥0.8 of selected support and≥12
points causes ambiguity skip, never query switching. This rule can conservatively
reject useful examples and does not guarantee identity in an unannotated scene.

Center coordinate j is observed only when both source-opposite face bands have≥6
supported points, each face spans≥0.4×source-size in BOTH tangential axes, pooled
span along j≥0.8×source-size, each face residual MAD≤0.03m, and two face median
translation residuals differ≤0.04m. Band halfwidth=min(0.15×size_j,0.30m). Correction
is mean of the two median residuals. Require |delta_j|≤min(0.1×size_j,0.2m), otherwise
reject that axis. No partial-point centroid is interpreted as complete box center.
Unobserved center axes are identity/w0; accepted axes w1. All three size coordinates
are source-prior identity/w0. No yaw output exists. A visible silhouette/face band is
only a conservative source-conditioned geometry cue, not proof of actual box faces;
same-color clutter, inaccurate source extents and missing opposite sides remain
failure conditions. Single-view opaque geometry may reject most examples.

All constants were chosen before real-data/GT execution, and are intentionally not
relaxed to achieve nonzero acceptance. Adapter Gtarget/w are stop-gradient. Required
arms: source, direct G, final-regression parameter SGD, equivalent output-space SGD;
PC-only and shuffled-color are feedback controls. Only synthetic CPU fixtures are
authorized here. No real model/checkpoint/data/GT access or GPU run occurs.
