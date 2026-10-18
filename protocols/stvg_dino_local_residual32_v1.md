# Cached local residual experiment, original 32

Question: after content/mask alignment restores feedback, does retaining each
anchor's correction avoid information loss from shared4 aggregation and global
application? This is a development experiment, not independent confirmation.

Freeze the original HC16/Vid16 roster (one query per parent), TA-STVG cross-domain
Source checkpoints, E trajectory, final-distribution MBR interval, four planned
decoder positions, original 24 sampled correction states, mask-aligned feedback,
0.35 threshold, whole-candidate legality and source geometry scales. No new
model/expert calls, feature extraction, training, network updates, resampling,
parameter search, identity labeling or dynamic OPD are permitted.

Three outputs, plus E baseline:

1. Shared4_global: exact previous B_aligned_once trajectory.
2. Shared4_observed: copy E, replace only the four planned decoder nodes with
   those exact Shared4_global nodes. Same correction, restricted application.
3. Independent4_observed: at each planned node independently, average accepted
   teacher-minus-sampled-input residuals in (cx,cy,log w,log h), dividing by 24
   including invalid and no-feedback states. Add that residual to the E node.
   Every other decoder node remains bit-exact E. No-feedback residual is zero.

Sampled inputs are reconstructed in float64 from the saved action and Source,
exactly as in the prior update and accounting. Detector boxes were stored in
float32 and are promoted to float64 before z conversion. No clipping, new
legality test, extrapolation or extra normalization is introduced.

Inactive queries remain no-op in every arm. Planned nodes are never selected by
GT visibility. All 32 complete outputs and input hashes are sealed before new GT
scoring. Existing exposure is acknowledged; sealing prevents new evaluation
from influencing these outputs, not from making this cohort unseen.

Evaluate official complete vIoU/tIoU/dense sIoU and actual decoder observed and
unobserved sIoU. Unknown GT support remains null, not failure or correctness.
All tIoUs must match. Independently verify exact nonobserved-node invariance.
Separate benchmark xyxy-interpolated frames from actual decoder frames, including
their exact additive contributions to vIoU changes within temporal overlap.
Report sample counts, all signed query outcomes, >5pp losses, displacement norms
in physical center and log-size coordinates, and 10000 paired parent bootstrap
intervals (seed 20261018). Each domain has 16 distinct parents. No GT-based
selection, checkpoints or deployment rule.

This finite CPU experiment stops after the three comparisons and publication.
Even a local improvement does not establish dynamic OPD, identity correctness or
new evidence at unobserved frames. Domain-specific outcomes remain separate.
Current deployment, historical predictions and all paused queues are unchanged.
