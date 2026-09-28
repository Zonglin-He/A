# DESTA-3D v2: one branch removal after the full-output anchor tradeoff

The completed, sealed 8-parent development screen found that the combined
time/coordinate anchor improved clean vIoU by 0.381681 pp relative to the old
calibration-alignment arm, while reducing noise vIoU by 0.228963 pp. It restored
HC64's clean time boundary but left spatial damage, and all noise differences
were spatial. Final time/coordinate KL did not uniformly decrease. This motivates
a branch ablation; it does not establish coordinate KL as the cause.

Add exactly one arm, `calibration_alignment_temporal_anchor`: retain the time
output KL and set coordinate output KL to zero. Keep coefficient
6.248522551708088, without re-normalization or new target/source tuning. Reuse
the previous seven arms' 168 predictions after full seal and input checks;
generate only 24 new predictions, 72 new Adam steps, for 192 total predictions.
All 8 historical development parents and all clean/noise/blur episodes remain.
Both previous screens already exposed their GT for offline scoring. This is
an explicitly development-driven ablation, not independent confirmation.

The fixed B1 checkpoint, post-query source moments, observed-input teacher,
photometric pre-gate student view, alignment .01, other pre-gate weights,
FiLM/LN 66816, frozen gates/backbone/heads/projections/text pools, fixed3 steps,
AdamW1e-5 wd0 clip1, seed and episodic reset are identical. Both native teacher
branches are still captured for matching and readout; only the event/time
branch is replayed differentiably. Shared LN remains trainable, so time-only
gradients can indirectly change spatial predictions. No GT/lowest-loss state
choice, no sample filtering, and no clean input teacher for corrupt episodes.

Missing event teacher means unchanged pre-gate-only3 steps and retained case.
Coordinate support is irrelevant to this ablation's loss, but missing/invalid
boxes remain in scoring. Actual time and coordinate distributions, intervals,
fixed-prefix coordinate KL, cast injection, raw gradients and actual Adam
counters are retained. Use proven memory-v7 event replay, 16GiB activation
offload / 6GiB host reserve / allocator cleanup; no numerical/cache change.

Primary contrast is time-only minus both-output anchor, with old no-output
calibration-alignment and B1/Frozen also reported. If clean protection persists
and noise loss reduces, this supports the usefulness of branch-specific
protection in this limited setting. If spatial damage remains, do not claim
the spatial anchor was responsible. If clean protection disappears, retain
the tradeoff. If no discriminating evidence emerges, lower this branch's
priority and audit view/alignment signal construction rather than sweep lambda.
Even favorable results do not automatically expand to64 or promote CURRENT.

All192 predictions and inputs must be sealed before re-opening historical GT.
Independent scalar/P3 geometry plus a second parent/condition/CI/retention/tail
calculation are required. CIs are descriptive on8 exposed parents, one seed.
HC is cross-dataset transfer from Vid-only source. GPU serial,3600s engineering
allocation,8GiB disk floor, cumulative cap=null; all failed/replayed overhead
remains in receipts. Preserve prior runs and pins; no automatic GitHub push.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
