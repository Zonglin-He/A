# Reproduction boundary

This is a grounding task adaptation suite, not original-task replication. See METHOD_CARDS.md.
Use the existing 3EED workspace, raw licensed data, the three epoch100 source-only checkpoints,
and the prior immutable source/preprocessing interfaces. Private input registries, captions,
media, point/box coordinates, embeddings, weights and third-party snapshots are not distributed.
Download author repositories at OFFICIAL_REPOSITORIES.json commits; MM-TTA uses Latte reference code.

Ordered stages: prepare128_cpu; train128_prepared (initial attempt retained);
train128_mate_chamfer; calibrate128_roles; run128_fast (engineering then formal);
run128_prepared (engineering then formal); run128_cloudfixer (engineering then formal);
run128_zero_controls; prepare128_independent_RGB; run128_mmtta_cpu --independent;
audit_score128 only AFTER every matrix sealed; audit128_scalars; report128; export128;
audit_public128. Every inference/training/scoring job uses experiment_guard_round128.py.
Prepared RGB/source calibration and all algorithm repairs preceded target-GT scoring.
No post-score parameter selection. Corrected one-forward/independent-RGB MM-TTA is primary;
the older diagnostics are retained as disclosed implementation attempts, not paper comparisons.

Prior own-code dependency: studies/3eed_fixed_R91_three_source_six_direction_TTA_20261114
(core97/factory/interface), plus earlier R76/R91 modules. Native upstream 3EED is obtained
from https://github.com/worldbench/3EED, not republished here. CloudFixer author diffusion
weights provenance/hash are provided; checkpoint bytes are intentionally excluded.
