# Saved-pipeline and dataset metadata analysis, CPU v1

Authorized task: explain where correct predictions are lost and relate failures to VidSTG/HC-v2 properties. This is a posthoc analysis of completed, sealed `tastvg_best_quick_v1`, not a model experiment or optimization/promotion.

Read existing ROWS/PIPELINE_DIAGNOSIS and deep REFERENCE/DEEP_STEPS scalar JSONs. Join original PLAN parents with previously exposed VidSTG annotation temporal spans via the saved official pointer and HC-v2 metadata key; verify annotation SHA256 against old GT_EXPOSURE. Read temporal metadata and bbox count, not box-coordinate metrics. Never load media, model weights, prediction payloads or GPU libraries.

Fixed descriptive groups: query type and scorer GT-event/observed sampled-envelope fractions <=.25, (.25,.5], >.5. These groups are posthoc and not GT-dependent online thresholds. Do not choose a formula, cohort, corruption or parameter using them. Report strict vIoU>.3/.5 damage, source counts, gross gain/loss, >5/>20pp tails, expert-conditioned denominators and mutually exclusive final threshold-loss paths. Do not sum sequential damage counts or conditional oracle headroom.

The full/query/event-bin corruption cohorts have equal condition/order counts per source, so cell means equal source macro there. Expert-source membership differs across orders; expert denominators and counts are cell-conditional, not independent-query or source-macro estimates. No new subgroup confidence intervals or causal tests are claimed.

Use inclusive dense annotation support for Uniform5 geometric hit counts, matching REFERENCE_ROWS. Existing HC scorer's last GT box frame is an exclusive span end; annotation length includes that frame. Report both durations without rewriting old scoring. Observed envelope is [first sampled frame,last sampled frame+1), not original video duration or exact official input used segment. Retain GT outside-envelope coverage; do not clip annotations.

Public outputs: anonymous parent-index/query-type/duration/coverage metadata and aggregate results, code, source references and audits. Exclude original GT endpoint coordinates, boxes, captions, media identifiers, paths, raw caches, credentials and private conversations. Old outputs, configurations, production registration and paused queues remain untouched. Independently reproduce saved stage counts and public grouped aggregates, update research archive, publish and verify remote file bytes before completion.
