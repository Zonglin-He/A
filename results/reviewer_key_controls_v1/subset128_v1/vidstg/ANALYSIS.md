# Fixed128 reviewer regional and uniform-strength controls: vidstg

User explicitly replaced the remaining full-domain acquisition with a subset on 2026-11-03. This report uses the unchanged original matched-pipeline128-query/128-distinct-parent panel. It is previously exposed development data, not a new unknown test. The completed full CPU experiments remain separate and unchanged.

The original full dispatcher and GPU worker were intentionally stopped after289 committed HC queries (260 new Source forwards /29 retained raw /0 new expert), with no uncommitted paid attempt. Those outputs are preserved and unscored as a full set. This subset needs no GPU or new acquisition: all256 queries have retained exact raw grids.

|Arm|tIoU %|vIoU %|
|---|---:|---:|
|Source|32.121535|13.322903|
|SourceROI|35.528603|14.912319|
|CorrectedROI|35.202435|14.815209|
|GlobalAppearance|35.560990|14.995460|
|BackgroundAppearance|35.070563|14.666547|
|ShuffledBoxes|35.675272|14.905602|
|UniformGamma05|34.988905|14.741875|
|UniformGamma1|35.202435|15.339259|
|Current|35.202435|15.339259|

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union-once|Rescue union-once|
|---|---|---|---|---:|
|CorrectedROI-SourceROI|-0.326168 [-1.025934, +0.461678]|-0.097110 [-0.355101, +0.149646]|14/2/14|6|
|CorrectedROI-GlobalAppearance|-0.358556 [-1.334083, +0.685216]|-0.180251 [-0.528901, +0.121262]|15/3/15|6|
|CorrectedROI-BackgroundAppearance|+0.131871 [-1.072934, +1.385601]|+0.148662 [-0.356672, +0.617729]|18/5/18|12|
|CorrectedROI-ShuffledBoxes|-0.472838 [-1.148537, +0.194076]|-0.090393 [-0.381584, +0.197301]|12/3/12|7|
|GlobalAppearance-SourceROI|+0.032388 [-0.966766, +0.925138]|+0.083141 [-0.198347, +0.413815]|7/0/7|10|
|BackgroundAppearance-SourceROI|-0.458039 [-1.648785, +0.654755]|-0.245773 [-0.705958, +0.246635]|16/7/16|14|
|ShuffledBoxes-SourceROI|+0.146670 [-0.675842, +0.996025]|-0.006718 [-0.311220, +0.300281]|14/1/14|12|
|UniformGamma05-Source|+2.867370 [+0.581494, +5.362818]|+1.418972 [-0.400737, +3.263426]|21/20/33|48|
|UniformGamma1-Source|+3.080900 [+0.795909, +5.611244]|+2.016356 [-0.114303, +4.169645]|21/21/34|56|
|UniformGamma05-Current|-0.213530 [-0.769131, +0.317275]|-0.597384 [-1.252646, +0.063371]|7/8/15|12|
|UniformGamma1-Current|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|0/0/0|0|

## Short and long events

|Group|Queries|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---|---|---|
|short_le5s|78|CorrectedROI-SourceROI|+0.092551 [-0.713344, +1.082189]|+0.148539 [-0.106014, +0.414080]|
|short_le5s|78|CorrectedROI-GlobalAppearance|-0.311756 [-1.244755, +0.610226]|-0.186388 [-0.692920, +0.224523]|
|short_le5s|78|CorrectedROI-BackgroundAppearance|+0.615448 [-0.601202, +1.943819]|+0.321512 [-0.279227, +0.863711]|
|short_le5s|78|CorrectedROI-ShuffledBoxes|-0.156936 [-0.956007, +0.592992]|+0.078779 [-0.211941, +0.410684]|
|short_le5s|78|GlobalAppearance-SourceROI|+0.404306 [-0.384260, +1.234245]|+0.334926 [-0.062639, +0.826243]|
|short_le5s|78|BackgroundAppearance-SourceROI|-0.522897 [-1.596381, +0.485188]|-0.172973 [-0.701868, +0.395567]|
|short_le5s|78|ShuffledBoxes-SourceROI|+0.249486 [-0.785734, +1.412715]|+0.069759 [-0.336719, +0.453321]|
|short_le5s|78|UniformGamma05-Source|-0.817745 [-2.710732, +1.074315]|-0.476492 [-2.125131, +1.092972]|
|short_le5s|78|UniformGamma1-Source|-0.552505 [-2.401444, +1.301291]|+0.116031 [-1.598349, +1.851574]|
|short_le5s|78|UniformGamma05-Current|-0.265240 [-0.915270, +0.360849]|-0.592523 [-1.108994, -0.121957]|
|short_le5s|78|UniformGamma1-Current|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|
|long_gt5s|50|CorrectedROI-SourceROI|-0.979369 [-2.157619, +0.182405]|-0.480323 [-0.970674, -0.046474]|
|long_gt5s|50|CorrectedROI-GlobalAppearance|-0.431564 [-2.455016, +1.773618]|-0.170678 [-0.607280, +0.232301]|
|long_gt5s|50|CorrectedROI-BackgroundAppearance|-0.622508 [-3.019623, +1.855897]|-0.120983 [-1.003694, +0.760562]|
|long_gt5s|50|CorrectedROI-ShuffledBoxes|-0.965645 [-2.243948, +0.246880]|-0.354302 [-0.922338, +0.180048]|
|long_gt5s|50|GlobalAppearance-SourceROI|-0.547805 [-2.828467, +1.301614]|-0.309645 [-0.662651, +0.006455]|
|long_gt5s|50|BackgroundAppearance-SourceROI|-0.356861 [-2.910832, +1.970214]|-0.359340 [-1.242222, +0.513844]|
|long_gt5s|50|ShuffledBoxes-SourceROI|-0.013725 [-1.303871, +1.343737]|-0.126022 [-0.632196, +0.351385]|
|long_gt5s|50|UniformGamma05-Source|+8.616150 [+4.178417, +13.776374]|+4.375896 [+0.509230, +8.328673]|
|long_gt5s|50|UniformGamma1-Source|+8.749011 [+4.203408, +13.965948]|+4.980864 [+0.225516, +9.716227]|
|long_gt5s|50|UniformGamma05-Current|-0.132861 [-1.061313, +0.792348]|-0.604968 [-2.091622, +0.936141]|
|long_gt5s|50|UniformGamma1-Current|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|

The five region arms all output the SAME shared-corrected E trajectory and independently update their temporal posterior/interval. Source-box, corrected-box, global, background-outside-expanded-E and whole-box-permutation controls isolate temporal pooling support. They are diagnostic expert-derived fixed-space comparisons, not no-expert deployment rankings. Every whole shuffled box and size multiset is preserved; fixed output geometry is unchanged.

Both uniform gamma values .5 and1 are reported in BOTH domains. Each rebuilds shared-corrected boxes, actual native-grid ROI features, original postfusion propagation geometry, temporal posterior, its own interval and final DP. The default HC.5 / Vid1 output is array-exact Current. These are two fixed cross-domain controls, not a sweep, new selected setting, or source-only proof for the historical target-exposed gamma selection. No method, budget, source model or parameters were changed.

Original per-query Source/region/Current scalar scores are reused only after their old independent audits, exact input identities and array equality; only the two newly derived gamma arms are newly officially scored after the whole128 seal. Another process replays all arrays and checks input/prediction/compact/receipt SHA and every reused score/mean. All positive and negative signed results and short/long groups are retained. Bootstrap is query-equal,10000 paired-parent resamples, pointwise95% intervals; no independent-frame, familywise, equivalence, action-semantic or full-target claim follows a crossing-zero CI.

## Pooling diagnostics

{
  "empty_background_frames": 386,
  "shuffled_fallback_frames": 386,
  "queries": 128,
  "parents": 128
}

## Cost and stopped full-prefix preservation

{
  "subset_cached_prediction_elapsed_wall_seconds": 89.19236744008958,
  "subset_query_elapsed_wall_sum_seconds": 89.08952013915405,
  "subset_reuse_counts": {
    "retained_raw_actual_repool": 128
  },
  "new_Source": 0,
  "new_expert": 0,
  "new_model": 0,
  "new_optimizer": 0,
  "GPU": false,
  "original_full_acquisition": {
    "status": "intentionally_stopped_unscored_after_user_scope_reduction",
    "completed_queries": 289,
    "counts": {
      "queries": 289,
      "new_Source": 260,
      "retained_raw": 29,
      "new_expert": 0
    },
    "original_Source_forward_sum_seconds": 925.3756501269527,
    "model_load": {
      "worker": {
        "pid": 4108873,
        "startticks": 324227881
      },
      "elapsed_wall_seconds": 11.003426148556173,
      "loads": 1
    },
    "whole_worker_completion_clock": null,
    "all_paid_commits_preserved": true,
    "no_uncommitted_paid_attempt_at_stop": true
  },
  "interpretation": "Cached elapsed wall includes read/hash/pool/replay/serialization and is not deployment latency or process CPU time. Source/expert original acquisition nonzero. Original full-prefix Source sum is nested in original query wall; partial-prefix query costs can overlap this subset reuse and must not be added as disjoint acquisition. Original full worker was user-stopped, unknown complete-worker clock remains null."
}

Media, captions, GT coordinates, raw features/masks, weights and credentials remain private. No manuscript/Fig1/current registry or old frozen source/plan changed.
