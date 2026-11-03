# Fixed128 reviewer regional and uniform-strength controls: hc2

User explicitly replaced the remaining full-domain acquisition with a subset on 2026-11-03. This report uses the unchanged original matched-pipeline128-query/128-distinct-parent panel. It is previously exposed development data, not a new unknown test. The completed full CPU experiments remain separate and unchanged.

The original full dispatcher and GPU worker were intentionally stopped after289 committed HC queries (260 new Source forwards /29 retained raw /0 new expert), with no uncommitted paid attempt. Those outputs are preserved and unscored as a full set. This subset needs no GPU or new acquisition: all256 queries have retained exact raw grids.

|Arm|tIoU %|vIoU %|
|---|---:|---:|
|Source|46.509478|25.124253|
|SourceROI|54.833053|30.470815|
|CorrectedROI|54.968548|30.453799|
|GlobalAppearance|54.479768|30.045855|
|BackgroundAppearance|53.499251|29.179448|
|ShuffledBoxes|53.563567|29.910041|
|UniformGamma05|54.968548|31.791298|
|UniformGamma1|54.309866|31.570678|
|Current|54.968548|31.791298|

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union-once|Rescue union-once|
|---|---|---|---|---:|
|CorrectedROI-SourceROI|+0.135495 [-0.479292, +0.770855]|-0.017017 [-0.232810, +0.217617]|8/1/8|9|
|CorrectedROI-GlobalAppearance|+0.488781 [-0.705089, +1.671980]|+0.407944 [-0.223229, +1.084151]|19/8/19|25|
|CorrectedROI-BackgroundAppearance|+1.469298 [-0.168165, +3.233836]|+1.274351 [+0.191013, +2.523046]|25/11/25|34|
|CorrectedROI-ShuffledBoxes|+1.404981 [+0.174056, +2.660755]|+0.543757 [-0.125456, +1.230776]|19/6/19|25|
|GlobalAppearance-SourceROI|-0.353285 [-1.661451, +0.896349]|-0.424961 [-1.151091, +0.231765]|19/8/19|19|
|BackgroundAppearance-SourceROI|-1.333803 [-3.218391, +0.358570]|-1.291368 [-2.548030, -0.187202]|30/22/31|26|
|ShuffledBoxes-SourceROI|-1.269486 [-2.611873, +0.052833]|-0.560774 [-1.251126, +0.111098]|26/14/26|20|
|UniformGamma05-Source|+8.459071 [+5.507059, +11.495124]|+6.667045 [+3.904815, +9.443270]|18/18/30|76|
|UniformGamma1-Source|+7.800389 [+4.796948, +10.900275]|+6.446425 [+3.633390, +9.286084]|23/21/35|77|
|UniformGamma05-Current|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|0/0/0|0|
|UniformGamma1-Current|-0.658682 [-1.265610, -0.124017]|-0.220620 [-0.787487, +0.328363]|11/6/13|9|

## Short and long events

|Group|Queries|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---|---|---|
|short_le5s|42|CorrectedROI-SourceROI|+0.478163 [-0.649461, +1.856214]|+0.102002 [-0.109272, +0.395107]|
|short_le5s|42|CorrectedROI-GlobalAppearance|+0.424168 [-1.099598, +1.944006]|+0.032531 [-0.630512, +0.646460]|
|short_le5s|42|CorrectedROI-BackgroundAppearance|+0.753400 [-1.445431, +3.026843]|+0.361142 [-0.912568, +1.631444]|
|short_le5s|42|CorrectedROI-ShuffledBoxes|+0.369322 [-1.212717, +2.139971]|-0.036807 [-0.693738, +0.694570]|
|short_le5s|42|GlobalAppearance-SourceROI|+0.053995 [-1.687538, +1.765625]|+0.069470 [-0.505821, +0.686023]|
|short_le5s|42|BackgroundAppearance-SourceROI|-0.275237 [-2.734464, +2.109711]|-0.259140 [-1.511642, +1.008206]|
|short_le5s|42|ShuffledBoxes-SourceROI|+0.108841 [-1.957645, +2.133909]|+0.138808 [-0.624684, +0.844295]|
|short_le5s|42|UniformGamma05-Source|+3.124530 [-1.631175, +7.812485]|+3.471243 [-0.162025, +7.252910]|
|short_le5s|42|UniformGamma1-Source|+2.690938 [-2.126860, +7.485237]|+3.003368 [-0.917029, +7.035771]|
|short_le5s|42|UniformGamma05-Current|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|
|short_le5s|42|UniformGamma1-Current|-0.433592 [-1.320363, +0.444725]|-0.467874 [-1.198564, +0.230551]|
|long_gt5s|86|CorrectedROI-SourceROI|-0.031854 [-0.697191, +0.683294]|-0.075142 [-0.366329, +0.246398]|
|long_gt5s|86|CorrectedROI-GlobalAppearance|+0.520335 [-1.106335, +2.151661]|+0.591285 [-0.300759, +1.520682]|
|long_gt5s|86|CorrectedROI-BackgroundAppearance|+1.818922 [-0.364509, +4.184855]|+1.720337 [+0.255382, +3.405631]|
|long_gt5s|86|CorrectedROI-ShuffledBoxes|+1.910768 [+0.301282, +3.536555]|+0.827288 [-0.122309, +1.761063]|
|long_gt5s|86|GlobalAppearance-SourceROI|-0.552190 [-2.274523, +1.144952]|-0.666427 [-1.678186, +0.280517]|
|long_gt5s|86|BackgroundAppearance-SourceROI|-1.850777 [-4.275091, +0.455596]|-1.795479 [-3.506913, -0.265615]|
|long_gt5s|86|ShuffledBoxes-SourceROI|-1.942623 [-3.591706, -0.312788]|-0.902430 [-1.804626, +0.020465]|
|long_gt5s|86|UniformGamma05-Source|+11.064312 [+7.308223, +14.900771]|+8.227786 [+4.614316, +11.886558]|
|long_gt5s|86|UniformGamma1-Source|+10.295702 [+6.431163, +14.212422]|+8.127918 [+4.420000, +11.856895]|
|long_gt5s|86|UniformGamma05-Current|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|
|long_gt5s|86|UniformGamma1-Current|-0.768610 [-1.571362, -0.129173]|-0.099869 [-0.893759, +0.643307]|

The five region arms all output the SAME shared-corrected E trajectory and independently update their temporal posterior/interval. Source-box, corrected-box, global, background-outside-expanded-E and whole-box-permutation controls isolate temporal pooling support. They are diagnostic expert-derived fixed-space comparisons, not no-expert deployment rankings. Every whole shuffled box and size multiset is preserved; fixed output geometry is unchanged.

Both uniform gamma values .5 and1 are reported in BOTH domains. Each rebuilds shared-corrected boxes, actual native-grid ROI features, original postfusion propagation geometry, temporal posterior, its own interval and final DP. The default HC.5 / Vid1 output is array-exact Current. These are two fixed cross-domain controls, not a sweep, new selected setting, or source-only proof for the historical target-exposed gamma selection. No method, budget, source model or parameters were changed.

Original per-query Source/region/Current scalar scores are reused only after their old independent audits, exact input identities and array equality; only the two newly derived gamma arms are newly officially scored after the whole128 seal. Another process replays all arrays and checks input/prediction/compact/receipt SHA and every reused score/mean. All positive and negative signed results and short/long groups are retained. Bootstrap is query-equal,10000 paired-parent resamples, pointwise95% intervals; no independent-frame, familywise, equivalence, action-semantic or full-target claim follows a crossing-zero CI.

## Pooling diagnostics

{
  "empty_background_frames": 22,
  "shuffled_fallback_frames": 22,
  "queries": 128,
  "parents": 128
}

## Cost and stopped full-prefix preservation

{
  "subset_cached_prediction_elapsed_wall_seconds": 51.70966928778216,
  "subset_query_elapsed_wall_sum_seconds": 51.60914105596021,
  "subset_reuse_counts": {
    "retained_raw_actual_repool": 112,
    "committed_full_compact_reuse": 16
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
