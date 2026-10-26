# Same eight observations, matched current temporal interval: hc2

Complete 1902 queries / 227 parents; 9510 official/dense checks, max error 1.3322676295501878e-15; 3804 prediction/receipt SHA checks plus bound parent/current references, exact cold arrays and all means. Root analysis reads audited scalar metrics only, zero new GT scoring/model/Source/expert.

|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|---:|
|SourceMatchedTime|49.764548|25.604197|50.685778|38.801262|14.773922|
|EightReplacement|49.764548|26.229192|51.744875|39.800210|15.089380|
|ExpertInterpolation|49.764548|25.214291|48.980489|37.749737|14.037855|
|Ours|49.764548|27.349044|53.283819|41.219769|16.876972|
|FinalIndependent|49.764548|26.281532|51.636175|39.484753|15.036803|

|Comparison|Delta v pp [95% CI]|Severe v|Rescue union|
|---|---:|---:|---:|
|Ours-SourceMatchedTime|+1.744846 [1.233989, 2.268511]|224|456|
|Ours-EightReplacement|+1.119852 [0.702703, 1.551092]|212|319|
|Ours-ExpertInterpolation|+2.134753 [1.814042, 2.473029]|131|416|
|Ours-FinalIndependent|+1.067512 [0.958161, 1.183652]|9|106|

All intervals and tIoU are exactly identical across these rows: same audited current selected q1 MAP interval. This isolates spatial trajectory conditional on Ours temporal evidence; it is not a claim that each competitor has independent equal-quality temporal inference. SourceMatchedTime is Source boxes under this common refined interval, not the native Source main-table baseline. EightReplacement keeps Source outside eight exact original teacher positions. ExpertInterpolation exactly reuses original center/log-size interpolation and E outside coverage; neither competitor receives our DP. FinalIndependent is the already-measured internal fixed-candidate control, not a new external propagation method. Ours and FinalIndependent reproduce the current main/final-DP references. SAM2 is not in this table and is still pending; no full mature-propagation superiority claim follows yet.

|Duration / comparison|Queries|Delta v pp [95% CI]|
|---|---:|---:|
|short_le5s / Ours-SourceMatchedTime|544|+1.697195 [0.945276, 2.427603]|
|short_le5s / Ours-EightReplacement|544|+1.009242 [0.451603, 1.561076]|
|short_le5s / Ours-ExpertInterpolation|544|+1.978059 [1.473458, 2.525647]|
|short_le5s / Ours-FinalIndependent|544|+0.872858 [0.686613, 1.063996]|
|long_gt5s / Ours-SourceMatchedTime|1358|+1.763935 [1.121647, 2.413193]|
|long_gt5s / Ours-EightReplacement|1358|+1.164161 [0.621129, 1.710607]|
|long_gt5s / Ours-ExpertInterpolation|1358|+2.197523 [1.810813, 2.605119]|
|long_gt5s / Ours-FinalIndependent|1358|+1.145488 [1.009648, 1.284256]|

Short is GT duration <=5s; long >5s. Pointwise parent-cluster10000 bootstrap95%, not familywise; fullsets were previously exposed, no parameter/sample selection. All signed positive/negative metrics, fixed annotated support observed/unobserved sIoU, duration CIs, severe >5pp union once and rescues remain in report/TSVs. A positive mean is not every-query improvement. Native temporal Source-relative severe counts from the main table must not be confused with this fixed-time spatial table.

## Costs and preserved engineering failures

{
  "actual_prediction_worker": {
    "pid": 3933783,
    "startticks": 254381366
  },
  "cached_prediction_elapsed_wall_seconds": 53.20010706782341,
  "actual_score_worker": {
    "pid": 3958430,
    "startticks": 254409946
  },
  "actual_independent_audit_worker": {
    "pid": 3964415,
    "startticks": 254416946
  },
  "new_Source": 0,
  "new_expert": 0,
  "new_model": 0,
  "inherited_Source_and_expert_acquisition_nonzero": true,
  "cost_scope": "Saved predictions/evidence reused; cached conditional replay/checks/CPU official scoring are not uncached deployment. Source/teacher acquisition was previously paid and is nonzero. Original failed report score passed official per-query loops but was not saved; recovery repeated scoring, disclosed separately.",
  "interface_failure": {
    "preservation_sha256": "e99b2bb4e6967008c4dff20d23e0fa59f8143d8598b2e7ae63598260cc951a1c",
    "original_score_exit": {
      "code": 1,
      "worker": {
        "pid": 3938728,
        "startticks": 254387367
      }
    },
    "original_score_wall_seconds": null,
    "missing_clock": "No standalone full-stage score clock was saved; no invented zero cost."
  },
  "SAM2_model_forward": false,
  "full_external_propagation_comparison_complete": false
}
