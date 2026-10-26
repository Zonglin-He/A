# Same eight observations, matched current temporal interval: vidstg

Complete 10303 queries / 732 parents; 51515 official/dense checks, max error 3.219646771412954e-15; 20606 prediction/receipt SHA checks plus bound parent/current references, exact cold arrays and all means. Root analysis reads audited scalar metrics only, zero new GT scoring/model/Source/expert.

|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|---:|
|SourceMatchedTime|34.918192|12.227364|32.729055|14.966515|4.241483|
|EightReplacement|34.918192|12.806780|33.882320|15.500340|4.668543|
|ExpertInterpolation|34.918192|14.665107|36.072277|19.596234|8.541202|
|Ours|34.918192|14.558151|36.090153|19.411822|8.463554|
|FinalIndependent|34.918192|13.997841|35.147280|18.615937|7.551199|

|Comparison|Delta v pp [95% CI]|Severe v|Rescue union|
|---|---:|---:|---:|
|Ours-SourceMatchedTime|+2.330787 [1.985568, 2.683873]|1136|2593|
|Ours-EightReplacement|+1.751372 [1.472721, 2.044449]|1067|2223|
|Ours-ExpertInterpolation|-0.106955 [-0.250579, 0.036605]|691|677|
|Ours-FinalIndependent|+0.560311 [0.512204, 0.612344]|19|303|

All intervals and tIoU are exactly identical across these rows: same audited current selected q1 MAP interval. This isolates spatial trajectory conditional on Ours temporal evidence; it is not a claim that each competitor has independent equal-quality temporal inference. SourceMatchedTime is Source boxes under this common refined interval, not the native Source main-table baseline. EightReplacement keeps Source outside eight exact original teacher positions. ExpertInterpolation exactly reuses original center/log-size interpolation and E outside coverage; neither competitor receives our DP. FinalIndependent is the already-measured internal fixed-candidate control, not a new external propagation method. Ours and FinalIndependent reproduce the current main/final-DP references. SAM2 is not in this table and is still pending; no full mature-propagation superiority claim follows yet.

|Duration / comparison|Queries|Delta v pp [95% CI]|
|---|---:|---:|
|short_le5s / Ours-SourceMatchedTime|6060|+1.237772 [0.913986, 1.584225]|
|short_le5s / Ours-EightReplacement|6060|+0.706170 [0.472149, 0.956333]|
|short_le5s / Ours-ExpertInterpolation|6060|-0.114356 [-0.250874, 0.011892]|
|short_le5s / Ours-FinalIndependent|6060|+0.331101 [0.287258, 0.376284]|
|long_gt5s / Ours-SourceMatchedTime|4243|+3.891869 [3.321844, 4.470748]|
|long_gt5s / Ours-EightReplacement|4243|+3.244165 [2.739792, 3.756858]|
|long_gt5s / Ours-ExpertInterpolation|4243|-0.096385 [-0.358340, 0.164588]|
|long_gt5s / Ours-FinalIndependent|4243|+0.887675 [0.800044, 0.980530]|

Short is GT duration <=5s; long >5s. Pointwise parent-cluster10000 bootstrap95%, not familywise; fullsets were previously exposed, no parameter/sample selection. All signed positive/negative metrics, fixed annotated support observed/unobserved sIoU, duration CIs, severe >5pp union once and rescues remain in report/TSVs. A positive mean is not every-query improvement. Native temporal Source-relative severe counts from the main table must not be confused with this fixed-time spatial table.

## Costs and preserved engineering failures

{
  "actual_prediction_worker": {
    "pid": 3969222,
    "startticks": 254421947
  },
  "cached_prediction_elapsed_wall_seconds": 1356.5785580137745,
  "actual_score_worker": {
    "pid": 4008724,
    "startticks": 254557958
  },
  "actual_independent_audit_worker": {
    "pid": 4027735,
    "startticks": 254631797
  },
  "new_Source": 0,
  "new_expert": 0,
  "new_model": 0,
  "inherited_Source_and_expert_acquisition_nonzero": true,
  "cost_scope": "Saved predictions/evidence reused; cached conditional replay/checks/CPU official scoring are not uncached deployment. Source/teacher acquisition was previously paid and is nonzero. Original failed report score passed official per-query loops but was not saved; recovery repeated scoring, disclosed separately.",
  "interface_failure": {
    "preservation_sha256": "e99b2bb4e6967008c4dff20d23e0fa59f8143d8598b2e7ae63598260cc951a1c",
    "original_score_exit": null,
    "original_score_wall_seconds": null,
    "missing_clock": "No standalone full-stage score clock was saved; no invented zero cost."
  },
  "SAM2_model_forward": false,
  "full_external_propagation_comparison_complete": false
}
