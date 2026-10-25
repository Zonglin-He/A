# Full-target Tent/SAR: hc2

Complete **1902 queries / 227 parents**, two baselines / 3804 method-query configurations. Whole-domain seal preceded official labels: 9510 original/dense checks, max discrepancy 1.1102230246251565e-15. All 5706 prediction/logit/receipt SHA checks, native/joint readout replay, Source parent arrays, episodic resets and means passed independent audit. Root additionally checks saved-logit diagnostics and every scored mean, with zero model/Source/expert/optimizer/GT-scoring calls.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|
|---|---:|---:|---:|---:|---|
|Source|40.141475|20.704678|29.022082|8.937960|0/0/0|
|TentNative|40.145299|20.704678|29.022082|8.937960|0/0/0|
|TentJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|SARNative|40.145299|20.704678|29.022082|8.937960|0/0/0|
|SARJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|SourceJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|FullMAP|49.514796|27.214726|41.062040|16.614090|364/305/467|

## Actual optimization and matched readout

{
  "Tent": {
    "changed_logits_queries": 1902,
    "nonzero_parameter_delta_queries": 1902,
    "changed_native_interval_queries": 3,
    "changed_joint_interval_queries": 0
  },
  "SAR": {
    "changed_logits_queries": 1676,
    "nonzero_parameter_delta_queries": 1676,
    "changed_native_interval_queries": 2,
    "changed_joint_interval_queries": 0
  }
}

{
  "Tent": {
    "all_query_scored_metrics_equal_SourceJointMAP": true,
    "changed_joint_intervals": 0
  },
  "SAR": {
    "all_query_scored_metrics_equal_SourceJointMAP": true,
    "changed_joint_intervals": 0
  }
}

Tent and SAR use the pinned official implementations with the previously frozen lr1e-4/five episodic steps; SAR margin fraction .8. Only temporal-decoder LayerNorm affine parameters are adapted, Source boxes and cached upstream contexts stay fixed, all decoder/optimizer state resets per method/query. This is a disclosed STVG adapter, not canonical BatchNorm/full-network/continual TTA or a general TTA ranking. Nonempty update counts and actual interval changes are distinguished; SAR reliable filtering may yield empty steps. Neither unchanged argmax nor this restricted negative result licenses retuning from full labels or a universal TTA-failure claim.

Primary Native uses original per-offset MAP plus physical hull. Secondary JointMAP uses the same joint readout as SourceJointMAP. Exact equality to that matched control must not be presented as optimizer gain over nativeSource. Ours uses current user-selected kappa0, eight expert observations and zero optimizer steps; TTA uses zero expert and five update steps. These are different budgets, and the user selected kappa0 after prior full-result exposure. No independent unseen confirmation claim.

## All paired effects and tails

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|TentNative-Source|+0.003824 [0.000000, 0.012258]|+0.000000 [0.000000, 0.000000]|0/0/0|1|
|TentJointMAP-Source|+6.551818 [5.857056, 7.264109]|+3.334419 [2.930885, 3.750103]|220/118/221|877|
|SARNative-Source|+0.003824 [0.000000, 0.012258]|+0.000000 [0.000000, 0.000000]|0/0/0|1|
|SARJointMAP-Source|+6.551818 [5.857056, 7.264109]|+3.334419 [2.930885, 3.750103]|220/118/221|877|
|FullMAP-Source|+9.373320 [8.373270, 10.392112]|+6.510048 [5.779450, 7.252761]|364/305/467|1124|
|FullMAP-TentNative|+9.369497 [8.367731, 10.388207]|+6.510048 [5.779450, 7.252761]|364/305/467|1124|
|FullMAP-TentJointMAP|+2.821502 [2.097354, 3.544531]|+3.175629 [2.559407, 3.810588]|468/374/597|896|
|FullMAP-SARNative|+9.369497 [8.367731, 10.388207]|+6.510048 [5.779450, 7.252761]|364/305/467|1124|
|FullMAP-SARJointMAP|+2.821502 [2.097354, 3.544531]|+3.175629 [2.559407, 3.810588]|468/374/597|896|
|FullMAP-SourceJointMAP|+2.821502 [2.097354, 3.544531]|+3.175629 [2.559407, 3.810588]|468/374/597|896|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / TentNative-Source|544|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|short_le5s / TentJointMAP-Source|544|+4.097325 [2.717493, 5.493670]|+1.895244 [1.117750, 2.676004]|
|short_le5s / SARNative-Source|544|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|short_le5s / SARJointMAP-Source|544|+4.097325 [2.717493, 5.493670]|+1.895244 [1.117750, 2.676004]|
|short_le5s / FullMAP-Source|544|+3.585731 [1.767606, 5.419920]|+3.417933 [2.090251, 4.723997]|
|short_le5s / FullMAP-TentNative|544|+3.585731 [1.767606, 5.419920]|+3.417933 [2.090251, 4.723997]|
|short_le5s / FullMAP-TentJointMAP|544|-0.511595 [-1.668386, 0.652049]|+1.522689 [0.477723, 2.570922]|
|short_le5s / FullMAP-SARNative|544|+3.585731 [1.767606, 5.419920]|+3.417933 [2.090251, 4.723997]|
|short_le5s / FullMAP-SARJointMAP|544|-0.511595 [-1.668386, 0.652049]|+1.522689 [0.477723, 2.570922]|
|short_le5s / FullMAP-SourceJointMAP|544|-0.511595 [-1.668386, 0.652049]|+1.522689 [0.477723, 2.570922]|
|long_gt5s / TentNative-Source|1358|+0.005355 [0.000000, 0.017427]|+0.000000 [0.000000, 0.000000]|
|long_gt5s / TentJointMAP-Source|1358|+7.535061 [6.728364, 8.313545]|+3.910936 [3.433406, 4.404103]|
|long_gt5s / SARNative-Source|1358|+0.005355 [0.000000, 0.017427]|+0.000000 [0.000000, 0.000000]|
|long_gt5s / SARJointMAP-Source|1358|+7.535061 [6.728364, 8.313545]|+3.910936 [3.433406, 4.404103]|
|long_gt5s / FullMAP-Source|1358|+11.691766 [10.525444, 12.887558]|+7.748716 [6.815321, 8.703140]|
|long_gt5s / FullMAP-TentNative|1358|+11.686410 [10.521165, 12.885989]|+7.748716 [6.815321, 8.703140]|
|long_gt5s / FullMAP-TentJointMAP|1358|+4.156705 [3.289639, 5.042575]|+3.837779 [3.032479, 4.644635]|
|long_gt5s / FullMAP-SARNative|1358|+11.686410 [10.521165, 12.885989]|+7.748716 [6.815321, 8.703140]|
|long_gt5s / FullMAP-SARJointMAP|1358|+4.156705 [3.289639, 5.042575]|+3.837779 [3.032479, 4.644635]|
|long_gt5s / FullMAP-SourceJointMAP|1358|+4.156705 [3.289639, 5.042575]|+3.837779 [3.032479, 4.644635]|

Short is native-GT duration <=5s. Confidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Severe is a signed loss greater than5pp in t or v, union counted once; rescues use greater than5pp gain. All signed positive and negative queries and fixed-support/boundary metrics remain in report.json and both TSVs.

## Actual costs and provenance

{
  "actual_new_counts": {
    "queries": 1902,
    "configs": 3804,
    "reused_completed_queries": 128,
    "new_optimizer_iterations": 17740,
    "new_Source_queries": 1608,
    "shared_Source_queries": 0,
    "new_expert": 0
  },
  "current_worker_segments": {
    "2558667:246462743": {
      "queries": 1902,
      "reused_completed_queries": 128,
      "new_Source_queries": 1608,
      "new_optimizer_iterations": 17740,
      "Source_forward_seconds": 3442.181676385924,
      "acquisition_wall_seconds": 4577.71326941764,
      "conditional_query_wall_seconds": 6245.79539268557,
      "adaptation_seconds": 444.79006355721503
    }
  },
  "current_model_load": {
    "worker": {
      "pid": 2558667,
      "startticks": 246462743
    },
    "seconds": 4.856266333255917,
    "loads": 1,
    "shared_with_structural_collector": false,
    "nested_not_additive": true
  },
  "failed_actual_model_load": {
    "worker": {
      "pid": 2550108,
      "startticks": 246454024
    },
    "seconds": 6.825765760149807,
    "loads": 1,
    "shared_with_structural_collector": false,
    "nested_not_additive": true
  },
  "original_reused_optimizer_source_cost_nonzero": true,
  "original_completed_TTA_audit_sha256": "43586ee897da72a82cff7306d2216e1e5350eb207b7c8a7a8ba6fbb65261a962",
  "Source_native_parent_acquisition_nonzero": true,
  "scope": "Each PID/start segment is separate. Failed model load retained. Source/acquisition/query/adaptation timers can be nested; do not sum them. Existing-context/reused-result costs are inherited, not zero-cost deployment. Full worker setup/outer wall is not independently reconstructed from overlapping timers; these cached-context baseline timings are not matched end-to-end deployment latency."
}

No private captions/media/coordinates/features/weights are published.
