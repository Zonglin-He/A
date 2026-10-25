# Full-target Tent/SAR: vidstg

Complete **10303 queries / 732 parents**, two baselines / 20606 method-query configurations. Whole-domain seal preceded official labels: 51515 original/dense checks, max discrepancy 2.2204460492503131e-15. All 30909 prediction/logit/receipt SHA checks, native/joint readout replay, Source parent arrays, episodic resets and means passed independent audit. Root additionally checks saved-logit diagnostics and every scored mean, with zero model/Source/expert/optimizer/GT-scoring calls.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|
|---|---:|---:|---:|---:|---|
|Source|31.728454|11.138484|12.831214|3.348539|0/0/0|
|TentNative|31.728070|11.138224|12.831214|3.348539|1/0/1|
|TentJointMAP|33.450602|11.641976|13.345627|3.630011|1014/272/1014|
|SARNative|31.728357|11.138446|12.831214|3.348539|0/0/0|
|SARJointMAP|33.450797|11.641976|13.345627|3.630011|1014/272/1014|
|SourceJointMAP|33.450806|11.641627|13.345627|3.630011|1015/272/1015|
|FullMAP|34.927982|14.548371|19.450645|8.492672|1487/1221/2238|

## Actual optimization and matched readout

{
  "Tent": {
    "changed_logits_queries": 10303,
    "nonzero_parameter_delta_queries": 10303,
    "changed_native_interval_queries": 7,
    "changed_joint_interval_queries": 7
  },
  "SAR": {
    "changed_logits_queries": 9530,
    "nonzero_parameter_delta_queries": 9530,
    "changed_native_interval_queries": 5,
    "changed_joint_interval_queries": 6
  }
}

{
  "Tent": {
    "all_query_scored_metrics_equal_SourceJointMAP": false,
    "changed_joint_intervals": 7,
    "queries_with_any_scored_metric_difference": 7,
    "paired_effects": {
      "v": {
        "mean": 0.0003492676987714764,
        "CI95": [
          -0.0004684208406748089,
          0.0014449322169063034
        ],
        "queries": 10303,
        "parents": 732,
        "improved": 2,
        "harmed": 2,
        "unchanged": 10299,
        "declined_over5pp": 0
      },
      "t": {
        "mean": -0.0002041953838946545,
        "CI95": [
          -0.002035129531874918,
          0.0016808612038190895
        ],
        "queries": 10303,
        "parents": 732,
        "improved": 2,
        "harmed": 4,
        "unchanged": 10297,
        "declined_over5pp": 1
      },
      "observed_actual_sIoU": {
        "mean": 0.0,
        "CI95": [
          0.0,
          0.0
        ],
        "queries": 8507,
        "parents": 715,
        "improved": 0,
        "harmed": 0,
        "unchanged": 8507,
        "declined_over5pp": 0
      },
      "unobserved_actual_sIoU": {
        "mean": 0.0,
        "CI95": [
          0.0,
          0.0
        ],
        "queries": 10300,
        "parents": 732,
        "improved": 0,
        "harmed": 0,
        "unchanged": 10300,
        "declined_over5pp": 0
      },
      "boundary_MAE_seconds": {
        "mean": 8.905982076417855e-06,
        "CI95": [
          -0.00022606887301667217,
          0.00018402534445308938
        ],
        "queries": 10303,
        "parents": 732,
        "improved": 2,
        "harmed": 5,
        "unchanged": 10296,
        "declined_over5pp": null
      }
    }
  },
  "SAR": {
    "all_query_scored_metrics_equal_SourceJointMAP": false,
    "changed_joint_intervals": 6,
    "queries_with_any_scored_metric_difference": 6,
    "paired_effects": {
      "v": {
        "mean": 0.0003492676987714764,
        "CI95": [
          -0.0004684208406748089,
          0.0014449322169063034
        ],
        "queries": 10303,
        "parents": 732,
        "improved": 2,
        "harmed": 2,
        "unchanged": 10299,
        "declined_over5pp": 0
      },
      "t": {
        "mean": -8.680631957486599e-06,
        "CI95": [
          -0.0017911088832582337,
          0.001833394124962078
        ],
        "queries": 10303,
        "parents": 732,
        "improved": 2,
        "harmed": 3,
        "unchanged": 10298,
        "declined_over5pp": 1
      },
      "observed_actual_sIoU": {
        "mean": 0.0,
        "CI95": [
          0.0,
          0.0
        ],
        "queries": 8507,
        "parents": 715,
        "improved": 0,
        "harmed": 0,
        "unchanged": 8507,
        "declined_over5pp": 0
      },
      "unobserved_actual_sIoU": {
        "mean": 0.0,
        "CI95": [
          0.0,
          0.0
        ],
        "queries": 10300,
        "parents": 732,
        "improved": 0,
        "harmed": 0,
        "unchanged": 10300,
        "declined_over5pp": 0
      },
      "boundary_MAE_seconds": {
        "mean": -1.9431233621275392e-05,
        "CI95": [
          -0.0002480436291823769,
          0.0001415724782856561
        ],
        "queries": 10303,
        "parents": 732,
        "improved": 2,
        "harmed": 4,
        "unchanged": 10297,
        "declined_over5pp": null
      }
    }
  }
}

Tent and SAR use the pinned official implementations with the previously frozen lr1e-4/five episodic steps; SAR margin fraction .8. Only temporal-decoder LayerNorm affine parameters are adapted, Source boxes and cached upstream contexts stay fixed, all decoder/optimizer state resets per method/query. This is a disclosed STVG adapter, not canonical BatchNorm/full-network/continual TTA or a general TTA ranking. Nonempty update counts and actual interval changes are distinguished; SAR reliable filtering may yield empty steps. Neither unchanged argmax nor this restricted negative result licenses retuning from full labels or a universal TTA-failure claim.

Primary Native uses original per-offset MAP plus physical hull. Secondary JointMAP uses the same joint readout as SourceJointMAP. Optimizer effects are the paired differences against the SAME readout before adaptation, reported separately; the jointMAP-versus-native decoding change is not optimizer gain. The matched control need not be exactly unchanged on every domain. Ours uses current user-selected kappa0, eight expert observations and zero optimizer steps; TTA uses zero expert and five update steps. These are different budgets, and the user selected kappa0 after prior full-result exposure. No independent unseen confirmation claim.

## All paired effects and tails

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|TentNative-Source|-0.000384 [-0.001762, 0.000599]|-0.000260 [-0.000840, 0.000056]|1/0/1|0|
|TentJointMAP-Source|+1.722147 [1.431544, 2.028579]|+0.503493 [0.391701, 0.624077]|1014/272/1014|2155|
|SARNative-Source|-0.000098 [-0.000303, 0.000000]|-0.000038 [-0.000116, 0.000000]|0/0/0|0|
|SARJointMAP-Source|+1.722343 [1.431663, 2.028917]|+0.503493 [0.391701, 0.624077]|1014/272/1014|2155|
|FullMAP-Source|+3.199528 [2.774632, 3.659093]|+3.409887 [2.994156, 3.838533]|1487/1221/2238|4405|
|FullMAP-TentNative|+3.199912 [2.774590, 3.659835]|+3.410146 [2.994103, 3.838541]|1487/1221/2238|4405|
|FullMAP-TentJointMAP|+1.477381 [1.074740, 1.889145]|+2.906394 [2.508355, 3.311569]|1895/1282/2642|3944|
|FullMAP-SARNative|+3.199626 [2.774632, 3.659288]|+3.409924 [2.994159, 3.838533]|1487/1221/2238|4405|
|FullMAP-SARJointMAP|+1.477185 [1.074534, 1.888951]|+2.906394 [2.508355, 3.311569]|1895/1282/2642|3944|
|FullMAP-SourceJointMAP|+1.477176 [1.074696, 1.889017]|+2.906744 [2.508579, 3.311928]|1896/1282/2643|3944|
|TentJointMAP-SourceJointMAP|-0.000204 [-0.002035, 0.001681]|+0.000349 [-0.000468, 0.001445]|1/0/1|1|
|SARJointMAP-SourceJointMAP|-0.000009 [-0.001791, 0.001833]|+0.000349 [-0.000468, 0.001445]|1/0/1|1|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / TentNative-Source|6060|+0.000272 [-0.000489, 0.001356]|-0.000016 [-0.000192, 0.000142]|
|short_le5s / TentJointMAP-Source|6060|-0.475661 [-0.719715, -0.221438]|-0.270939 [-0.368010, -0.171617]|
|short_le5s / SARNative-Source|6060|-0.000166 [-0.000520, 0.000000]|-0.000064 [-0.000200, 0.000000]|
|short_le5s / SARJointMAP-Source|6060|-0.475661 [-0.719715, -0.221438]|-0.270939 [-0.368010, -0.171617]|
|short_le5s / FullMAP-Source|6060|-0.077952 [-0.381935, 0.234039]|+1.128568 [0.774518, 1.512491]|
|short_le5s / FullMAP-TentNative|6060|-0.078224 [-0.382056, 0.233976]|+1.128584 [0.774521, 1.512492]|
|short_le5s / FullMAP-TentJointMAP|6060|+0.397709 [0.073938, 0.727543]|+1.399507 [1.039366, 1.794589]|
|short_le5s / FullMAP-SARNative|6060|-0.077786 [-0.381652, 0.234194]|+1.128632 [0.774522, 1.512543]|
|short_le5s / FullMAP-SARJointMAP|6060|+0.397709 [0.073938, 0.727543]|+1.399507 [1.039366, 1.794589]|
|short_le5s / FullMAP-SourceJointMAP|6060|+0.397731 [0.073838, 0.727991]|+1.399819 [1.039258, 1.794239]|
|short_le5s / TentJointMAP-SourceJointMAP|6060|+0.000022 [-0.002852, 0.002850]|+0.000313 [-0.001103, 0.002033]|
|short_le5s / SARJointMAP-SourceJointMAP|6060|+0.000022 [-0.002852, 0.002850]|+0.000313 [-0.001103, 0.002033]|
|long_gt5s / TentNative-Source|4243|-0.001320 [-0.004146, 0.000000]|-0.000607 [-0.001907, 0.000000]|
|long_gt5s / TentJointMAP-Source|4243|+4.861134 [4.408173, 5.370019]|+1.609562 [1.420903, 1.822134]|
|long_gt5s / SARNative-Source|4243|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|long_gt5s / SARJointMAP-Source|4243|+4.861609 [4.408666, 5.370373]|+1.609562 [1.420903, 1.822134]|
|long_gt5s / FullMAP-Source|4243|+7.880539 [7.042697, 8.733172]|+6.668145 [5.967036, 7.399667]|
|long_gt5s / FullMAP-TentNative|4243|+7.881859 [7.045178, 8.733261]|+6.668753 [5.967589, 7.399829]|
|long_gt5s / FullMAP-TentJointMAP|4243|+3.019405 [2.157809, 3.880820]|+5.058583 [4.389985, 5.738814]|
|long_gt5s / FullMAP-SARNative|4243|+7.880539 [7.042697, 8.733172]|+6.668145 [5.967036, 7.399667]|
|long_gt5s / FullMAP-SARJointMAP|4243|+3.018930 [2.156858, 3.880808]|+5.058583 [4.389985, 5.738814]|
|long_gt5s / FullMAP-SourceJointMAP|4243|+3.018877 [2.157248, 3.881630]|+5.058985 [4.389985, 5.739213]|
|long_gt5s / TentJointMAP-SourceJointMAP|4243|-0.000528 [-0.002834, 0.001855]|+0.000402 [-0.000004, 0.001264]|
|long_gt5s / SARJointMAP-SourceJointMAP|4243|-0.000053 [-0.002207, 0.002307]|+0.000402 [-0.000004, 0.001264]|

Short is native-GT duration <=5s. Confidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Severe is a signed loss greater than5pp in t or v, union counted once; rescues use greater than5pp gain. All signed positive and negative queries and fixed-support/boundary metrics remain in report.json and both TSVs.

## Actual costs and provenance

{
  "actual_new_counts": {
    "queries": 10303,
    "configs": 20606,
    "reused_completed_queries": 128,
    "new_optimizer_iterations": 101750,
    "new_Source_queries": 0,
    "shared_Source_queries": 9988,
    "new_expert": 0
  },
  "current_worker_segments": {
    "3432864:247190991": {
      "queries": 10303,
      "reused_completed_queries": 128,
      "new_Source_queries": 0,
      "shared_Source_queries": 9988,
      "new_optimizer_iterations": 101750,
      "Source_forward_seconds": 21097.39725106815,
      "acquisition_wall_seconds": 0.0,
      "conditional_query_wall_seconds": 7550.737348041963,
      "adaptation_seconds": 3357.7014066115953
    }
  },
  "current_model_load": {
    "worker": {
      "pid": 3432864,
      "startticks": 247190991
    },
    "seconds": 6.475113641005009,
    "loads": 1,
    "shared_with_structural_collector": true,
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
  "shared_Source_charged_once_to_original_collector": true,
  "scope": "Each PID/start segment is separate. Failed model load retained. Source/acquisition/query/adaptation timers can be nested; do not sum them. Existing-context/reused-result costs are inherited, not zero-cost deployment. Full worker setup/outer wall is not independently reconstructed from overlapping timers; these cached-context baseline timings are not matched end-to-end deployment latency."
}

No private captions/media/coordinates/features/weights are published.
