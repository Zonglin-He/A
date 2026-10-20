# Current state guided observation: cached local-feature probe

HC128 exposed unique parents; Source Vid frozen, same65/2080/MBR and fixed spatial E/Live41/42/43. Eight revealed cached Dense nodes in2x4 rounds, no new encoding/RGB/expert/training. Existing frozen fusion/full decoder/head replay is charged. Actual Strong preserved:0.5 old Source448 +0.5 mean6 current224 second-pass posteriors. No posterior weight sweep.

Selection proxy is posterior risk-weighted boundary coverage, NOT calibrated actual observation value or event probability. CurrentState uses its real after4 posterior for round2; InitialState holds initial q only for selection, not output. Reflected selection uses wrong phase; actual feedback/output are never reflected. Uniform covers fixed8. Identical masks may reuse exact deterministic suffix results; logical observations equal8 but actual joint experiment replays/cache hits disclosed.

| Output | full vIoU percent | tIoU percent | boundary MAE frames |
|---|---:|---:|---:|
| Native | 27.928474 | 51.439514 | 77.238281 |
| HiResAuxMix | 29.534908 | 55.920522 | 70.367188 |
| DenseLocalMix | 28.901490 | 53.597804 | 74.714844 |
| Uniform8 | 29.393969 | 55.627776 | 70.566406 |
| InitialState8 | 29.350968 | 55.466485 | 71.644531 |
| CurrentState8 | 29.525327 | 55.781158 | 70.628906 |
| ReflectedState8 | 29.615096 | 56.101371 | 69.679688 |

| Comparison | full-v pp [CI95] | tIoU pp [CI95] | losses<-5pp |
|---|---|---|---:|
| CurrentState8_minus_HiResAuxMix | -0.009581 [-0.35373231273796873, 0.3344390907520011] | -0.139364 [-0.763640811469287, 0.4775825665385861] | 4 |
| CurrentState8_minus_Uniform8 | +0.131357 [-0.18444752476213322, 0.445404850542203] | +0.153382 [-0.4391301214026756, 0.7232233425737619] | 3 |
| CurrentState8_minus_InitialState8 | +0.174359 [-0.08014352244879279, 0.5854625594009477] | +0.314673 [-0.12872410587920616, 1.0076028449196788] | 0 |
| CurrentState8_minus_ReflectedState8 | -0.089769 [-0.4680168031176434, 0.2782977406340424] | -0.320213 [-1.099563392764112, 0.42938397094866576] | 3 |
| CurrentState8_minus_Native | +1.596853 [0.4814952381483028, 2.745925568889461] | +4.341644 [2.170793747885589, 6.602198725448349] | 14 |
| InitialState8_minus_Uniform8 | -0.043001 [-0.6690964733057903, 0.4347856906091267] | -0.161291 [-1.2180592349557116, 0.6444917255834103] | 3 |
| InitialState8_minus_HiResAuxMix | -0.183940 [-0.8148714803131226, 0.30135260003423064] | -0.454037 [-1.5209727973907534, 0.37792568166009394] | 4 |
| Uniform8_minus_HiResAuxMix | -0.140939 [-0.358655383991383, 0.06511436566761876] | -0.292746 [-0.7538092692064599, 0.09982221760932215] | 1 |
| ReflectedState8_minus_HiResAuxMix | +0.080188 [-0.07704030914236393, 0.2422300788504252] | +0.180849 [-0.20617552113017498, 0.6301836211867808] | 0 |

```json
{
  "primary_over_Strong": {
    "mean": -0.009581242049984784,
    "CI95": [
      -0.35373231273796873,
      0.3344390907520011
    ],
    "parents": 128
  },
  "position_value_over_Uniform": {
    "mean": 0.13135731309549825,
    "CI95": [
      -0.18444752476213322,
      0.445404850542203
    ],
    "parents": 128
  },
  "refresh_over_initial_state": {
    "mean": 0.1743587735608918,
    "CI95": [
      -0.08014352244879279,
      0.5854625594009477
    ],
    "parents": 128
  },
  "state_selection_candidate": false,
  "refresh_candidate": false,
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_parameter_sweep": true,
  "no_automatic_next_experiment": true,
  "scope": "cached-feature partial observation; not fresh RGB acquisition or deployment",
  "goal_achieved": false
}
{
  "mean_first4_L1": 0.14405823157370484,
  "changed_round2_final_sets": 80,
  "same_first4_all128": true,
  "mean_current_unique_frames": 61.609375
}
{
  "GPU_seconds": 174.16763572697528,
  "measured_lease_seconds": 171.16763572697528,
  "cleanup_reserve_seconds": 3,
  "success": true,
  "actual_calls": {
    "fusion_attempts": 1956,
    "fusion_completed": 1956,
    "decoder_attempts": 3912,
    "decoder_completed": 3912,
    "head_attempts": 1956,
    "head_completed": 1956,
    "image_attempts": 0,
    "image_completed": 0,
    "motion_attempts": 0,
    "motion_completed": 0,
    "text_attempts": 0,
    "text_completed": 0,
    "image_projection_attempts": 0,
    "image_projection_completed": 0,
    "motion_projection_attempts": 0,
    "motion_projection_completed": 0
  },
  "logical_requests": 1154,
  "actual_unique_replays": 978,
  "deterministic_cache_hits": 176,
  "new_RGB": 0,
  "new_visual_motion_text_encoding": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_training": 0,
  "new_student_updates": 0,
  "spatial_updates": 0,
  "not_deployment_latency": true,
  "GPU_peak_allocated_bytes": 1091377152,
  "GPU_peak_reserved_bytes": 1138753536,
  "CPU_audit_and_scoring_seconds": 128.02219740208238,
  "publication_and_archive_excluded": true
}
{
  "status": "pass",
  "checks": {
    "selection_steps": 3072,
    "independent_acquisition_scores": 199680,
    "selected_only_fusion_inputs": 1956,
    "layer_posteriors": 5868,
    "independent_final_layer_CDF": 978,
    "direct_MBR_sample_utilities": 13674,
    "space_exact": 512,
    "native_empty_exact": 128,
    "first_round_Fixed_Live_exact": 128,
    "logical_policy_budgets": 512
  },
  "max_independent_acquisition_score_error": 8.326672684688674e-17,
  "max_final_layer_CDF_error": 2.8261198036858077e-16,
  "max_direct_utility_error": 6.661338147750939e-16,
  "selected_local_features_only": true,
  "zero_new_RGB_encoding": true,
  "new_GT": true,
  "official_Dense_output_checks": 5632,
  "old_metrics_exact": 1536,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "scalar_candidate_quality_calls": 798720
}
```

All128 predictions/reveal histories sealed before GT; all parents/seed41,42,43 retained, seeds averaged within parent, paired parent bootstrap10000. First4-stage metrics and all final signed outputs, selected physical-node IDs, severe tails are in RESULTS_PUBLIC. Candidate quality is offline diagnostic and not a substitute for actual full-v. Acquisition comparisons use identical local clip budget; full global original inputs remain. Cached fixed-B8 local features are not proof of exact fresh sparse clip packing;64 real hypothetical RGB slots/128 padded slots are not measured encoder time. This is adaptive inference/observation capability, no optimized student policy state, no OPD contribution. No training/mixture/observation-budget scan or automatic confirmation/new experiment. Spatial/CURRENT/MAIN/PAPER unchanged.

## Interpretation and supplementary audit

Current does not exceed Strong or the reflected-state control on the mean; all three main pairwise intervals cross zero. Mean first4 posterior L1 change0.144058 and80/128 changed second-round selected sets show a real state refresh, but only22 final full-v scores differ versus Initial (12positive/10negative/106flat). Removing the largest positive refresh case leaves+0.007642pp mean; this is a post-hoc concentration diagnostic, not a sample exclusion or alternative endpoint. The signed top cases and first4-stage readouts are retained in INTERPRETATION_PUBLIC. Close this exact proxy, retain Strong, do not enter OPD or scan weights. This is not proof that all adaptive observation policies fail.

Supplementary GT-before audit explicitly locks the dynamically imported tails helper after original measurement locking (EXTERNAL_AUDIT_PIN001), independently checks both full65 qualifications against old Dense logits/q (SUPPLEMENTARY_PRE_GT_AUDIT001), and checks actual terminal budget/shared-lock release after barrier (POST_SEAL_ENGINEERING_AUDIT001). These supplement the original653pins; they do not pretend to be original prelocks or a second runtime model hash. ROOT_RECOMPUTE.py independently checks4688 saved-score scalars and54 paired parent bootstrap outputs, all signed tails/worst/LOO; maximum error5.56e-17, no new GT/model. Original inference/scorer/publisher bytes and predictions unchanged.
