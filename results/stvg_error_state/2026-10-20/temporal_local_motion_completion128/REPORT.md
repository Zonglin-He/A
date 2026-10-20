# Budget-only completion of fixed-batch local motion128

Actual execution entry: scripts/run_stvg_temporal_local_motion_completion128_v1.py; scoring entry: scripts/score_stvg_temporal_local_motion_completion128_v1.py. User explicitly reopened the unfinished task measurement. New finite GPU cap1800s; old900s budget stop, two unscored pilots and cost remain immutable. All128 rerun in one loaded-model process so terminal weight/flag verification covers this run. Two old pilot features/posteriors/intervals/spatial arrays independently replay exactly before GT. No changed numerical gate, clip, input, arm, score, threshold, teacher, spatial state or parameter. Previous engineering and old fixedbatch costs separately disclosed; no automatic confirmation/OPD.

Actual entry scripts/score_stvg_temporal_local_motion_fixedbatch128_v1.py. Engineering qualification and execution specification explicitly revised before any task scores: fixed B8 with repeat-tail padding, same-shape full-clip permutation/repeat exact. Original R30/recovery001 remain failed, old single tolerance is disclosed without passing it retroactively. Prior engineering costs separately retained.
# Fixed-batch local motion, unchanged global context: RC128

Source Vid frozen; original65 decoder timestamps, full-image appearance/text and spatial E/Live41/42/43 exact. Dense8×stride2 and Sparse8×same-offset-neighbors have the same T=8, center3, no temporal shift. Higher density and shorter physical context are coupled; no isolated aliasing or pure motion-vs-local-appearance claim. No spatial crop, new expert, training, student/spatial update. Not OPD.

| Comparison | full vIoU pp [CI95] | expected candidate GTv pp [CI95] |
|---|---|---|
| DenseLocalMix_minus_Native | +0.973016 [-0.26830602840913303, 2.1939391405345567] | +0.942654 [0.24125103424907482, 1.6315168542489655] |
| DenseLocalMix_minus_HiResAuxMix | -0.633418 [-1.7832233127768082, 0.5060435228891965] | +1.132889 [0.43008770704877813, 1.8724356087948848] |
| DenseLocalMix_minus_SparseLocalMix | +0.917487 [-0.20220100759451037, 1.9853450886325708] | +0.709712 [0.1003250315744179, 1.303929352282108] |
| DenseLocalMix_minus_DenseLocal | +0.163111 [-0.5394415840689017, 0.9162067393553119] | -0.942654 [-1.6315168542489653, -0.24125103424907512] |
| DenseLocal_minus_Native | +0.809905 [-0.9219511126346946, 2.4938090489574485] | +1.885307 [0.4825020684981496, 3.2630337084979306] |
| DenseLocal_minus_HiResAuxMix | -0.796529 [-2.2742733240767508, 0.6726029472226221] | +2.075542 [0.9382344976502751, 3.2080344899390583] |
| DenseLocal_minus_SparseLocal | +0.584671 [-1.1191387616332011, 2.278292568458221] | +1.419425 [0.20065006314883488, 2.6078587045642165] |
| SparseLocalMix_minus_Native | +0.055529 [-0.4384988970650762, 0.668647797176634] | +0.232941 [-0.09910998394847205, 0.5984213763653761] |
| SparseLocal_minus_Native | +0.225235 [-0.6310199485111515, 1.184992463908304] | +0.465883 [-0.19821996789694374, 1.1968427527307517] |
| HiResAuxMix_minus_Native | +1.606434 [0.5915612220621151, 2.666302924676544] | -0.190235 [-1.0486135913597796, 0.6718222797720916] |

All128 repeatedly exposed parents; spatial seeds averaged within parent, paired parent bootstrap. All losses/tails and no-change parents retained. Strong is an existing stronger development reference. Hard readout failure and transfer qualification are assessed separately; no method/OPD claim without matched actual transfer and independent confirmation.

```json
{
  "primary_hard": {
    "mean": 0.9730162912507514,
    "CI95": [
      -0.26830602840913303,
      2.1939391405345567
    ],
    "parents": 128
  },
  "pure_teacher_expected": {
    "mean": 1.8853072141216942,
    "CI95": [
      0.4825020684981496,
      3.2630337084979306
    ],
    "parents": 128
  },
  "Dense_versus_Sparse_expected": {
    "mean": 1.4194247061731258,
    "CI95": [
      0.20065006314883488,
      2.6078587045642165
    ],
    "parents": 128
  },
  "decision": "close_this_local_motion_configuration",
  "versus_Strong": {
    "mean": -0.6334178159929663,
    "CI95": [
      -1.7832233127768082,
      0.5060435228891965
    ],
    "parents": 128
  },
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_clip_stride_mixture_temperature_sweep": true,
  "goal_achieved": false
}
{
  "positive": 70,
  "negative": 49,
  "flat": 9,
  "below_minus3": 23,
  "below_minus5": 16,
  "leave_one_out_mean_range_pp": [
    0.7852166149835078,
    1.2364404580223975
  ],
  "worst": [
    {
      "alias": "RC042",
      "delta_pp": -32.4818528887483
    },
    {
      "alias": "RC080",
      "delta_pp": -22.46286601594561
    },
    {
      "alias": "RC014",
      "delta_pp": -14.27661553020802
    },
    {
      "alias": "RC125",
      "delta_pp": -12.886400829482517
    },
    {
      "alias": "RC030",
      "delta_pp": -10.766417020905966
    },
    {
      "alias": "RC111",
      "delta_pp": -10.638717443606877
    },
    {
      "alias": "RC082",
      "delta_pp": -8.921954563120526
    },
    {
      "alias": "RC100",
      "delta_pp": -8.188912640017332
    }
  ]
}
{
  "GPU_seconds": 1531.9604845391586,
  "prior_engineering_GPU_seconds": 34.14319452410564,
  "RGB_slots": 148130,
  "measured_lease_seconds": 1528.9604845391586,
  "cleanup_reserve_seconds": 3,
  "actual_calls": {
    "motion_attempts": 2320,
    "motion_completed": 2320,
    "projection_attempts": 2320,
    "projection_completed": 2320,
    "fusion_attempts": 768,
    "fusion_completed": 768,
    "decoder_attempts": 1536,
    "decoder_completed": 1536,
    "head_attempts": 768,
    "head_completed": 768
  },
  "GPU_peak_allocated_bytes": 4842234368,
  "GPU_peak_reserved_bytes": 5769265152,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "new_student_updates": 0,
  "spatial_updates": 0,
  "success": true
}
{
  "status": "pass",
  "frozen_spatial_arrays_exact": 512,
  "old_posteriors_intervals_exact": 256,
  "independent_clip_indices": 133120,
  "actual_input_and_center_feature_SHA_checks": 3072,
  "independent_CDF_checks": 384,
  "max_CDF_error": 2.7145838948547905e-16,
  "independent_MBR_sample_utilities": 10743,
  "Native_motion_exact": true,
  "same_batch_repeat_exact": true,
  "batch_single_relative_errors": [
    0.1318359375,
    0.04150390625,
    0.05607476457953453,
    0.068115234375
  ],
  "original_R30_qualification_passed": false,
  "new_exact_qualification_checks": 68,
  "independent_full_clip_permutation_checks": 64,
  "tail_padding_outputs_excluded": true,
  "new_GT": true,
  "official_Dense_output_checks": 3072,
  "scalar_candidate_quality_calls": 798720,
  "old_metrics_exact": 1024,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "all_temporal_arms_sIoU_identical": true
}
```

No clip/stride/mix/temperature rescue; close only exact negative configuration, not all OPD. No CURRENT/MAIN/PAPER or spatial config change. Candidate quality uses Dense scalar scoring, final outputs official/Dense double checked; MBR independent utility checks are predeclared samples, not all-action double calculation.


## Current user decision boundary

The inherited fixedbatch decision above is retained as its original diagnostic. Current final decision is COMPLETION_DECISION_PUBLIC.json: actual full-v must improve Native and Strong; pure expected quality cannot authorize OPD. No automatic next experiment.
{
  "primary_hard": {
    "mean": 0.9730162912507514,
    "CI95": [
      -0.26830602840913303,
      2.1939391405345567
    ],
    "parents": 128
  },
  "pure_teacher_expected": {
    "mean": 1.8853072141216942,
    "CI95": [
      0.4825020684981496,
      3.2630337084979306
    ],
    "parents": 128
  },
  "Dense_versus_Sparse_expected": {
    "mean": 1.4194247061731258,
    "CI95": [
      0.20065006314883488,
      2.6078587045642165
    ],
    "parents": 128
  },
  "decision": "close_this_local_motion_configuration",
  "versus_Strong": {
    "mean": -0.6334178159929663,
    "CI95": [
      -1.7832233127768082,
      0.5060435228891965
    ],
    "parents": 128
  },
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_clip_stride_mixture_temperature_sweep": true,
  "goal_achieved": false,
  "previous_fixedbatch_rule_decision": "close_this_local_motion_configuration",
  "pure_expected_quality_is_diagnostic_only": true,
  "actual_full_v_and_Strong_required": true,
  "no_automatic_OPD_or_next_run": true
}

## Actual full-v means, same frozen spatial

| Output | full vIoU percent | tIoU percent | boundary MAE frames |
|---|---:|---:|---:|
| Native | 27.928474 | 51.439514 | 77.238281 |
| HiResAuxMix | 29.534908 | 55.920522 | 70.367188 |
| DenseLocal | 28.738379 | 53.792321 | 76.757812 |
| SparseLocal | 28.153708 | 51.569949 | 74.960938 |
| DenseLocalMix | 28.901490 | 53.597804 | 74.714844 |
| SparseLocalMix | 27.984003 | 51.414314 | 75.691406 |

The new observation has a positive offline expected-quality signal (Dense minus Native +1.885307pp, CI[+0.482502,+3.263034]; Dense minus Sparse +1.419425pp, CI[+0.200650,+2.607859]). This is not zero information, deployable GT selection, or an OPD result. Actual DenseLocalMix full-v +0.973016pp versus Native has CI crossing zero and is -0.633418pp versus Strong. Pure Dense hard +0.809905pp also has CI crossing zero. Neither the hard criterion nor the older expected>=2pp criterion passed. Close this exact observation configuration under the declared decision; do not infer all OPD is impossible, do not sweep the same local parameters, and do not automatically begin OPD/confirmation or a new fusion.

Costs separately retained: this run1531.960485s GPU lease including3s cleanup; prior fixedbatch stop28.194746s; earlier engineering34.143195s. Cumulative scopes1594.298425s, not this run's elapsed time. CPU independent audit/task scoring47.639808s; publication/archive and this saved-score audit are separate, no new GPU. Current Source/teacher/checkpoint collection unchanged, zero new DINO/ReferDINO/training/student or spatial updates.
