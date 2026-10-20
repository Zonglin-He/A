# Fixed-batch local motion: budget stop after two complete pilots

This round is closed at its prelocked resource gate. Two engineering pilots completed, all new exact checks passed, but the conservative128-parent estimate exceeds900s. No full128 barrier, GT access, effect scoring, static component/OPD claim or new experiment follows. User requested uploading existing results and stopping.

| Item | Actual |
|---|---|
| Requested / completed parents |128 /2|
| Non-control pilot seconds | [7.719566258834675, 6.89391440525651] |
| Loading / control seconds | 6.007290 / 4.258476 |
| Conservative128 estimate / cap | 1058.370247 /900s |
| Actual GPU lease+reserve | 28.194746s |
| Prior separately charged engineering | 34.143195s |
| New GT/effect scoring/expert/training/spatial or student updates |0|

```json
{
  "status": "stopped_at_prelocked_budget_qualification",
  "requested_parents": 128,
  "completed_pilot_parents": 2,
  "full_round_completed": false,
  "new_exact_pilot_qualification_passed": true,
  "estimated_GPU_seconds": 1058.370246994542,
  "budget_GPU_seconds": 900,
  "new_task_effect_measurement": false,
  "GT_read": false,
  "original_R30_qualification_passed": false,
  "user_requested_stop_after_publication": true,
  "no_further_exploration": true,
  "goal_achieved": false
}
{
  "status": "pass",
  "scope": "CPU independent pilot artifact audit only, not full128 or task-effect evaluation",
  "pins": 123,
  "all_pins_exact": true,
  "frozen_spatial_arrays_exact": 8,
  "old_posteriors_intervals_exact": 4,
  "Native_prefix_logit_checks": 4,
  "independent_CDF_checks": 6,
  "max_CDF_error": 2.3245294578089215e-16,
  "independent_clip_indices": 2080,
  "input_center_SHA_checks": 48,
  "fullclip_permutation_checks": 64,
  "Native_repeat_tail_exact_checks": 68,
  "independent_MBR_sample_utilities": 168,
  "new_GT": false,
  "terminal_model_weight_flags_hash_check_reached": false,
  "Source_checkpoint_file_exact": true
}
{
  "GPU_seconds": 28.194746180903167,
  "prior_engineering_GPU_seconds": 34.14319452410564,
  "RGB_slots": 2978,
  "measured_lease_seconds": 25.194746180903167,
  "cleanup_reserve_seconds": 3,
  "actual_calls": {
    "motion_attempts": 52,
    "motion_completed": 52,
    "projection_attempts": 52,
    "projection_completed": 52,
    "fusion_attempts": 12,
    "fusion_completed": 12,
    "decoder_attempts": 24,
    "decoder_completed": 24,
    "head_attempts": 12,
    "head_completed": 12
  },
  "GPU_peak_allocated_bytes": 4545404416,
  "GPU_peak_reserved_bytes": 5398069248,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "new_student_updates": 0,
  "spatial_updates": 0,
  "success": false
}
```

Both pilots retained all six outputs, all four frozen spatial arrays and complete first2 qualification stages. Original motion/prefix/logits/q exact, fullclip permutation/repeat and tail duplicate checks passed. Single/batch differences and originalR30 failure remain disclosed; this revision does not retroactively pass the old8eps16 gate. Padding never entered65-node readout. The returned pilot intervals are included in RESULTS_PUBLIC with all GT metrics explicitly unknown. They are not performance results.

Terminal whole-model weight/flags hash validation was after the full128 loop and was not reached because of the resource assertion. Source checkpoint/config/code pins are exact and the execution path has no optimizer/network/student/spatial updates; do not substitute that for an executed terminal runtime hash check. The CPU audit validates saved pilot tensors/receipts and declared exact controls only. No claim about temporal or OPD effectiveness is possible from this round. No budget extension, smaller scoring panel, parameter changes or continuation. Existing spatial configuration and CURRENT/MAIN/PAPER remain unchanged.
