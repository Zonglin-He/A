# Fixed local-clip engineering audit, no task scores

All six complete raw768/projection256 stages saved before numeric comparison, then independently read back using numpy. Same frozen Source, failed first batch input, AMP/batch8/clip8/center3 unchanged. Native original controls now retained. No fusion, temporal task prediction, new expert, training, student/spatial update or GT.

```json
{
  "status": "completed_engineering_audit_only",
  "runtime": {
    "torch": "2.7.0+cu128",
    "torch_git": "134179474539648ba7dee1317959529fbd0e7f89",
    "cuda": "12.8",
    "GPU": "NVIDIA GeForce RTX 5090",
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "cudnn_allow_tf32": true,
    "matmul_allow_tf32": false
  },
  "audit_completed": true,
  "independence_pass": true,
  "checks": {
    "Native0_exact": true,
    "Native1_exact": true,
    "same_batch_raw": {
      "max_relative": 0.0,
      "max_absolute": 0.0,
      "RMS": 0.0,
      "relative_L2": 0.0,
      "exact": true
    },
    "first_clip_independence_raw": {
      "max_relative": 0.0,
      "max_absolute": 0.0,
      "RMS": 0.0,
      "relative_L2": 0.0,
      "exact": true
    },
    "single_versus_batch_raw": {
      "max_relative": 0.20166015625,
      "max_absolute": 4.875,
      "RMS": 0.011360723621234872,
      "relative_L2": 0.0019305757787645322,
      "exact": false
    },
    "same_batch_projected": {
      "max_relative": 0.0,
      "max_absolute": 0.0,
      "RMS": 0.0,
      "relative_L2": 0.0,
      "exact": true
    },
    "first_clip_independence_projected": {
      "max_relative": 0.0,
      "max_absolute": 0.0,
      "RMS": 0.0,
      "relative_L2": 0.0,
      "exact": true
    },
    "single_versus_batch_projected": {
      "max_relative": 0.1318359375,
      "max_absolute": 0.3671875,
      "RMS": 0.0062427391092100635,
      "relative_L2": 0.0019283603004220215,
      "exact": false
    }
  },
  "original_R30_qualification_passed": false,
  "current_comparison_with_original_bound_passed": false,
  "old_batch_centers_reproduced": true,
  "old_single_reproduced": true,
  "task_predictions": 0,
  "GT_read": false,
  "actual_calls": {
    "motion_attempts": 6,
    "motion_completed": 6,
    "projection_attempts": 6,
    "projection_completed": 6
  },
  "cost": {
    "GPU_seconds": 12.888298632809892,
    "measured_lease_seconds": 9.888298632809892,
    "cleanup_reserve_seconds": 3,
    "actual_calls": {
      "motion_attempts": 6,
      "motion_completed": 6,
      "projection_attempts": 6,
      "projection_completed": 6
    },
    "success": true,
    "GPU_peak_allocated_bytes": 4534534656,
    "GPU_peak_reserved_bytes": 5398069248,
    "execution_completed": true,
    "not_task_qualification": true,
    "new_experts": 0,
    "new_training": 0,
    "spatial_updates": 0,
    "student_updates": 0
  }
}
```

The original R30 max-error qualification remains failed. Same-shape repeatability and first-clip invariance under permutation of the other seven clips support these checked engineering properties only. They do not prove exhaustive sample independence, numerical root cause, harmless downstream error, teacher quality or temporal gain. Raw stages distinguish where the batch/single discrepancy is present, without a precision/kernel/batch sweep. No automatic old R30 resume or tolerance change. Independent confirmation and OPD remain untested.
