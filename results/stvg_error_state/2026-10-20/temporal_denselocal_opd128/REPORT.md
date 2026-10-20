# Pure DenseLocal driven temporal OPD: actual HC128 experiment

Only supervision changes from the existing Strong-start temporal OPD. Initial B=exact saved HiResAuxMix; T=exact pure DenseLocal final-layer posterior, no new mixture or teacher. Actual130-dimensional temporary state,40 Adam0.1 updates,32 intervals per stochastic step,seeds41/42/43. Fixed freshly samples B each step; Live freshly samples CURRENT p_eta and queries the fixed teacher conditional tables at its sampled boundaries. Original fitter, loss, optimizer and finalMBR reused unchanged.

| Output | full vIoU percent | tIoU percent | boundaryMAE frames |
|---|---:|---:|---:|
| Native | 27.928474 | 51.439514 | 77.238281 |
| Strong | 29.534908 | 55.920522 | 70.367188 |
| TeacherDirect | 28.738379 | 53.792321 | 76.757812 |
| DenseLocalMix | 28.901490 | 53.597804 | 74.714844 |
| DirectJoint | 28.754328 | 53.781354 | 76.792969 |
| DirectConditional | 28.746162 | 53.757570 | 76.789062 |
| Fixed | 28.787339 | 53.654366 | 76.174479 |
| Live | 28.762386 | 53.785570 | 76.709635 |

| Comparison | full-v pp [CI95] | tIoU pp [CI95] | losses below-5pp |
|---|---|---|---:|
| Mean_Live_minus_Mean_Strong | -0.772521 [-2.219428943421424, 0.6671980943984536] | -2.134951 [-4.946140976871118, 0.6346396866871085] | 27 |
| Mean_Live_minus_Mean_Native | +0.833913 [-0.8763785968691581, 2.4956193379748988] | +2.346056 [-0.46434585649166354, 5.087553215138674] | 24 |
| Mean_Live_minus_Mean_TeacherDirect | +0.024008 [-0.03696130316755833, 0.09752993427873874] | -0.006751 [-0.11348138525904138, 0.10132638360138205] | 0 |
| Mean_Live_minus_Mean_DenseLocalMix | -0.139104 [-0.8516213106394953, 0.5340997605206586] | +0.187766 [-1.0324729598775715, 1.4165061153363128] | 14 |
| Mean_Live_minus_Mean_DirectJoint | +0.008059 [-0.04408153856258868, 0.07711257948117996] | +0.004216 [-0.06959452140537595, 0.08915343866852729] | 0 |
| Mean_Live_minus_Mean_DirectConditional | +0.016224 [-0.03538022066725351, 0.06687684118056585] | +0.028000 [-0.05279565170596042, 0.1139857421727535] | 0 |
| Mean_Live_minus_Mean_Fixed | -0.024952 [-0.4111533211789124, 0.2620463747417726] | +0.131204 [-0.42354486114281525, 0.5762879906942615] | 1 |
| Mean_DirectJoint_minus_Mean_Strong | -0.780580 [-2.2586145152211903, 0.6777384107552292] | -2.139168 [-4.970344719440893, 0.6421049934662355] | 27 |
| Mean_DirectConditional_minus_Mean_Strong | -0.788746 [-2.235313829213991, 0.649658921249071] | -2.162952 [-4.955691525387357, 0.5968153621799788] | 27 |
| Mean_Fixed_minus_Mean_Strong | -0.747569 [-2.260535730159965, 0.747983356821047] | -2.266156 [-5.174490890365002, 0.5640400996864492] | 27 |
| Live41_Live41_minus_Live41_Fixed41 | -0.044342 [-0.4343194064594881, 0.24626916970391355] | +0.156657 [-0.4121750071579185, 0.616596726824217] | 1 |
| Live42_Live42_minus_Live42_Fixed42 | -0.012752 [-0.3974784829981668, 0.2773965400402915] | +0.120419 [-0.4331780769335967, 0.5721900176541763] | 1 |
| Live43_Live43_minus_Live43_Fixed43 | -0.017763 [-0.404602634818272, 0.28336960280133267] | +0.116536 [-0.44334192116429977, 0.5804795220777106] | 1 |

All128 development parents included, repeatedly exposed; seed paired temporal/spatial then mean within parent. Static/Direct time outputs average the same three frozen spatial trajectories. Other crossseed/E values retained and never selected. No independent generalization claim; no hard-readout/previous diagnostic prerequisite gate. Fixed/Live equal update/candidate budget;32 sampled intervals each ask TWO full teacher conditional vectors (up to65 classes), not32 scalar rewards or teacher network forwards. Teacher conditional labels are derived posterior tables, not native autoregressive network predictions.

```json
{
  "primary_net_over_current_unadapted_Strong": {
    "mean": -0.7725214708839447,
    "CI95": [
      -2.219428943421424,
      0.6671980943984536
    ],
    "parents": 128
  },
  "versus_same_teacher_controls": {
    "TeacherDirect": {
      "mean": 0.024007544555787665,
      "CI95": [
        -0.03696130316755833,
        0.09752993427873874
      ],
      "parents": 128
    },
    "DirectJoint": {
      "mean": 0.008058852658588488,
      "CI95": [
        -0.04408153856258868,
        0.07711257948117996
      ],
      "parents": 128
    },
    "DirectConditional": {
      "mean": 0.01622415491506305,
      "CI95": [
        -0.03538022066725351,
        0.06687684118056585
      ],
      "parents": 128
    },
    "Fixed": {
      "mean": -0.024952356984206572,
      "CI95": [
        -0.4111533211789124,
        0.2620463747417726
      ],
      "parents": 128
    }
  },
  "point_gain_over_Strong": false,
  "net_gain_interval_above_zero": false,
  "independent_Live_value_intervals_above_zero": false,
  "actual_OPD_executed": true,
  "not_confirmation": true,
  "no_parameter_search": true,
  "no_automatic_next_experiment": true,
  "scope": "one unchanged Strong-start conditional OPD configuration; only teacher replaced, no universal OPD verdict"
}
{
  "success": true,
  "CPU_prediction_and_updates_seconds": 48.13839913206175,
  "actual_fits": 1024,
  "fit_attempts": 1024,
  "actual_updates": 40960,
  "completed_trace_steps": 40960,
  "completed_fit_updates": 40960,
  "scheduled_fits": 1024,
  "scheduled_updates": 40960,
  "first_Fixed_Live_exact": 384,
  "max_manual_gradient_error": 4.3151246464923076e-16,
  "temporary_parameters_per_fit": 130,
  "GPU_seconds": 0,
  "new_RGB": 0,
  "new_model_forward": 0,
  "new_Source_encoder": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "spatial_updates": 0,
  "upstream_cache_generation_not_free_deployment": true,
  "CPU_audit_and_scoring_seconds": 68.1919718971476,
  "publication_and_archive_excluded": true,
  "upstream_cache_cost": {
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
}
{
  "status": "pass",
  "independent_numpy_gradient_Adam_sampling_steps": 40960,
  "maxerror": 6.217248937900877e-15,
  "actual_fits": 1024,
  "changed_posterior_fits": 1024,
  "first_Fixed_Live_exact": 384,
  "Live_steps_differ_from_initial_sampler": 14976,
  "source_initial_and_teacher_exact": 256,
  "spatial_arrays_exact": 512,
  "independent_MBR_sample_utilities": 21489,
  "max_direct_utility_error": 8.881784197001252e-16,
  "actual_updates": 40960,
  "new_model_or_GT": false,
  "new_target_GT_scored": true,
  "official_DenseTube_checks": 6144,
  "previous_static_metrics_exact": 2048,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true
}
```

All predictions, actual samples/teacher digests/gradients/eta chains sealed before GT; independent NumPy audit reconstructs the existing gradient and Adam/sampling algorithm. Spatial arrays and fixed unobserved actual-node sIoU identical. Network weights never loaded or updated; CPU probability-state adaptation is real, with perquery/fit optimizer reset. Complete signed means, each seed, no-change cases and severe tails retained. No new RGB, GPU, models or experts. CPU adaptation latency excludes prior1531.960485s multi-arm Dense-cache generation and other upstream Strong/spatial costs; it is not an end-to-end deployment latency. No scan, automatic confirmation, new route or registry/spatial promotion follows this result.

## Measured interpretation and final aggregation audit

Actual Live learns the cached DenseLocal distribution: average joint KL(T||student) starts0.879460 and ends0.031400/0.031484/0.031530 for seeds41/42/43; this is not proof of better event localization. Live full-v28.762386 is near pure TeacherDirect28.738379 and DirectJoint28.754328, below Strong29.534908 on the mean. Live−Strong56positive/68negative/4flat,37losses below3pp and27below5pp; all retained. Live−Fixed is negative on each of the three seed point estimates, with all paired intervals crossing zero. Current one unchanged configuration did not establish net temporal gain or independent Live advantage; no parameter rescue/confirmation follows and no universal OPD claim is made.

Independent ROOT_RECOMPUTE.py checked5360 saved-score scalars,52 paired bootstrap outputs, all signed tails/worst/LOO and within-parent seed aggregation; maximum error4.45e-16, no new GT/model/fitting. The final audit originally carried the preaudit-only field new_model_or_GT=False alongside new_target_GT_scored=True. AUDIT_PHASE_LABEL_CLARIFICATION001_PUBLIC preserves original bytes by SHA and the full original metadata, renames the preaudit field explicitly and marks final new_GT_scoring=True. Original pinned scorer/optimizer/predictions/results remain unchanged.
