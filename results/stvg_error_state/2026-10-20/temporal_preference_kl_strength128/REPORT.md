# Finite Strong KL protection-strength verification: full HC128
**Result:** Reducing protection did not produce the requested2–3pp or an established positive increment over Strong. Lambda1 remains the maximum development point estimate (+0.058941pp); lambda0.3 gives+0.030064pp and0.1 gives+0.012780pp, all95%intervals cross0. Differences versus1 also cross0; this does not prove1 globally optimal.

Relaxing1 to0.3 increases positive contribution+0.171849pp and adds negative contribution−0.200726pp, net−0.028877pp. Relaxing1 to0.1 increases positive contribution+0.273795pp and adds negative contribution−0.319956pp, net−0.046161pp. Paired descriptive contribution intervals are in CONTRIBUTION_INTERVALS_PUBLIC.json. This is the predicted expansion of both helpful and harmful corrections, not evidence of sufficiently selective task improvement. Below−5pp parents increase from1 to5 at either weaker strength. Raw policy KL before the last update rises.021848→.048976/.071957 and final posterior L1.133833→.216679/.276735, with nearly unchanged admission fractions, confirming effective strength changed.

Two engineering issues are retained: (1) exact first-step float assertion failed before GT, then mathematically equal steps were checked with declared strict tolerances without changing updates; initial failed1fit/40updates7.229496s is separate from768validfits/30720updates119.735569s. (2) All scores and audits completed, but final cost export requested the wrong prior cost filename. EXPORT_RECOVERY002 regenerated only cost/report/provenance from saved scores, without new GT or fitting. Original perf_counter elapsed was not saved; the whole scorer invocation wall interval to export failure is153.199357s, reconstructed from the log filesystem birth and failure timestamp (includes imports; not the same timing definition as the original intended perf_counter). No task scores changed, and original errors/source are preserved.

Only the continuous global Strong KL coefficient changes. Native offset evidence, binary agreement target, learner initialization/state, sampling/seed streams, optimizer40Adam0.1, spatial arrays and final MBR are frozen. New actual Live fits at0.1/0.3 for128parents x3seeds;1 is reused from the completed previous experiment. This is development sensitivity/selection, not confirmation. No new Direct/Fixed experiment and no claim of independent on-policy improvement at newly selected strength.

| Coefficient | complete vIoU percent | tIoU percent | full-v delta vs Strong pp [CI95] | positive contribution pp | negative contribution pp | below-5pp parents |
|---|---:|---:|---|---:|---:|---:|
| Strong | 29.534908 | 55.920522 | reference | - | - | - |
| 1.0 | 29.593849 | 55.411054 | +0.058941 [-0.21345309062151321, 0.33045221071867137] | +0.466339 | -0.407398 | 1 |
| 0.3 | 29.564972 | 55.162517 | +0.030064 [-0.3285072627034055, 0.41333666216175163] | +0.638187 | -0.608124 | 5 |
| 0.1 | 29.547688 | 55.115751 | +0.012780 [-0.3839352628090536, 0.4394038058512199] | +0.740134 | -0.727354 | 5 |

| Comparison | complete-v pp [CI95] | tIoU pp [CI95] |
|---|---|---|
| KL0p1 minus KL1 | -0.046161 [-0.2615788073500243, 0.1787958083883053] | -0.295303 [-0.7873363517376767, 0.1769246934932579] |
| KL0p3 minus KL1 | -0.028877 [-0.1887628145084629, 0.14777386408699777] | -0.248537 [-0.5875144384908689, 0.083731466347827] |

All128 parents retained, including no-change/time-no-overlap cases. Match temporal/spatial seeds then mean within parent; bootstrap parents10000times, not384 independent videos. Fixed-unobserved actual-node sIoU and all512 spatial arrays remain exact. Posthoc sign groups never choose output/feedback/parameters online. Development selection maximizes complete-v only, exact ties choose stronger protection; this does not promote a method or authorize more coefficients/confirmation.

Process and contribution expansion:
```json
{
  "KL0p1": {
    "admission_fraction": 0.51220703125,
    "unique_sampled_intervals": 601.421875,
    "unique_admitted_intervals": 449.6927083333333,
    "initial_preference_KL": 0.07390732257883652,
    "final_preupdate_preference_KL": 0.03466933607222627,
    "final_preupdate_global_KL": 0.07195699956822671,
    "final_preupdate_weighted_global_KL": 0.007195699956822672,
    "eta_norm": 14.09490260706734,
    "posterior_L1": 0.2767353422118402
  },
  "KL0p3": {
    "admission_fraction": 0.5135233561197917,
    "unique_sampled_intervals": 600.2890625,
    "unique_admitted_intervals": 448.6927083333333,
    "initial_preference_KL": 0.07390732257883652,
    "final_preupdate_preference_KL": 0.034493938647592266,
    "final_preupdate_global_KL": 0.04897585280221208,
    "final_preupdate_weighted_global_KL": 0.014692755840663624,
    "eta_norm": 12.091926904594857,
    "posterior_L1": 0.21667869797292902
  },
  "KL1": {
    "admission_fraction": 0.5155110677083333,
    "unique_sampled_intervals": 599.21875,
    "unique_admitted_intervals": 447.6223958333333,
    "initial_preference_KL": 0.07390732257883652,
    "final_preupdate_preference_KL": 0.04303136343487594,
    "final_preupdate_global_KL": 0.021848347290224644,
    "eta_norm": 8.706530274222663,
    "posterior_L1": 0.1338328916512199,
    "final_preupdate_weighted_global_KL": 0.021848347290224644
  }
}
{
  "KL0p1": {
    "groups": {
      "coefficient1_positive": {
        "parents": 37,
        "coefficient1_contribution_to_all128_pp": 0.46633865585492223,
        "new_contribution_to_all128_pp": 0.669015330538078,
        "new_positive": 35,
        "new_negative": 1,
        "new_flat": 1
      },
      "coefficient1_negative": {
        "parents": 32,
        "coefficient1_contribution_to_all128_pp": -0.4073980398696192,
        "new_contribution_to_all128_pp": -0.5613615938707296,
        "new_positive": 1,
        "new_negative": 30,
        "new_flat": 1
      },
      "coefficient1_flat": {
        "parents": 59,
        "coefficient1_contribution_to_all128_pp": 0.0,
        "new_contribution_to_all128_pp": -0.09487403392308011,
        "new_positive": 13,
        "new_negative": 13,
        "new_flat": 33
      }
    },
    "positive_contribution_increase_pp": 0.27379548358492023,
    "negative_contribution_change_pp": -0.3199563968259549,
    "net_change_pp": -0.04616091324103462,
    "posthoc_GT_explanation_only": true
  },
  "KL0p3": {
    "groups": {
      "coefficient1_positive": {
        "parents": 37,
        "coefficient1_contribution_to_all128_pp": 0.46633865585492223,
        "new_contribution_to_all128_pp": 0.6004814002156016,
        "new_positive": 37,
        "new_negative": 0,
        "new_flat": 0
      },
      "coefficient1_negative": {
        "parents": 32,
        "coefficient1_contribution_to_all128_pp": -0.4073980398696192,
        "new_contribution_to_all128_pp": -0.51477668314837,
        "new_positive": 0,
        "new_negative": 31,
        "new_flat": 1
      },
      "coefficient1_flat": {
        "parents": 59,
        "coefficient1_contribution_to_all128_pp": 0.0,
        "new_contribution_to_all128_pp": -0.05564113120489026,
        "new_positive": 6,
        "new_negative": 12,
        "new_flat": 41
      }
    },
    "positive_contribution_increase_pp": 0.17184867756071104,
    "negative_contribution_change_pp": -0.2007257076836727,
    "net_change_pp": -0.028877030122961694,
    "posthoc_GT_explanation_only": true
  }
}
{
  "development_selection_only": "KL1",
  "selection_rule": "max parent-mean complete vIoU, ties stronger protection1 then.3 then.1; development only",
  "method_net_over_Strong": {
    "KL0p1": {
      "mean": 0.012779702744268366,
      "CI95": [
        -0.3839352628090536,
        0.4394038058512199
      ],
      "parents": 128
    },
    "KL0p3": {
      "mean": 0.03006358586234127,
      "CI95": [
        -0.3285072627034055,
        0.41333666216175163
      ],
      "parents": 128
    },
    "KL1": {
      "mean": 0.05894061598530293,
      "CI95": [
        -0.21345309062151321,
        0.33045221071867137
      ],
      "parents": 128
    }
  },
  "selected_point_gain_at_least2pp": false,
  "selected_gain_CI_above_zero": false,
  "no_new_independent_on_policy_contribution_claim": true,
  "new_Direct_or_Fixed_controls": false,
  "no_promotion": true,
  "no_automatic_next_experiment": true,
  "no_coefficients_beyond_predeclared_three": true,
  "not_confirmation": true,
  "scope": "only protection-strength range0.1/0.3/1, all other algorithm choices fixed; contribution expansion examined, not universal OPD verdict"
}
{
  "success": true,
  "CPU_prediction_and_updates_seconds": 119.7355688509997,
  "actual_fits": 768,
  "actual_updates": 30720,
  "coefficient_fit_CPU_seconds": {
    "KL0p1": 11.916114561958238,
    "KL0p3": 11.943171369377524
  },
  "scheduled_fits": 768,
  "scheduled_updates": 30720,
  "reused_coefficient1_fits": 384,
  "first_step_equals_saved_coefficient1": 768,
  "coefficient1_nonzero_state_objective_exact": 384,
  "first_gradient_maxerror": 3.8163916471489756e-17,
  "first_state_maxerror": 3.7813901394698533e-10,
  "initial_failed_attempt_cost": {
    "success": false,
    "CPU_prediction_and_updates_seconds": 7.2294962098822,
    "actual_fits": 1,
    "actual_updates": 40,
    "coefficient_fit_CPU_seconds": {
      "KL0p1": 0.0315736448392272,
      "KL0p3": 0.0
    },
    "scheduled_fits": 768,
    "scheduled_updates": 30720,
    "reused_coefficient1_fits": 384,
    "first_step_equals_saved_coefficient1": 0,
    "coefficient1_nonzero_state_objective_exact": 3,
    "GPU_seconds": 0,
    "new_RGB": 0,
    "new_model_forward": 0,
    "new_DINO": 0,
    "new_ReferDINO": 0,
    "new_network_training": 0,
    "spatial_updates": 0,
    "upstream_cache_cost_not_free": true
  },
  "GPU_seconds": 0,
  "new_RGB": 0,
  "new_model_forward": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "spatial_updates": 0,
  "upstream_cache_cost_not_free": true,
  "CPU_audit_and_scoring_seconds": null,
  "CPU_audit_score_invocation_wall_seconds_from_filesystem": 153.19935703277588,
  "wall_timing_source": {
    "log_birth": "2026-10-20 11:06:54.371904981 +0800",
    "export_failure_epoch": 1792465767.571261,
    "scope": "whole scorer invocation to export failure, includes Python imports, not original perf_counter interval"
  },
  "scorer_process_exit_code": 1,
  "all_scientific_scores_and_audits_completed_before_export_error": true,
  "prior_coefficient1_experiment_cost": {
    "success": true,
    "CPU_prediction_and_updates_seconds": 111.01661336910911,
    "actual_fits": 1152,
    "actual_updates": 46080,
    "mode_fit_CPU_seconds": {
      "Direct": 12.137743700528517,
      "Fixed": 11.038785012671724,
      "Live": 11.135764163220301
    },
    "scheduled_fits": 1152,
    "scheduled_updates": 46080,
    "first_Fixed_Live_exact": 384,
    "GPU_seconds": 0,
    "new_RGB": 0,
    "new_model_forward": 0,
    "new_DINO": 0,
    "new_ReferDINO": 0,
    "new_network_training": 0,
    "spatial_updates": 0,
    "upstream_cache_generation_not_free_deployment": true,
    "Direct_pair_arithmetic_larger_than_Fixed_Live": true
  },
  "original_source_cache_generation_cost": {
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
  },
  "publication_archive_excluded": true
}
{
  "status": "pass",
  "independent_numpy_weighted_gradient_Adam_steps": 30720,
  "maxerror": 1.9984014443252818e-15,
  "source_teacher_evidence_and_mapping_byte_exact": true,
  "original_source_evidence_validation": "reused prior independently validated sealed native-logit evidence; not re-encoded",
  "first_step_matches_saved_coefficient1_with_declared_tolerance": 768,
  "spatial_arrays_exact": 512,
  "reused_Strong_and_coefficient1_posteriors_exact": 512,
  "independent_MBR_utilities": 12797,
  "max_direct_utility_error": 6.661338147750939e-16,
  "pre_GT_audit_new_model_or_GT": false,
  "new_target_GT_scored": true,
  "official_DenseTube_checks": 5120,
  "previous_baseline_metrics_exact": 2048,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true
}
```

All new predictions/traces sealed before new GT; independent audit verifies every weighted gradient, sampled pair/label, Adam update and MBR decision. Raw0.1/0.3 signed outputs, all seeds, negative tails, worst cases and leave-one-out retained. Zero new experts/models/checkpoints/RGB/visual forwards/GPU/neural weight training or spatial updates. CPU-only cached adaptation/scoring time excludes existing source/expert cache generation and is not end-to-end deployment latency. Both offsets remain correlated and B_o not a marginal of global Strong. No follow-up search or registry promotion.
