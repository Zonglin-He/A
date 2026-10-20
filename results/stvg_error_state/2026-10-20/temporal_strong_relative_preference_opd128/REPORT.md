# Strong-relative DenseLocal preference OPD: actual HC128 experiment
The completed configuration does **not establish the requested2–3pp gain**. Live−Strong full-v is +0.058941pp,95%CI[−0.213453,+0.330452]; tIoU−0.509468pp,CI crosses0. Live−Direct+0.000419pp and Live−Fixed−0.032829pp, both CIs cross0. Fixed has the largest point estimate (+0.091769pp), also uncertain. No independent on-policy advantage is established. Strong stays unchanged.

Relative to Strong, Live has37 positive/32 negative/59 unchanged parent means;6 lose more than3pp and1 more than5pp (RC079−6.412825pp). The previous whole-teacher Live had27 below−5pp; this difference concerns the combined new design, not an isolated causal attribution to filtering or KL. All1152 posterior fits actually changed. Live admits51.5511% of comparisons and touches447.62 distinct admitted intervals per fit on average, so this is not simply zero supervision or no updates. Positive-support policy size1953 versus2080 allowed final decisions is preserved.

One user-revised configuration: strict four-sign agreement, t=sigmoid(weakest signed log preference), binary admitted w=1; otherwise abstain. Joint interval residual r with continuing KL(pi||Strong), coefficient1;40 Adam0.1 steps,32 pairs per stochastic step, seeds41/42/43. No new models, visual observations, experts or neural training. Local B_o is a reference recipe, NOT the marginal of global Strong. Both offsets are correlated.

| Output | full vIoU percent | tIoU percent | boundary MAE frames |
|---|---:|---:|---:|
| Native | 27.928474 | 51.439514 | 77.238281 |
| Strong | 29.534908 | 55.920522 | 70.367188 |
| TeacherDirect | 28.738379 | 53.792321 | 76.757812 |
| DenseLocalMix | 28.901490 | 53.597804 | 74.714844 |
| Direct | 29.593430 | 55.431422 | 70.186198 |
| Fixed | 29.626677 | 55.446474 | 70.117188 |
| Live | 29.593849 | 55.411054 | 70.147135 |
| WholeDenseLive | 28.762386 | 53.785570 | 76.709635 |

| Comparison | full-v pp [CI95] | tIoU pp [CI95] | losses below-5pp |
|---|---|---|---:|
| Mean_Direct_minus_Mean_Strong | +0.058522 [-0.21196216543508606, 0.3264836813183349] | -0.489100 [-1.3800423142584102, 0.19941038135961267] | 2 |
| Mean_Fixed_minus_Mean_Strong | +0.091769 [-0.17374087469693056, 0.36081797269780913] | -0.474048 [-1.3785460741965956, 0.22999667029117798] | 1 |
| Mean_Live_minus_Mean_Strong | +0.058941 [-0.21345309062151321, 0.33045221071867137] | -0.509468 [-1.4208340651770586, 0.18677739796773207] | 1 |
| Mean_Live_minus_Mean_Direct | +0.000419 [-0.07775181497555593, 0.07173337532292112] | -0.020368 [-0.1376968535605335, 0.09096120776378436] | 0 |
| Mean_Live_minus_Mean_Fixed | -0.032829 [-0.08560501502257609, 0.013280980467143657] | -0.035420 [-0.11096141535747674, 0.03608291589781148] | 0 |
| Mean_Live_minus_Mean_Native | +1.665375 [0.6955910779280892, 2.693699904149568] | +3.971540 [2.1823337731669876, 5.854100442516427] | 9 |
| Mean_Live_minus_Mean_TeacherDirect | +0.855470 [-0.5006883951407014, 2.2149512187912155] | +1.618733 [-0.8763701281800416, 4.144253115775156] | 25 |
| Mean_Live_minus_Mean_WholeDenseLive | +0.831462 [-0.4905795235302656, 2.1611632149368645] | +1.625483 [-0.8433084852880399, 4.106932828810977] | 24 |
| Live41_Live41_minus_Live41_Fixed41 | -0.024576 [-0.08371223699719545, 0.02840085167457745] | -0.038247 [-0.15555475333051158, 0.06524340217623467] | 0 |
| Live42_Live42_minus_Live42_Fixed42 | -0.009074 [-0.11625960075636443, 0.08509435054537134] | +0.004362 [-0.15165551364010368, 0.16021889780584794] | 0 |
| Live43_Live43_minus_Live43_Fixed43 | -0.064837 [-0.17506582239217794, 0.011393478126746357] | -0.072375 [-0.21847934461446653, 0.044046256307174896] | 1 |

Method net gain and independent on-policy contribution are assessed separately. No Direct/Fixed gain is erased because Live does not beat it. Goal +2–3pp is relative to Strong, not Native; no independent confirmation or universal OPD verdict. All128 parents retained; temporal/spatial seeds paired then averaged within each parent. Repeatedly exposed development, not384 independent examples. Spatial arrays and actual unobserved-node sIoU identical across temporal arms.

Fixed draws fresh BxB pairs per step; Live draws I from current student and J from B using paired uniforms. Direct uses the same1280-pair Fixed bank every step and has larger pair arithmetic; it is not compute-FLOP matched. All outputs use unchanged2080-action MBR. Only B-positive support adapted; no epsilon support. Teacher targets, mappings, sampling and candidate coordinates detached.

Process (mean per fit; Direct revisits full bank each step):
```json
{
  "Direct": {
    "admission_fraction": 0.5218017578125,
    "unique_sampled_intervals": 598.9036458333334,
    "unique_admitted_intervals": 447.7291666666667,
    "initial_preference_KL": 0.07499677584872425,
    "final_preupdate_preference_KL": 0.02335010848252858,
    "initial_global_KL": 0.0,
    "final_preupdate_global_KL": 0.01715558278488608,
    "eta_norm": 13.022131689394536,
    "posterior_L1": 0.1013038447889958
  },
  "Fixed": {
    "admission_fraction": 0.5218017578125,
    "unique_sampled_intervals": 598.9036458333334,
    "unique_admitted_intervals": 447.7291666666667,
    "initial_preference_KL": 0.07390732257883652,
    "final_preupdate_preference_KL": 0.04483148319521032,
    "initial_global_KL": 0.0,
    "final_preupdate_global_KL": 0.02223300455004203,
    "eta_norm": 8.833306810681462,
    "posterior_L1": 0.13435262031810827
  },
  "Live": {
    "admission_fraction": 0.5155110677083333,
    "unique_sampled_intervals": 599.21875,
    "unique_admitted_intervals": 447.6223958333333,
    "initial_preference_KL": 0.07390732257883652,
    "final_preupdate_preference_KL": 0.04303136343487594,
    "initial_global_KL": 0.0,
    "final_preupdate_global_KL": 0.021848347290224644,
    "eta_norm": 8.706530274222663,
    "posterior_L1": 0.1338328916512199
  }
}
{
  "method_net_over_Strong": {
    "Direct": {
      "mean": 0.05852198024405982,
      "CI95": [
        -0.21196216543508606,
        0.3264836813183349
      ],
      "parents": 128
    },
    "Fixed": {
      "mean": 0.09176938670813194,
      "CI95": [
        -0.17374087469693056,
        0.36081797269780913
      ],
      "parents": 128
    },
    "Live": {
      "mean": 0.05894061598530293,
      "CI95": [
        -0.21345309062151321,
        0.33045221071867137
      ],
      "parents": 128
    }
  },
  "Live_independent_value": {
    "Direct": {
      "mean": 0.0004186357412431056,
      "CI95": [
        -0.07775181497555593,
        0.07173337532292112
      ],
      "parents": 128
    },
    "Fixed": {
      "mean": -0.03282877072282902,
      "CI95": [
        -0.08560501502257609,
        0.013280980467143657
      ],
      "parents": 128
    }
  },
  "method_point_gain": {
    "Direct": true,
    "Fixed": true,
    "Live": true
  },
  "method_CI_above_zero": {
    "Direct": false,
    "Fixed": false,
    "Live": false
  },
  "method_two_pp_goal": {
    "Direct": false,
    "Fixed": false,
    "Live": false
  },
  "on_policy_CI_above_both_controls": false,
  "actual_OPD_executed": true,
  "not_confirmation": true,
  "no_parameter_search": true,
  "no_automatic_next_experiment": true,
  "no_registry_promotion": true,
  "scope": "one binary-weight cross-offset preference and continuing Strong KL configuration; method gain is distinct from on-policy gain"
}
{
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
  "Direct_pair_arithmetic_larger_than_Fixed_Live": true,
  "CPU_audit_and_scoring_seconds": 175.17414302914403,
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
  },
  "publication_and_archive_excluded": true
}
{
  "status": "pass",
  "independent_numpy_steps": 46080,
  "maxerror": 3.552713678800501e-15,
  "independent_native_logp_and_physical_mapping": true,
  "first_Fixed_Live_exact": 384,
  "spatial_arrays_exact": 512,
  "independent_MBR_utilities": 16638,
  "max_direct_utility_error": 6.661338147750939e-16,
  "pre_GT_audit_new_model_or_GT": false,
  "new_target_GT_scored": true,
  "official_DenseTube_checks": 6656,
  "previous_static_metrics_exact": 2048,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true
}
```

Predictions and complete actual samples/feedback/gradients/Adam chains sealed before GT, then independent NumPy reconstruction of logits, physical mapping, strict signs, losses and every update; official task metrics cross-checked. New CPU timing excludes upstream cache generation, spatial and Source inference and is not deployment latency. Complete signed query results, negative tails and all three seeds retained. No scan, automatic next round, or CURRENT/MAIN/PAPER promotion.
