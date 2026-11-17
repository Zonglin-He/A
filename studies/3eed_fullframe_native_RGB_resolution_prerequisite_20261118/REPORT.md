# R152 — Full-frame native RGB resolution prerequisite

## Decision

**stop_fullframe2x_recipe_no_new_TTA**. The predeclared full-frame fresh-detail gate fails. No new source calibration or parameter adaptation is run. Keep the prior R145 development candidate (13.819603% versus intact Frozen-A 13.394463%, +0.425140 pp); the requested >=2 pp improvement is NOT achieved.

## Input mechanism and controls

Full original A and registered B images are both resized to 768x512, producing 96x64=6144 native CNN tokens per bank and 12288 total. There is no candidate-dependent crop selection. Original physical visibility, point positions, fused 1024 point features, normalized position references, original A256 and all source-prepared weights remain fixed. The native model is fully convolutional on images; no image encoder weights, extra semantic model, source training or new architecture are added.

The low-detail control first downsizes each original frame to 384x256, then upsizes it to 768x512. It uses exactly the same high-grid graphs as the fresh-pixel reader. This separates genuine additional pixel information from image lattice, spatial neighborhood, normalization and token-budget changes. All original media are at least as large as the high-resolution input.

Point-to-image edges use the original 3x3 grid-neighborhood rule at the new 96x64 lattice. Image-to-point edges use up to eight projected points with the same distance and depth discontinuity criteria. Offsets are derived from each actual image token count, not the old hardcoded1536. The unchanged CNN is evaluated separately per image with GroupNorm, and outputs are concatenated in exactly the same order as graph offsets. No 32x48 image-mask auxiliary task is used. The physical extent of a 3x3 kernel is smaller than before; this is part of the input/grid change and is controlled by the low-detail arm, not claimed to be pure photometric enhancement.

The wrong-binding control deterministically permutes visible points within exact forward-degree buckets independently per camera and inversely remaps image-to-point edges. It preserves images, visibility, lattice and the exact number of forward edges for every point. This changes content/location binding, not just synchronous K/V ordering. Cases with no effective permutation are retained. Original joint zero, R145 TTA and all112 historical arms remain. Fixed half-means with zero or old TTA are ensemble controls, not new parameter adaptation or independent confirmation.

## Results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.40304063926085 | 25.899514674225376 | 5.056806006058658 |
| raw | norm_joint_zero | 13.716845698214941 | 26.56221517419685 | 5.249643946003637 |
| raw | norm_Aimage | 13.661544723534073 | 26.46619191640889 | 5.249643946003637 |
| raw | norm_Bimage | 13.711942296165214 | 26.471140492958238 | 5.48324893792589 |
| raw | norm_TTA | 13.801914656549181 | 26.834112929821703 | 5.528786278545199 |
| raw | full_high | 13.563567239622799 | 26.09120461398668 | 5.20410660538433 |
| raw | full_low | 13.570889375694442 | 26.136741954605984 | 5.20410660538433 |
| raw | full_wrong | 13.768451051332212 | 26.74798682513244 | 5.249643946003637 |
| raw | full_high_zero_mean | 13.661052157357801 | 26.56088888610586 | 5.249643946003637 |
| raw | full_high_parent_mean | 13.650334236203165 | 26.46124333985954 | 5.249643946003637 |
| raw | full_low_parent_mean | 13.652056954215313 | 26.46124333985954 | 5.249643946003637 |
| raw | full_wrong_parent_mean | 13.716664933850794 | 26.50695896838616 | 5.249643946003637 |
| design_weighted | Frozen_A | 13.394463465154152 | 26.26330584715755 | 4.599296733787808 |
| design_weighted | norm_joint_zero | 13.739713257337598 | 27.044149189023084 | 4.793982350866708 |
| design_weighted | norm_Aimage | 13.659170072455359 | 26.859601705238706 | 4.793982350866708 |
| design_weighted | norm_Bimage | 13.733419725521841 | 26.93794211447542 | 5.029399635265411 |
| design_weighted | norm_TTA | 13.81960316688109 | 27.305335032677846 | 5.063686987738404 |
| design_weighted | full_high | 13.559553923524968 | 26.472878647553927 | 4.7484450102474005 |
| design_weighted | full_low | 13.564491905732693 | 26.51841598817323 | 4.7484450102474005 |
| design_weighted | full_wrong | 13.782591074750764 | 27.170129257494608 | 4.793982350866708 |
| design_weighted | full_high_zero_mean | 13.675059465001093 | 27.016978595652944 | 4.793982350866708 |
| design_weighted | full_high_parent_mean | 13.643923936387061 | 26.844212052167695 | 4.793982350866708 |
| design_weighted | full_low_parent_mean | 13.645863805792382 | 26.844212052167695 | 4.793982350866708 |
| design_weighted | full_wrong_parent_mean | 13.710929627362153 | 26.888880038001343 | 4.793982350866708 |

{
  "phase": "frozen_fullframe_2x_RGB_prerequisite",
  "primary": "full_high",
  "source_parameters_updated": 0,
  "FINAL_access": false,
  "target_exposed_development": true,
  "endpoints": {
    "full_high": {
      "comparisons": {
        "raw": {
          "norm_joint_zero": -0.15327845859214229,
          "full_low": -0.007322136071643115,
          "full_wrong": -0.20488381170941317,
          "norm_TTA": -0.2383474169263824,
          "Frozen_A": 0.16052660036194943
        },
        "design_weighted": {
          "norm_joint_zero": -0.1801593338126306,
          "full_low": -0.0049379822077249,
          "full_wrong": -0.22303715122579604,
          "norm_TTA": -0.26004924335612145,
          "Frozen_A": 0.16509045837081615
        }
      }
    },
    "full_high_parent_mean": {
      "comparisons": {
        "raw": {
          "norm_TTA": -0.1515804203460167,
          "full_high_zero_mean": -0.010717921154636656,
          "full_high": 0.08676699658036569,
          "full_low_parent_mean": -0.0017227180121484054,
          "full_wrong_parent_mean": -0.06633069764762922,
          "Frozen_A": 0.24729359694231512
        },
        "design_weighted": {
          "norm_TTA": -0.175679230494028,
          "full_high_zero_mean": -0.03113552861403157,
          "full_high": 0.08437001286209345,
          "full_low_parent_mean": -0.001939869405321204,
          "full_wrong_parent_mean": -0.06700569097509224,
          "Frozen_A": 0.2494604712329096
        }
      }
    },
    "full_low": {
      "comparisons": {
        "raw": {
          "norm_joint_zero": -0.14595632252049917,
          "full_high": 0.007322136071643115,
          "full_wrong": -0.19756167563777005,
          "norm_TTA": -0.23102528085473928,
          "Frozen_A": 0.16784873643359255
        },
        "design_weighted": {
          "norm_joint_zero": -0.1752213516049057,
          "full_high": 0.0049379822077249,
          "full_wrong": -0.21809916901807114,
          "norm_TTA": -0.25511126114839655,
          "Frozen_A": 0.17002844057854105
        }
      }
    }
  },
  "proceed_sourcecal_and_TTA": false,
  "retained_oldTTA_plus_high_ensemble": false,
  "disposition": "stop_fullframe2x_recipe_no_new_TTA"
}

## Positive and negative effects

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| full_high | norm_joint_zero | 30 | 42 | -0.1801593338126292 | 0.018245761786296468 | -0.04124910190536532 | 56 | 69 | 8 |
| full_high | full_low | 8 | 8 | -0.004937982207722801 | 0.009441728502658717 | -0.014600800518350844 | 16 | 16 | 4 |
| full_high | full_wrong | 44 | 64 | -0.22303715122579718 | 0.012009308911707088 | -0.03996118331421444 | 72 | 95 | 9 |
| full_high | norm_TTA | 58 | 76 | -0.2600492433561208 | 0.018245761786296468 | -0.04124910190536532 | 87 | 123 | 10 |
| full_high | Frozen_A | 108 | 106 | 0.16509045837081843 | 0.053304207197498364 | -0.04909164122994028 | 123 | 185 | 19 |
| full_high_parent_mean | norm_TTA | 31 | 31 | -0.17567923049402884 | 0.018186894743547708 | -0.04124910190536532 | 53 | 60 | 9 |
| full_high_parent_mean | full_high_zero_mean | 34 | 24 | -0.03113552861403132 | 0.016561661845913917 | -0.04124910190536532 | 46 | 55 | 6 |
| full_high_parent_mean | full_high | 51 | 28 | 0.08437001286209192 | 0.027732986665551986 | -0.018245761786296468 | 61 | 77 | 7 |
| full_high_parent_mean | full_low_parent_mean | 2 | 3 | -0.0019398694053215178 | 0.0020313566061384864 | -0.0027159078579605408 | 5 | 5 | 3 |
| full_high_parent_mean | full_wrong_parent_mean | 26 | 25 | -0.06700569097509114 | 0.018186894743547708 | -0.026944995983505412 | 40 | 47 | 6 |
| full_high_parent_mean | Frozen_A | 116 | 96 | 0.24946047123291037 | 0.053304207197498364 | -0.04909164122994028 | 117 | 179 | 18 |
| full_low | norm_joint_zero | 28 | 43 | -0.17522135160490646 | 0.018245761786296468 | -0.04124910190536532 | 56 | 67 | 8 |
| full_low | full_high | 8 | 8 | 0.004937982207722801 | 0.014600800518350844 | -0.009441728502658717 | 16 | 16 | 4 |
| full_low | full_wrong | 40 | 60 | -0.21809916901807433 | 0.014600800518350844 | -0.03996118331421444 | 68 | 89 | 9 |
| full_low | norm_TTA | 56 | 77 | -0.255111261148398 | 0.018245761786296468 | -0.04124910190536532 | 87 | 121 | 10 |
| full_low | Frozen_A | 107 | 107 | 0.17002844057854127 | 0.053304207197498364 | -0.04909164122994028 | 123 | 184 | 18 |

The primary gate requires full_high to beat original joint zero, full_low and full_wrong in both raw and design-weighted six-direction means; it does not require all directions to rise. Low detail and full-frame point-binding permutation are matched controls. Parent-TTA ensemble retention is separately gated against parent TTA, high zero-mean, pure high, low and wrong parent means. No per-input, per-platform or per-direction GT choice is made. A failed fixed2x full-frame recipe does not exclude all RGB or TTA methods.

## Measurement, resources and limitations

The scalar checker independently recomputes all saved task values.

All variable-grid geometry and actual raw-pixel provenance are independently checked before any model, across the complete1798 input registry. Producer, independent graph/geometry auditor, CNN engineering auditor, streamed full matching/competition auditor and task scorer are pinned before the first forward. Engineering covers12 inputs,18 new full-reader forwards plus6 original-zero bridge forwards. Actual CNN pre-hooks check every normalized 512x768 RGB tensor and point-adapter pre-hooks check exact frozen F1024. Independent CPU reconstruction regenerates engineering CNN outputs, physical graphs and complete final learned readout. Source and prepared weights stay unchanged.

Formal execution seals chunks of two cases, independently audits native full matching/competition and variable-grid geometry, then deletes only its own new TEMP features/graphs after PASS. Predictions, raw-pixel/input hashes, graph recipes, persistent records and audit receipts remain. Full transformer forward is not independently regenerated on CPU; no backprop occurs in this prerequisite. Formal actual pixel tensors are provenance-checked by hooks, while engineering CNN outputs are separately recomputed on CPU.

All1798 inputs,764 eligible B and1034 no-B completeFrozen-A fallbacks, remain in every denominator. Predictions are sealed and audited before taskGT evaluation. Shared recording/frame/content clusters and all positive/negative results are retained; IoU alone is not an instance-identity annotation. This is exposed development, not new independent validation; no single-scene population CI or per-input/per-direction GT selection is claimed. FINAL is untouched.

Resource profile includes retained engineering CNN arrays, all new predictions and the formal streamed chunk. New model payload cap0.18GiB, free floor12GiB plus1GiB safety. No old data, weights or experiment artifacts are deleted. Cached-pipeline runtime plus CPU audit is not a cold deployment latency benchmark. The fixed09:00 Beijing Nov18 deadline remains in effect.

{
  "decision": {
    "phase": "frozen_fullframe_2x_RGB_prerequisite",
    "primary": "full_high",
    "source_parameters_updated": 0,
    "FINAL_access": false,
    "target_exposed_development": true,
    "endpoints": {
      "full_high": {
        "comparisons": {
          "raw": {
            "norm_joint_zero": -0.15327845859214229,
            "full_low": -0.007322136071643115,
            "full_wrong": -0.20488381170941317,
            "norm_TTA": -0.2383474169263824,
            "Frozen_A": 0.16052660036194943
          },
          "design_weighted": {
            "norm_joint_zero": -0.1801593338126306,
            "full_low": -0.0049379822077249,
            "full_wrong": -0.22303715122579604,
            "norm_TTA": -0.26004924335612145,
            "Frozen_A": 0.16509045837081615
          }
        }
      },
      "full_high_parent_mean": {
        "comparisons": {
          "raw": {
            "norm_TTA": -0.1515804203460167,
            "full_high_zero_mean": -0.010717921154636656,
            "full_high": 0.08676699658036569,
            "full_low_parent_mean": -0.0017227180121484054,
            "full_wrong_parent_mean": -0.06633069764762922,
            "Frozen_A": 0.24729359694231512
          },
          "design_weighted": {
            "norm_TTA": -0.175679230494028,
            "full_high_zero_mean": -0.03113552861403157,
            "full_high": 0.08437001286209345,
            "full_low_parent_mean": -0.001939869405321204,
            "full_wrong_parent_mean": -0.06700569097509224,
            "Frozen_A": 0.2494604712329096
          }
        }
      },
      "full_low": {
        "comparisons": {
          "raw": {
            "norm_joint_zero": -0.14595632252049917,
            "full_high": 0.007322136071643115,
            "full_wrong": -0.19756167563777005,
            "norm_TTA": -0.23102528085473928,
            "Frozen_A": 0.16784873643359255
          },
          "design_weighted": {
            "norm_joint_zero": -0.1752213516049057,
            "full_high": 0.0049379822077249,
            "full_wrong": -0.21809916901807114,
            "norm_TTA": -0.25511126114839655,
            "Frozen_A": 0.17002844057854105
          }
        }
      }
    },
    "proceed_sourcecal_and_TTA": false,
    "retained_oldTTA_plus_high_ensemble": false,
    "disposition": "stop_fullframe2x_recipe_no_new_TTA"
  },
  "preparation": {
    "status": "PASS",
    "n": 1798,
    "B": 764,
    "max_graph_weight_error": 0.0,
    "high_image_size": [
      768,
      512
    ],
    "per_bank_tokens": 6144,
    "total_AB_tokens": 12288,
    "independent_camera_grid_reconstruction": true,
    "GT_read": false,
    "models_run": false
  },
  "native_audit": {
    "status": "PASS",
    "n": 1798,
    "new_reader_forwards": 2292,
    "max_CPU_probability_error": 8.100316521431061e-07,
    "max_graph_weight_error": 0.0,
    "max_engineering_CNN_error": 0.0,
    "all_native_chunks_verified_before_temporary_deletion": true,
    "source_parameters_updated": 0,
    "raw_pixel_SHA_and_actual_CNN_input_provenance": true,
    "fixed_1024point_F_input_verified": true,
    "engineering_CNN_regenerated_CPU": false,
    "full_transformer_not_CPU_regenerated": true,
    "FINAL_access": false
  },
  "engineering_audit": {
    "status": "PASS",
    "n": 12,
    "new_reader_forwards": 18,
    "max_CPU_probability_error": 5.040882825047e-07,
    "max_graph_weight_error": 0.0,
    "max_engineering_CNN_error": 4.76837158203125e-06,
    "all_native_chunks_verified_before_temporary_deletion": true,
    "source_parameters_updated": 0,
    "raw_pixel_SHA_and_actual_CNN_input_provenance": true,
    "fixed_1024point_F_input_verified": true,
    "engineering_CNN_regenerated_CPU": true,
    "full_transformer_not_CPU_regenerated": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "arms": 119,
    "task_values": 213962,
    "max_error": 0.0,
    "all_historical_and_new_negative_cases_kept": true
  },
  "source_calibration_run": false,
  "TTA_run": false,
  "FINAL_access": false,
  "exposed_development_not_independent": true
}
