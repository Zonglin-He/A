# R151 — Native original-RGB crop reader prerequisite

## Decision

**stop_native_crop_recipe_no_new_TTA**. The predeclared frozen-evidence gate fails. No new source calibration or parameter adaptation is run. Keep the prior R145 development candidate (13.819603% versus intact Frozen-A 13.394463%, +0.425140 pp); the requested >=2 pp improvement is NOT achieved.

## Input mechanism and controls

Original full A and B images remain present. Up to three local banks are selected from the original supported candidate groups, ranked by the fixed half-mean of frozen A-image and B-image predictions. Support Jaccard must be <=0.5 relative to previously selected regions. At least four distinct visible points are required in one camera; choose the camera with more supported points, breaking ties toward A. This selects uncertain predicted regions, not certified instances, and does not read task GT.

Each ROI comes from those projected points with 1.5 margin, 3:2 aspect ratio, raw minimum 48x32, shifted inside the image. High-resolution evidence is cropped from the original licensed RGB frame and resized to 384x256. The low-resolution control uses exactly the same floating ROI on the existing downsized full-frame image, then resizes to the same bank size. Their point/pixel graphs are identical. Local 3x3 point-to-image and depth-gated eight-point image-to-point associations are independently recomputed after the crop affine. Physical coordinates, camera visibility/depth, 1024 fused point features, normalized positional references, original A256 and trained weights remain unchanged.

Every bank passes through the same trained three-convolution RGB encoder with independent per-bank GroupNorm, then the existing native bidirectional interaction and full language/competition reader. No new image encoder or source training is added. Crop banks are an input distribution change; trained scale invariance is not assumed.

Controls: (1) low-resolution crops isolate fresh detail from magnification/context and new graph support; (2) within-crop point-binding permutation preserves images and visibility but may retain the same object evidence; (3) duplicate corresponding full frames match bank count but differ in graph extent; (4) cyclic crop-image exchange across selected regions preserves the entire crop-image multiset and original high-crop graphs while breaking region/content binding. Original joint zero-update, both single-image readers, original R145 TTA and all 101 parent arms remain available. Fixed half-means with the parent TTA are ensemble controls, not new parameter updates or independent replications.

## Results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.40304063926085 | 25.899514674225376 | 5.056806006058658 |
| raw | norm_joint_zero | 13.716845698214941 | 26.56221517419685 | 5.249643946003637 |
| raw | norm_Aimage | 13.661544723534073 | 26.46619191640889 | 5.249643946003637 |
| raw | norm_Bimage | 13.711942296165214 | 26.471140492958238 | 5.48324893792589 |
| raw | norm_TTA | 13.801914656549181 | 26.834112929821703 | 5.528786278545199 |
| raw | high_local | 13.690932436335705 | 26.508106968569844 | 5.3407186272422535 |
| raw | low_local | 13.68499826308873 | 26.508106968569844 | 5.295181286622945 |
| raw | wrong_local | 13.801280271557594 | 26.745512536857763 | 5.249643946003637 |
| raw | duplicate_local | 13.718027530249836 | 26.608900514999842 | 5.249643946003637 |
| raw | region_swap_local | 13.715492283594527 | 26.418358575422218 | 5.295181286622945 |
| raw | high_zero_mean | 13.67801243639318 | 26.461421627766853 | 5.295181286622945 |
| raw | high_parent_mean | 13.675283865299905 | 26.41588428714755 | 5.295181286622945 |
| raw | low_parent_mean | 13.664717215888494 | 26.41588428714755 | 5.295181286622945 |
| raw | wrong_parent_mean | 13.719295461689162 | 26.606426226725166 | 5.249643946003637 |
| raw | duplicate_parent_mean | 13.700150401809713 | 26.745334248950453 | 5.249643946003637 |
| raw | region_swap_parent_mean | 13.683663219060035 | 26.46504391622521 | 5.249643946003637 |
| design_weighted | Frozen_A | 13.394463465154152 | 26.26330584715755 | 4.599296733787808 |
| design_weighted | norm_joint_zero | 13.739713257337598 | 27.044149189023084 | 4.793982350866708 |
| design_weighted | norm_Aimage | 13.659170072455359 | 26.859601705238706 | 4.793982350866708 |
| design_weighted | norm_Bimage | 13.733419725521841 | 26.93794211447542 | 5.029399635265411 |
| design_weighted | norm_TTA | 13.81960316688109 | 27.305335032677846 | 5.063686987738404 |
| design_weighted | high_local | 13.690910180510295 | 26.90178759284809 | 4.8850570321053235 |
| design_weighted | low_local | 13.684977544327829 | 26.90178759284809 | 4.839519691486016 |
| design_weighted | wrong_local | 13.82328747622004 | 27.212573094123126 | 4.793982350866708 |
| design_weighted | duplicate_local | 13.740904633329372 | 27.091093465574257 | 4.793982350866708 |
| design_weighted | region_swap_local | 13.721472816030726 | 26.811250492755676 | 4.839519691486016 |
| design_weighted | high_zero_mean | 13.677957648866693 | 26.854843316296922 | 4.839519691486016 |
| design_weighted | high_parent_mean | 13.674026466673004 | 26.824438368986677 | 4.839519691486016 |
| design_weighted | low_parent_mean | 13.663409846243354 | 26.824438368986677 | 4.839519691486016 |
| design_weighted | wrong_parent_mean | 13.738827844193347 | 27.077648329581308 | 4.793982350866708 |
| design_weighted | duplicate_parent_mean | 13.715233990577891 | 27.217074223302955 | 4.793982350866708 |
| design_weighted | region_swap_parent_mean | 13.682376245097872 | 26.873327162615908 | 4.793982350866708 |

{
  "phase": "frozen_native_RGB_crop_prerequisite",
  "primary": "high_local",
  "source_parameters_updated": 0,
  "FINAL_access": false,
  "target_exposed_development": true,
  "endpoints": {
    "high_local": {
      "comparisons": {
        "raw": {
          "norm_joint_zero": -0.025913261879235705,
          "region_swap_local": -0.02455984725882132,
          "wrong_local": -0.11034783522188896,
          "duplicate_local": -0.027095093914130786,
          "low_local": 0.00593417324697576,
          "norm_TTA": -0.11098222021347581,
          "Frozen_A": 0.287891797074856
        },
        "design_weighted": {
          "norm_joint_zero": -0.048803076827303116,
          "region_swap_local": -0.030562635520430703,
          "wrong_local": -0.13237729570974466,
          "duplicate_local": -0.04999445281907633,
          "low_local": 0.005932636182466311,
          "norm_TTA": -0.12869298637079396,
          "Frozen_A": 0.29644671535614364
        }
      }
    },
    "high_parent_mean": {
      "comparisons": {
        "raw": {
          "norm_TTA": -0.12663079124927634,
          "high_zero_mean": -0.0027285710932751783,
          "high_local": -0.01564857103580053,
          "region_swap_parent_mean": -0.008379353760130215,
          "wrong_parent_mean": -0.044011596389257335,
          "duplicate_parent_mean": -0.024866536509808412,
          "low_parent_mean": 0.010566649411410722,
          "Frozen_A": 0.2722432260390555
        },
        "design_weighted": {
          "norm_TTA": -0.14557670020808544,
          "high_zero_mean": -0.003931182193689153,
          "high_local": -0.01688371383729148,
          "region_swap_parent_mean": -0.008349778424868504,
          "wrong_parent_mean": -0.0648013775203431,
          "duplicate_parent_mean": -0.041207523904887466,
          "low_parent_mean": 0.010616620429649615,
          "Frozen_A": 0.27956300151885216
        }
      }
    },
    "low_local": {
      "comparisons": {
        "raw": {
          "norm_joint_zero": -0.031847435126211465,
          "high_local": -0.00593417324697576,
          "duplicate_local": -0.033029267161106546,
          "norm_TTA": -0.11691639346045157,
          "Frozen_A": 0.28195762382788025
        },
        "design_weighted": {
          "norm_joint_zero": -0.05473571300976943,
          "high_local": -0.005932636182466311,
          "duplicate_local": -0.05592708900154264,
          "norm_TTA": -0.13462562255326027,
          "Frozen_A": 0.29051407917367733
        }
      }
    }
  },
  "proceed_sourcecal_and_TTA": false,
  "retained_oldTTA_plus_crop_ensemble": false,
  "disposition": "stop_native_crop_recipe_no_new_TTA"
}

## Positive and negative effects

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| high_local | norm_joint_zero | 26 | 21 | -0.04880307682730307 | 0.018245761786296468 | -0.04124910190536532 | 39 | 45 | 6 |
| high_local | region_swap_local | 21 | 18 | -0.030562635520433132 | 0.018245761786296468 | -0.026944995983505412 | 37 | 39 | 7 |
| high_local | wrong_local | 22 | 21 | -0.1323772957097451 | 0.009094021615666054 | -0.04124910190536532 | 38 | 43 | 9 |
| high_local | duplicate_local | 25 | 21 | -0.04999445281907552 | 0.018245761786296468 | -0.04124910190536532 | 38 | 44 | 6 |
| high_local | low_local | 4 | 6 | 0.005932636182465801 | 0.018245761786296468 | -0.014600800518350844 | 9 | 10 | 3 |
| high_local | norm_TTA | 60 | 62 | -0.12869298637079463 | 0.03206973218682594 | -0.04124910190536532 | 82 | 109 | 10 |
| high_local | Frozen_A | 116 | 98 | 0.29644671535614453 | 0.053304207197498364 | -0.04909164122994028 | 125 | 186 | 19 |
| high_parent_mean | norm_TTA | 27 | 29 | -0.14557670020808633 | 0.013663444475686917 | -0.04124910190536532 | 49 | 54 | 6 |
| high_parent_mean | high_zero_mean | 29 | 26 | -0.003931182193689611 | 0.040535899868321686 | -0.03206973218682594 | 46 | 54 | 6 |
| high_parent_mean | high_local | 35 | 33 | -0.016883713837291713 | 0.040535899868321686 | -0.03206973218682594 | 54 | 64 | 6 |
| high_parent_mean | region_swap_parent_mean | 7 | 7 | -0.008349778424868969 | 0.009094021615666054 | -0.010912258782887742 | 13 | 14 | 3 |
| high_parent_mean | wrong_parent_mean | 10 | 9 | -0.06480137752034754 | 0.009094021615666054 | -0.04124910190536532 | 19 | 19 | 6 |
| high_parent_mean | duplicate_parent_mean | 13 | 11 | -0.0412075239048868 | 0.013701725203984182 | -0.04124910190536532 | 24 | 24 | 4 |
| high_parent_mean | low_parent_mean | 2 | 1 | 0.010616620429650963 | 0.009730543939810134 | -0.0006709478189828494 | 3 | 3 | 2 |
| high_parent_mean | Frozen_A | 114 | 96 | 0.2795630015188529 | 0.053304207197498364 | -0.04909164122994028 | 115 | 178 | 17 |
| low_local | norm_joint_zero | 24 | 18 | -0.05473571300976886 | 0.013701725203984182 | -0.04124910190536532 | 36 | 39 | 6 |
| low_local | high_local | 6 | 4 | -0.005932636182465801 | 0.014600800518350844 | -0.018245761786296468 | 9 | 10 | 3 |
| low_local | duplicate_local | 24 | 19 | -0.05592708900154131 | 0.013701725203984182 | -0.04124910190536532 | 37 | 40 | 6 |
| low_local | norm_TTA | 59 | 59 | -0.13462562255326044 | 0.03206973218682594 | -0.04124910190536532 | 83 | 108 | 10 |
| low_local | Frozen_A | 117 | 95 | 0.2905140791736788 | 0.053304207197498364 | -0.04909164122994028 | 123 | 182 | 19 |

The primary gate requires high_local to beat original joint zero, duplicate frames and region-image swap in both raw and design-weighted six-direction means; it does not require all directions to rise. Low resolution and within-crop point permutation are fully reported diagnostics. Parent-TTA ensemble retention is separately gated against parent TTA, high zero-mean, pure high, and low/swap/duplicate parent means. No per-input, per-platform or per-direction GT choice is made. A failed crop recipe does not exclude all higher-resolution, RGB or TTA methods.

## Measurement, resources and limitations

The first package-only scalar checker used a wrong comprehension variable and raised before export. It was repaired without changing any predictions, scoring definitions or results; the failed code/log are retained privately.

Producer, crop selection, independent geometry auditor, formal auditor and task scorer were pinned before the first network forward. Before any model or task score, a control amendment added the stronger region-image swap while retaining the weaker within-region point permutation; the old preparation/profile attempts are retained privately and the independent preparation audit was rerun. It is not a post-score gate change.

Engineering executes 12 inputs: 30 new reader forwards plus six original-joint bridge forwards. Independent CPU checks regenerate actual CNN inputs from raw media, the CNN outputs and all geometry, and reconstruct the full learned matching/competition readout. Actual RGB pre-hooks verify every normalized tensor in the native CNN call order; point-adapter pre-hooks verify bit-identical frozen point features. All prepared and source states are checked unchanged. Formal execution streams chunks of eight, seals new features/graphs/predictions, independently reconstructs the native head and crop geometry on CPU, and only after PASS deletes its own new TEMP feature/graph files. Persistent predictions, recipe provenance, pixel/input hashes and audit receipts remain. Full transformer forward and backprop are not independently regenerated on CPU. Formal CNN tensors are provenance-checked in the actual forward; engineering CNN outputs are independently regenerated.

All 1798 inputs, 764 with original B and 1034 without B, remain in every denominator. No-B inputs reproduce complete Frozen-A. Predictions are sealed before task scoring; post-seal GT is for evaluation only. All positive and negative results and recording/frame/content clusters are retained. IoU changes do not establish object identity. This is repeated exposed-development analysis, not independent confirmation, and no single-scene population CI is claimed. FINAL is untouched.

Resource cap is 0.4 GiB of new model payload, with 12 GiB free floor plus 1 GiB safety and streamed audit/deletion of only new temporary files. No historical experiments, data, weights or checkpoints are deleted. Runtime tables include CPU audit and cached-pipeline overhead; they are not cold end-to-end deployment latency.

{
  "decision": {
    "phase": "frozen_native_RGB_crop_prerequisite",
    "primary": "high_local",
    "source_parameters_updated": 0,
    "FINAL_access": false,
    "target_exposed_development": true,
    "endpoints": {
      "high_local": {
        "comparisons": {
          "raw": {
            "norm_joint_zero": -0.025913261879235705,
            "region_swap_local": -0.02455984725882132,
            "wrong_local": -0.11034783522188896,
            "duplicate_local": -0.027095093914130786,
            "low_local": 0.00593417324697576,
            "norm_TTA": -0.11098222021347581,
            "Frozen_A": 0.287891797074856
          },
          "design_weighted": {
            "norm_joint_zero": -0.048803076827303116,
            "region_swap_local": -0.030562635520430703,
            "wrong_local": -0.13237729570974466,
            "duplicate_local": -0.04999445281907633,
            "low_local": 0.005932636182466311,
            "norm_TTA": -0.12869298637079396,
            "Frozen_A": 0.29644671535614364
          }
        }
      },
      "high_parent_mean": {
        "comparisons": {
          "raw": {
            "norm_TTA": -0.12663079124927634,
            "high_zero_mean": -0.0027285710932751783,
            "high_local": -0.01564857103580053,
            "region_swap_parent_mean": -0.008379353760130215,
            "wrong_parent_mean": -0.044011596389257335,
            "duplicate_parent_mean": -0.024866536509808412,
            "low_parent_mean": 0.010566649411410722,
            "Frozen_A": 0.2722432260390555
          },
          "design_weighted": {
            "norm_TTA": -0.14557670020808544,
            "high_zero_mean": -0.003931182193689153,
            "high_local": -0.01688371383729148,
            "region_swap_parent_mean": -0.008349778424868504,
            "wrong_parent_mean": -0.0648013775203431,
            "duplicate_parent_mean": -0.041207523904887466,
            "low_parent_mean": 0.010616620429649615,
            "Frozen_A": 0.27956300151885216
          }
        }
      },
      "low_local": {
        "comparisons": {
          "raw": {
            "norm_joint_zero": -0.031847435126211465,
            "high_local": -0.00593417324697576,
            "duplicate_local": -0.033029267161106546,
            "norm_TTA": -0.11691639346045157,
            "Frozen_A": 0.28195762382788025
          },
          "design_weighted": {
            "norm_joint_zero": -0.05473571300976943,
            "high_local": -0.005932636182466311,
            "duplicate_local": -0.05592708900154264,
            "norm_TTA": -0.13462562255326027,
            "Frozen_A": 0.29051407917367733
          }
        }
      }
    },
    "proceed_sourcecal_and_TTA": false,
    "retained_oldTTA_plus_crop_ensemble": false,
    "disposition": "stop_native_crop_recipe_no_new_TTA"
  },
  "preparation": {
    "B": 764,
    "regions": 2292,
    "A_regions": 2291,
    "B_regions": 1,
    "n": 1798,
    "new_models": 0,
    "GT_read": false
  },
  "native_audit": {
    "status": "PASS",
    "n": 1798,
    "new_reader_forwards": 3820,
    "max_CPU_probability_error": 1.1570169483610115e-06,
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
    "new_reader_forwards": 30,
    "max_CPU_probability_error": 3.447058953831572e-07,
    "max_graph_weight_error": 0.0,
    "max_engineering_CNN_error": 6.67572021484375e-06,
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
    "arms": 112,
    "task_values": 201376,
    "max_error": 0.0,
    "all_historical_and_new_negative_cases_kept": true
  },
  "source_calibration_run": false,
  "TTA_run": false,
  "FINAL_access": false,
  "exposed_development_not_independent": true
}
