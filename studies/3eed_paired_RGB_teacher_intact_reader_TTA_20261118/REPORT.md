# R142 — Paired RGB teacher with intact full-reader adaptation

## Decision

**retain_as_development_candidate_for_separately_locked_expansion**. R141 native-point expansion remains rejected. R142 follows its positive image-only ablation with a matched image-binding control; this is exposed-target method development, not independent confirmation. The frozen prerequisite and conditional source-reader/TTA design were registered before new controls/task scores. Original A256, A1024, all six query positions, three intact source-prepared model states and complete live competition remain fixed; no new expert, source-weight training or FINAL. All1798 inputs and1034 completeFrozenA noB fallbacks remain.

## Why this is a different bounded comparison

R141 joint nativeB+RGBB failed against oldFrozenF and extraA. Its Fpoints+actualRGBB image-only ablation scored13.670700% versusFrozenF13.580861% in design-weighted six-direction mean, but lacked an exact image-only wrongbinding control. Old R105 imagebanks also failed on its earlier protocol; it is retained as contrary evidence, not described as untested. Here every source Adapter/gate/matching/interface path is intact, raw Fpoint input remains the original validA/B average, and actualB images are encoded separately by the same small source CNN. RGBB-only means an additionalB image bank alongsideA; it does not mean an RGB-only grounding network.

Phase1 reuses exactR141 FrozenA/F,R135A/F,R136correct/direct and correct F_RGBB_only predictions. New frozen arms are the matching F_RGBB_only_wrong, F_Bimage_only_teacher and F_Bimage_only_wrong_teacher. Only visibleB point-to-pixel bindings permute, with the inverse reverse graph updated; pointfeatures/imagepixels/visibility/sequence budget stay fixed. DuplicateA imagebank equivalence is independently engineering-checked againstFrozenF. Both source and original A geometry/role priors retain cameraA coordinates; no B query numbers or objectIDs are copied between different text forwards.

## Frozen prerequisite results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.403041 | 25.899515 | 5.056806 |
| raw | Frozen_F | 13.605854 | 25.898010 | 5.154947 |
| raw | R135_A | 13.506393 | 26.220572 | 5.102343 |
| raw | R135_F | 13.625244 | 26.175826 | 5.343193 |
| raw | C_B_A | 13.472877 | 25.937807 | 5.060428 |
| raw | direct_C_B_A | 13.483204 | 26.126845 | 5.103491 |
| raw | F_RGBB_only | 13.670413 | 26.087404 | 5.154947 |
| raw | F_RGBB_only_wrong | 13.601882 | 25.899158 | 5.249644 |
| raw | F_Bimage_only_teacher | 13.666392 | 25.959363 | 5.484575 |
| raw | F_Bimage_only_wrong_teacher | 13.575428 | 25.911530 | 5.437890 |
| design_weighted | Frozen_A | 13.394463 | 26.263306 | 4.599297 |
| design_weighted | Frozen_F | 13.580861 | 26.099350 | 4.688056 |
| design_weighted | R135_A | 13.498989 | 26.600013 | 4.644834 |
| design_weighted | R135_F | 13.606614 | 26.393334 | 4.859257 |
| design_weighted | C_B_A | 13.459843 | 26.270810 | 4.601034 |
| design_weighted | direct_C_B_A | 13.477422 | 26.444000 | 4.639679 |
| design_weighted | F_RGBB_only | 13.670700 | 26.377786 | 4.688056 |
| design_weighted | F_RGBB_only_wrong | 13.575010 | 26.100757 | 4.753882 |
| design_weighted | F_Bimage_only_teacher | 13.681997 | 26.281908 | 5.002193 |
| design_weighted | F_Bimage_only_wrong_teacher | 13.602154 | 26.321172 | 4.956105 |

{
  "phase": "frozen_image_only_binding_prerequisite",
  "passed": true,
  "comparisons": {
    "raw": {
      "Frozen_A": 0.2673724439049927,
      "Frozen_F": 0.06455930928016507,
      "F_RGBB_only_wrong": 0.06853154034108577
    },
    "design_weighted": {
      "Frozen_A": 0.2762363211418748,
      "Frozen_F": 0.08983854504866429,
      "F_RGBB_only_wrong": 0.0956898961124324
    }
  },
  "parameters_updated": 0,
  "exposed_target_ablation_followup_not_independent_confirmation": true,
  "TTA_not_yet_tested": true,
  "next": "separately_pin_source_image_reader_calibration_and_sameUp4896_conditional_TTA",
  "FINAL_access": false
}

The registered frozen gate requires correctF_RGBB_only to beatFrozenF and matchedwrong in both raw and design-weighted six-direction means. No requirement that every direction rise is added. The pure Bimage teacher is also reported, not assumed correct. Geometry-consistent graph construction does not independently certify target-instance identity.

## Conditional parameter experiment

Executed: **True**. Phase2 is permitted only after the frozen prerequisite passes; if it fails, sourcecalibration and allTTA are skipped, even though conditional implementation files are available. No target-selected imagefusion weight, confidence threshold or new loss is introduced.

Source-only recalibration uses exactly the previous313 source-own inputs/pairs and source-independent qualification, now with the correct Fpoint/Aimage and Fpoint/Bimage native full readers. Source predictions are sealed and CPUaudited before source labels identify groups with bestIoU>.25. Fixedq90 supplies one own-source loggap kappa for each source checkpoint, shared by its two outgoing directions. These labels calibrate thresholds, never train source weights or choose a target output. Eligible sourcecounts remain22/18/37; limited source sample sizes and domain shift do not imply a target coverage guarantee.

{
  "status": "LOCKED_SOURCE_ONLY_IMAGE_READER",
  "quantile": 0.9,
  "source_eligible_counts": {
    "waymo": 22,
    "drone": 18,
    "quad": 37
  },
  "kappa": {
    "waymo": 2.2682414032518876,
    "drone": 1.264869783419025,
    "quad": 0.6166806578636176
  },
  "source_labels_used": true,
  "target_GT_parameter_choice": false,
  "source_weight_training": false,
  "source_reader": "Fpoints_Aimage_and_Fpoints_Bimage_full_native_competition",
  "source_prediction_seal_sha256": "85266a796c18e6c94c7cff6c8e5e5c07cae6405e5050876f289530e28afe89d7",
  "FINAL_access": false,
  "utc": "2026-11-17T17:27:10.714962+00:00"
}

C is the union of source-threshold-qualified F/A-image and F/B-image groupMEAN candidates in the original supported scopeS. Teacher probabilities are the frozen F/B-image groupMAX distribution conditionalC. Student and final reader both use Fpoints plus actualRGB A/B; this removes the old pointB/Aimage teacher-versus-A-only final mismatch without declaring B trustworthy. Original R136 objective partial(S,C)+KL(teacherB|C || adaptedAB|C)+source-adaptedF featurekeep1, Up4896only, sourceinitialization perinput, originalrho0/3steps/9Armijo remain unchanged. No hardunknownoutput protection, EMA, newlayer or sourceweight training. C/teacher/routing are fixed before adaptation.

WrongTTA permutes teacher weights within equal-size candidate groups in C, retaining S,C,mass,entropy,groupmultiplicity and update budget. This isolates within-C ordering from eligibility. It is distinct from the physical wrongimage binding frozen control. Empty permutation degrees are explicitly reused, never fabricated. The same-info direct projection uses exactly the same C/teacher and zeroABimage distribution. Pure Bimage frozen readout is retained as a simple strong control; passing the TTA gate also requires beating that readout.

## Complete adaptation results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.403041 | 25.899515 | 5.056806 |
| raw | Frozen_F | 13.605854 | 25.898010 | 5.154947 |
| raw | R135_A | 13.506393 | 26.220572 | 5.102343 |
| raw | R135_F | 13.625244 | 26.175826 | 5.343193 |
| raw | C_B_A | 13.472877 | 25.937807 | 5.060428 |
| raw | direct_C_B_A | 13.483204 | 26.126845 | 5.103491 |
| raw | F_RGBB_only | 13.670413 | 26.087404 | 5.154947 |
| raw | F_RGBB_only_wrong | 13.601882 | 25.899158 | 5.249644 |
| raw | F_Bimage_only_teacher | 13.666392 | 25.959363 | 5.484575 |
| raw | direct_image_B | 13.560909 | 25.806935 | 5.341867 |
| raw | direct_wrong_image_B | 12.884816 | 25.160585 | 5.157421 |
| raw | image_B_TTA | 13.701826 | 26.224194 | 5.344341 |
| raw | wrong_image_B_TTA | 13.624579 | 26.178479 | 5.342045 |
| design_weighted | Frozen_A | 13.394463 | 26.263306 | 4.599297 |
| design_weighted | Frozen_F | 13.580861 | 26.099350 | 4.688056 |
| design_weighted | R135_A | 13.498989 | 26.600013 | 4.644834 |
| design_weighted | R135_F | 13.606614 | 26.393334 | 4.859257 |
| design_weighted | C_B_A | 13.459843 | 26.270810 | 4.601034 |
| design_weighted | direct_C_B_A | 13.477422 | 26.444000 | 4.639679 |
| design_weighted | F_RGBB_only | 13.670700 | 26.377786 | 4.688056 |
| design_weighted | F_RGBB_only_wrong | 13.575010 | 26.100757 | 4.753882 |
| design_weighted | F_Bimage_only_teacher | 13.681997 | 26.281908 | 5.002193 |
| design_weighted | direct_image_B | 13.560986 | 26.101944 | 4.849543 |
| design_weighted | direct_wrong_image_B | 12.877941 | 25.395895 | 4.696966 |
| design_weighted | image_B_TTA | 13.695324 | 26.498397 | 4.859808 |
| design_weighted | wrong_image_B_TTA | 13.620423 | 26.454585 | 4.883328 |

{
  "primary": "image_B_TTA",
  "comparisons": {
    "raw": {
      "Frozen_A": 0.2987853829835956,
      "Frozen_F": 0.09597224835876794,
      "R135_F": 0.07658182705889516,
      "F_RGBB_only": 0.031412939078602875,
      "direct_image_B": 0.14091746451068943,
      "wrong_image_B_TTA": 0.07724682358514556,
      "F_RGBB_only_wrong": 0.09994447941968865,
      "F_Bimage_only_teacher": 0.03543397206116694
    },
    "design_weighted": {
      "Frozen_A": 0.30086041266935126,
      "Frozen_F": 0.11446263657614075,
      "R135_F": 0.08871030662177581,
      "F_RGBB_only": 0.024624091527476466,
      "direct_image_B": 0.13433796051542402,
      "wrong_image_B_TTA": 0.07490092001379445,
      "F_RGBB_only_wrong": 0.12031398763990886,
      "F_Bimage_only_teacher": 0.013326786581735917
    }
  },
  "passed": true,
  "same_information_parameter_increment_established_in_scope": true,
  "source_checkpoint_weights_retrained": false,
  "interface_parameters": 4896,
  "steps": 3,
  "original_rho0": true,
  "source_only_image_reader_kappa": true,
  "target_GT_hyperparameter_selection": false,
  "independent_validation": false,
  "all1798_negatives_unknowns_and_noB_retained": true,
  "FINAL_access": false,
  "disposition": "retain_as_development_candidate_for_separately_locked_expansion"
}

## Changed-outcome partitions

| source | target | n | eligible | better | worse | best_in_C_eligible | good25_in_C_eligible | zero_best_already | best_outside_S | best_outside_C | C_internal_order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | 357 | 178 | 12 | 11 | 94 | 77 | 4 | 4 | 1 | 14 |
| drone | waymo | 176 | 50 | 3 | 1 | 16 | 26 | 0 | 2 | 1 | 1 |
| quad | drone | 366 | 154 | 12 | 8 | 27 | 65 | 1 | 2 | 10 | 7 |
| quad | waymo | 176 | 50 | 3 | 0 | 3 | 15 | 0 | 1 | 1 | 1 |
| waymo | drone | 366 | 154 | 10 | 11 | 67 | 70 | 0 | 0 | 10 | 11 |
| waymo | quad | 357 | 178 | 3 | 11 | 113 | 106 | 1 | 2 | 1 | 10 |

These GT partitions are postseal diagnostic descriptions, not deployment filters or independently checked instance identity. Every gain and negative comparison remains; no input/direction/source threshold is selected by target GT. The comparison separates extra RGB evidence from the incremental value of parameter updates.

## Supplementary canonical same-information means

The originalTTA gate above remains unchanged. Beforepublication we identified an omitted necessary simplecounter: equal probability means of the frozen singleA/singleB readers, and of the exact frozenjointAB/Bteacher predictions used byadaptation. These controls were added after originalTTA results, so they are explicitly posthoc. Both fixed1/2 formulas were locked before their new queries/task scores; no weightgrid or GT-selected view was used. All1798 newfused predictions were sealed, byte-equivalent CPUaudited, then scored. No model, input observation or TTA prediction changed.

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | FA_FB_probability_mean | 13.535399 | 25.724610 | 5.343193 |
| raw | FAB_FB_probability_mean | 13.547931 | 25.816833 | 5.343193 |
| design_weighted | FA_FB_probability_mean | 13.537643 | 26.025512 | 4.859257 |
| design_weighted | FAB_FB_probability_mean | 13.550181 | 26.117994 | 4.859257 |

{
  "status": "POSTHOC_CANONICAL_SAME_INFORMATION_STRONG_CONTROL",
  "canonical_equal_probability_mean_no_GT_weight_search": true,
  "original_preregistered_TTA_gate_preserved": true,
  "comparisons": {
    "raw": {
      "FA_FB_probability_mean": 0.16642684559137244,
      "FAB_FB_probability_mean": 0.1538952638339648
    },
    "design_weighted": {
      "FA_FB_probability_mean": 0.15768058323865652,
      "FAB_FB_probability_mean": 0.1451428837888038
    }
  },
  "TTA_exceeds_both_means": true,
  "parameter_increment_over_supplementary_strong_controls_supported": true,
  "source_checkpoint_OR_predictions_changed": false,
  "FINAL_access": false
}

## What the small gain does and does not establish

The design-weighted total gain overintactFrozenA is+.300860pp. Frozen paired-image evidence contributes+.276236pp; parameter adaptation adds+.024624pp beyondthat exactzero. It also exceeds the pureBimage reader by+.013327pp. The mainincrement has43 improved and42 worsened IoUs; the largestpositive contribution+.053304pp exceeds the net+.024624pp. This is a valid small development effect with fragile contribution structure, not independent generalization or the requested>=2pp outcome. Four ofsix directions decrease relative toownjointABzero; bothQuad-source directions increase. The full direction, scene and negativecase tables remain.

Postseal box-quality diagnosis: among764 Binputs,611 have an Acc25-capable A256 candidate and192 zero predictions already passAcc25. Of419 recoverable zeroerrors, S retains a passingcandidate in349, C in209, and the teacherTopC chosenquery passes inonly27. Thus bothset retention andwithinC entityselection remain substantive limitations. DirectBimage information has value, but conditioning its distribution onC does not make it a reliable targetlabel. The same Up4896 actually changesparameters and optimizes the native finalreader; zero-gradient or a disconnected readout is not an explanation for this small increment.

Simple qualification rules are not established repairs: both-image-qualified comparisons have3 updateimprovements/7 harms; positive true-versus-wrong binding contrasts have13/15. These are descriptive postseal strata, not rulesused forprediction. Image qualification at the existing4points/.5coverage threshold reduces supported currentleaders539to114 and addsnone. Context/reference RGBmaystillbeusefulwhen the target hasno directBpixels, so visibilityabsence is not a negativeidentity label. Neither visibility nor contrast is installed as a hardgate.

## Positive and negative contribution structure

| phase | method | control | better | worse | changed_content_groups | changed_frame_groups | changed_scene_groups | design_weighted_gain_pp | largest_positive_contribution_pp | largest_negative_contribution_pp |
|---|---|---|---|---|---|---|---|---|---|---|
| image | F_RGBB_only | Frozen_A | 89 | 79 | 110 | 150 | 18 | 0.276236321141875 | 0.047564337916906944 | -0.04909164122994028 |
| image | F_RGBB_only | Frozen_F | 6 | 2 | 8 | 8 | 5 | 0.08983854504866348 | 0.04124910190536532 | -0.013701725203984182 |
| image | F_RGBB_only | F_RGBB_only_wrong | 17 | 15 | 25 | 31 | 8 | 0.09568989611243159 | 0.04124910190536532 | -0.008529554290294058 |
| TTA | image_B_TTA | F_RGBB_only | 43 | 42 | 66 | 80 | 7 | 0.02462409152747715 | 0.053304207197498364 | -0.046091117295781324 |
| TTA | image_B_TTA | direct_image_B | 90 | 89 | 106 | 151 | 14 | 0.13433796051542501 | 0.053304207197498364 | -0.05407119513360017 |
| TTA | image_B_TTA | wrong_image_B_TTA | 42 | 47 | 70 | 84 | 12 | 0.07490092001379187 | 0.031667559544968506 | -0.02741550046085794 |
| TTA | image_B_TTA | F_Bimage_only_teacher | 88 | 87 | 108 | 149 | 14 | 0.01332678658173593 | 0.036117173255342604 | -0.05407119513360017 |
| supplementary_mean | image_B_TTA | FA_FB_probability_mean | 72 | 59 | 93 | 119 | 11 | 0.15768058323865675 | 0.027361131901579055 | -0.046091117295781324 |
| supplementary_mean | image_B_TTA | FAB_FB_probability_mean | 71 | 59 | 91 | 117 | 11 | 0.14514288378880233 | 0.027361131901579055 | -0.046091117295781324 |

Contribution sums use each direction's fixed design-weight denominator and six-direction equal weighting. They are descriptive sensitivity summaries, not deleting large cases, new selection rules or a significance test. All positive/negative records are retained in ALL_CHANGED_CASES.

## Audits, cost and scope

{
  "decision": {
    "primary": "image_B_TTA",
    "comparisons": {
      "raw": {
        "Frozen_A": 0.2987853829835956,
        "Frozen_F": 0.09597224835876794,
        "R135_F": 0.07658182705889516,
        "F_RGBB_only": 0.031412939078602875,
        "direct_image_B": 0.14091746451068943,
        "wrong_image_B_TTA": 0.07724682358514556,
        "F_RGBB_only_wrong": 0.09994447941968865,
        "F_Bimage_only_teacher": 0.03543397206116694
      },
      "design_weighted": {
        "Frozen_A": 0.30086041266935126,
        "Frozen_F": 0.11446263657614075,
        "R135_F": 0.08871030662177581,
        "F_RGBB_only": 0.024624091527476466,
        "direct_image_B": 0.13433796051542402,
        "wrong_image_B_TTA": 0.07490092001379445,
        "F_RGBB_only_wrong": 0.12031398763990886,
        "F_Bimage_only_teacher": 0.013326786581735917
      }
    },
    "passed": true,
    "same_information_parameter_increment_established_in_scope": true,
    "source_checkpoint_weights_retrained": false,
    "interface_parameters": 4896,
    "steps": 3,
    "original_rho0": true,
    "source_only_image_reader_kappa": true,
    "target_GT_hyperparameter_selection": false,
    "independent_validation": false,
    "all1798_negatives_unknowns_and_noB_retained": true,
    "FINAL_access": false,
    "disposition": "retain_as_development_candidate_for_separately_locked_expansion"
  },
  "image_decision": {
    "phase": "frozen_image_only_binding_prerequisite",
    "passed": true,
    "comparisons": {
      "raw": {
        "Frozen_A": 0.2673724439049927,
        "Frozen_F": 0.06455930928016507,
        "F_RGBB_only_wrong": 0.06853154034108577
      },
      "design_weighted": {
        "Frozen_A": 0.2762363211418748,
        "Frozen_F": 0.08983854504866429,
        "F_RGBB_only_wrong": 0.0956898961124324
      }
    },
    "parameters_updated": 0,
    "exposed_target_ablation_followup_not_independent_confirmation": true,
    "TTA_not_yet_tested": true,
    "next": "separately_pin_source_image_reader_calibration_and_sameUp4896_conditional_TTA",
    "FINAL_access": false
  },
  "image_audit": {
    "status": "PASS",
    "n": 1798,
    "B_inputs": 764,
    "effective_wrong_B_inputs": 754,
    "max_CPU_full_reader_error": 1.3797565064921358e-06,
    "max_independent_graph_weight_error": 0.0,
    "native_A256_noB_and_reused_references_exact": true,
    "source_parameters_updated": 0,
    "fullTransformer_not_independently_CPU_recomputed": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "image_phase": {
      "status": "PASS",
      "n": 1798,
      "task_values": 17980,
      "max_error": 0.0,
      "all_negative_fallback_rows_retained": true
    },
    "TTA_phase": {
      "status": "PASS",
      "n": 1798,
      "task_values": 23374,
      "max_error": 0.0,
      "all_negative_fallback_rows_retained": true
    },
    "supplementary_mean_phase": {
      "status": "PASS",
      "n": 1798,
      "task_values": 3596,
      "max_error": 0.0,
      "all_negative_fallback_rows_retained": true
    },
    "total_task_values_checked": 44950,
    "FINAL_access": false
  },
  "conditional_TTA_executed": true,
  "runtime": {
    "image": {
      "n": 1798,
      "complete_reader_forwards": 3820,
      "max_reference_bridge_error": 0.0,
      "max_engineering_duplicate_A_image_error": 0.0,
      "parameters_updated": 0,
      "GT_read": false,
      "FINAL_access": false,
      "private_artifact_manifest_not_exported": true
    },
    "source": {
      "n": 313,
      "B_inputs": 113,
      "complete_reader_forwards": 626,
      "max_original_A_bridge_error": 0.0,
      "source_or_target_GT_read": false,
      "source_training": false,
      "private_artifact_manifest_not_exported": true
    },
    "method": {
      "n": 1798,
      "fits": 1504,
      "GT_read": false,
      "primary": "image_B_TTA",
      "source_kappas_sha256": "44a28ce1ea03951d2238006744d285937bc8743935726fba1a3a8f7ec1e8db00",
      "FINAL_access": false,
      "private_artifact_manifest_not_exported": true
    }
  },
  "source_training": false,
  "FINAL_access": false,
  "source_calibration": {
    "status": "LOCKED_SOURCE_ONLY_IMAGE_READER",
    "quantile": 0.9,
    "source_eligible_counts": {
      "waymo": 22,
      "drone": 18,
      "quad": 37
    },
    "kappa": {
      "waymo": 2.2682414032518876,
      "drone": 1.264869783419025,
      "quad": 0.6166806578636176
    },
    "source_labels_used": true,
    "target_GT_parameter_choice": false,
    "source_weight_training": false,
    "source_reader": "Fpoints_Aimage_and_Fpoints_Bimage_full_native_competition",
    "source_prediction_seal_sha256": "85266a796c18e6c94c7cff6c8e5e5c07cae6405e5050876f289530e28afe89d7",
    "FINAL_access": false,
    "utc": "2026-11-17T17:27:10.714962+00:00"
  },
  "source_audit": {
    "status": "PASS",
    "n": 313,
    "B_inputs": 113,
    "source_views": 626,
    "max_CPU_full_reader_probability_error": 1.0167141276107294e-06,
    "max_independent_B_graph_weight_error": 0.0,
    "source_Fpoints_Aimage_Bimage_reader": true,
    "same_qualification_as_R132": true,
    "GT_read": false,
    "fullTransformer_not_CPU_regenerated": true,
    "FINAL_access": false
  },
  "TTA_audit": {
    "status": "PASS",
    "n": 1798,
    "fit_count": 1504,
    "max_CPU_probability_error": 1.110235305967766e-06,
    "max_state_error": 0.0,
    "max_Ffeature_error": 1.9073486328125e-06,
    "max_objective_error": 4.656617313969491e-10,
    "full_encoder_gradients_not_CPU_regenerated": true,
    "evidence_C_target_and_direct_reconstructed": true,
    "original_rho0_3steps_Up4896_and_source_only_initialization": true,
    "FINAL_access": false
  },
  "GTfree_frozen_supervision_scope": {
    "status": "FROZEN_GT_FREE_SUPERVISION_SCOPE_METADATA_ONLY",
    "n_all": 1798,
    "B_eligible": 764,
    "counts": {
      "eligible": 764,
      "top_supported": 539,
      "top_in_C": 539,
      "supported_top_excluded": 0,
      "teacher_top_C_differs_student_top_C": 272,
      "teacher_top_C_differs_global_student_group": 419
    },
    "initial_teacher_KL_mean": 0.08132163641057558,
    "initial_teacher_KL_median": 0.019244964657322304,
    "median_Bimage_visible_points": 158.0,
    "teacher_TopC_has_zero_Bimage_support": 278,
    "student_global_Top_has_zero_Bimage_support": 363,
    "GT_or_TTA_predictions_read": false,
    "new_model_or_task_scoring": false,
    "current_TTA_predictor_and_teacher_unchanged": true,
    "not_semantic_target_identity_or_performance_evidence": true,
    "FINAL_access": false
  },
  "supplementary_mean_decision": {
    "status": "POSTHOC_CANONICAL_SAME_INFORMATION_STRONG_CONTROL",
    "canonical_equal_probability_mean_no_GT_weight_search": true,
    "original_preregistered_TTA_gate_preserved": true,
    "comparisons": {
      "raw": {
        "FA_FB_probability_mean": 0.16642684559137244,
        "FAB_FB_probability_mean": 0.1538952638339648
      },
      "design_weighted": {
        "FA_FB_probability_mean": 0.15768058323865652,
        "FAB_FB_probability_mean": 0.1451428837888038
      }
    },
    "TTA_exceeds_both_means": true,
    "parameter_increment_over_supplementary_strong_controls_supported": true,
    "source_checkpoint_OR_predictions_changed": false,
    "FINAL_access": false
  },
  "supplementary_mean_audit": {
    "status": "PASS",
    "n": 1798,
    "task_GT_read": false,
    "fixed_A256_and_same_observations": true,
    "probability_arithmetic_bitexact": true,
    "FINAL_access": false
  }
}

Independent CPU reconstructs the final native matching/competition from saved features, actualB sparse projection/visibility/permutation graphs, all originalA256/zero/old-reference/noB identities. Sourcecalibration predictions and labels are separated by seal/audit. Where executed, CPU reconstructs C/teacher/permutation/direct projection, every accepted delta and Armijo decision, source-adaptedF keep and finalobjective; engineered finite differences check actual differentiable whole-reader paths. FullTransformer gradients and PointNet are not independently regenerated onCPU. Reusing the same audited backbone/CNN and matchedpoint budget is explicit. All model/score jobs use the unique-GPU/deadline guard. New pureB-image teacher forward costs are included; no new pretrained network is hidden. Measured jobcosts are incremental archived-work timings, not cold-start end-to-end latency: commonprefix/registration/backbone/geometry/routing and compatiblehistoricalpredictions are reused. Offline sourcecalibration cost is separate fromtarget adaptation.

All taskscalars are independently reread from sealedargmax plus the existing fixedA256 IoU cache. Public tasks/changed-cases use stable row/frame/content/scene aliases, preserving repeated physical observations. Recording/scene independence is limited; raw and design-weighted means are reported, not a single-scene populationCI. These are internal controlled ablations, not published-paper baselines. No physical-media/description/coordinate/queryID/feature/weight/key/thirdparty snapshots are exported. A positive frozen component alone cannot meet the parentgoal of validTTA>=2pp over intactFrozenA13.394463%; no oracle or weakerreset baseline is substituted.
