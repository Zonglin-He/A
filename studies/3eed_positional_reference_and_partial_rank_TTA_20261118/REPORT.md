# R145 — Fixed positional reference with unchanged partial/rank adaptation

## Decision

**retain_normalized_reader_TTA_development_candidate**. This registered combination tests actual complementarity; R144 inputstate and R142 adaptation increments are never added arithmetically. Bothpreviousround decisions remain unchanged. All1798 exposed ordinaryrecords, sixdirections, fixedweights/physicalclusters andallnegative comparisons remain. No independent confirmation, per-directionbestoutput or sourceweighttraining.

## Fixed versus changed

Original three intact source-prepared models, originalA256/outputboxes/queryidentities/rawFpoint1024/physicalA+Bcamera graphs/sourceArole-routing/geometricpriors, sameUp4896, partial(S,C)+KL(frozenBgroupMAX|C || adaptedjointABgroupMAX|C)+source-adaptedF featurekeep1, 3steps/originalrho0/9Armijo. No new expert, architecture, layer, regression, EMA, Bselection or auxiliaryloss. All1034 noB targetrecords are exactcompleteFrozenA.

The single changeddependency is the positiveR144 frozen positional reference: Hsource fixed source-ownmedian q10height over302sourcecontentgroups; delta=Hsource-q10(exactoriginaltargetA16384sample). Point pos_embed andsixquerycenterZ areadjusted together, whilephysicalxyz/graphs/outputboxes stayoriginal. This isolates learnedpositionalconditioning, not fullPointNettranslation or a certifiedgroundplane. The sameactualdelta is used by Aimage/Fpoints, Bimage/Fpoints andjointABstudent readouts. OriginalR144positions andzero are bytechecked/reused; no offset/quantile/strengthsearch.

Source-reader calibration must match thatdependency. Same313source-own inputs/113originalpairs, sameq90 andoldqualification; exactsourceperprefix16384sampledraws are now read individually. Thereare302physicalsourcecontents but313distinct actualsampledraws, so a q10 measured foranotherdescription cannot silentlyreplace currentdraw. Sourcepredictionsseal andindependentCPUaudits precede sourceGTgroupgoodness used onlyfor q90. Oneown-sourcekappa sharedbytwooutgoingdirections; no targetGTnumericparameterchoice. The sourceHstatistic itself remains fixedR144, not refit to this target score.

## Source-only calibration

{
  "status": "LOCKED_SOURCE_ONLY_IMAGE_READER",
  "quantile": 0.9,
  "source_eligible_counts": {
    "waymo": 22,
    "drone": 18,
    "quad": 37
  },
  "kappa": {
    "waymo": 2.29706174619496,
    "drone": 1.4053959552580049,
    "quad": 0.6205128759145743
  },
  "source_labels_used": true,
  "target_GT_parameter_choice": false,
  "source_weight_training": false,
  "source_reader": "R144positional_reference_Fpoints_Aimage_and_Fpoints_Bimage_full_native_competition",
  "source_prediction_seal_sha256": "beffcf4dc2fbf0610dc725918210be04c952b13f74733363d0f7cfbb2efa3a36",
  "FINAL_access": false,
  "utc": "2026-11-17T19:08:32.174787+00:00"
}

C uses existing unionofnormalizedF/Aimage andnormalizedF/Bimage groupMEAN loggap sets in originalS. Target is frozennormalizedBimage groupMAXconditionalC. C, teacher,positions,routing are fixed beforefits. Featurekeep anchors intactsource-adaptedrawFfeatures, not an untrainedreset. Fulljointnative matching andlivecompetition are recalculated aftereveryUp change.

WrongconditionalTTA permutes teacherweights withinC/equalquerymultiplicity, retainingC/mass/entropy/budget. FrozenwrongBbinding separately permutes physicalvisibleB associations with inversereversegraph and sameimages/visibility/pointbudget. These are distinctcontrols, neither certifies entityidentity. Ineffectivewrongtarget permutations reusecorrectfit and remaincounted.

## Same-information zero-update comparisons

AllR144/R142refs byte-reused. Fournewfrozenreadouts use normalizedF/Aimage, normalizedF/Bimage, normalizedjointwithwrongBbinding andnormalizedBimagewithwrongbinding. Bothcanonical fixedhalf-probabilitymeans are registered BEFORE TTA: normalizedFA/FB andnormalizedjointAB/FB. SameCdirect uses exactlytheteacher/scope/set usedbyUpupdates. These are internalcontrolled comparisons, not publishedTTA baselines.

## Whole-cohort results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.40304063926085 | 25.899514674225376 | 5.056806006058658 |
| raw | Frozen_F | 13.605853773885677 | 25.898010098227072 | 5.154946976306667 |
| raw | R135_A | 13.506393386808027 | 26.220572058927896 | 5.102343346677966 |
| raw | R135_F | 13.62524419518555 | 26.175826142677636 | 5.343192915516926 |
| raw | C_B_A | 13.472876819351491 | 25.937807437927976 | 5.0604282945170125 |
| raw | direct_C_B_A | 13.483204067665383 | 26.12684480150729 | 5.103491346861646 |
| raw | F_RGBB_only | 13.670413083165842 | 26.08740403762101 | 5.154946976306667 |
| raw | F_RGBB_only_wrong | 13.601881542824756 | 25.89915809841075 | 5.249643946003637 |
| raw | F_Bimage_only_teacher | 13.666392050183278 | 25.959362880770787 | 5.484575226016884 |
| raw | direct_image_B | 13.560908557733756 | 25.806935416988456 | 5.341866627425934 |
| raw | direct_wrong_image_B | 12.884815787950117 | 25.160585222663332 | 5.157421264581342 |
| raw | image_B_TTA | 13.701826022244445 | 26.22419434738625 | 5.344340915700607 |
| raw | wrong_image_B_TTA | 13.6245791986593 | 26.178478718859623 | 5.342044915333247 |
| raw | FA_FB_probability_mean | 13.535399176653073 | 25.724609888664855 | 5.343192915516926 |
| raw | FAB_FB_probability_mean | 13.54793075841048 | 25.816832570087147 | 5.343192915516926 |
| raw | aligned_reference | 13.716845698214941 | 26.56221517419685 | 5.249643946003637 |
| raw | opposite_reference | 13.577198921362617 | 25.7592803639091 | 5.247347945636277 |
| raw | half_zero_aligned | 13.630067729578494 | 26.18457529559265 | 5.154946976306667 |
| raw | norm_joint_zero | 13.716845698214941 | 26.56221517419685 | 5.249643946003637 |
| raw | norm_Aimage | 13.661544723534073 | 26.46619191640889 | 5.249643946003637 |
| raw | norm_Bimage | 13.711942296165214 | 26.471140492958238 | 5.48324893792589 |
| raw | norm_joint_wrong | 13.59665405392136 | 26.41818028751491 | 5.154946976306667 |
| raw | norm_B_wrong | 13.657651326585928 | 26.421980863880574 | 5.436563597122903 |
| raw | norm_FA_FB_mean | 13.598653219807787 | 26.23753550103599 | 5.343192915516926 |
| raw | norm_FAB_FB_mean | 13.624958676076995 | 26.283072841655294 | 5.389878256319915 |
| raw | norm_direct | 13.61923556074858 | 26.46848791677625 | 5.48324893792589 |
| raw | norm_wrong_direct | 12.890186442322497 | 24.74217829208186 | 5.346815203975281 |
| raw | norm_TTA | 13.801914656549181 | 26.834112929821703 | 5.528786278545199 |
| raw | norm_wrong_TTA | 13.671675192005827 | 26.552496309005473 | 5.294033286439266 |
| design_weighted | Frozen_A | 13.394463465154152 | 26.26330584715755 | 4.599296733787808 |
| design_weighted | Frozen_F | 13.580861241247362 | 26.099350265391482 | 4.688055597703281 |
| design_weighted | R135_A | 13.498989254903094 | 26.600013496665493 | 4.6448340744071155 |
| design_weighted | R135_F | 13.606613571201727 | 26.393334446143836 | 4.859257252330195 |
| design_weighted | C_B_A | 13.459843379670076 | 26.27080991908078 | 4.601034064844766 |
| design_weighted | direct_C_B_A | 13.477421661915146 | 26.44400022063034 | 4.639678907165158 |
| design_weighted | F_RGBB_only | 13.670699786296026 | 26.377786302278206 | 4.688055597703281 |
| design_weighted | F_RGBB_only_wrong | 13.575009890183594 | 26.100757201323347 | 4.753881508597696 |
| design_weighted | F_Bimage_only_teacher | 13.681997091241767 | 26.281907798110392 | 5.002192919558421 |
| design_weighted | direct_image_B | 13.560985917308079 | 26.101943522949373 | 4.849542504149657 |
| design_weighted | direct_wrong_image_B | 12.877941295183637 | 25.395895406548494 | 4.696965748443405 |
| design_weighted | image_B_TTA | 13.695323877823503 | 26.49839657604141 | 4.859808261761123 |
| design_weighted | wrong_image_B_TTA | 13.620422957809708 | 26.454584516708696 | 4.883327911208479 |
| design_weighted | FA_FB_probability_mean | 13.537643294584846 | 26.0255123491001 | 4.859257252330195 |
| design_weighted | FAB_FB_probability_mean | 13.550180994034699 | 26.11799396627058 | 4.859257252330195 |
| design_weighted | aligned_reference | 13.739713257337598 | 27.044149189023084 | 4.793982350866708 |
| design_weighted | opposite_reference | 13.566661145279816 | 26.053982412669857 | 4.75277948973584 |
| design_weighted | half_zero_aligned | 13.64252453806843 | 26.525001795123256 | 4.688055597703281 |
| design_weighted | norm_joint_zero | 13.739713257337598 | 27.044149189023084 | 4.793982350866708 |
| design_weighted | norm_Aimage | 13.659170072455359 | 26.859601705238706 | 4.793982350866708 |
| design_weighted | norm_Bimage | 13.733419725521841 | 26.93794211447542 | 5.029399635265411 |
| design_weighted | norm_joint_wrong | 13.588304121042142 | 26.80061922862645 | 4.688055597703281 |
| design_weighted | norm_B_wrong | 13.684126964029248 | 26.889053320846184 | 4.982455358714241 |
| design_weighted | norm_FA_FB_mean | 13.616389042580542 | 26.70494601300616 | 4.89027865861377 |
| design_weighted | norm_FAB_FB_mean | 13.642479898952432 | 26.75048335362547 | 4.936367008664006 |
| design_weighted | norm_direct | 13.643483266302125 | 26.90261572620692 | 5.029399635265412 |
| design_weighted | norm_wrong_direct | 12.904351468797019 | 24.995010514649366 | 4.944884792156855 |
| design_weighted | norm_TTA | 13.81960316688109 | 27.305335032677846 | 5.063686987738404 |
| design_weighted | norm_wrong_TTA | 13.686396172517528 | 27.007052866504143 | 4.8446748587279735 |

{
  "primary": "norm_TTA",
  "same_R142_loss_steps_and_Up": true,
  "R144_sourceonly_positionformula_fixed": true,
  "FINAL_access": false,
  "no_target_GT_numeric_parameter_or_per_direction_choice": true,
  "exposed_development_not_independent_confirmation": true,
  "endpoints": {
    "norm_TTA": {
      "comparisons": {
        "raw": {
          "norm_joint_zero": 0.08506895833424011,
          "norm_direct": 0.1826790958006015,
          "norm_wrong_TTA": 0.13023946454335444,
          "norm_Bimage": 0.0899723603839675,
          "norm_FA_FB_mean": 0.20326143674139452,
          "norm_FAB_FB_mean": 0.17695598047218652,
          "norm_joint_wrong": 0.20526060262782053,
          "image_B_TTA": 0.10008863430473625,
          "Frozen_A": 0.39887401728833183,
          "Frozen_F": 0.1960608826635042
        },
        "design_weighted": {
          "norm_joint_zero": 0.07988990954349084,
          "norm_direct": 0.1761199005789642,
          "norm_wrong_TTA": 0.13320699436356165,
          "norm_Bimage": 0.08618344135924794,
          "norm_FA_FB_mean": 0.20321412430054764,
          "norm_FAB_FB_mean": 0.17712326792865696,
          "norm_joint_wrong": 0.23129904583894678,
          "image_B_TTA": 0.12427928905758634,
          "Frozen_A": 0.4251397017269376,
          "Frozen_F": 0.2387419256337271
        }
      },
      "retained_development_candidate": true
    }
  },
  "disposition": "retain_normalized_reader_TTA_development_candidate",
  "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
}

The registeredprimary mustexceed itsownnormalizedjointzero, sameCdirect, matchedwrongTTA, normalizedpureB/bothcanonicalmeans andreport allpreviousR142/R144/FrozenA references inbothraw anddesignweightedmeans. No eachdirectionpositive requirement or endpoint-specific GTselection. Exposed-target development is disclosed; source-onlynumericcalibration doesnotmake this an independenttest. Ifcombinationfails, keepbestfrozeninputstate andstopthiscombination withoutloss/offsetgrid.

## Gains, harms and contribution concentration

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| norm_TTA | Frozen_A | 117 | 102 | 0.4251397017269392 | 0.053304207197498364 | -0.04909164122994028 | 119 | 181 | 17 |
| norm_TTA | norm_joint_zero | 52 | 47 | 0.07988990954349155 | 0.030260612572096602 | -0.03206973218682594 | 72 | 93 | 7 |
| norm_TTA | norm_direct | 86 | 86 | 0.1761199005789658 | 0.043932121024601284 | -0.03647242010323247 | 107 | 150 | 16 |
| norm_TTA | norm_wrong_TTA | 45 | 46 | 0.1332069943635605 | 0.043932121024601284 | -0.027670598778255808 | 69 | 85 | 9 |
| norm_TTA | norm_Bimage | 77 | 88 | 0.08618344135924885 | 0.034984151162616633 | -0.03205222459287011 | 105 | 144 | 11 |
| norm_TTA | norm_FA_FB_mean | 67 | 65 | 0.2032141243005483 | 0.034984151162616633 | -0.03205222459287011 | 94 | 122 | 11 |
| norm_TTA | norm_FAB_FB_mean | 66 | 66 | 0.1771232679286554 | 0.034984151162616633 | -0.03205222459287011 | 94 | 122 | 11 |
| norm_TTA | image_B_TTA | 53 | 49 | 0.12427928905758709 | 0.040096090214912596 | -0.036117173255342604 | 75 | 94 | 17 |
| norm_joint_zero | norm_Aimage | 6 | 2 | 0.0805431848822385 | 0.04124910190536532 | -0.0014437455841802508 | 7 | 8 | 4 |
| norm_joint_zero | norm_joint_wrong | 14 | 7 | 0.1514091362954546 | 0.053304207197498364 | -0.0018507138090753316 | 19 | 21 | 5 |
| norm_Bimage | norm_B_wrong | 20 | 11 | 0.049292761492592554 | 0.031481163809365754 | -0.012802773570742389 | 24 | 30 | 5 |

Allchangedrecord/scene/direction tables remain. Counts are boxIoU outcomes, not independentlyverified physicalobject identities. Repeated descriptions/source directions may shareframes andrecordings; no singleDrone scene populationCI or removaloflargestnegative. Acc25/50 are reported alongside mIoU, not assumed jointlyimproved.

## Postseal set and ranking diagnostics

| source | target | n | eligible | recoverable_zero_errors | recoverable_with_good25_in_S | recoverable_with_good25_in_C | changed | zero_best_already | best_outside_S | best_outside_C | C_internal_order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | 357 | 178 | 202 | 67 | 60 | 20 | 1 | 4 | 1 | 14 |
| drone | waymo | 176 | 50 | 71 | 14 | 7 | 2 | 0 | 0 | 1 | 1 |
| quad | drone | 366 | 154 | 276 | 116 | 47 | 20 | 1 | 1 | 12 | 6 |
| quad | waymo | 176 | 50 | 106 | 25 | 13 | 7 | 1 | 2 | 4 | 0 |
| waymo | drone | 366 | 154 | 179 | 65 | 32 | 25 | 0 | 0 | 11 | 14 |
| waymo | quad | 357 | 178 | 153 | 59 | 51 | 25 | 0 | 4 | 0 | 21 |

These use onlysealedexistingA256IoUs afterpredictions. They do notselectthresholds, candidategroups, gradients oroutputs. Good25 denotes a geometricboxquality proxy, not an instanceidentity label; changed-subsetclassifications cannotbe extrapolatedto allerrors.

## Measurement, resources and limits

{
  "decision": {
    "primary": "norm_TTA",
    "same_R142_loss_steps_and_Up": true,
    "R144_sourceonly_positionformula_fixed": true,
    "FINAL_access": false,
    "no_target_GT_numeric_parameter_or_per_direction_choice": true,
    "exposed_development_not_independent_confirmation": true,
    "endpoints": {
      "norm_TTA": {
        "comparisons": {
          "raw": {
            "norm_joint_zero": 0.08506895833424011,
            "norm_direct": 0.1826790958006015,
            "norm_wrong_TTA": 0.13023946454335444,
            "norm_Bimage": 0.0899723603839675,
            "norm_FA_FB_mean": 0.20326143674139452,
            "norm_FAB_FB_mean": 0.17695598047218652,
            "norm_joint_wrong": 0.20526060262782053,
            "image_B_TTA": 0.10008863430473625,
            "Frozen_A": 0.39887401728833183,
            "Frozen_F": 0.1960608826635042
          },
          "design_weighted": {
            "norm_joint_zero": 0.07988990954349084,
            "norm_direct": 0.1761199005789642,
            "norm_wrong_TTA": 0.13320699436356165,
            "norm_Bimage": 0.08618344135924794,
            "norm_FA_FB_mean": 0.20321412430054764,
            "norm_FAB_FB_mean": 0.17712326792865696,
            "norm_joint_wrong": 0.23129904583894678,
            "image_B_TTA": 0.12427928905758634,
            "Frozen_A": 0.4251397017269376,
            "Frozen_F": 0.2387419256337271
          }
        },
        "retained_development_candidate": true
      }
    },
    "disposition": "retain_normalized_reader_TTA_development_candidate",
    "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
  },
  "source_calibration": {
    "status": "LOCKED_SOURCE_ONLY_IMAGE_READER",
    "quantile": 0.9,
    "source_eligible_counts": {
      "waymo": 22,
      "drone": 18,
      "quad": 37
    },
    "kappa": {
      "waymo": 2.29706174619496,
      "drone": 1.4053959552580049,
      "quad": 0.6205128759145743
    },
    "source_labels_used": true,
    "target_GT_parameter_choice": false,
    "source_weight_training": false,
    "source_reader": "R144positional_reference_Fpoints_Aimage_and_Fpoints_Bimage_full_native_competition",
    "source_prediction_seal_sha256": "beffcf4dc2fbf0610dc725918210be04c952b13f74733363d0f7cfbb2efa3a36",
    "FINAL_access": false,
    "utc": "2026-11-17T19:08:32.174787+00:00"
  },
  "source_preparation": {
    "n": 313,
    "sampledraws": 313,
    "max_seed_bridge_error": 3.885696710752532e-06,
    "source_own_H_from_R144": {
      "waymo": -0.07376708984375,
      "drone": -1.8500963568687439,
      "quad": -0.8063075244426727
    },
    "no_source_or_target_GT_or_model_read": true,
    "noB_source_delta_zero": true
  },
  "source_audit": {
    "status": "PASS",
    "n": 313,
    "B_inputs": 113,
    "source_views": 626,
    "max_CPU_full_reader_probability_error": 1.0001617029575804e-06,
    "max_independent_B_graph_weight_error": 0.0,
    "max_CPU_position_MLP_error": 1.1444091796875e-05,
    "source_Fpoints_Aimage_Bimage_reader": true,
    "same_qualification_as_R132": true,
    "GT_read": false,
    "fullTransformer_not_CPU_regenerated": true,
    "FINAL_access": false
  },
  "image_audit": {
    "status": "PASS",
    "n": 1798,
    "B_inputs": 764,
    "effective_wrong_B_inputs": 754,
    "max_CPU_full_reader_error": 1.6016652828465539e-06,
    "max_independent_graph_weight_error": 0.0,
    "max_CPU_position_MLP_error": 1.1444091796875e-05,
    "native_A256_noB_and_reused_references_exact": true,
    "source_parameters_updated": 0,
    "fullTransformer_not_independently_CPU_recomputed": true,
    "FINAL_access": false
  },
  "TTA_audit": {
    "status": "PASS",
    "n": 1798,
    "fit_count": 1506,
    "max_CPU_probability_error": 1.1006910001776227e-06,
    "max_state_error": 0.0,
    "max_Ffeature_error": 1.430511474609375e-06,
    "max_objective_error": 4.656618979304028e-10,
    "full_encoder_gradients_not_CPU_regenerated": true,
    "evidence_C_target_and_direct_reconstructed": true,
    "original_rho0_3steps_Up4896_and_source_only_initialization": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "task_values": 52142,
    "max_error": 0.0,
    "all_negative_noB_and_identitywrong_kept": true
  },
  "runtime": {
    "source": {
      "n": 313,
      "B_inputs": 113,
      "complete_reader_forwards": 626,
      "max_original_A_bridge_error": 0.0,
      "source_or_target_GT_read": false,
      "source_training": false
    },
    "image": {
      "n": 1798,
      "full_reader_forwards": 3056,
      "max_R144_zero_bridge_error": 0.0,
      "parameters_updated": 0,
      "target_GT_read": false,
      "FINAL_access": false
    },
    "method": {
      "n": 1798,
      "fits": 1506,
      "GT_read": false,
      "primary": "norm_TTA",
      "source_kappas_sha256": "1c9066c93e4ea7bd70b2f57a585ca659f6574d45f57a044914b0c59e1a4b6ba6",
      "FINAL_access": false
    }
  },
  "parameter_updates": "only original Up4896, sourceonlyinitialize perinput",
  "source_weight_training": false,
  "FINAL_access": false,
  "ancillary_metadata": {
    "VISUAL_SCALE_METADATA_ONLY": {
      "status": "FROZEN_VISIBLE_SUPPORT_METADATA_ONLY",
      "summary": [
        {
          "cohort": "source_own",
          "source": "drone",
          "target": "drone",
          "n": 100,
          "at_least_two_projected_support_points": 91,
          "median_min_side_cells": 1.3687621895223856,
          "min_side_le_one_cell_fraction": 0.32967032967032966,
          "min_side_le_two_cells_fraction": 0.6263736263736264
        },
        {
          "cohort": "source_own",
          "source": "quad",
          "target": "quad",
          "n": 94,
          "at_least_two_projected_support_points": 84,
          "median_min_side_cells": 2.0452922033146024,
          "min_side_le_one_cell_fraction": 0.19047619047619047,
          "min_side_le_two_cells_fraction": 0.47619047619047616
        },
        {
          "cohort": "source_own",
          "source": "waymo",
          "target": "waymo",
          "n": 119,
          "at_least_two_projected_support_points": 117,
          "median_min_side_cells": 3.6552812941372395,
          "min_side_le_one_cell_fraction": 0.07692307692307693,
          "min_side_le_two_cells_fraction": 0.2222222222222222
        },
        {
          "cohort": "target_exposed",
          "source": "drone",
          "target": "quad",
          "n": 357,
          "at_least_two_projected_support_points": 319,
          "median_min_side_cells": 2.0567580237984657,
          "min_side_le_one_cell_fraction": 0.1974921630094044,
          "min_side_le_two_cells_fraction": 0.46394984326018807
        },
        {
          "cohort": "target_exposed",
          "source": "drone",
          "target": "waymo",
          "n": 176,
          "at_least_two_projected_support_points": 158,
          "median_min_side_cells": 3.818581659346819,
          "min_side_le_one_cell_fraction": 0.13924050632911392,
          "min_side_le_two_cells_fraction": 0.2848101265822785
        },
        {
          "cohort": "target_exposed",
          "source": "quad",
          "target": "drone",
          "n": 366,
          "at_least_two_projected_support_points": 301,
          "median_min_side_cells": 2.3986296225339174,
          "min_side_le_one_cell_fraction": 0.2059800664451827,
          "min_side_le_two_cells_fraction": 0.42857142857142855
        },
        {
          "cohort": "target_exposed",
          "source": "quad",
          "target": "waymo",
          "n": 176,
          "at_least_two_projected_support_points": 159,
          "median_min_side_cells": 2.0372394034639,
          "min_side_le_one_cell_fraction": 0.23270440251572327,
          "min_side_le_two_cells_fraction": 0.4779874213836478
        },
        {
          "cohort": "target_exposed",
          "source": "waymo",
          "target": "drone",
          "n": 366,
          "at_least_two_projected_support_points": 355,
          "median_min_side_cells": 4.0388518664985895,
          "min_side_le_one_cell_fraction": 0.08169014084507042,
          "min_side_le_two_cells_fraction": 0.19436619718309858
        },
        {
          "cohort": "target_exposed",
          "source": "waymo",
          "target": "quad",
          "n": 357,
          "at_least_two_projected_support_points": 345,
          "median_min_side_cells": 1.9985993430018425,
          "min_side_le_one_cell_fraction": 0.12753623188405797,
          "min_side_le_two_cells_fraction": 0.5043478260869565
        }
      ],
      "image_input": [
        384,
        256
      ],
      "CNN_grid": [
        48,
        32
      ],
      "CNN_stride": 8,
      "CNN_receptive_field_pixels": 15,
      "support_span_not_true_object_extent": true,
      "point_sparse_support_can_underestimate_true_extent": true,
      "winner_can_be_wrong_no_identity_certification": true,
      "GT_or_new_model_images_or_taskscore_read": false,
      "R145_predictor_unchanged": true,
      "no_resolution_method_registered_or_run": true,
      "FINAL_access": false
    },
    "VISUAL_SCALE_METADATA_REPAIR": {
      "ancillary_only": true,
      "first_source_flag_mismatches": 197,
      "cause": "candidatepairpresence was incorrectly named hasB; validregistration is actualvalid_B.any",
      "summary_n_or_span_statistics_unchanged": true,
      "all_primary_source_image_TTA_producers_use_actualvalidB_already": true,
      "no_model_or_taskscore_rerun": true,
      "failed_metadata_attempt_retained_PRIVATE": true
    },
    "VISUAL_SCALE_METADATA_AUDIT": {
      "status": "PASS",
      "source_rows": 313,
      "source_actual_B": 113,
      "old_summary_span_statistics_exactly_unchanged": true,
      "source_hasB_flag_now_actual_validmask": true,
      "primary_TTA_or_image_or_source_producer_changed": false,
      "no_model_or_taskscore_rerun": true
    },
    "COMPONENT_CONFLICT_METADATA_ONLY": {
      "status": "POSTSEAL_HISTORICAL_COMPONENT_METADATA_ONLY",
      "rows": [
        {
          "gradient_bin": "negative_cosine",
          "old_R143_reference_control": "F_RGBB_only",
          "n": 310,
          "full_R142_better": 20,
          "full_R142_worse": 13,
          "weighted_contribution_with_original_full_denominator_pp": 0.06286015589166782
        },
        {
          "gradient_bin": "negative_cosine",
          "old_R143_reference_control": "partial_only",
          "n": 310,
          "full_R142_better": 17,
          "full_R142_worse": 17,
          "weighted_contribution_with_original_full_denominator_pp": 0.10829103505575004
        },
        {
          "gradient_bin": "negative_cosine",
          "old_R143_reference_control": "rank_only",
          "n": 310,
          "full_R142_better": 17,
          "full_R142_worse": 12,
          "weighted_contribution_with_original_full_denominator_pp": 0.032758988833121505
        },
        {
          "gradient_bin": "nonnegative_cosine",
          "old_R143_reference_control": "F_RGBB_only",
          "n": 412,
          "full_R142_better": 23,
          "full_R142_worse": 28,
          "weighted_contribution_with_original_full_denominator_pp": -0.03686713982774976
        },
        {
          "gradient_bin": "nonnegative_cosine",
          "old_R143_reference_control": "partial_only",
          "n": 412,
          "full_R142_better": 10,
          "full_R142_worse": 12,
          "weighted_contribution_with_original_full_denominator_pp": -0.03535912623186491
        },
        {
          "gradient_bin": "nonnegative_cosine",
          "old_R143_reference_control": "rank_only",
          "n": 412,
          "full_R142_better": 24,
          "full_R142_worse": 20,
          "weighted_contribution_with_original_full_denominator_pp": 0.00861653205899407
        },
        {
          "gradient_bin": "one_component_zero",
          "old_R143_reference_control": "F_RGBB_only",
          "n": 42,
          "full_R142_better": 0,
          "full_R142_worse": 1,
          "weighted_contribution_with_original_full_denominator_pp": -0.0013689245364409436
        },
        {
          "gradient_bin": "one_component_zero",
          "old_R143_reference_control": "partial_only",
          "n": 42,
          "full_R142_better": 0,
          "full_R142_worse": 0,
          "weighted_contribution_with_original_full_denominator_pp": 0.0
        },
        {
          "gradient_bin": "one_component_zero",
          "old_R143_reference_control": "rank_only",
          "n": 42,
          "full_R142_better": 0,
          "full_R142_worse": 1,
          "weighted_contribution_with_original_full_denominator_pp": -0.0013689245364409436
        }
      ],
      "firstgradient_only_not_all_later_steps": true,
      "not_R145_gradients_or_results": true,
      "no_new_model_or_GT_cache_or_taskscore": true,
      "not_deployment_gradient_gate_or_new_optimizer": true,
      "FINAL_access": false
    }
  }
}

IndependentCPU audits source/targetactualBprojection/permutation/visibility graphs, positionalMLP andsixpositions, completefinalmatching/livecompetition fromnativefeatures, originalA256/oldreference/noB/meanidentities. It reconstructs targetC/teacher/wrongpermutation/directreadout, acceptedparameterdeltas/Armijo/sourcefeaturekeep/objectives; engineeringfinite differences verify the actualfull differentiable path. FullTransformerbackprop isnot independentlyregeneratedCPU. Predictionsseal beforetaskGTscore; allsavedscalars independentlyrecalculated. UniqueGPU/deadlineguard, nonewmodel/training/scoringafter09:00BeijingNov18. Jobtimings are incrementalresearchcosts, not coldstartdeploymentlatency; commonprefix/backbone/registration/routing andpriorcompatiblepredictions are reused andnotfreein deployment. No newsourceweighttraining/pretrainedexpert/FINAL. Rawmedia/descriptions/coordinates/IDs/features/weights/secrets/thirdpartysnapshots remainprivate. Parent>=2pp overintactFrozenA13.394463% remains aseparateverifiedgoal; GTdiagnostic or a weakerresetbaseline nevercounts.

## Ancillary evidence, no predictor changes

Frozen visible-support spans are estimated in the48x32 CNNgrid, not certifiedobjectextent; sparsepoints andwrongcurrentwinners can bias the measurement. No universal targetsmallness premise is established, and no resolutionmethod isregistered. The first metadataattempt incorrectly named candidatepairpresence hasB;197sourceflags were corrected to actualvalid_B.any. Allspanaggregate statistics areidentical, andprimarysource/image/TTA producers alreadyusedactualvalid_B, so no model/taskscore wasrerun. Bothlocalmetadataattempts are retained.

HistoricalR143 first-gradient descriptions do not support treating everynegative partial-versus-rank cosine as harmful: the310negativecosine cases contribute+.062860pp forfullR142 overownzero, while412nonnegativecases contribute-.036867pp and42one-zero cases-.001369pp usingoriginalfull denominators. This is exposedpostseal association/firststep only, notR145gradient evidence, a deploymentgate or a proof to applygradientsurgery. No optimizer/gate is changed fromthese descriptions.


## Task-effect limits and next-action boundary

Relative to the normalizedjointzero, primaryTTA has52IoU improvements and47harms, withpositivecontributions+.320097pp andnegative-.240207pp, net+.079890pp. Largestpositive+.030261pp isbelowthenet. Three directions improve andthree decline; allremain. Waymo-to-Drone+.373212pp/Waymo-to-Quad+.266064pp, Quad-to-Drone+.066181pp; Drone-to-Quad-.077257pp/Drone-to-Waymo-.014674pp/Quad-to-Waymo-.134187pp.

An intentionallynondeployable GTdiagnostic bounds a narrowstrategy: choosing thebetter ofcompleteFrozenA andthe CURRENT R145endpoint perrecord wouldgain only+.968593pp, reaching14.363057%. Itcannot reachthe15.394463% (2pp)goal merelybyperfectlyaccepting/revertingthis fixedendpoint. ThisdoesNOTbound newviews, differentvalidupdates orallTTA. Therefore furtheracceptancegatesalone cannotbe advertisedasasufficient2ppplan; additionalcorrectendpoints/evidence wouldbeneeded. Labelsnever implement thisbinarychoice.
