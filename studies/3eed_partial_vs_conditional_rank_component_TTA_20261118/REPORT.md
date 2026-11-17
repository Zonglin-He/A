# R143 — Existing partial and conditional-rank objective components

## Decision

**keep_R142_full_reference_no_component_replacement**. R142 full partial+rank+keep remains the predeclared reference, unchanged. Correctcomponent endpoints and controls were registered before any new model/taskscore. All1798 exposed ordinary inputs and allnegative comparisons remain; no new independent confirmation or per-direction bestobjective is claimed.

## Fixed versus changed

Same intact three source-prepared model states, Fpoint1024, realRGB A/B, full native live matching/competition, source-own R142 image-reader kappas, originalS/C/Bteacher/grouping/routing, Up4896, originalrho0/3steps/9Armijo and source-adapted featurekeep1. No source training, expert, EMA, module, newframe/Bchoice, candidatebox/queryposition change, visibilityhardgate or new auxiliary loss. All1034 noB cases remain exactcompleteFrozenA.

Two ablations remove oneexistingtaskterm: partial_only usesLpartial(S,C)+keep; rank_only usesKL(frozenBgroupMAX|C || studentABgroupMAX|C)+keep. This tests dependence/interference, not newteacher correctness or a new architecture. Full R142 correct/wrong andallcompatible no-update/reference predictions are byte-reused instead of rerun.

The partial-only wrongcontrol permutes C membership withinS, fixedgroupmultiplicity and one-bitlog2 frozenstudent-groupMEAN confidence bins. Ccardinality, group-size distribution and S stayfixed; originalgroupmass ismatched withinfactor2, notexactly. Source/targetGT never chooses the permutation, binwidth, sample, output or update. Ineffective wrongmembership permutations are honestly identical/reused. This is a closer-confidence setcontrol than unconstrainedset derangement, still not identity certification. Rank-only wrongcontrol is the exactold withinC teacher-probability permutation withmass/entropy/multiplicity fixed.

Partialdirect moves onlySprobabilitymass ontoC; rankdirect changes onlywithinCgroupMAX proportions, preservingCmass/alloutsideC. Bothwrongdirectcontrols use their own exactcorrespondingwrong signal. FrozenpureBimage and bothcanonical equal-probability means remain asstrong simpleinformation controls. These are internalcontrols, not published baselines.

IfC=S thepartialtask isidenticallyzero; ifCsingleton theconditionalranktask isidenticallyzero. With source-anchoredkeep minimized atθ0, those cases exactlyreusezero instead of forcing a gradient. NoB, algebraiczero andidentitywrong cases remain in every evaluation denominator.

## Whole-cohort outcomes

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
| raw | FA_FB_probability_mean | 13.535399 | 25.724610 | 5.343193 |
| raw | FAB_FB_probability_mean | 13.547931 | 25.816833 | 5.343193 |
| raw | direct_partial | 13.672663 | 26.128171 | 5.154947 |
| raw | direct_wrong_partial | 13.679496 | 26.128171 | 5.154947 |
| raw | direct_rank | 13.603451 | 25.996329 | 5.389878 |
| raw | direct_wrong_rank | 12.909893 | 24.972518 | 5.252118 |
| raw | partial_only | 13.649047 | 26.404839 | 5.200484 |
| raw | partial_only_wrong_C | 13.631413 | 26.265753 | 5.200484 |
| raw | rank_only | 13.651443 | 26.042045 | 5.439038 |
| raw | rank_only_wrong | 13.405093 | 25.945052 | 5.250970 |
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
| design_weighted | FA_FB_probability_mean | 13.537643 | 26.025512 | 4.859257 |
| design_weighted | FAB_FB_probability_mean | 13.550181 | 26.117994 | 4.859257 |
| design_weighted | direct_partial | 13.673157 | 26.418865 | 4.688056 |
| design_weighted | direct_wrong_partial | 13.679989 | 26.418865 | 4.688056 |
| design_weighted | direct_rank | 13.604416 | 26.286712 | 4.896266 |
| design_weighted | direct_wrong_rank | 12.891376 | 25.148566 | 4.789778 |
| design_weighted | partial_only | 13.622392 | 26.606426 | 4.733593 |
| design_weighted | partial_only_wrong_C | 13.591164 | 26.441237 | 4.733593 |
| design_weighted | rank_only | 13.655317 | 26.342880 | 4.944550 |
| design_weighted | rank_only_wrong | 13.410991 | 26.250984 | 4.792253 |

{
  "predeclared_reference": "R142 full partial+rank+keep",
  "original_R142_adoption_unchanged": true,
  "new_architecture_or_loss": false,
  "endpoints": {
    "partial_only": {
      "comparisons": {
        "raw": {
          "image_B_TTA": -0.05277903054689048,
          "F_RGBB_only": -0.021366091468287607,
          "direct_partial": -0.023616032104254003,
          "partial_only_wrong_C": 0.017634215006660625,
          "F_Bimage_only_teacher": -0.01734505848572354,
          "FA_FB_probability_mean": 0.11364781504448196,
          "FAB_FB_probability_mean": 0.10111623328707431,
          "Frozen_A": 0.2460063524367051,
          "Frozen_F": 0.04319321781187746
        },
        "design_weighted": {
          "image_B_TTA": -0.072931908823886,
          "F_RGBB_only": -0.04830781729640954,
          "direct_partial": -0.05076497207675423,
          "partial_only_wrong_C": 0.03122780571548489,
          "F_Bimage_only_teacher": -0.05960512224215009,
          "FA_FB_probability_mean": 0.08474867441477052,
          "FAB_FB_probability_mean": 0.0722109749649178,
          "Frozen_A": 0.22792850384546526,
          "Frozen_F": 0.04153072775225475
        }
      },
      "retained_development_candidate": false
    },
    "rank_only": {
      "comparisons": {
        "raw": {
          "image_B_TTA": -0.050383409386490996,
          "F_RGBB_only": -0.01897047030788812,
          "direct_rank": 0.04799127403685155,
          "rank_only_wrong": 0.24634925656013706,
          "F_Bimage_only_teacher": -0.014949437325324055,
          "FA_FB_probability_mean": 0.11604343620488144,
          "FAB_FB_probability_mean": 0.1035118544474738,
          "Frozen_A": 0.2484019735971046,
          "Frozen_F": 0.045588838972276946
        },
        "design_weighted": {
          "image_B_TTA": -0.04000659635567416,
          "F_RGBB_only": -0.015382504828197696,
          "direct_rank": 0.050901439027057904,
          "rank_only_wrong": 0.24432613167139827,
          "F_Bimage_only_teacher": -0.026679809773938246,
          "FA_FB_probability_mean": 0.11767398688298236,
          "FAB_FB_probability_mean": 0.10513628743312964,
          "Frozen_A": 0.2608538163136771,
          "Frozen_F": 0.07445604022046659
        }
      },
      "retained_development_candidate": false
    }
  },
  "FINAL_access": false,
  "no_target_GT_hyperparameter_or_per_direction_output_choice": true,
  "exposed_development_structure_ablation_not_independent_test": true,
  "disposition": "keep_R142_full_reference_no_component_replacement",
  "partial_wrong_confidence_mass_only_factor2_matched_not_exact_identity_certainty": true
}

No ablation retrospectively replaces R142 or becomes an independent test. A positiveendpoint must exceedfullreference, ownzero/direct, appropriatewrong andstrongsimplefrozen controls inboth means under the registered rule. Labels are used only aftersealedpredictions fordevelopmentassessment; numeric source-only parameters stayfixed, but choosingobjective structure fromexposedtarget results isexplicitly target-guideddevelopment. Ifneithercomponentaloneimproves, retainR142 ratherthan open a new lossweightgrid.

## First-gradient and control-mass checks

| source | target | eligible | both_gradients_nonzero | cosine_mean | cosine_negative_fraction | partial_gradient_mean | rank_gradient_mean | max_sum_residual | effective_wrong_C | correct_C_mass_mean | wrong_C_mass_mean | effective_C_mass_ratio_min | effective_C_mass_ratio_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | 178 | 166 | 0.1978161705809816 | 0.28313253012048195 | 2.8910471460522653 | 0.17310116575679246 | 6.362674865239623e-07 | 95 | 0.5111040159352895 | 0.5058848560787248 | 0.8931774761555374 | 1.0010878237831606 |
| drone | waymo | 50 | 45 | 0.22781908144439594 | 0.28888888888888886 | 2.456362052380458 | 0.2652266668490356 | 6.730054424171743e-07 | 34 | 0.49278739019816925 | 0.4853088775668459 | 0.9291754334550416 | 0.9943944974099043 |
| quad | drone | 154 | 148 | -0.029695611514327615 | 0.5472972972972973 | 2.518274892039862 | 1.1229787171546217 | 1.175509658319971e-06 | 107 | 0.38846133673536676 | 0.3788964792272048 | 0.8434627568080809 | 1.0891998113453287 |
| quad | waymo | 50 | 39 | -0.045231234557714654 | 0.5128205128205128 | 3.4791573099724995 | 2.1180544542921624 | 1.222955984302248e-06 | 28 | 0.3915180416302577 | 0.3806935278450556 | 0.745319705503788 | 1.0471129897921159 |
| waymo | drone | 154 | 154 | 0.007591599484545002 | 0.4935064935064935 | 0.9726730642638485 | 0.8109886820733619 | 1.1000157761702512e-06 | 107 | 0.8518067242457825 | 0.8482961944822998 | 0.9311774742596618 | 1.0029106841492252 |
| waymo | quad | 178 | 170 | 0.13451561719250313 | 0.4294117647058823 | 1.1432057559625637 | 0.8956863913936854 | 2.0446400058616875e-06 | 107 | 0.805573424708426 | 0.8020156910645252 | 0.9080673660540503 | 1.0012046921447386 |

At the same source state the keepgradient iszero, so separatelycomputed partialgradient+rankgradient must equal the archived R142 fullfirstgradient. This check confirms a truecomponent ablation; it is not a claim gradientagreementmeans bettergrounding. The normalizedresidual uses the sumofcomponentnorms, remaining well-defined undercancellation. Negativecosine describes an optimization interaction, not proof eitherterm iswrong. Allsource initializations andlate/competitionpaths remainactive.

## All positive and negative outcomes

| method | control | better | worse | changed_content_groups | changed_frames | changed_scene_groups | weighted_gain_pp | largest_positive_pp | largest_negative_pp |
|---|---|---|---|---|---|---|---|---|---|
| partial_only | image_B_TTA | 29 | 27 | 45 | 55 | 6 | -0.07293190882388514 | 0.025596129035886747 | -0.053304207197498364 |
| partial_only | F_RGBB_only | 44 | 38 | 64 | 78 | 5 | -0.04830781729640799 | 0.026007183165523858 | -0.046091117295781324 |
| partial_only | direct_partial | 58 | 57 | 84 | 103 | 12 | -0.05076497207675347 | 0.040535899868321686 | -0.05407119513360017 |
| partial_only | partial_only_wrong_C | 11 | 8 | 19 | 19 | 6 | 0.03122780571548691 | 0.040535899868321686 | -0.01426499326467251 |
| rank_only | image_B_TTA | 33 | 41 | 64 | 70 | 7 | -0.040006596355674634 | 0.046091117295781324 | -0.026007183165523858 |
| rank_only | F_RGBB_only | 42 | 50 | 69 | 86 | 8 | -0.015382504828197478 | 0.053304207197498364 | -0.02121788318256053 |
| rank_only | direct_rank | 57 | 63 | 81 | 105 | 8 | 0.0509014390270578 | 0.053304207197498364 | -0.022334397879461065 |
| rank_only | rank_only_wrong | 74 | 58 | 91 | 116 | 9 | 0.24432613167139627 | 0.04153914610760538 | -0.02121788318256053 |

Changedrecord counts are boxIoU comparisons, not verifiedinstance identities or independent scenes. Direction/recording tables and every changedrecord use stable anonymous clusters; repeateddescriptions/source directions may sharephysicalframes. Largecasecontributions are reported without deletion. No per-input/perscene bestendpoint or annotation-filtered adaptation.

## Verification and resource boundary

{
  "status": "PASS",
  "n": 13,
  "fit_count": 21,
  "max_CPU_full_reader_error": 6.645248878944798e-07,
  "max_state_error": 0.0,
  "max_Ffeature_error": 1.430511474609375e-06,
  "max_objective_error": 2.3283065753165744e-10,
  "max_component_gradient_sum_relative_error": 8.173162917002113e-07,
  "max_independent_direct_error": 0.0,
  "all_R142_references_exact_and_noB_complete_A": true,
  "fullTransformer_gradients_not_independently_CPU_regenerated": true,
  "new_loss_or_sourceweight_training": false,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "fit_count": 2704,
  "max_CPU_full_reader_error": 1.0579377210662155e-06,
  "max_state_error": 0.0,
  "max_Ffeature_error": 1.9073486328125e-06,
  "max_objective_error": 4.656617313969491e-10,
  "max_component_gradient_sum_relative_error": 2.0446400058616875e-06,
  "max_independent_direct_error": 0.0,
  "all_R142_references_exact_and_noB_complete_A": true,
  "fullTransformer_gradients_not_independently_CPU_regenerated": true,
  "new_loss_or_sourceweight_training": false,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "task_values": 41354,
  "max_error": 0.0,
  "all_negative_unknown_and_noB_rows_retained": true,
  "FINAL_access": false
}

CPUindependently checks allnative finalhead probabilities, acceptedparameter deltas andArmijo decisions, source-adaptedF featurekeep, enabledobjective term, confidence strata/wrongmembership/directcontrols and thegradientcomponent sum. Everyfrozen/reference/nobox/noB identity ischecked. WholeTransformerbackprop isnot independently regenerated onCPU; engineeringfinite differences check the actualpath. Predictionsseal before newtaskIoUs. Newfits andacceptedpathnorms are reported separately fromcompatiblecache reuse. Sourcecalibration/frozenbackbone/registration/commonimage encoding costs inheritedfromR142 are not silentlyfree in a cold-startdeployment; measuredjobs areincrementalresearchruntime. Allmodels use auniqueGPU andharddeadlineguard.

Sourceweights, rawmedia/descriptions/coordinates/IDs/features/gradientvectors/keys andthirdpartysnapshots remainprivate. NoFINAL. The parentgoal>=2pp overintactFrozenA13.394463 isseparate; GTdiagnostics, weakerreset baselines andsourcepreparation gains do not count asnewTTAincrement.
