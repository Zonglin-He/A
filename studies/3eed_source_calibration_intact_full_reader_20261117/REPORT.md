# R140 — Source calibration matched to intact complete readout

## Decision

**matched_calibration_repair_not_sufficient_for_TTA_task_gain**. The primary was fixed as fullreader_ownsource before source thresholds and new target results. Other factorial levels are ablations, not candidates selected by target score. All1798 target inputs are exposed ordinary development, with every no-B/no-op/negative retained; FINAL untouched.

## Concrete dependency mismatch

R132 source132 explicitly calls start123, resetting source-trained pointAdapterUp and gateP, then reads body127 with fixedsourceA lateRGB/residual. R135 repaired initialization and complete live competition, but R135–R139 retained that older calibrationkappa. This experiment repairs that dependency while keeping weights, architecture, originalA256/querypositions, registration, qualification and update math fixed. A mismatch in calibration provenance alone does not prove its repair improves grounding.

## Source-only threshold factorial

Exact313 previousR132 source-own inputs, same sourceA/B sensor pairs and geometric registrations; existing stored source-prepared routes frozen. SourceR91 preparation trained only interface after those routes, so no labels are used to reconstruct or choose routing. Intact source-adapter/gates and complete live matching/competition are replayed for A/B. Sourcepredictionproducer never parses source labels; predictions sealed and independentlyCPUaudited first. Source labels then identify groups whose best candidateIoU>.25, exactly as priorcalibration. For each eligible view calculate max(log(groupMEANprobability)) minus bestgoodgroup logmeanprobability; use fixed90thpercentile. No sourceweight training, target-GT threshold choice, expert or new network.

Oldreader pooled/own-source gaps reuse sealed source calibration; fullreader pooled/own-source use new complete-readout gaps. Pooling all three source labelsets is not strict per-direction source-only; only own-source variants satisfy that calibration boundary. Each own-source threshold is shared by both outgoing directions, with the same quantile and qualification rule. Small source eligible counts limit calibration precision; no target-fitted confidence interval or bestdirection output.

| reader | pooling | source | kappa |
|---|---|---|---|
| oldreader | pooled | waymo | 2.742430260548657 |
| oldreader | pooled | drone | 2.742430260548657 |
| oldreader | pooled | quad | 2.742430260548657 |
| oldreader | ownsource | waymo | 2.7793937504291555 |
| oldreader | ownsource | drone | 2.866890507412273 |
| oldreader | ownsource | quad | 2.0174857914447784 |
| fullreader | pooled | waymo | 1.58659678828853 |
| fullreader | pooled | drone | 1.58659678828853 |
| fullreader | pooled | quad | 1.58659678828853 |
| fullreader | ownsource | waymo | 2.237389163672925 |
| fullreader | ownsource | drone | 1.211397770880211 |
| fullreader | ownsource | quad | 0.39297581911087054 |

## Unchanged GT-free update and matched controls

Same R136 episodic source-centered Up4896 adaptation: partial(S,C)+KL(frozenB groupMAX conditionalC || completeA groupMAX conditionalC)+source-adapted featurekeep1. Original3steps/rho0/9-levelArmijo and fullA final output. C is the union of source-threshold-qualified groupMEAN candidates from frozen A/B, preserving UNKNOWN. Fixed group/box geometry, support>=4 and Bcoverage>=.5. Correct and wrong target controls share C/mass/entropy/groupmultiplicity; only B conditional probabilities permute within equal-size groups. This tests internal ordering and does not certify target identity.

All1034 noB inputs fall back exactly to completeFrozenA. FrozenA/F,ABmean,R135A/F,R136conditionalB-to-A/itswrong/direct are compatible reused references, never warmstarts. Each new threshold has its own direct projection and same-budget wrong update. If complete C and teacher are byteidentical to R136 or an earlier factorial level, reuse exactly compatible fitted prediction rather than rerun an unchanged experiment; every reuse has explicit provenance. Engineering still tests changed evidence in all6directions with native fullreadout/gradientfinite differences before formal.

## Whole-cohort results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.403041 | 25.899515 | 5.056806 |
| raw | Frozen_F | 13.605854 | 25.898010 | 5.154947 |
| raw | AB_mean | 13.476794 | 25.712417 | 5.061576 |
| raw | R135_A | 13.506393 | 26.220572 | 5.102343 |
| raw | R135_F | 13.625244 | 26.175826 | 5.343193 |
| raw | C_B_A | 13.472877 | 25.937807 | 5.060428 |
| raw | wrong_C_B_A | 13.235829 | 25.620194 | 5.061576 |
| raw | direct_C_B_A | 13.483204 | 26.126845 | 5.103491 |
| raw | oldreader_ownsource_direct | 13.447401 | 26.081307 | 5.103491 |
| raw | oldreader_ownsource_wrong_direct | 11.923689 | 22.684301 | 4.777664 |
| raw | oldreader_ownsource_correct | 13.454797 | 25.846733 | 5.060428 |
| raw | oldreader_ownsource_wrong | 13.288771 | 25.670502 | 5.061576 |
| raw | fullreader_pooled_direct | 13.446130 | 25.987758 | 5.103491 |
| raw | fullreader_pooled_wrong_direct | 12.615515 | 24.001923 | 5.013743 |
| raw | fullreader_pooled_correct | 13.487916 | 25.801374 | 5.105966 |
| raw | fullreader_pooled_wrong | 13.502572 | 25.709151 | 5.292885 |
| raw | fullreader_ownsource_direct | 13.496916 | 26.078833 | 5.103491 |
| raw | fullreader_ownsource_wrong_direct | 12.706742 | 24.183102 | 5.153977 |
| raw | fullreader_ownsource_correct | 13.453385 | 25.751066 | 4.965731 |
| raw | fullreader_ownsource_wrong | 13.539567 | 26.086434 | 5.103491 |
| design_weighted | Frozen_A | 13.394463 | 26.263306 | 4.599297 |
| design_weighted | Frozen_F | 13.580861 | 26.099350 | 4.688056 |
| design_weighted | AB_mean | 13.447193 | 25.914692 | 4.595879 |
| design_weighted | R135_A | 13.498989 | 26.600013 | 4.644834 |
| design_weighted | R135_F | 13.606614 | 26.393334 | 4.859257 |
| design_weighted | C_B_A | 13.459843 | 26.270810 | 4.601034 |
| design_weighted | wrong_C_B_A | 13.214720 | 25.962793 | 4.601585 |
| design_weighted | direct_C_B_A | 13.477422 | 26.444000 | 4.639679 |
| design_weighted | oldreader_ownsource_direct | 13.441630 | 26.398463 | 4.639679 |
| design_weighted | oldreader_ownsource_wrong_direct | 11.900546 | 22.948130 | 4.386947 |
| design_weighted | oldreader_ownsource_correct | 13.445389 | 26.179735 | 4.601034 |
| design_weighted | oldreader_ownsource_wrong | 13.272893 | 26.013089 | 4.601585 |
| design_weighted | fullreader_pooled_direct | 13.440020 | 26.305444 | 4.639679 |
| design_weighted | fullreader_pooled_wrong_direct | 12.576334 | 24.298051 | 4.552928 |
| design_weighted | fullreader_pooled_correct | 13.477875 | 26.133329 | 4.646571 |
| design_weighted | fullreader_pooled_wrong | 13.485098 | 26.040500 | 4.822938 |
| design_weighted | fullreader_ownsource_direct | 13.510591 | 26.453013 | 4.639679 |
| design_weighted | fullreader_ownsource_wrong_direct | 12.725273 | 24.568062 | 4.716498 |
| design_weighted | fullreader_ownsource_correct | 13.439139 | 26.084647 | 4.508222 |
| design_weighted | fullreader_ownsource_wrong | 13.540094 | 26.483505 | 4.645385 |

{
  "primary": "fullreader_ownsource_correct",
  "all_levels_reported_no_target_best_selection": true,
  "strict_source_only_calibration_primary": true,
  "calibration_is_source_label_assisted_no_weight_training": true,
  "comparisons": {
    "raw": {
      "Frozen_A": 0.05034434735168425,
      "Frozen_F": -0.1524687872731434,
      "C_B_A": -0.01949183273895727,
      "R135_F": -0.17185920857301618,
      "fullreader_ownsource_direct": -0.04353086334441514,
      "fullreader_ownsource_wrong": -0.08618198422779066
    },
    "design_weighted": {
      "Frozen_A": 0.044675286598359065,
      "Frozen_F": -0.14172248949485144,
      "C_B_A": -0.020704627917565332,
      "R135_F": -0.16747481944921638,
      "fullreader_ownsource_direct": -0.07145256978272485,
      "fullreader_ownsource_wrong": -0.10095544919655097
    }
  },
  "FINAL_access": false,
  "adopted": false,
  "disposition": "matched_calibration_repair_not_sufficient_for_TTA_task_gain"
}

## Scope, changed cases and computation

| source | target | level | eligible | changed_IoU | better | worse | best_in_C_eligible | good25_in_C_eligible | zero_best_already | best_outside_S | best_outside_C | C_internal_order |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | oldreader_ownsource | 178 | 12 | 8 | 4 | 112 | 84 | 2 | 2 | 0 | 8 |
| drone | waymo | oldreader_ownsource | 50 | 4 | 1 | 3 | 29 | 32 | 2 | 0 | 0 | 2 |
| quad | drone | oldreader_ownsource | 154 | 20 | 11 | 9 | 78 | 113 | 1 | 2 | 4 | 13 |
| quad | waymo | oldreader_ownsource | 50 | 4 | 1 | 3 | 7 | 19 | 0 | 2 | 2 | 0 |
| waymo | drone | oldreader_ownsource | 154 | 26 | 14 | 12 | 82 | 76 | 1 | 0 | 6 | 19 |
| waymo | quad | oldreader_ownsource | 178 | 28 | 15 | 13 | 120 | 108 | 4 | 3 | 0 | 21 |
| drone | quad | fullreader_pooled | 178 | 14 | 5 | 9 | 98 | 79 | 1 | 2 | 0 | 11 |
| drone | waymo | fullreader_pooled | 50 | 4 | 2 | 2 | 23 | 30 | 1 | 0 | 0 | 3 |
| quad | drone | fullreader_pooled | 154 | 18 | 11 | 7 | 66 | 101 | 1 | 2 | 7 | 8 |
| quad | waymo | fullreader_pooled | 50 | 3 | 1 | 2 | 6 | 19 | 0 | 2 | 1 | 0 |
| waymo | drone | fullreader_pooled | 154 | 25 | 15 | 10 | 63 | 67 | 1 | 0 | 10 | 14 |
| waymo | quad | fullreader_pooled | 178 | 29 | 14 | 15 | 110 | 104 | 4 | 2 | 1 | 22 |
| drone | quad | fullreader_ownsource | 178 | 14 | 6 | 8 | 95 | 77 | 1 | 3 | 0 | 10 |
| drone | waymo | fullreader_ownsource | 50 | 5 | 2 | 3 | 19 | 26 | 1 | 0 | 0 | 4 |
| quad | drone | fullreader_ownsource | 154 | 19 | 12 | 7 | 22 | 58 | 0 | 2 | 14 | 3 |
| quad | waymo | fullreader_ownsource | 50 | 4 | 2 | 2 | 3 | 13 | 0 | 2 | 2 | 0 |
| waymo | drone | fullreader_ownsource | 154 | 24 | 13 | 11 | 72 | 73 | 1 | 0 | 8 | 15 |
| waymo | quad | fullreader_ownsource | 178 | 28 | 13 | 15 | 113 | 105 | 5 | 2 | 1 | 20 |

These postseal IoU partitions are box-quality diagnoses, not independently verified instance identity. Denominators are explicit: changedrecords are not allerrors, source directions/descriptions may share physicalframes/content. Private identities remain available for local audit; public aliases retain content/frame/scene clusters. No single-scene populationCI and no GTfiltered deployment.

## Audits and boundaries

{
  "status": "PASS",
  "n": 313,
  "source_views": 626,
  "max_CPU_full_reader_probability_error": 1.116984955029654e-06,
  "same_source_qualification_and_assignment_as_R132": true,
  "only_source_own_inputs": true,
  "GT_read": false,
  "FINAL_access": false,
  "full_transformer_not_CPU_regenerated": true
}

{
  "status": "PASS",
  "n": 6,
  "new_fits": 36,
  "compatible_reuses_or_exact_fallbacks": 0,
  "max_CPU_native_probability_error": 4.198376958169936e-07,
  "max_parameter_delta_error": 0.0,
  "max_objective_error": 2.32830976720777e-10,
  "source_kappas_and_geometry_exact": true,
  "full_transformer_gradients_not_CPU_regenerated": true,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "new_fits": 3512,
  "compatible_reuses_or_exact_fallbacks": 7276,
  "max_CPU_native_probability_error": 1.2424360581464455e-06,
  "max_parameter_delta_error": 0.0,
  "max_objective_error": 4.656618424192516e-10,
  "source_kappas_and_geometry_exact": true,
  "full_transformer_gradients_not_CPU_regenerated": true,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "task_values": 35960,
  "max_error": 0.0,
  "all_negative_and_fallback_rows_in_denominator": true,
  "FINAL_access": false
}

IndependentCPU reconstructs every complete native matching/competition probability from savedfinalfeatures and unchanged sourcehead, every newdelta/Armijo decision, directprojection/sourcekappa/C/target/permutation, source-adapted keep and finalobjective; all compatible/fallback predictions and originalboxes are exact. Later fullTransformer gradients are not independently regenerated onCPU; nativeengineeringfinite differences provide a bounded implementation check. All targetpredictions sealed before any new taskscore; no GTwarmstart/teacher/acceptance/runtimefilter. Sourcecalibration labels are explicitly used. The parentovernight >=2pp goal remains distinct from these bounded method results and cannot be counted from GTdiagnostics.

A useful calibration repair may still fail as a performance method: rank supervision can preserve the wrong withinC ordering, missingB and fixedA256 geometry remain limits. Report allfactorlevels and negative directions; do not claim universal correction, independent confirmation or global success from a favorable ablation. No further threshold/quantile/step/grid is selected in this round. Sourceweights/private media/descriptions/coordinates/queryIDs/features/rawlabels remain unpublished. A separately locked sourcecached-coordinate lower-envelope plane diagnostic is included as prospective metadata only: it never changes the R140 predictor or supplies supervision and is not ground-identity or task-efficacy evidence.
