# R139 — Episodic batch8 mean-gradient adaptation with matched direct batch sharing

## Decision

**batch8_recipe_not_adopted**. Primary is fixed partition0; three-partition mean is a robustness summary, not a deployed probability ensemble or best-partition selector. All target data are exposed development; no independent validation is claimed. The parent overnight goal remains separately active until Beijing2026-11-18 09:00.

## Why this test follows the positive control

R136 established finite-budget GT-assisted Up4896 correction capability but weak noGT B-to-A direction. This round tests a narrow alternative: whether other physical observations share a helpful update component that is obscured by per-input teacher noise. A prelocked cached diagnostic compares GT-best-set gradients with the mean noGT first gradient of other batchmembers, excluding both the same frame and point-content group. GT is used for research diagnosis only, never a teacher, initial state, runtime qualification, loss, acceptance or output selector. The fixed diagnostic gate passed: recoverable-case peer cosine mean0.049779 versus own0.029536, positive fraction0.562092. These are repeated cases across three partitions, not1224 independent examples; this small surrogate gain does not establish task improvement. OriginalR136source-state gradients are reused; no new oracle fitting.

## Fixed model, observations and loss

Same three intact source-prepared R91 models, originalA256/positions/route, originalregisteredB/S/C/kappa, source-trained pointAdapter/gates/complete competition. Only originalUp weights+bias4896 can change; all remaining source weights frozen. Loss is unchanged R136 partial(S,C)+KL(frozenB groupMAX conditionalC || completeA groupMAX conditionalC)+source-adapted featurekeep1. No newloss,source training,EMA,expert,BN adaptation or FINAL. All1798 inputs remain;1034noB use exact completeFrozenA.

Within each source-target direction, three fixed hashpartitions group whole content blocks into batches up to8rows. Content blocks never split. A model resets to source at everybatch, with no online history. Each step averages raw per-input taskgradients at the same parameterstate, then uses originalrho0 normalized proposal and9-levelArmijo on meanloss, three steps total. Meanloss acceptance may increase individual losses; those cases and task damages remain. Original per-input targets/unknown scope are unchanged. There is no cross-input queryID or targetlabel averaging.

This is transductive batch adaptation: each input can use other unlabeled inputs in its batch. It has more context than R136episodic adaptation and must not be presented as an identical per-input protocol. Controls receive the same available batch; the per-input source/AB/F/direct readouts and an explicit batch-feature-sharing zero-parameter control are reported. Three partitions are all evaluated, not tuned or selected by targetIoU.

## Strong direct batch-context control

Before any R139task scoring, an addendum locked a stronger no-update comparison: for each input, average(Bvis-Avis) over physically registered validB seeds; average those288D means across the same batch; add half of that common vector to each eligible A rawseed bank, then execute the intact complete sourceA reader. Half matches the existing equalA/B feature mixture; it is not selected by tasklabels. Model parameters never change. Frozen raw feature channels are shared only within the same source checkpoint; different frames have no shared candidate IDs. The assumed transferable common feature shift is itself uncertain, not an instance-identity measurement. NoB remains completeFrozenA.

Primary and replicate-mean gates also require exceeding the corresponding direct-batch control in raw anddesignweighted metrics. This addition changes no running gradient producer, primary, grouping, loss or updatebudget. Wrong labels retain same Cscope/probability multiset/group multiplicity, original3step/radius and batchmembership. Compatible R135A/F predictions are reused as historical references, not warmstarts.

## Whole-cohort results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.403041 | 25.899515 | 5.056806 |
| raw | Frozen_F | 13.605854 | 25.898010 | 5.154947 |
| raw | AB_mean | 13.476794 | 25.712417 | 5.061576 |
| raw | direct_C_B_A | 13.483204 | 26.126845 | 5.103491 |
| raw | C_B_A | 13.472877 | 25.937807 | 5.060428 |
| raw | wrong_C_B_A | 13.235829 | 25.620194 | 5.061576 |
| raw | correct_order0 | 13.404928 | 25.798899 | 5.011269 |
| raw | wrong_order0 | 13.319701 | 25.622668 | 5.057954 |
| raw | correct_order1 | 13.502069 | 25.898367 | 5.105966 |
| raw | wrong_order1 | 13.313670 | 25.665731 | 5.011269 |
| raw | correct_order2 | 13.480474 | 25.988293 | 5.105966 |
| raw | wrong_order2 | 13.301416 | 25.663257 | 4.965731 |
| raw | batch_direct_order0 | 13.401701 | 25.899515 | 5.056806 |
| raw | batch_direct_order1 | 13.406462 | 25.899515 | 5.056806 |
| raw | batch_direct_order2 | 13.412983 | 25.899515 | 5.010121 |
| raw | R135_A | 13.506393 | 26.220572 | 5.102343 |
| raw | R135_F | 13.625244 | 26.175826 | 5.343193 |
| raw | correct_partition_mean | 13.462490 | 25.895186 | 5.074400 |
| raw | wrong_partition_mean | 13.311596 | 25.650552 | 5.011651 |
| raw | batch_direct_partition_mean | 13.407049 | 25.899515 | 5.041244 |
| design_weighted | Frozen_A | 13.394463 | 26.263306 | 4.599297 |
| design_weighted | Frozen_F | 13.580861 | 26.099350 | 4.688056 |
| design_weighted | AB_mean | 13.447193 | 25.914692 | 4.595879 |
| design_weighted | direct_C_B_A | 13.477422 | 26.444000 | 4.639679 |
| design_weighted | C_B_A | 13.459843 | 26.270810 | 4.601034 |
| design_weighted | wrong_C_B_A | 13.214720 | 25.962793 | 4.601585 |
| design_weighted | correct_order0 | 13.383216 | 26.136630 | 4.553759 |
| design_weighted | wrong_order0 | 13.322714 | 26.023653 | 4.599848 |
| design_weighted | correct_order1 | 13.481392 | 26.230569 | 4.646571 |
| design_weighted | wrong_order1 | 13.297995 | 26.019678 | 4.553759 |
| design_weighted | correct_order2 | 13.460015 | 26.319380 | 4.646571 |
| design_weighted | wrong_order2 | 13.282141 | 26.017734 | 4.508222 |
| design_weighted | batch_direct_order0 | 13.393117 | 26.263306 | 4.599297 |
| design_weighted | batch_direct_order1 | 13.397885 | 26.263306 | 4.599297 |
| design_weighted | batch_direct_order2 | 13.404493 | 26.263306 | 4.553208 |
| design_weighted | R135_A | 13.498989 | 26.600013 | 4.644834 |
| design_weighted | R135_F | 13.606614 | 26.393334 | 4.859257 |
| design_weighted | correct_partition_mean | 13.441541 | 26.228860 | 4.615634 |
| design_weighted | wrong_partition_mean | 13.300950 | 26.020355 | 4.553943 |
| design_weighted | batch_direct_partition_mean | 13.398498 | 26.263306 | 4.583934 |

| source | target | method | control | gain_pp | better | worse | query_changed |
|---|---|---|---|---|---|---|---|
| drone | quad | correct_order0 | Frozen_A | -0.226301 | 4 | 5 | 12 |
| drone | quad | correct_order0 | Frozen_F | -0.480104 | 16 | 18 | 49 |
| drone | quad | correct_order0 | direct_C_B_A | -0.254755 | 22 | 29 | 68 |
| drone | quad | correct_order0 | C_B_A | -0.373041 | 4 | 9 | 18 |
| drone | quad | correct_order0 | wrong_order0 | -0.047413 | 9 | 8 | 25 |
| drone | quad | correct_order0 | batch_direct_order0 | -0.218220 | 5 | 5 | 13 |
| drone | quad | correct_order1 | Frozen_A | -0.128229 | 4 | 3 | 11 |
| drone | quad | correct_order1 | Frozen_F | -0.382032 | 14 | 15 | 47 |
| drone | quad | correct_order1 | direct_C_B_A | -0.156683 | 24 | 28 | 70 |
| drone | quad | correct_order1 | C_B_A | -0.274968 | 3 | 6 | 14 |
| drone | quad | correct_order1 | wrong_order1 | -0.033306 | 10 | 9 | 30 |
| drone | quad | correct_order1 | batch_direct_order1 | -0.128229 | 4 | 3 | 11 |
| drone | quad | correct_order2 | Frozen_A | -0.176336 | 2 | 5 | 11 |
| drone | quad | correct_order2 | Frozen_F | -0.430139 | 13 | 17 | 49 |
| drone | quad | correct_order2 | direct_C_B_A | -0.204790 | 22 | 31 | 72 |
| drone | quad | correct_order2 | C_B_A | -0.323076 | 4 | 11 | 22 |
| drone | quad | correct_order2 | wrong_order2 | -0.154095 | 9 | 15 | 33 |
| drone | quad | correct_order2 | batch_direct_order2 | -0.176336 | 2 | 5 | 11 |
| drone | waymo | correct_order0 | Frozen_A | -0.437164 | 1 | 3 | 9 |
| drone | waymo | correct_order0 | Frozen_F | -0.096258 | 7 | 6 | 17 |
| drone | waymo | correct_order0 | direct_C_B_A | -0.252738 | 6 | 11 | 23 |
| drone | waymo | correct_order0 | C_B_A | -0.038867 | 2 | 2 | 10 |
| drone | waymo | correct_order0 | wrong_order0 | -0.415058 | 1 | 3 | 9 |
| drone | waymo | correct_order0 | batch_direct_order0 | -0.437164 | 1 | 3 | 9 |
| drone | waymo | correct_order1 | Frozen_A | -0.061602 | 2 | 2 | 8 |
| drone | waymo | correct_order1 | Frozen_F | 0.279303 | 7 | 5 | 16 |
| drone | waymo | correct_order1 | direct_C_B_A | 0.122823 | 6 | 10 | 21 |
| drone | waymo | correct_order1 | C_B_A | 0.336694 | 2 | 0 | 6 |
| drone | waymo | correct_order1 | wrong_order1 | 0.232947 | 2 | 1 | 7 |
| drone | waymo | correct_order1 | batch_direct_order1 | -0.061602 | 2 | 2 | 8 |
| drone | waymo | correct_order2 | Frozen_A | -0.075106 | 2 | 3 | 9 |
| drone | waymo | correct_order2 | Frozen_F | 0.265800 | 7 | 6 | 14 |
| drone | waymo | correct_order2 | direct_C_B_A | 0.109320 | 6 | 10 | 20 |
| drone | waymo | correct_order2 | C_B_A | 0.323191 | 2 | 1 | 8 |
| drone | waymo | correct_order2 | wrong_order2 | 0.390212 | 3 | 2 | 9 |
| drone | waymo | correct_order2 | batch_direct_order2 | -0.075106 | 2 | 3 | 9 |
| quad | drone | correct_order0 | Frozen_A | 0.311917 | 9 | 4 | 22 |
| quad | drone | correct_order0 | Frozen_F | 0.029656 | 17 | 14 | 57 |
| quad | drone | correct_order0 | direct_C_B_A | 0.079983 | 24 | 25 | 90 |
| quad | drone | correct_order0 | C_B_A | -0.026005 | 8 | 6 | 31 |
| quad | drone | correct_order0 | wrong_order0 | 0.395455 | 13 | 9 | 35 |
| quad | drone | correct_order0 | batch_direct_order0 | 0.311917 | 9 | 4 | 22 |
| quad | drone | correct_order1 | Frozen_A | 0.063996 | 8 | 9 | 30 |
| quad | drone | correct_order1 | Frozen_F | -0.218265 | 14 | 18 | 57 |
| quad | drone | correct_order1 | direct_C_B_A | -0.167938 | 20 | 27 | 87 |
| quad | drone | correct_order1 | C_B_A | -0.273926 | 4 | 9 | 29 |
| quad | drone | correct_order1 | wrong_order1 | 0.076075 | 7 | 10 | 38 |
| quad | drone | correct_order1 | batch_direct_order1 | 0.061106 | 7 | 9 | 29 |
| quad | drone | correct_order2 | Frozen_A | 0.070499 | 8 | 7 | 30 |
| quad | drone | correct_order2 | Frozen_F | -0.211761 | 18 | 19 | 62 |
| quad | drone | correct_order2 | direct_C_B_A | -0.161434 | 23 | 28 | 90 |
| quad | drone | correct_order2 | C_B_A | -0.267422 | 7 | 9 | 30 |
| quad | drone | correct_order2 | wrong_order2 | 0.076653 | 12 | 11 | 43 |
| quad | drone | correct_order2 | batch_direct_order2 | 0.082922 | 6 | 7 | 28 |
| quad | waymo | correct_order0 | Frozen_A | -0.175574 | 1 | 1 | 6 |
| quad | waymo | correct_order0 | Frozen_F | -0.071659 | 5 | 2 | 19 |
| quad | waymo | correct_order0 | direct_C_B_A | 0.149703 | 3 | 0 | 14 |
| quad | waymo | correct_order0 | C_B_A | 0.018801 | 4 | 1 | 9 |
| quad | waymo | correct_order0 | wrong_order0 | 0.029035 | 1 | 0 | 5 |
| quad | waymo | correct_order0 | batch_direct_order0 | -0.175574 | 1 | 1 | 6 |
| quad | waymo | correct_order1 | Frozen_A | 0.192102 | 2 | 1 | 7 |
| quad | waymo | correct_order1 | Frozen_F | 0.296016 | 6 | 1 | 17 |
| quad | waymo | correct_order1 | direct_C_B_A | 0.517378 | 5 | 0 | 15 |
| quad | waymo | correct_order1 | C_B_A | 0.386476 | 4 | 0 | 8 |
| quad | waymo | correct_order1 | wrong_order1 | 0.328401 | 3 | 1 | 10 |
| quad | waymo | correct_order1 | batch_direct_order1 | 0.192102 | 2 | 1 | 7 |
| quad | waymo | correct_order2 | Frozen_A | 0.080777 | 1 | 1 | 7 |
| quad | waymo | correct_order2 | Frozen_F | 0.184692 | 5 | 1 | 17 |
| quad | waymo | correct_order2 | direct_C_B_A | 0.406054 | 4 | 0 | 15 |
| quad | waymo | correct_order2 | C_B_A | 0.275152 | 3 | 0 | 6 |
| quad | waymo | correct_order2 | wrong_order2 | 0.379736 | 5 | 1 | 12 |
| quad | waymo | correct_order2 | batch_direct_order2 | 0.080777 | 1 | 1 | 7 |
| waymo | drone | correct_order0 | Frozen_A | 0.319369 | 12 | 7 | 31 |
| waymo | drone | correct_order0 | Frozen_F | 0.102339 | 16 | 17 | 54 |
| waymo | drone | correct_order0 | direct_C_B_A | -0.169434 | 21 | 26 | 85 |
| waymo | drone | correct_order0 | C_B_A | -0.003826 | 11 | 8 | 32 |
| waymo | drone | correct_order0 | wrong_order0 | 0.229946 | 10 | 11 | 33 |
| waymo | drone | correct_order0 | batch_direct_order0 | 0.319369 | 12 | 7 | 31 |
| waymo | drone | correct_order1 | Frozen_A | 0.416212 | 10 | 7 | 30 |
| waymo | drone | correct_order1 | Frozen_F | 0.199182 | 19 | 17 | 60 |
| waymo | drone | correct_order1 | direct_C_B_A | -0.072591 | 24 | 26 | 93 |
| waymo | drone | correct_order1 | C_B_A | 0.093017 | 10 | 9 | 31 |
| waymo | drone | correct_order1 | wrong_order1 | 0.313694 | 5 | 10 | 24 |
| waymo | drone | correct_order1 | batch_direct_order1 | 0.398571 | 10 | 8 | 31 |
| waymo | drone | correct_order2 | Frozen_A | 0.391964 | 11 | 7 | 33 |
| waymo | drone | correct_order2 | Frozen_F | 0.174934 | 18 | 14 | 54 |
| waymo | drone | correct_order2 | direct_C_B_A | -0.096840 | 22 | 25 | 90 |
| waymo | drone | correct_order2 | C_B_A | 0.068769 | 7 | 5 | 27 |
| waymo | drone | correct_order2 | wrong_order2 | 0.212076 | 5 | 10 | 27 |
| waymo | drone | correct_order2 | batch_direct_order2 | 0.264552 | 10 | 8 | 33 |
| waymo | quad | correct_order0 | Frozen_A | 0.140270 | 10 | 10 | 28 |
| waymo | quad | correct_order0 | Frozen_F | -0.669844 | 18 | 29 | 63 |
| waymo | quad | correct_order0 | direct_C_B_A | -0.117991 | 28 | 36 | 85 |
| waymo | quad | correct_order0 | C_B_A | -0.036826 | 7 | 10 | 23 |
| waymo | quad | correct_order0 | wrong_order0 | 0.171050 | 13 | 12 | 32 |
| waymo | quad | correct_order0 | batch_direct_order0 | 0.140270 | 10 | 10 | 27 |
| waymo | quad | correct_order1 | Frozen_A | 0.039091 | 9 | 8 | 26 |
| waymo | quad | correct_order1 | Frozen_F | -0.771023 | 18 | 27 | 60 |
| waymo | quad | correct_order1 | direct_C_B_A | -0.219170 | 28 | 36 | 85 |
| waymo | quad | correct_order1 | C_B_A | -0.138005 | 9 | 8 | 19 |
| waymo | quad | correct_order1 | wrong_order1 | 0.182567 | 12 | 9 | 30 |
| waymo | quad | correct_order1 | batch_direct_order1 | 0.039091 | 9 | 8 | 26 |
| waymo | quad | correct_order2 | Frozen_A | 0.101512 | 9 | 12 | 28 |
| waymo | quad | correct_order2 | Frozen_F | -0.708602 | 16 | 29 | 62 |
| waymo | quad | correct_order2 | direct_C_B_A | -0.156749 | 26 | 36 | 85 |
| waymo | quad | correct_order2 | C_B_A | -0.075584 | 8 | 11 | 24 |
| waymo | quad | correct_order2 | wrong_order2 | 0.162662 | 10 | 13 | 32 |
| waymo | quad | correct_order2 | batch_direct_order2 | 0.156323 | 10 | 11 | 27 |

{
  "primary": "correct_order0",
  "three_orders_not_cherry_picked": true,
  "partition_mean_is_replicate_mean_not_a_deployed_ensemble": true,
  "comparisons": {
    "raw": {
      "correct_order0": {
        "Frozen_A": 0.0018876711510777966,
        "Frozen_F": -0.20092546347374984,
        "direct_C_B_A": -0.07827575725345604,
        "C_B_A": -0.06794850893956372,
        "wrong_order0": 0.08522703277449217,
        "batch_direct_order0": 0.0032271748054490956
      },
      "correct_partition_mean": {
        "Frozen_A": 0.05944967977031901,
        "Frozen_F": -0.14336345485450863,
        "direct_C_B_A": -0.020713748634214824,
        "C_B_A": -0.010386500320322511,
        "wrong_partition_mean": 0.15089427556428525,
        "batch_direct_partition_mean": 0.05544140149288346
      }
    },
    "design_weighted": {
      "correct_order0": {
        "Frozen_A": -0.011247282640210798,
        "Frozen_F": -0.1976450587334213,
        "direct_C_B_A": -0.0942054794012055,
        "C_B_A": -0.0766271971561352,
        "wrong_order0": 0.06050252559273872,
        "batch_direct_order0": -0.009900349557776522
      },
      "correct_partition_mean": {
        "Frozen_A": 0.04707756298733479,
        "Frozen_F": -0.13932021310587572,
        "direct_C_B_A": -0.03588063377365991,
        "C_B_A": -0.01830235152858961,
        "wrong_partition_mean": 0.14059102828006154,
        "batch_direct_partition_mean": 0.0430427186401392
      }
    }
  },
  "FINAL_access": false,
  "source_training": false,
  "transductive_batch_context_exceeds_individual_R136": true,
  "GT_used_only_diagnostic_gate_and_postseal_scoring": true,
  "retained": false,
  "disposition": "batch8_recipe_not_adopted"
}

## Update behavior and cost

| order | wrong | batches | case_fits | accepted_steps | trials | mean_parameter_L2 | individual_pseudo_loss_increases |
|---|---|---|---|---|---|---|---|
| 0 | False | 113 | 764 | 339 | 346 | 0.10030534756461462 | 197 |
| 0 | True | 113 | 764 | 339 | 339 | 0.1052857565244514 | 46 |
| 1 | False | 111 | 764 | 333 | 342 | 0.10034896511045188 | 189 |
| 1 | True | 111 | 764 | 333 | 333 | 0.10516564396848942 | 57 |
| 2 | False | 113 | 764 | 339 | 344 | 0.10084324627222871 | 192 |
| 2 | True | 113 | 764 | 339 | 339 | 0.1050763066603119 | 48 |

There are337batchpartitions and674correct/wrong fits,4584case fits;3steps imply13752case-level gradient evaluations. Directsharing adds337zero-parameter batches/2292case forwards. Source/backbone extraction and historicalregistration costs remain dependencies, not free deployment work. Jobreceipts give actual bounded engineering/formal/audit costs. No favorable-order timing or loss-only success claim.

## Independent audits and engineering records

{
  "status": "PASS",
  "engineering": true,
  "batch_fits": 18,
  "case_fits": 46,
  "max_first_gradient_error_vs_prior_source_state_mean": 1.4901161193847656e-08,
  "max_delta_error": 0.0,
  "max_CPU_native_probability_error": 2.4252097394206196e-07,
  "max_CPU_objective_error": 2.328314763211381e-10,
  "singleton_native_bridge_passed": true,
  "source_restore_and_onlyUp4896_native_asserted": true,
  "full_encoder_later_gradients_not_CPU_regenerated": true,
  "FINAL_access": false
}

{
  "status": "PASS",
  "engineering": false,
  "batch_fits": 674,
  "case_fits": 4584,
  "max_first_gradient_error_vs_prior_source_state_mean": 4.470348358154297e-08,
  "max_delta_error": 0.0,
  "max_CPU_native_probability_error": 1.2373040687574388e-06,
  "max_CPU_objective_error": 4.65662175486159e-10,
  "singleton_native_bridge_passed": true,
  "source_restore_and_onlyUp4896_native_asserted": true,
  "full_encoder_later_gradients_not_CPU_regenerated": true,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n_batch": 337,
  "case_fits": 2292,
  "source_shift_exact": true,
  "probability_max_CPU_error": 1.2379107684479251e-06,
  "no_parameters_updated": true,
  "FULL_encoder_not_CPU_regenerated": true,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "task_scalar_values": 30566,
  "max_error": 0.0,
  "partition_mean_Acc_is_mean_of_indicators_not_threshold_mean_IoU": true,
  "FINAL_access": false
}

Engineering singleton fits recover exactR136deltas and probabilities; thus batching, not a hidden change to source initialization/objective/readout, is the principal difference. IndependentCPU validates mean firstgradients against existing source-state gradients, every normalizeddelta/acceptance, complete native matching/competition and final source-adapter featurekeep/mean objectives. It explicitly joins each batchmember/order/kind prediction to the final per-input arrays, preserving all noB fallbacks. Later transformer gradients are not independently CPU regenerated; native engineering directional derivatives provide a bounded check.

All formal producer predictions and direct-control predictions are sealed before task scoring. The auditorjoin extension and directcontrol/historical-evaluation addenda were locked before anyR139IoU evaluation; the running batchpredictor never changed. A draft of the new directcontrol had a missing whitespace syntax error, caught and fixed before any controlmodel execution; old draft retained. Original auditor/evaluator versions and all real failed attempts are preserved. No GT-conditioned threshold/step/batch/direction/output choice.

## Limits and interpretation

The sourceB teacher, sparse registration and Cscope remain fallible; positive GT-gradient cosine alone does not establish correct instance supervision. Candidate geometry never changes. Different groups are not certified different objects, GTIoU is a box-quality proxy rather than independent semantic identity, and targetcontent/descriptions/directions are correlated. All negative/no-op cases and anonymous physicalframe/content/scene clusters are retained; no populationCI from a singlerecording. The originalpooledsourcekappa is inherited, not strict per-direction source-only calibration. Codeinspection also established it was calibrated under the oldzeroUp/zeroGate fixedresidual reader, whereas currentpredictions use intactsource/livecompetition; matched-reader recalibration is a separate prospective repair, never a changed threshold in this lockedround. Targetdevelopmentlabels informed a research prerequisite, not deployedupdates; this is not an unexposed test.

Prospective source-only camera metadata was inspected separately without models/media/taskannotations: it is not part of this batchalgorithm or an experimental normalization result. No calibration or coordinate transformations were changed. Raw transforms remain private.

If only the direct batchcorrection helps, parametergradient adaptation has not established its extra value. If batchupdates help sourceA but lose toF/direct/wrong, do not adopt them by ignoring controls. A positive fixed-development result still requires a locked broader evaluation before generalization claims. No newlayer/radius/grid/source checkpoint is selected here.

All sourceweights,rawmedia/descriptions/coordinates/candidateIDs/featurebanks/GTgradientvectors remain private. Public export includes own code/config,anonymous complete taskscalars,all negativechanges and cluster links, excluding thirdparty snapshots.
