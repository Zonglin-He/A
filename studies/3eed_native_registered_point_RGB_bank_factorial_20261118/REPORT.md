# R141 — Native registered point and RGB evidence banks

## Decision

**native_paired_bank_recipe_not_adopted_no_TTA_loss_added**. This is a frozen evidence-acquisition prerequisite, not a parameter-TTA result. Primary F_nativeB_RGBB and the raw/design-weighted gate were registered before target task scores. All1798 exposed ordinary inputs, allnegative comparisons and all1034 no-B completeFrozenA fallbacks are retained. No FINAL access, no new labels in prediction, no source training or external expert.

## Question and controls

Earlier R105 added real B/C RGB banks at A point locations; R106/109/122 interpolated registered B point features back onto A1024 locations. Neither alone tested preserving native B spatial tokens jointly with their physical RGB. This experiment uses the same source-trained PointNet, small RGB CNN, bidirectional fusion, complete live competition and original A256 boxes. It changes the input evidence budget, not learned weights or network modules. The first1024 F anchors are unchanged; up to512 static-compatible nativeB tokens are appended. Such expanded observation context must not be reported as a parameter-TTA increment.

Five fresh arms: F_RGBB_only (F anchors, actualB image added), F_nativeB_only (nativeB spatial tokens, imageA only), F_nativeB_RGBB (both, primary), F_nativeB_RGBB_wrong (same point/image budgets, visible extraRGB associations permuted), F_extraA_duplicateA (same extra-token count and image-bank count using a freshA sample and duplicatedA image). The six exactly compatible historical references are Frozen_A,Frozen_F,R135_A,R135_F,C_B_A,direct_C_B_A. They are reused, not rerun or used as adaptation warmstarts. The A-resample control uses one additional PointNet call; it is a stronger resource control, not identical runtime.

NativeB point sampling reproduces the old provider exactly before expansion. Selected native points must be within.25m of actualA returns and farther than1e-4m from originalA seeds. These are static-surface compatibility tests, not verified instance labels. First512 native-order qualified points are deterministic; missing or ambiguous evidence remains in the denominator. RGB banks are encoded separately with the same CNN before concatenation, preserving per-image GroupNorm statistics. Additional image associations use actual camera transforms, depth/return checks and bounds; wrong bindings preserve the visible set and invert the reverse graph consistently.

OriginalA physical geometry, relation priors and validity remain frozen for every arm. Candidate/role masks are explicitly expanded using the same original hypothesis boxes; no stale1024 mask is broadcast. Geometry statistics are not claimed to incorporate B. CameraA UV relation convention, all original query positions and A256 coordinates stay fixed. This is a bounded variable-sequence-input test of a source-trained model; success is not guaranteed outside its training token budget.

## Engineering failure and recovery

Attempt001 failed before fresh bank predictions or scoring because the old R135 route archive intentionally omitted hypothesis query IDs. Its incomplete artifacts, log and source versions are preserved locally; code is included publicly. No unique object/query ID was inferred from an ambiguous mask. Before attempt002, replay the original GT-free role97 A forward, assert all12 archived route arrays byte-identical and recover original hypothesis IDs/attention. The CPU auditor independently reconstructs original top3 selection from that attention and old own-box supports. This costs one extra originalA role forward per B input, is not a new routing rule, and is recorded in the addendum. Attempt002 then passed12 engineered inputs spanning six directions and B/noB states before formal execution.

## Whole-cohort results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.403041 | 25.899515 | 5.056806 |
| raw | Frozen_F | 13.605854 | 25.898010 | 5.154947 |
| raw | R135_A | 13.506393 | 26.220572 | 5.102343 |
| raw | R135_F | 13.625244 | 26.175826 | 5.343193 |
| raw | C_B_A | 13.472877 | 25.937807 | 5.060428 |
| raw | direct_C_B_A | 13.483204 | 26.126845 | 5.103491 |
| raw | F_RGBB_only | 13.670413 | 26.087404 | 5.154947 |
| raw | F_nativeB_only | 13.464022 | 25.900484 | 5.062724 |
| raw | F_nativeB_RGBB | 13.518589 | 25.898188 | 5.014891 |
| raw | F_nativeB_RGBB_wrong | 13.504854 | 25.993855 | 5.155125 |
| raw | F_extraA_duplicateA | 13.595276 | 26.037096 | 5.108262 |
| design_weighted | Frozen_A | 13.394463 | 26.263306 | 4.599297 |
| design_weighted | Frozen_F | 13.580861 | 26.099350 | 4.688056 |
| design_weighted | R135_A | 13.498989 | 26.600013 | 4.644834 |
| design_weighted | R135_F | 13.606614 | 26.393334 | 4.859257 |
| design_weighted | C_B_A | 13.459843 | 26.270810 | 4.601034 |
| design_weighted | direct_C_B_A | 13.477422 | 26.444000 | 4.639679 |
| design_weighted | F_RGBB_only | 13.670700 | 26.377786 | 4.688056 |
| design_weighted | F_nativeB_only | 13.442617 | 26.165101 | 4.596430 |
| design_weighted | F_nativeB_RGBB | 13.510342 | 26.214594 | 4.549791 |
| design_weighted | F_nativeB_RGBB_wrong | 13.483742 | 26.259846 | 4.661154 |
| design_weighted | F_extraA_duplicateA | 13.578919 | 26.265396 | 4.641967 |

{
  "primary": "F_nativeB_RGBB",
  "new_parameters_updated": 0,
  "TTA_increment_not_tested_in_frozen_prerequisite": true,
  "comparisons": {
    "raw": {
      "Frozen_A": 0.11554856087409071,
      "Frozen_F": -0.08726457375073693,
      "F_RGBB_only": -0.151823883030902,
      "F_nativeB_only": 0.054566808997257965,
      "F_nativeB_RGBB_wrong": 0.013734964767728286,
      "F_extraA_duplicateA": -0.07668661932736676
    },
    "design_weighted": {
      "Frozen_A": 0.11587813173465733,
      "Frozen_F": -0.07051964435855318,
      "F_RGBB_only": -0.16035818940721747,
      "F_nativeB_only": 0.06772508594125881,
      "F_nativeB_RGBB_wrong": 0.026599661341855096,
      "F_extraA_duplicateA": -0.06857759336000768
    }
  },
  "factorial_interaction": {
    "raw": -0.009992500282907102,
    "design_weighted": -0.022113459107407252
  },
  "FINAL_access": false,
  "all1798_negatives_and_noB_fallbacks_retained": true,
  "prerequisite_passed": false,
  "both_components_incremental": false,
  "disposition": "native_paired_bank_recipe_not_adopted_no_TTA_loss_added"
}

The primary must exceed intactFrozenA, same-observation FrozenF, wrongRGB binding and the extraA/duplicate-image control in both aggregation conventions. Single-component increments and the factorial interaction are reported separately: joint dependence is not established merely by beating source. No best arm, source direction or input is selected using task GT. No requirement that every direction be positive is added after scoring.

## Coverage, outcomes and costs

| source | target | n | with_B | no_B | native_B_tokens_total | native_B_tokens_median | A_resample_padding_total | reader_seconds | Frozen_A__better | Frozen_A__worse | Frozen_F__better | Frozen_F__worse | F_nativeB_RGBB_wrong__better | F_nativeB_RGBB_wrong__worse | F_extraA_duplicateA__better | F_extraA_duplicateA__worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | 357 | 178 | 179 | 90755 | 512.0 | 0 | 101.7028433426749 | 16 | 20 | 3 | 9 | 8 | 4 | 8 | 10 |
| drone | waymo | 176 | 50 | 126 | 25600 | 512.0 | 0 | 31.205250985978637 | 7 | 7 | 2 | 1 | 1 | 4 | 3 | 1 |
| quad | drone | 366 | 154 | 212 | 78798 | 512.0 | 0 | 85.27750973636284 | 19 | 19 | 3 | 5 | 3 | 3 | 4 | 3 |
| quad | waymo | 176 | 50 | 126 | 25600 | 512.0 | 0 | 30.762950069911312 | 2 | 5 | 1 | 0 | 1 | 0 | 1 | 1 |
| waymo | drone | 366 | 154 | 212 | 78798 | 512.0 | 0 | 85.25142448017141 | 19 | 14 | 3 | 2 | 2 | 2 | 4 | 3 |
| waymo | quad | 357 | 178 | 179 | 90755 | 512.0 | 0 | 101.94451934465906 | 26 | 22 | 3 | 9 | 6 | 4 | 4 | 10 |

All per-input scalars and all changed cases against all predefined controls are exported with anonymous content/frame/scene clusters. Differences in boxIoU are not independent instance-identity annotations. Multiple descriptions/directions may share physical observations; no single-scene population confidence interval is claimed. Walltime includes different engineering/formal work; complete job costs are exported and extra-role/PointNet forwards are disclosed.

## Independent checks and limits

{
  "status": "PASS",
  "n": 12,
  "B_inputs": 6,
  "max_CPU_full_reader_probability_error": 6.23641104702255e-07,
  "max_graph_weight_error": 0.0,
  "max_engineering_CPU_RGB_CNN_error": 3.814697265625e-06,
  "max_engineering_CPU_expanded_local_read_error": 1.6689300537109375e-06,
  "max_B_native_to_actual_sample_distance_m": 0.0,
  "fixed_A256_and_noB_complete_fallback": true,
  "source_parameters_updated": 0,
  "independent_full_transformer_not_CPU_recomputed": true,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "B_inputs": 764,
  "max_CPU_full_reader_probability_error": 1.0874520555148592e-06,
  "max_graph_weight_error": 0.0,
  "max_engineering_CPU_RGB_CNN_error": 0.0,
  "max_engineering_CPU_expanded_local_read_error": 0.0,
  "max_B_native_to_actual_sample_distance_m": 0.0,
  "fixed_A256_and_noB_complete_fallback": true,
  "source_parameters_updated": 0,
  "independent_full_transformer_not_CPU_recomputed": true,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "task_values": 19778,
  "max_error": 0.0,
  "all_negative_fallback_rows_retained": true,
  "FINAL_access": false
}

Independent CPU reconstructs every final native matching/competition probability and sparse projection graph, original source route hypothesis selection, nativeB-to-actual-sample coordinates, all historical reference/fallback/box/query identities. Engineering additionally reconstructs the source RGB CNN and expanded strict local attention reads from original checkpoint Q/K/V parameters. The fullTransformer and PointNet are not independently regenerated onCPU; this limitation is explicit. Source weights/buffers remain unchanged. All formal predictions are sealed before task IoU scoring. Every published scalar is independently reread from sealed argmax and cached fixed-box evaluation.

## Ancillary text-only prerequisite

The current1798 records contain899 unique expressions. A separately locked existing dependency-parser/tokenizer-only check found839 unique parseable subject heads with zero changes to the current subject-token masks;60 unique unknown parses retain the existing mask. No grounding/Roberta/parser-model inference or task labels were used. This extends the old DEV336 P0 scope but does not prove full language/entity semantics correct. The R141 predictor was unchanged.

## Resource decision

If the frozen evidence gate fails, stop this exact native-bank recipe without adding a TTA loss or target-selected bank size. A negative result bounds this source-prepared fixed-readout/input-expansion protocol, not every physical correspondence or multimodal-TTA architecture. If it passes, a separately registered matched no-update versus Up4896 experiment is still required. The parent >=2pp goal remains distinct and cannot be claimed from a weak baseline, an oracle or additional observations alone.

## Source mechanisms

- [OpenScene CVPR2023 supplementary feature fusion](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Peng_OpenScene_3D_Scene_CVPR_2023_supplemental.pdf): we use existing source-trained smallCNN and sparse measured-return/depth guards, no CLIP/OpenSeg/LSeg/distillation/new3Dmodel
- [OpenScene official PointCloudToImageMapper](https://github.com/pengsongyou/openscene/blob/main/scripts/feature_fusion/fusion_util.py): reference mechanism only; implementation stays originalR105/R131 geometry with selected actualB sensor, sparse checks. No external snapshot/weights vendored
- [WildRefer ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06306.pdf): not an off-the-shelf TTA guarantee or reproduction; no WildReferweights/network loaded

These primary papers/code motivate physical point-pixel correspondence and complementary observations. Their networks, pretrained experts and third-party source snapshots are not imported; no reproduction or novelty claim follows.
