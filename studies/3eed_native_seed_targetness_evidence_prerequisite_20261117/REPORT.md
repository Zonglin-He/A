# R138 — Frozen native seed-targetness evidence prerequisite

## Decision

**stop_native_seed_product_no_TTA**. No new parameter adaptation, no source training and no FINAL. Parent overnight exploration separately continues until Beijing2026-11-18 09:00.

## Signal and exact contrast

The existing epoch100 points_obj_cls Conv/BN head reads the language-conditioned seed bank. In the actual 3EED source loss, up to four seedpoints near the annotated target center, within target point support, receive positive focal labels; remaining points receive negatives. Its output is a learned seed priority, not a certified object mask, target identity or final-box IoU estimate. Original query generation selects the top256 sigmoid scores. The current test keeps those original A256 query IDs/boxes/positions and only reads scores at their anchors. All1798 original selectors match exactly.

This differs from R56 stage-wise query/text contrastive consensus and R108–R110 soft-token CE classification/BN transport; those failed controls remain closed and were not rerun. Source-trained bidirectional RGB–LiDAR modules and complete competition readout remain intact, frozen. Only the seed-priority evidence is added to source complete log-probabilities, coefficient1 in raw logit units, no scale/threshold/temperature scan. Shared representations and focal calibration prevent interpreting this sum as independent or calibrated likelihoods.

Primary F_seed_product uses source-prepared F seedfeatures with the original FrozenF distribution. A_seed_product/seed_only are separately reported. Wrong controls derange all256 anchor scores while preserving their multiset. NoB1034 inputs retain completeFrozenA exactly. Existing1798 exposeddevelopment inputs, sixdirections and allnegative/no-op cases remain in denominator.

## Original-input compatibility addendum

After predictor locking and before any R138 target scoring, a separately timestamped addendum registered original-source input controls. Raw-source seedbank was already captured to bridge originalqueryselection; no additional inference/labels were used to choose controls. Frozen original seed logits are paired with A/F distributions. A paired comparison isolates source-prepared head-input changes; F vs originalA also changes view and is not a pure compatibility contrast. These controls are not retroactively substituted for the primary. The original unused evaluator and original native auditor are retained, alongside the extended pre-score evaluation lock. No original predictions changed.

## Full cohort results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.403041 | 25.899515 | 5.056806 |
| raw | Frozen_F | 13.605854 | 25.898010 | 5.154947 |
| raw | AB_mean | 13.476794 | 25.712417 | 5.061576 |
| raw | R135_A | 13.506393 | 26.220572 | 5.102343 |
| raw | R135_F | 13.625244 | 26.175826 | 5.343193 |
| raw | A_seed_only | 12.747660 | 25.118135 | 4.452193 |
| raw | A_seed_product | 13.252659 | 26.036918 | 4.917898 |
| raw | A_wrong_seed_product | 13.326763 | 25.443963 | 5.475035 |
| raw | F_seed_only | 12.995364 | 25.114691 | 4.639112 |
| raw | F_seed_product | 13.491543 | 26.087226 | 5.200663 |
| raw | F_wrong_seed_product | 13.423349 | 25.535394 | 5.476183 |
| raw | original_seed_only | 12.410847 | 24.012611 | 4.591892 |
| raw | A_original_seed_product | 13.270028 | 25.125915 | 5.011269 |
| raw | F_original_Aseed_product | 13.392508 | 25.547587 | 5.151325 |
| design_weighted | Frozen_A | 13.394463 | 26.263306 | 4.599297 |
| design_weighted | Frozen_F | 13.580861 | 26.099350 | 4.688056 |
| design_weighted | AB_mean | 13.447193 | 25.914692 | 4.595879 |
| design_weighted | R135_A | 13.498989 | 26.600013 | 4.644834 |
| design_weighted | R135_F | 13.606614 | 26.393334 | 4.859257 |
| design_weighted | A_seed_only | 12.733081 | 25.343395 | 4.064913 |
| design_weighted | A_seed_product | 13.233921 | 26.290978 | 4.527814 |
| design_weighted | A_wrong_seed_product | 13.309249 | 25.719516 | 5.027629 |
| design_weighted | F_seed_only | 12.953724 | 25.252820 | 4.223221 |
| design_weighted | F_seed_product | 13.462212 | 26.250432 | 4.752498 |
| design_weighted | F_wrong_seed_product | 13.422598 | 25.852384 | 5.028180 |
| design_weighted | original_seed_only | 12.431004 | 24.354521 | 4.186757 |
| design_weighted | A_original_seed_product | 13.272615 | 25.447920 | 4.620846 |
| design_weighted | F_original_Aseed_product | 13.415144 | 25.902070 | 4.753405 |

| source | target | method | control | gain_pp | better | worse | query_changed |
|---|---|---|---|---|---|---|---|
| drone | quad | F_seed_product | Frozen_A | -0.057931 | 40 | 36 | 111 |
| drone | quad | F_seed_product | Frozen_F | -0.311734 | 37 | 29 | 96 |
| drone | quad | F_seed_product | AB_mean | -0.204582 | 39 | 30 | 99 |
| drone | quad | F_seed_product | R135_A | -0.241684 | 41 | 36 | 110 |
| drone | quad | F_seed_product | R135_F | -0.370653 | 35 | 29 | 93 |
| drone | quad | F_seed_product | F_wrong_seed_product | -0.156187 | 48 | 47 | 153 |
| drone | quad | F_seed_product | A_seed_product | 0.098466 | 21 | 19 | 65 |
| drone | waymo | F_seed_product | Frozen_A | -0.629939 | 14 | 11 | 36 |
| drone | waymo | F_seed_product | Frozen_F | -0.289033 | 12 | 9 | 30 |
| drone | waymo | F_seed_product | AB_mean | -0.289033 | 12 | 9 | 31 |
| drone | waymo | F_seed_product | R135_A | -0.505519 | 14 | 10 | 36 |
| drone | waymo | F_seed_product | R135_F | -0.129979 | 12 | 8 | 29 |
| drone | waymo | F_seed_product | F_wrong_seed_product | -0.315997 | 12 | 15 | 41 |
| drone | waymo | F_seed_product | A_seed_product | 0.236623 | 6 | 5 | 17 |
| quad | drone | F_seed_product | Frozen_A | -0.109592 | 25 | 26 | 86 |
| quad | drone | F_seed_product | Frozen_F | -0.391852 | 17 | 22 | 67 |
| quad | drone | F_seed_product | AB_mean | -0.147156 | 19 | 19 | 65 |
| quad | drone | F_seed_product | R135_A | -0.407643 | 22 | 30 | 87 |
| quad | drone | F_seed_product | R135_F | -0.421417 | 20 | 23 | 76 |
| quad | drone | F_seed_product | F_wrong_seed_product | 0.146933 | 36 | 33 | 123 |
| quad | drone | F_seed_product | A_seed_product | -0.323449 | 16 | 23 | 66 |
| quad | waymo | F_seed_product | Frozen_A | 0.381221 | 4 | 4 | 22 |
| quad | waymo | F_seed_product | Frozen_F | 0.485135 | 3 | 0 | 13 |
| quad | waymo | F_seed_product | AB_mean | 0.491567 | 4 | 1 | 16 |
| quad | waymo | F_seed_product | R135_A | 0.381221 | 4 | 4 | 23 |
| quad | waymo | F_seed_product | R135_F | 0.077451 | 2 | 1 | 14 |
| quad | waymo | F_seed_product | F_wrong_seed_product | 0.463286 | 6 | 6 | 29 |
| quad | waymo | F_seed_product | A_seed_product | -0.082537 | 2 | 4 | 16 |
| waymo | drone | F_seed_product | Frozen_A | 0.275130 | 23 | 23 | 88 |
| waymo | drone | F_seed_product | Frozen_F | 0.058100 | 14 | 18 | 65 |
| waymo | drone | F_seed_product | AB_mean | 0.081962 | 17 | 20 | 72 |
| waymo | drone | F_seed_product | R135_A | 0.042223 | 20 | 28 | 91 |
| waymo | drone | F_seed_product | R135_F | 0.013121 | 14 | 18 | 70 |
| waymo | drone | F_seed_product | F_wrong_seed_product | -0.534951 | 33 | 29 | 120 |
| waymo | drone | F_seed_product | A_seed_product | 0.395582 | 17 | 13 | 63 |
| waymo | quad | F_seed_product | Frozen_A | 0.547602 | 37 | 34 | 92 |
| waymo | quad | F_seed_product | Frozen_F | -0.262512 | 23 | 25 | 61 |
| waymo | quad | F_seed_product | AB_mean | 0.157356 | 27 | 24 | 65 |
| waymo | quad | F_seed_product | R135_A | 0.510738 | 38 | 33 | 92 |
| waymo | quad | F_seed_product | R135_F | -0.034933 | 28 | 21 | 66 |
| waymo | quad | F_seed_product | F_wrong_seed_product | 0.634601 | 49 | 44 | 115 |
| waymo | quad | F_seed_product | A_seed_product | 1.045061 | 30 | 19 | 67 |

{
  "primary": "F_seed_product",
  "comparisons": {
    "raw": {
      "Frozen_A": 0.08850186741848276,
      "Frozen_F": -0.11431126720634488,
      "F_wrong_seed_product": 0.0681935436710237
    },
    "design_weighted": {
      "Frozen_A": 0.06774850009895239,
      "Frozen_F": -0.11864927599425812,
      "F_wrong_seed_product": 0.039614181982386754
    }
  },
  "GT_only_postseal_scoring": true,
  "FINAL_access": false,
  "TTA_run": false,
  "direct_prerequisite_passed": false,
  "disposition": "stop_native_seed_product_no_TTA"
}

## Independent verification and limits

{
  "status": "PASS",
  "n": 1798,
  "no_B_complete_fallback": 1034,
  "source_head_state_items_verified": 48,
  "exact_original_query_selector": 1798,
  "CPU_seed_logits_max_error": 5.245208740234375e-06,
  "CPU_probability_max_error": 3.936324390885204e-07,
  "full_encoder_not_CPU_replayed": true,
  "GT_used": false,
  "FINAL_access": false
}

{
  "status": "PASS",
  "n": 1798,
  "task_scalar_values": 25172,
  "max_error": 0.0,
  "FINAL_access": false
}

A separate CPU Conv/BatchNorm/softmax implementation reconstructs all seed logits and candidate readouts. All48 saved source-head stateitems match immutable source checkpoints. Full cross-encoder banks are captured from native source execution and are not independently CPU regenerated; originalqueryselection provides an exact bridge, not a proof of final identity. Every prediction and extra control is sealed before cachedGT scoring. No fresh target GT geometry, GT masks in evidence, GT-selected per-input/direction output or oracle warmstart.

Gate: primary must exceed completeFrozenA,FrozenF and matchedwrong in rawanddesignweighted six-direction mIoU. Failure stops this fixed additive seed recipe without fitting its labels, not all seed-level semantic adaptation. Primary literature/implementation: [3EED official source](https://github.com/worldbench/3EED), specifically points_obj_cls and its focalqueryselection loss; [Group-Free3D (ICCV2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Group-Free_3D_Object_Detection_via_Transformers_ICCV_2021_paper.html) provides related querysampling/attention context, not validation of our language-conditioned use.

All positives/negatives/scenes and anonymous frame/content clusters are retained. Inputs are repeatedly exposed development, not independent validation. Same-frame descriptions and incoming directions do not provide independent physical scenes. No scene-population CI from a single recording. All source weights/banks/RGB/text/coordinates/privateIDs remain local; source/architecture/A256 are unchanged.
