# R137 — Frozen native RGB association intervention

## Decision

**stop_visible_box_late_rebinding_no_TTA**. This is a fixed-head evidence/readout intervention, not a parameter-TTA gain. Its prerequisite is tested before any new adaptation. Parent exploration remains separately authorized until Beijing2026-11-18 09:00.

## Hypothesis and exact scope

R135 found that original query seed anchors sometimes lie outside their final predictedA256 boxes, including403/1437 globally Acc25-recoverable examples. That GT-derived descriptive statistic motivated a test; no GT chooses runtime candidates. An outside anchor may still supply legitimate context, and predicted-box point supports may include road/neighbors, so neither is an instance-identity label.

Keep existing three source models, source-prepared RGB–LiDAR bidirectional architecture, h/u/local/own/text features, trained matching/late gates/projection and full competition residual. Fixed originalA256, route, Bregistration and model initialization. Only lateRGB lookup changes: when anchor lies outside its own predictedbox or has no RGB, and >=4 visible mapped seedpoints lie inside thatbox, average those points native point_image(II,d) features and feed the existing trained lateprojection/gate. Otherqueries retain their originalanchor. This is a one-shot geometric association rule, no learned mask, no prompt/score heuristic/new visual expert/source training. All1034noB cases returncompleteFrozenA. NoFINAL.

PrimaryF_rebind uses the originalF frozenh/u/competition and its own source point-to-image features. A_rebind is a secondary diagnostic, not selected after outcomes. Wrong_rebind cyclically permutes the newly pooled late vectors only among edited queries, maintaining edited eligibility and vector multiset; repeat instances may make some mismatches ineffective and <=1edited query is identity. Every instance/group remains uncertain. Early fusion is unmodified, so results do not settle all RGB–LiDAR association questions.

## All full-cohort metrics

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.403041 | 25.899515 | 5.056806 |
| raw | Frozen_F | 13.605854 | 25.898010 | 5.154947 |
| raw | AB_mean | 13.476794 | 25.712417 | 5.061576 |
| raw | R135_A | 13.506393 | 26.220572 | 5.102343 |
| raw | R135_F | 13.625244 | 26.175826 | 5.343193 |
| raw | A_rebind | 13.430954 | 26.037275 | 5.056806 |
| raw | A_wrong_rebind | 13.421712 | 25.895892 | 5.056806 |
| raw | F_rebind | 13.583544 | 25.944695 | 5.200484 |
| raw | F_wrong_rebind | 13.607285 | 25.943547 | 5.200484 |
| design_weighted | Frozen_A | 13.394463 | 26.263306 | 4.599297 |
| design_weighted | Frozen_F | 13.580861 | 26.099350 | 4.688056 |
| design_weighted | AB_mean | 13.447193 | 25.914692 | 4.595879 |
| design_weighted | R135_A | 13.498989 | 26.600013 | 4.644834 |
| design_weighted | R135_F | 13.606614 | 26.393334 | 4.859257 |
| design_weighted | A_rebind | 13.422376 | 26.401325 | 4.599297 |
| design_weighted | A_wrong_rebind | 13.410271 | 26.203057 | 4.599297 |
| design_weighted | F_rebind | 13.558522 | 26.146295 | 4.733593 |
| design_weighted | F_wrong_rebind | 13.582321 | 26.144888 | 4.733593 |

| source | target | method | control | gain_pp | better | worse | query_changed |
|---|---|---|---|---|---|---|---|
| drone | quad | F_rebind | Frozen_A | 0.212787 | 14 | 15 | 47 |
| drone | quad | F_rebind | Frozen_F | -0.041016 | 2 | 3 | 8 |
| drone | quad | F_rebind | AB_mean | 0.066136 | 5 | 4 | 13 |
| drone | quad | F_rebind | R135_A | 0.029034 | 15 | 16 | 48 |
| drone | quad | F_rebind | R135_F | -0.099936 | 4 | 8 | 19 |
| drone | quad | F_rebind | F_wrong_rebind | -0.055617 | 7 | 6 | 21 |
| drone | quad | F_rebind | A_rebind | 0.198697 | 14 | 16 | 46 |
| drone | waymo | F_rebind | Frozen_A | -0.361134 | 5 | 8 | 16 |
| drone | waymo | F_rebind | Frozen_F | -0.020228 | 0 | 1 | 1 |
| drone | waymo | F_rebind | AB_mean | -0.020228 | 0 | 1 | 2 |
| drone | waymo | F_rebind | R135_A | -0.236714 | 4 | 7 | 13 |
| drone | waymo | F_rebind | R135_F | 0.138826 | 1 | 4 | 6 |
| drone | waymo | F_rebind | F_wrong_rebind | 0.000000 | 0 | 0 | 0 |
| drone | waymo | F_rebind | A_rebind | -0.371457 | 5 | 8 | 16 |
| quad | drone | F_rebind | Frozen_A | 0.177828 | 18 | 21 | 71 |
| quad | drone | F_rebind | Frozen_F | -0.104433 | 5 | 8 | 19 |
| quad | drone | F_rebind | AB_mean | 0.140263 | 12 | 9 | 33 |
| quad | drone | F_rebind | R135_A | -0.120223 | 18 | 26 | 75 |
| quad | drone | F_rebind | R135_F | -0.133998 | 12 | 13 | 43 |
| quad | drone | F_rebind | F_wrong_rebind | -0.012340 | 0 | 1 | 1 |
| quad | drone | F_rebind | A_rebind | 0.133400 | 15 | 17 | 58 |
| quad | waymo | F_rebind | Frozen_A | -0.103915 | 1 | 5 | 18 |
| quad | waymo | F_rebind | Frozen_F | 0.000000 | 0 | 0 | 2 |
| quad | waymo | F_rebind | AB_mean | 0.006432 | 1 | 1 | 7 |
| quad | waymo | F_rebind | R135_A | -0.103915 | 1 | 5 | 20 |
| quad | waymo | F_rebind | R135_F | -0.407684 | 0 | 3 | 9 |
| quad | waymo | F_rebind | F_wrong_rebind | 0.000000 | 0 | 0 | 0 |
| quad | waymo | F_rebind | A_rebind | -0.103915 | 1 | 5 | 17 |
| waymo | drone | F_rebind | Frozen_A | 0.250729 | 21 | 14 | 65 |
| waymo | drone | F_rebind | Frozen_F | 0.033700 | 10 | 4 | 21 |
| waymo | drone | F_rebind | AB_mean | 0.057562 | 13 | 5 | 30 |
| waymo | drone | F_rebind | R135_A | 0.017823 | 20 | 19 | 71 |
| waymo | drone | F_rebind | R135_F | -0.011279 | 12 | 6 | 35 |
| waymo | drone | F_rebind | F_wrong_rebind | -0.067023 | 7 | 3 | 18 |
| waymo | drone | F_rebind | A_rebind | 0.141960 | 20 | 16 | 65 |
| waymo | quad | F_rebind | Frozen_A | 0.808055 | 27 | 21 | 65 |
| waymo | quad | F_rebind | Frozen_F | -0.002059 | 0 | 1 | 1 |
| waymo | quad | F_rebind | AB_mean | 0.417809 | 9 | 4 | 14 |
| waymo | quad | F_rebind | R135_A | 0.771191 | 29 | 21 | 69 |
| waymo | quad | F_rebind | R135_F | 0.225520 | 10 | 4 | 21 |
| waymo | quad | F_rebind | F_wrong_rebind | -0.007815 | 0 | 1 | 1 |
| waymo | quad | F_rebind | A_rebind | 0.818192 | 27 | 20 | 63 |

{
  "primary": "F_rebind",
  "comparisons": {
    "raw": {
      "Frozen_A": 0.1805036596868419,
      "Frozen_F": -0.02230947493798574,
      "F_wrong_rebind": -0.023741070338084924
    },
    "design_weighted": {
      "Frozen_A": 0.1640583719531712,
      "Frozen_F": -0.02233940414003932,
      "F_wrong_rebind": -0.023799384049416616
    }
  },
  "GT_only_postseal_scoring": true,
  "FINAL_access": false,
  "TTA_run": false,
  "direct_prerequisite_passed": false,
  "disposition": "stop_visible_box_late_rebinding_no_TTA"
}

## Verification and limits

{
  "status": "PASS",
  "n": 1798,
  "no_B_complete_fallback": 1034,
  "edited_rows": 761,
  "edited_queries": 36356,
  "CPU_RGB_max_error": 4.76837158203125e-06,
  "CPU_late_max_error": 1.0728836059570312e-06,
  "CPU_probability_max_error": 1.214316047271069e-06,
  "FINAL_access": false,
  "GT_used": false
}

{
  "status": "PASS",
  "n": 1798,
  "task_scalar_values": 16182,
  "max_error": 0.0,
  "cached_GT_only_post_prediction": true,
  "FINAL_access": false
}

GPU engine verifies originalanchor late features against immutableR135. A separate CPU implementation reconstructs RGB CNN/GroupNorm/adapters, source point-to-image attention, projection, visibility/boxpoint support, pooling, wrongpermutations and the complete native competition readout. Source state/checkpoint hashes remainunchanged; all predictions sealed before cachedGT scoring. No rawGT/newboxIoU computations. Existing1798 exposeddevelopment rows, repeated descriptions/physicalcontent and sixdirections are not independent validation. Rawanddesignweighted means and all scenes/positive/negative/no-op outcomes retained; noCI over a singleDrone recording.

The predeclared direct gate requiresF_rebind>FrozenA,FrozenF,matchedwrong inboth rawanddesignweighted. Failure stops this exact latepool recipe; it neither disproves calibratedmultimodalfusion nor licenses fitting its labels harder. No per-case/per-direction GT output switch, no multiplicity of strengths/support thresholds.

## Primary literature used for mechanism, not a reproduction

[TransFusion (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.html) motivates query-conditioned image association; it learns softassociation and does not validate this uniformpool. [OpenScene official project (CVPR2023)](https://pengsongyou.github.io/openscene) motivates calibrated image-feature association and aggregation; its pretrained image embeddings/multiviewfusion are absent here.

Complete public code/config/scalars and anonymous frame/content/scene clusters are provided. Rawimages,descriptions,coordinates,queryIDs,weights,features,privateIDs and thirdparty snapshots staylocal. These controls are internal interventions, not published baseline reproductions.
