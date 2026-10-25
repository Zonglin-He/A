# Direct shared-corrected-box ROI structural controls: hc2

All 1902 queries / 227 parents sealed before independent official labels; 28530 official dense checks (max error 1.3322676295501878e-15), 3804 new prediction/receipt SHA checks plus all bound collector/capture/compact/teacher provenance. Every array cold rederived exactly and all scalar means checked. This export reads already-scored scalars only: zero model, optimizer, Source, expert or GT-scoring calls.

Kappa0 and removal of the first auxiliary DP in favor of direct shared-corrected-box ROI pooling were adopted by explicit user instruction after exposed full-domain ablations. Final trajectory DP and other method parameters remain frozen; these historically exposed full sets are not new independent confirmation. Native Source remains per-offset MAP plus physical hull, and SourceJointMAP isolates readout.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|
|---|---:|---:|---:|---:|---|
|Source|40.141475|20.704678|29.022082|8.937960|0/0/0|
|SourceJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|FullMAP|49.764548|27.349044|41.219769|16.876972|346/305/458|
|NoSharedCorrection|49.689566|26.557068|40.168244|15.720294|345/313/456|
|PastOnly|49.764548|26.872766|40.588854|15.562566|346/300/450|
|OneWayAffinity|49.764548|26.252758|40.063091|14.353312|346/313/455|
|AuxIndependent|49.764548|27.349044|41.219769|16.876972|346/305/458|
|AuxSourceROI|49.689566|27.315220|41.114616|16.824395|345/303/459|
|AuxE_ROI|49.764548|27.349044|41.219769|16.876972|346/305/458|
|NoSpatialTrajectory|49.764548|26.489749|40.063091|15.772871|346/260/421|
|NoTemporalInference|40.141475|22.383628|31.861199|11.041009|0/193/193|
|S0T0|46.693293|24.868872|37.013670|13.827550|220/171/304|
|S1T0|46.693293|25.847764|39.011567|15.036803|220/241/364|
|S0T1|49.764548|26.489749|40.063091|15.772871|346/260/421|
|S1T1|49.764548|27.349044|41.219769|16.876972|346/305/458|

## Paired effects and tails

Positive means first configuration exceeds second. All intervals are pointwise parent-cluster bootstrap CIs, not familywise. Severe is a loss greater than5pp in tIoU or vIoU; the union counts each query once. Rescue is a gain greater than5pp in either metric.

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|FullMAP-Source|+9.623073 [8.594747, 10.683117]|+6.644365 [5.902588, 7.408451]|346/305/458|1129|
|FullMAP-SourceJointMAP|+3.071255 [2.341218, 3.800142]|+3.309946 [2.692047, 3.938965]|460/365/588|903|
|FullMAP-NoSharedCorrection|+0.074982 [-0.094384, 0.236031]|+0.791975 [0.643565, 0.937926]|88/60/122|169|
|FullMAP-PastOnly|+0.000000 [0.000000, 0.000000]|+0.476278 [0.323659, 0.629222]|0/62/62|133|
|FullMAP-OneWayAffinity|+0.000000 [0.000000, 0.000000]|+1.096286 [0.899990, 1.293941]|0/61/61|196|
|FullMAP-AuxIndependent|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|0/0/0|0|
|FullMAP-AuxSourceROI|+0.074982 [-0.094384, 0.236031]|+0.033824 [-0.046807, 0.114818]|88/27/88|86|
|FullMAP-AuxE_ROI|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|0/0/0|0|
|FullMAP-NoSpatialTrajectory|+0.000000 [0.000000, 0.000000]|+0.859295 [0.502377, 1.226670]|0/198/198|260|
|FullMAP-NoTemporalInference|+9.623073 [8.594747, 10.683117]|+4.965416 [4.374584, 5.568715]|346/233/361|988|
|FullMAP-S0T0|+3.071255 [2.341218, 3.800142]|+2.480171 [1.962182, 3.005360]|460/371/557|843|
|FullMAP-S1T0|+3.071255 [2.341218, 3.800142]|+1.501279 [1.059534, 1.941792]|460/292/465|719|
|FullMAP-S0T1|+0.000000 [0.000000, 0.000000]|+0.859295 [0.502377, 1.226670]|0/198/198|260|
|S1T1-S1T0|+3.071255 [2.341218, 3.800142]|+1.501279 [1.059534, 1.941792]|460/292/465|719|
|S0T1-S0T0|+3.071255 [2.341218, 3.800142]|+1.620877 [1.201837, 2.041639]|460/272/464|718|
|S1T0-S0T0|+0.000000 [0.000000, 0.000000]|+0.978892 [0.637944, 1.331710]|0/180/180|263|

## Duration groups

Short is native-GT duration <=5s; long is >5s.

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / FullMAP-Source|544|+3.539258 [1.682666, 5.388260]|+3.434542 [2.062777, 4.774484]|
|short_le5s / FullMAP-SourceJointMAP|544|-0.558067 [-1.779814, 0.646902]|+1.539298 [0.450914, 2.615297]|
|short_le5s / FullMAP-NoSharedCorrection|544|+0.086879 [-0.227539, 0.394597]|+0.565003 [0.328017, 0.795969]|
|short_le5s / FullMAP-PastOnly|544|+0.000000 [0.000000, 0.000000]|+0.522053 [0.294651, 0.759792]|
|short_le5s / FullMAP-OneWayAffinity|544|+0.000000 [0.000000, 0.000000]|+0.944007 [0.706093, 1.177505]|
|short_le5s / FullMAP-AuxIndependent|544|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|short_le5s / FullMAP-AuxSourceROI|544|+0.086879 [-0.227539, 0.394597]|-0.013340 [-0.158176, 0.133147]|
|short_le5s / FullMAP-AuxE_ROI|544|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|short_le5s / FullMAP-NoSpatialTrajectory|544|+0.000000 [0.000000, 0.000000]|+1.174988 [0.627267, 1.719856]|
|short_le5s / FullMAP-NoTemporalInference|544|+3.539258 [1.682666, 5.388260]|+1.798734 [0.716385, 2.926437]|
|short_le5s / FullMAP-S0T0|544|-0.558067 [-1.779814, 0.646902]|+1.025723 [0.129634, 1.914737]|
|short_le5s / FullMAP-S1T0|544|-0.558067 [-1.779814, 0.646902]|-0.267863 [-0.979443, 0.462360]|
|short_le5s / FullMAP-S0T1|544|+0.000000 [0.000000, 0.000000]|+1.174988 [0.627267, 1.719856]|
|short_le5s / S1T1-S1T0|544|-0.558067 [-1.779814, 0.646902]|-0.267863 [-0.979443, 0.462360]|
|short_le5s / S0T1-S0T0|544|-0.558067 [-1.779814, 0.646902]|-0.149266 [-0.847347, 0.561062]|
|short_le5s / S1T0-S0T0|544|+0.000000 [0.000000, 0.000000]|+1.293586 [0.676245, 1.908987]|
|long_gt5s / FullMAP-Source|1358|+12.060182 [10.872964, 13.264188]|+7.930185 [7.010995, 8.893365]|
|long_gt5s / FullMAP-SourceJointMAP|1358|+4.525121 [3.666805, 5.403609]|+4.019249 [3.228219, 4.829502]|
|long_gt5s / FullMAP-NoSharedCorrection|1358|+0.070216 [-0.123939, 0.268090]|+0.882898 [0.686723, 1.077791]|
|long_gt5s / FullMAP-PastOnly|1358|+0.000000 [0.000000, 0.000000]|+0.457941 [0.271008, 0.649951]|
|long_gt5s / FullMAP-OneWayAffinity|1358|+0.000000 [0.000000, 0.000000]|+1.157287 [0.909080, 1.403437]|
|long_gt5s / FullMAP-AuxIndependent|1358|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|long_gt5s / FullMAP-AuxSourceROI|1358|+0.070216 [-0.123939, 0.268090]|+0.052717 [-0.053602, 0.162903]|
|long_gt5s / FullMAP-AuxE_ROI|1358|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|long_gt5s / FullMAP-NoSpatialTrajectory|1358|+0.000000 [0.000000, 0.000000]|+0.732831 [0.298838, 1.192886]|
|long_gt5s / FullMAP-NoTemporalInference|1358|+12.060182 [10.872964, 13.264188]|+6.233954 [5.537159, 6.939939]|
|long_gt5s / FullMAP-S0T0|1358|+4.525121 [3.666805, 5.403609]|+3.062808 [2.404371, 3.743435]|
|long_gt5s / FullMAP-S1T0|1358|+4.525121 [3.666805, 5.403609]|+2.209979 [1.675721, 2.743814]|
|long_gt5s / FullMAP-S0T1|1358|+0.000000 [0.000000, 0.000000]|+0.732831 [0.298838, 1.192886]|
|long_gt5s / S1T1-S1T0|1358|+4.525121 [3.666805, 5.403609]|+2.209979 [1.675721, 2.743814]|
|long_gt5s / S0T1-S0T0|1358|+4.525121 [3.666805, 5.403609]|+2.329976 [1.828225, 2.834367]|
|long_gt5s / S1T0-S0T0|1358|+0.000000 [0.000000, 0.000000]|+0.852829 [0.479383, 1.259232]|

## Controls and interpretation

NoSharedCorrection sets E=Source and rebuilds uncorrected final geometry, uses the actual saved Source-box ROI features, and recomputes posterior, MAP interval and final DP. Full uses retained exact features actually pooled in shared-corrected E boxes. No old auxiliary-trajectory feature is renamed as direct-ROI appearance. PastOnly removes future anchors; OneWayAffinity replaces the original bidirectional complete-box matching by the original forward term, keeping the other settings fixed. Their effects are assessed against Full in the paired table, not assumed beneficial.

The first auxiliary DP no longer exists in the selected path. AuxIndependent and AuxE_ROI are deliberately redundant identity controls and must be exactly Full; their zero differences are implementation checks, not independent removed-component contributions or statistical equivalence claims. AuxSourceROI changes only the ROI appearance input to actual saved Source-box features while retaining main final geometry. The historical two-DP strict deletion result remains separately archived. The retained final-DP benefit is measured by the separate same-upstream independent-box comparison, not these identity rows.

NoSpatialTrajectory removes the remaining trajectory propagation/candidates and final path DP, using E directly for boxes and ROI; it is not interpolation, greedy replacement or expert overwrite. NoTemporalInference removes visual posterior updating and restores native per-offset MAP/hull, then resolves the retained spatial DP on that interval. Its comparison also changes temporal readout, so it is not solely a posterior-evidence attribution.

The E-fixed factorial uses S0T0=E/q0jointMAP, S1T0=spatial/q0jointMAP, S0T1=E/updated q1 from E ROIs, S1T1=full spatial trajectory/updated q1. Every changed interval resolves the relevant final DP. Interaction is (S11-S10)-(S01-S00), rather than a sum of progressive gains.

Interaction delta t -0.000000 [-0.000000, 0.000000] pp; delta v -0.119597 [-0.245474, 0.009547] pp. The signed interaction and its confidence interval determine support for superadditivity. A zero-crossing interval does not establish a positive or negative interaction and is not an equivalence proof. Separate component effects are reported independently.

## Costs and reproducibility

{
  "current_cached_query_elapsed_wall_seconds_by_PID_start": {
    "3399047:253052109": 409.2811224623583
  },
  "parent_collector": {
    "seal_sha256": "54359f4539195a7ddbe53237a2dab172e20bee763b65e8ba02d112ffed6847a6",
    "worker": {
      "pid": 1518134,
      "startticks": 245498341
    },
    "actual_counts": {
      "queries": 1902,
      "shared_raw_grid_hits": 192,
      "legacy_raw_grid_hits": 0,
      "full_Source_queries": 1710,
      "appearance_prefix_queries": 0,
      "new_expert": 0
    },
    "whole_worker_seconds": 8521.806689498946,
    "model_load_seconds": 7.144109104759991,
    "Source_forward_seconds": 6037.778304519597,
    "query_pipeline_seconds": 8251.671000295319
  },
  "inherited_original_Source_and_eight_teacher_cost_nonzero": true,
  "selected_first_auxiliary_DP": false,
  "new_Source": 0,
  "new_expert": 0,
  "scope": "Conditional cached time.perf_counter elapsed wall is not process CPU time and includes provenance checks. For Vid the parent Source was shared with full TTA; do not add TTA Source subtotals again. Original raw-grid acquisition and cached CPU replay reported separately. Parent worker/query/Source/model-load intervals may be nested; do not add them or call replay deployment latency. Shared old raw-grid acquisitions retain original cost provenance."
}

All observed/unobserved fixed-support metrics, boundary errors, signed positives and negatives, severe/rescue counts and full duration CIs remain in report.json, all_paired_tails.json and both signed TSVs. Media, captions, coordinates, raw features and weights stay private.
