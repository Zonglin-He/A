# Current no-shrinkage structural controls: hc2

All 1902 queries / 227 parents sealed before independent official labels; 28530 official dense checks (max error 1.3322676295501878e-15), 3804 new prediction/receipt SHA checks plus all bound collector/capture/compact/teacher provenance. Every array cold rederived exactly and all scalar means checked. This export reads already-scored scalars only: zero model, optimizer, Source, expert or GT-scoring calls.

Current kappa0 was selected by direct user after full-result exposure. Other method parameters remain frozen; these historically exposed full sets are not new independent confirmation. Native Source remains per-offset MAP plus physical hull, and SourceJointMAP isolates readout.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|
|---|---:|---:|---:|---:|---|
|Source|40.141475|20.704678|29.022082|8.937960|0/0/0|
|SourceJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|FullMAP|49.514796|27.214726|41.062040|16.614090|364/305/467|
|NoSharedCorrection|49.519197|26.466523|39.484753|15.352261|367/320/473|
|PastOnly|49.673103|26.885075|40.851735|15.825447|362/304/470|
|OneWayAffinity|49.660457|26.236337|40.168244|14.195584|362/317/466|
|AuxIndependent|49.636693|27.297858|41.377497|16.876972|361/303/465|
|AuxSourceROI|49.689566|27.315220|41.114616|16.824395|345/303/459|
|AuxE_ROI|49.764548|27.349044|41.219769|16.876972|346/305/458|
|NoSpatialTrajectory|49.764548|26.489749|40.063091|15.772871|346/260/421|
|NoTemporalInference|40.141475|22.383628|31.861199|11.041009|0/193/193|
|S0T0|46.693293|24.868872|37.013670|13.827550|220/171/304|
|S1T0|46.693293|25.847764|39.011567|15.036803|220/241/364|
|S0T1|49.764548|26.489749|40.063091|15.772871|346/260/421|
|S1T1|49.514796|27.214726|41.062040|16.614090|364/305/467|

## Paired effects and tails

Positive means first configuration exceeds second. All intervals are pointwise parent-cluster bootstrap CIs, not familywise. Severe is a loss greater than5pp in tIoU or vIoU; the union counts each query once. Rescue is a gain greater than5pp in either metric.

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|FullMAP-Source|+9.373320 [8.373270, 10.392112]|+6.510048 [5.779450, 7.252761]|364/305/467|1124|
|FullMAP-SourceJointMAP|+2.821502 [2.097354, 3.544531]|+3.175629 [2.559407, 3.810588]|468/374/597|896|
|FullMAP-NoSharedCorrection|-0.004401 [-0.155025, 0.148652]|+0.748203 [0.594165, 0.905532]|76/63/107|146|
|FullMAP-PastOnly|-0.158308 [-0.356442, 0.035868]|+0.329651 [0.156588, 0.500300]|144/107/192|226|
|FullMAP-OneWayAffinity|-0.145662 [-0.344155, 0.054863]|+0.978390 [0.767558, 1.199906]|168/112/216|303|
|FullMAP-AuxIndependent|-0.121897 [-0.321227, 0.074240]|-0.083132 [-0.204191, 0.030577]|93/47/94|96|
|FullMAP-AuxSourceROI|-0.174770 [-0.465210, 0.108024]|-0.100493 [-0.274786, 0.067298]|221/90/223|197|
|FullMAP-AuxE_ROI|-0.249752 [-0.535223, 0.041523]|-0.134317 [-0.300683, 0.028280]|202/91/203|183|
|FullMAP-NoSpatialTrajectory|-0.249752 [-0.535223, 0.041523]|+0.724978 [0.325060, 1.131767]|202/240/354|390|
|FullMAP-NoTemporalInference|+9.373320 [8.373270, 10.392112]|+4.831099 [4.263833, 5.407858]|364/237/375|980|
|FullMAP-S0T0|+2.821502 [2.097354, 3.544531]|+2.345854 [1.823431, 2.877482]|468/387/574|833|
|FullMAP-S1T0|+2.821502 [2.097354, 3.544531]|+1.366962 [0.938541, 1.799398]|468/305/476|705|
|FullMAP-S0T1|-0.249752 [-0.535223, 0.041523]|+0.724978 [0.325060, 1.131767]|202/240/354|390|
|S1T1-S1T0|+2.821502 [2.097354, 3.544531]|+1.366962 [0.938541, 1.799398]|468/305/476|705|
|S0T1-S0T0|+3.071255 [2.341218, 3.800142]|+1.620877 [1.201837, 2.041639]|460/272/464|718|
|S1T0-S0T0|+0.000000 [0.000000, 0.000000]|+0.978892 [0.637944, 1.331710]|0/180/180|263|

## Duration groups

Short is native-GT duration <=5s; long is >5s.

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / FullMAP-Source|544|+3.585731 [1.767606, 5.419920]|+3.417933 [2.090251, 4.723997]|
|short_le5s / FullMAP-SourceJointMAP|544|-0.511595 [-1.668386, 0.652049]|+1.522689 [0.477723, 2.570922]|
|short_le5s / FullMAP-NoSharedCorrection|544|+0.044505 [-0.281176, 0.378879]|+0.504927 [0.248636, 0.750456]|
|short_le5s / FullMAP-PastOnly|544|-0.461287 [-0.860041, -0.068472]|+0.275724 [-0.047515, 0.593914]|
|short_le5s / FullMAP-OneWayAffinity|544|-0.323638 [-0.721643, 0.085551]|+0.766646 [0.444224, 1.083406]|
|short_le5s / FullMAP-AuxIndependent|544|-0.260515 [-0.595301, 0.063231]|-0.115867 [-0.320292, 0.063182]|
|short_le5s / FullMAP-AuxSourceROI|544|+0.133351 [-0.433957, 0.689984]|-0.029949 [-0.356632, 0.276074]|
|short_le5s / FullMAP-AuxE_ROI|544|+0.046472 [-0.453872, 0.539594]|-0.016610 [-0.319419, 0.273856]|
|short_le5s / FullMAP-NoSpatialTrajectory|544|+0.046472 [-0.453872, 0.539594]|+1.158379 [0.569619, 1.741954]|
|short_le5s / FullMAP-NoTemporalInference|544|+3.585731 [1.767606, 5.419920]|+1.782124 [0.740883, 2.873767]|
|short_le5s / FullMAP-S0T0|544|-0.511595 [-1.668386, 0.652049]|+1.009113 [0.158355, 1.846828]|
|short_le5s / FullMAP-S1T0|544|-0.511595 [-1.668386, 0.652049]|-0.284473 [-0.970635, 0.407222]|
|short_le5s / FullMAP-S0T1|544|+0.046472 [-0.453872, 0.539594]|+1.158379 [0.569619, 1.741954]|
|short_le5s / S1T1-S1T0|544|-0.511595 [-1.668386, 0.652049]|-0.284473 [-0.970635, 0.407222]|
|short_le5s / S0T1-S0T0|544|-0.558067 [-1.779814, 0.646902]|-0.149266 [-0.847347, 0.561062]|
|short_le5s / S1T0-S0T0|544|+0.000000 [0.000000, 0.000000]|+1.293586 [0.676245, 1.908987]|
|long_gt5s / FullMAP-Source|1358|+11.691766 [10.525444, 12.887558]|+7.748716 [6.815321, 8.703140]|
|long_gt5s / FullMAP-SourceJointMAP|1358|+4.156705 [3.289639, 5.042575]|+3.837779 [3.032479, 4.644635]|
|long_gt5s / FullMAP-NoSharedCorrection|1358|-0.023993 [-0.197055, 0.150299]|+0.845656 [0.656003, 1.043356]|
|long_gt5s / FullMAP-PastOnly|1358|-0.036937 [-0.245867, 0.176797]|+0.351254 [0.142986, 0.559028]|
|long_gt5s / FullMAP-OneWayAffinity|1358|-0.074366 [-0.295937, 0.154001]|+1.063212 [0.792121, 1.345534]|
|long_gt5s / FullMAP-AuxIndependent|1358|-0.066368 [-0.290072, 0.152245]|-0.070018 [-0.218853, 0.068524]|
|long_gt5s / FullMAP-AuxSourceROI|1358|-0.298200 [-0.642007, 0.061400]|-0.128752 [-0.343309, 0.103970]|
|long_gt5s / FullMAP-AuxE_ROI|1358|-0.368417 [-0.707777, -0.003753]|-0.181470 [-0.384981, 0.038542]|
|long_gt5s / FullMAP-NoSpatialTrajectory|1358|-0.368417 [-0.707777, -0.003753]|+0.551362 [0.068418, 1.058520]|
|long_gt5s / FullMAP-NoTemporalInference|1358|+11.691766 [10.525444, 12.887558]|+6.052484 [5.376777, 6.751077]|
|long_gt5s / FullMAP-S0T0|1358|+4.156705 [3.289639, 5.042575]|+2.881338 [2.215126, 3.565674]|
|long_gt5s / FullMAP-S1T0|1358|+4.156705 [3.289639, 5.042575]|+2.028509 [1.510372, 2.549535]|
|long_gt5s / FullMAP-S0T1|1358|-0.368417 [-0.707777, -0.003753]|+0.551362 [0.068418, 1.058520]|
|long_gt5s / S1T1-S1T0|1358|+4.156705 [3.289639, 5.042575]|+2.028509 [1.510372, 2.549535]|
|long_gt5s / S0T1-S0T0|1358|+4.525121 [3.666805, 5.403609]|+2.329976 [1.828225, 2.834367]|
|long_gt5s / S1T0-S0T0|1358|+0.000000 [0.000000, 0.000000]|+0.852829 [0.479383, 1.259232]|

## Controls and interpretation

NoSharedCorrection sets E=Source and rebuilds all affected geometry, auxiliary trajectory, actual raw-grid ROI pooling, posterior, MAP interval and final DP. PastOnly removes future anchors; OneWayAffinity replaces the original bidirectional complete-box matching by the original forward term, keeping the other settings fixed. Their effects are assessed against Full in the paired table, not assumed beneficial.

AuxIndependent fixes the auxiliary bank/validity/unary and removes only auxiliary path links, with every downstream step recomputed. AuxSourceROI and AuxE_ROI use Source or E directly as auxiliary ROI boxes. Their results do not establish an independent first-auxiliary-DP benefit when CIs cross0. This is distinct from the already completed final-DP comparison.

NoSpatialTrajectory removes propagation candidates and both path DPs, using E directly for boxes and ROI; it is not interpolation, greedy replacement or expert overwrite. NoTemporalInference removes visual posterior updating and restores native per-offset MAP/hull, then resolves the retained spatial DP on that interval. Its comparison also changes temporal readout, so it is not solely a posterior-evidence attribution.

The E-fixed factorial uses S0T0=E/q0jointMAP, S1T0=spatial/q0jointMAP, S0T1=E/updated q1 from E ROIs, S1T1=full spatial trajectory/updated q1. Every changed interval resolves the relevant final DP. Interaction is (S11-S10)-(S01-S00), rather than a sum of progressive gains.

Interaction delta t -0.249752 [-0.535223, 0.041523] pp; delta v -0.253915 [-0.444752, -0.059924] pp. A negative interaction rules out a positive superadditive claim under this defined control; it does not negate the separate positive component effects.

## Costs and reproducibility

{
  "current_cached_query_CPU_segments": {
    "2025353:245981057": 260.9727392923087
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
  "new_expert": 0,
  "scope": "Original raw-grid acquisition and cached CPU replay reported separately. Parent worker/query/Source/model-load intervals may be nested; do not add them or call replay deployment latency. Shared old raw-grid acquisitions retain original cost provenance."
}

All observed/unobserved fixed-support metrics, boundary errors, signed positives and negatives, severe/rescue counts and full duration CIs remain in report.json, all_paired_tails.json and both signed TSVs. Media, captions, coordinates, raw features and weights stay private.
