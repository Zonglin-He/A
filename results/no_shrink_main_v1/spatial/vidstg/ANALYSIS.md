# Current no-shrinkage structural controls: vidstg

All 10303 queries / 732 parents sealed before independent official labels; 154545 official dense checks (max error 3.7747582837255322e-15), 20606 new prediction/receipt SHA checks plus all bound collector/capture/compact/teacher provenance. Every array cold rederived exactly and all scalar means checked. This export reads already-scored scalars only: zero model, optimizer, Source, expert or GT-scoring calls.

Current kappa0 was selected by direct user after full-result exposure. Other method parameters remain frozen; these historically exposed full sets are not new independent confirmation. Native Source remains per-offset MAP plus physical hull, and SourceJointMAP isolates readout.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|
|---|---:|---:|---:|---:|---|
|Source|31.728454|11.138484|12.831214|3.348539|0/0/0|
|SourceJointMAP|33.450806|11.641627|13.345627|3.630011|1015/272/1015|
|FullMAP|34.927982|14.548371|19.450645|8.492672|1487/1221/2238|
|NoSharedCorrection|34.877961|13.554270|17.189168|6.425313|1457/1033/1974|
|PastOnly|34.840715|14.322819|19.062409|8.288848|1496/1240/2248|
|OneWayAffinity|34.911899|14.048485|18.606231|7.842376|1517/1233/2274|
|AuxIndependent|34.895972|14.528705|19.324469|8.492672|1476/1211/2217|
|AuxSourceROI|34.908528|14.552160|19.392410|8.385907|1531/1222/2276|
|AuxE_ROI|34.918192|14.558151|19.411822|8.463554|1562/1222/2294|
|NoSpatialTrajectory|34.918192|13.948734|18.722702|8.230612|1562/1306/2357|
|NoTemporalInference|31.728454|13.261107|16.995050|6.978550|0/1068/1068|
|S0T0|33.450806|13.243519|16.995050|7.366786|1015/1194/1917|
|S1T0|33.450806|13.871399|17.790935|7.590022|1015/1109/1837|
|S0T1|34.918192|13.948734|18.722702|8.230612|1562/1306/2357|
|S1T1|34.927982|14.548371|19.450645|8.492672|1487/1221/2238|

## Paired effects and tails

Positive means first configuration exceeds second. All intervals are pointwise parent-cluster bootstrap CIs, not familywise. Severe is a loss greater than5pp in tIoU or vIoU; the union counts each query once. Rescue is a gain greater than5pp in either metric.

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|FullMAP-Source|+3.199528 [2.774632, 3.659093]|+3.409887 [2.994156, 3.838533]|1487/1221/2238|4405|
|FullMAP-SourceJointMAP|+1.477176 [1.074696, 1.889017]|+2.906744 [2.508579, 3.311928]|1896/1282/2643|3944|
|FullMAP-NoSharedCorrection|+0.050022 [-0.056958, 0.154134]|+0.994101 [0.812697, 1.185677]|676/534/1035|1671|
|FullMAP-PastOnly|+0.087268 [-0.000850, 0.176891]|+0.225552 [0.163720, 0.288317]|418/163/485|682|
|FullMAP-OneWayAffinity|+0.016084 [-0.072948, 0.108834]|+0.499886 [0.412699, 0.589356]|559/240/667|1006|
|FullMAP-AuxIndependent|+0.032010 [-0.035059, 0.100511]|+0.019666 [-0.016242, 0.056182]|286/85/286|312|
|FullMAP-AuxSourceROI|+0.019455 [-0.120755, 0.157118]|-0.003790 [-0.062577, 0.056041]|1000/262/1005|1026|
|FullMAP-AuxE_ROI|+0.009790 [-0.107217, 0.128661]|-0.009781 [-0.060899, 0.042102]|759/198/762|773|
|FullMAP-NoSpatialTrajectory|+0.009790 [-0.107217, 0.128661]|+0.599637 [0.445527, 0.758889]|759/485/1049|1396|
|FullMAP-NoTemporalInference|+3.199528 [2.774632, 3.659093]|+1.287264 [1.073083, 1.519803]|1487/594/1499|3143|
|FullMAP-S0T0|+1.477176 [1.074696, 1.889017]|+1.304851 [1.068437, 1.546357]|1896/933/2105|2975|
|FullMAP-S1T0|+1.477176 [1.074696, 1.889017]|+0.676971 [0.472311, 0.885336]|1896/761/1905|2500|
|FullMAP-S0T1|+0.009790 [-0.107217, 0.128661]|+0.599637 [0.445527, 0.758889]|759/485/1049|1396|
|S1T1-S1T0|+1.477176 [1.074696, 1.889017]|+0.676971 [0.472311, 0.885336]|1896/761/1905|2500|
|S0T1-S0T0|+1.467386 [1.057285, 1.888033]|+0.705214 [0.500586, 0.914261]|1947/712/1951|2539|
|S1T0-S0T0|+0.000000 [0.000000, 0.000000]|+0.627880 [0.493177, 0.767989]|0/341/341|774|

## Duration groups

Short is native-GT duration <=5s; long is >5s.

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / FullMAP-Source|6060|-0.077952 [-0.381935, 0.234039]|+1.128568 [0.774518, 1.512491]|
|short_le5s / FullMAP-SourceJointMAP|6060|+0.397731 [0.073838, 0.727991]|+1.399819 [1.039258, 1.794239]|
|short_le5s / FullMAP-NoSharedCorrection|6060|+0.053432 [-0.049404, 0.154994]|+0.550598 [0.384739, 0.723616]|
|short_le5s / FullMAP-PastOnly|6060|+0.033118 [-0.055888, 0.120846]|+0.149736 [0.088346, 0.210248]|
|short_le5s / FullMAP-OneWayAffinity|6060|+0.045431 [-0.043347, 0.133597]|+0.301649 [0.224811, 0.381079]|
|short_le5s / FullMAP-AuxIndependent|6060|+0.005540 [-0.058837, 0.071017]|+0.001999 [-0.029339, 0.034310]|
|short_le5s / FullMAP-AuxSourceROI|6060|+0.024128 [-0.112994, 0.168167]|-0.013289 [-0.068724, 0.043262]|
|short_le5s / FullMAP-AuxE_ROI|6060|+0.051694 [-0.079223, 0.183422]|-0.007237 [-0.056879, 0.044845]|
|short_le5s / FullMAP-NoSpatialTrajectory|6060|+0.051694 [-0.079223, 0.183422]|+0.679474 [0.509365, 0.850034]|
|short_le5s / FullMAP-NoTemporalInference|6060|-0.077952 [-0.381935, 0.234039]|-0.156229 [-0.302979, -0.004428]|
|short_le5s / FullMAP-S0T0|6060|+0.397731 [0.073838, 0.727991]|+0.867934 [0.633750, 1.109440]|
|short_le5s / FullMAP-S1T0|6060|+0.397731 [0.073838, 0.727991]|+0.165978 [0.002471, 0.333395]|
|short_le5s / FullMAP-S0T1|6060|+0.051694 [-0.079223, 0.183422]|+0.679474 [0.509365, 0.850034]|
|short_le5s / S1T1-S1T0|6060|+0.397731 [0.073838, 0.727991]|+0.165978 [0.002471, 0.333395]|
|short_le5s / S0T1-S0T0|6060|+0.346037 [0.006013, 0.687903]|+0.188460 [0.029695, 0.349071]|
|short_le5s / S1T0-S0T0|6060|+0.000000 [0.000000, 0.000000]|+0.701956 [0.545885, 0.860638]|
|long_gt5s / FullMAP-Source|4243|+7.880539 [7.042697, 8.733172]|+6.668145 [5.967036, 7.399667]|
|long_gt5s / FullMAP-SourceJointMAP|4243|+3.018877 [2.157248, 3.881630]|+5.058985 [4.389985, 5.739213]|
|long_gt5s / FullMAP-NoSharedCorrection|4243|+0.045151 [-0.172502, 0.260595]|+1.627527 [1.319887, 1.950078]|
|long_gt5s / FullMAP-PastOnly|4243|+0.164606 [-0.007297, 0.340905]|+0.333835 [0.224173, 0.448323]|
|long_gt5s / FullMAP-OneWayAffinity|4243|-0.025832 [-0.224914, 0.170354]|+0.783015 [0.614091, 0.956334]|
|long_gt5s / FullMAP-AuxIndependent|4243|+0.069816 [-0.066495, 0.211537]|+0.044899 [-0.030589, 0.122377]|
|long_gt5s / FullMAP-AuxSourceROI|4243|+0.012779 [-0.260530, 0.279751]|+0.009777 [-0.108768, 0.130055]|
|long_gt5s / FullMAP-AuxE_ROI|4243|-0.050059 [-0.267030, 0.161602]|-0.013414 [-0.117418, 0.091333]|
|long_gt5s / FullMAP-NoSpatialTrajectory|4243|-0.050059 [-0.267030, 0.161602]|+0.485611 [0.244964, 0.733282]|
|long_gt5s / FullMAP-NoTemporalInference|4243|+7.880539 [7.042697, 8.733172]|+3.348910 [2.909447, 3.819269]|
|long_gt5s / FullMAP-S0T0|4243|+3.018877 [2.157248, 3.881630]|+1.928872 [1.476395, 2.389016]|
|long_gt5s / FullMAP-S1T0|4243|+3.018877 [2.157248, 3.881630]|+1.406790 [0.990124, 1.839240]|
|long_gt5s / FullMAP-S0T1|4243|-0.050059 [-0.267030, 0.161602]|+0.485611 [0.244964, 0.733282]|
|long_gt5s / S1T1-S1T0|4243|+3.018877 [2.157248, 3.881630]|+1.406790 [0.990124, 1.839240]|
|long_gt5s / S0T1-S0T0|4243|+3.068936 [2.217529, 3.932641]|+1.443261 [1.025859, 1.872107]|
|long_gt5s / S1T0-S0T0|4243|+0.000000 [0.000000, 0.000000]|+0.522082 [0.319007, 0.734358]|

## Controls and interpretation

NoSharedCorrection sets E=Source and rebuilds all affected geometry, auxiliary trajectory, actual raw-grid ROI pooling, posterior, MAP interval and final DP. PastOnly removes future anchors; OneWayAffinity replaces the original bidirectional complete-box matching by the original forward term, keeping the other settings fixed. Their effects are assessed against Full in the paired table, not assumed beneficial.

AuxIndependent fixes the auxiliary bank/validity/unary and removes only auxiliary path links, with every downstream step recomputed. AuxSourceROI and AuxE_ROI use Source or E directly as auxiliary ROI boxes. Their results do not establish an independent first-auxiliary-DP benefit when CIs cross0. This is distinct from the already completed final-DP comparison.

NoSpatialTrajectory removes propagation candidates and both path DPs, using E directly for boxes and ROI; it is not interpolation, greedy replacement or expert overwrite. NoTemporalInference removes visual posterior updating and restores native per-offset MAP/hull, then resolves the retained spatial DP on that interval. Its comparison also changes temporal readout, so it is not solely a posterior-evidence attribution.

The E-fixed factorial uses S0T0=E/q0jointMAP, S1T0=spatial/q0jointMAP, S0T1=E/updated q1 from E ROIs, S1T1=full spatial trajectory/updated q1. Every changed interval resolves the relevant final DP. Interaction is (S11-S10)-(S01-S00), rather than a sum of progressive gains.

Interaction delta t +0.009790 [-0.107217, 0.128661] pp; delta v -0.028243 [-0.090367, 0.032336] pp. The signed interaction and its confidence interval determine support for superadditivity. A zero-crossing interval does not establish a positive or negative interaction and is not an equivalence proof. Separate component effects are reported independently.

## Costs and reproducibility

{
  "current_cached_query_CPU_segments": {
    "3432872:247191005": 1994.859637080226
  },
  "parent_collector": {
    "seal_sha256": "fbd1eb745dbc2311ebe720ebb509e971a72a3c72272d82c964fd9174247ca964",
    "worker": {
      "pid": 3432864,
      "startticks": 247190991
    },
    "actual_counts": {
      "queries": 10303,
      "shared_raw_grid_hits": 192,
      "legacy_raw_grid_hits": 0,
      "full_Source_queries": 10111,
      "appearance_prefix_queries": 0,
      "new_expert": 0
    },
    "whole_worker_seconds": 48532.223635217175,
    "model_load_seconds": 6.475349193904549,
    "Source_forward_seconds": 26544.502293606754,
    "query_pipeline_seconds": 46851.946231667884
  },
  "inherited_original_Source_and_eight_teacher_cost_nonzero": true,
  "new_expert": 0,
  "scope": "Original raw-grid acquisition and cached CPU replay reported separately. Parent worker/query/Source/model-load intervals may be nested; do not add them or call replay deployment latency. Shared old raw-grid acquisitions retain original cost provenance."
}

All observed/unobserved fixed-support metrics, boundary errors, signed positives and negatives, severe/rescue counts and full duration CIs remain in report.json, all_paired_tails.json and both signed TSVs. Media, captions, coordinates, raw features and weights stay private.
