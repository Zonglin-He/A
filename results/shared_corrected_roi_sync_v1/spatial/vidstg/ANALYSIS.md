# Direct shared-corrected-box ROI structural controls: vidstg

All 10303 queries / 732 parents sealed before independent official labels; 154545 official dense checks (max error 3.7747582837255322e-15), 20606 new prediction/receipt SHA checks plus all bound collector/capture/compact/teacher provenance. Every array cold rederived exactly and all scalar means checked. This export reads already-scored scalars only: zero model, optimizer, Source, expert or GT-scoring calls.

Kappa0 and removal of the first auxiliary DP in favor of direct shared-corrected-box ROI pooling were adopted by explicit user instruction after exposed full-domain ablations. Final trajectory DP and other method parameters remain frozen; these historically exposed full sets are not new independent confirmation. Native Source remains per-offset MAP plus physical hull, and SourceJointMAP isolates readout.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|
|---|---:|---:|---:|---:|---|
|Source|31.728454|11.138484|12.831214|3.348539|0/0/0|
|SourceJointMAP|33.450806|11.641627|13.345627|3.630011|1015/272/1015|
|FullMAP|34.918192|14.558151|19.411822|8.463554|1562/1222/2294|
|NoSharedCorrection|34.908528|13.598841|17.092109|6.376783|1531/1039/2042|
|PastOnly|34.918192|14.345138|18.945938|8.317966|1562/1235/2300|
|OneWayAffinity|34.918192|14.066686|18.431525|7.920023|1562/1234/2294|
|AuxIndependent|34.918192|14.558151|19.411822|8.463554|1562/1222/2294|
|AuxSourceROI|34.908528|14.552160|19.392410|8.385907|1531/1222/2276|
|AuxE_ROI|34.918192|14.558151|19.411822|8.463554|1562/1222/2294|
|NoSpatialTrajectory|34.918192|13.948734|18.722702|8.230612|1562/1306/2357|
|NoTemporalInference|31.728454|13.261107|16.995050|6.978550|0/1068/1068|
|S0T0|33.450806|13.243519|16.995050|7.366786|1015/1194/1917|
|S1T0|33.450806|13.871399|17.790935|7.590022|1015/1109/1837|
|S0T1|34.918192|13.948734|18.722702|8.230612|1562/1306/2357|
|S1T1|34.918192|14.558151|19.411822|8.463554|1562/1222/2294|

## Paired effects and tails

Positive means first configuration exceeds second. All intervals are pointwise parent-cluster bootstrap CIs, not familywise. Severe is a loss greater than5pp in tIoU or vIoU; the union counts each query once. Rescue is a gain greater than5pp in either metric.

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|FullMAP-Source|+3.189738 [2.753594, 3.658697]|+3.419668 [3.009582, 3.851146]|1562/1222/2294|4460|
|FullMAP-SourceJointMAP|+1.467386 [1.057285, 1.888033]|+2.916524 [2.524561, 3.322600]|1947/1269/2679|3959|
|FullMAP-NoSharedCorrection|+0.009665 [-0.116164, 0.132882]|+0.959310 [0.782941, 1.142772]|748/573/1117|1762|
|FullMAP-PastOnly|+0.000000 [0.000000, 0.000000]|+0.213013 [0.166350, 0.260819]|0/83/83|242|
|FullMAP-OneWayAffinity|+0.000000 [0.000000, 0.000000]|+0.491465 [0.421118, 0.566518]|0/126/126|518|
|FullMAP-AuxIndependent|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|0/0/0|0|
|FullMAP-AuxSourceROI|+0.009665 [-0.116164, 0.132882]|+0.005991 [-0.038913, 0.051228]|748/188/751|754|
|FullMAP-AuxE_ROI|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|0/0/0|0|
|FullMAP-NoSpatialTrajectory|+0.000000 [0.000000, 0.000000]|+0.609417 [0.471700, 0.755210]|0/375/375|809|
|FullMAP-NoTemporalInference|+3.189738 [2.753594, 3.658697]|+1.297044 [1.075346, 1.534715]|1562/599/1572|3182|
|FullMAP-S0T0|+1.467386 [1.057285, 1.888033]|+1.314632 [1.080050, 1.552740]|1947/934/2169|3024|
|FullMAP-S1T0|+1.467386 [1.057285, 1.888033]|+0.686752 [0.481305, 0.895982]|1947/750/1957|2536|
|FullMAP-S0T1|+0.000000 [0.000000, 0.000000]|+0.609417 [0.471700, 0.755210]|0/375/375|809|
|S1T1-S1T0|+1.467386 [1.057285, 1.888033]|+0.686752 [0.481305, 0.895982]|1947/750/1957|2536|
|S0T1-S0T0|+1.467386 [1.057285, 1.888033]|+0.705214 [0.500586, 0.914261]|1947/712/1951|2539|
|S1T0-S0T0|+0.000000 [0.000000, 0.000000]|+0.627880 [0.493177, 0.767989]|0/341/341|774|

## Duration groups

Short is native-GT duration <=5s; long is >5s.

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / FullMAP-Source|6060|-0.129647 [-0.447674, 0.187245]|+1.135805 [0.785926, 1.511448]|
|short_le5s / FullMAP-SourceJointMAP|6060|+0.346037 [0.006013, 0.687903]|+1.407056 [1.050908, 1.795827]|
|short_le5s / FullMAP-NoSharedCorrection|6060|-0.027566 [-0.154578, 0.102475]|+0.518922 [0.354869, 0.689906]|
|short_le5s / FullMAP-PastOnly|6060|+0.000000 [0.000000, 0.000000]|+0.160061 [0.108826, 0.211496]|
|short_le5s / FullMAP-OneWayAffinity|6060|+0.000000 [0.000000, 0.000000]|+0.298703 [0.233434, 0.367405]|
|short_le5s / FullMAP-AuxIndependent|6060|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|short_le5s / FullMAP-AuxSourceROI|6060|-0.027566 [-0.154578, 0.102475]|-0.006052 [-0.055665, 0.043524]|
|short_le5s / FullMAP-AuxE_ROI|6060|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|short_le5s / FullMAP-NoSpatialTrajectory|6060|+0.000000 [0.000000, 0.000000]|+0.686711 [0.525217, 0.849912]|
|short_le5s / FullMAP-NoTemporalInference|6060|-0.129647 [-0.447674, 0.187245]|-0.148992 [-0.295221, -0.001035]|
|short_le5s / FullMAP-S0T0|6060|+0.346037 [0.006013, 0.687903]|+0.875171 [0.644216, 1.110823]|
|short_le5s / FullMAP-S1T0|6060|+0.346037 [0.006013, 0.687903]|+0.173215 [0.007694, 0.339635]|
|short_le5s / FullMAP-S0T1|6060|+0.000000 [0.000000, 0.000000]|+0.686711 [0.525217, 0.849912]|
|short_le5s / S1T1-S1T0|6060|+0.346037 [0.006013, 0.687903]|+0.173215 [0.007694, 0.339635]|
|short_le5s / S0T1-S0T0|6060|+0.346037 [0.006013, 0.687903]|+0.188460 [0.029695, 0.349071]|
|short_le5s / S1T0-S0T0|6060|+0.000000 [0.000000, 0.000000]|+0.701956 [0.545885, 0.860638]|
|long_gt5s / FullMAP-Source|4243|+7.930598 [7.102139, 8.789204]|+6.681559 [5.972095, 7.420513]|
|long_gt5s / FullMAP-SourceJointMAP|4243|+3.068936 [2.217529, 3.932641]|+5.072398 [4.395900, 5.758100]|
|long_gt5s / FullMAP-NoSharedCorrection|4243|+0.062838 [-0.180383, 0.310264]|+1.588287 [1.296313, 1.907157]|
|long_gt5s / FullMAP-PastOnly|4243|+0.000000 [0.000000, 0.000000]|+0.288641 [0.211942, 0.370107]|
|long_gt5s / FullMAP-OneWayAffinity|4243|+0.000000 [0.000000, 0.000000]|+0.766775 [0.636591, 0.901817]|
|long_gt5s / FullMAP-AuxIndependent|4243|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|long_gt5s / FullMAP-AuxSourceROI|4243|+0.062838 [-0.180383, 0.310264]|+0.023191 [-0.066368, 0.113517]|
|long_gt5s / FullMAP-AuxE_ROI|4243|+0.000000 [0.000000, 0.000000]|+0.000000 [0.000000, 0.000000]|
|long_gt5s / FullMAP-NoSpatialTrajectory|4243|+0.000000 [0.000000, 0.000000]|+0.499025 [0.288674, 0.715895]|
|long_gt5s / FullMAP-NoTemporalInference|4243|+7.930598 [7.102139, 8.789204]|+3.362324 [2.915842, 3.840324]|
|long_gt5s / FullMAP-S0T0|4243|+3.068936 [2.217529, 3.932641]|+1.942286 [1.496720, 2.399368]|
|long_gt5s / FullMAP-S1T0|4243|+3.068936 [2.217529, 3.932641]|+1.420204 [1.001527, 1.849466]|
|long_gt5s / FullMAP-S0T1|4243|+0.000000 [0.000000, 0.000000]|+0.499025 [0.288674, 0.715895]|
|long_gt5s / S1T1-S1T0|4243|+3.068936 [2.217529, 3.932641]|+1.420204 [1.001527, 1.849466]|
|long_gt5s / S0T1-S0T0|4243|+3.068936 [2.217529, 3.932641]|+1.443261 [1.025859, 1.872107]|
|long_gt5s / S1T0-S0T0|4243|+0.000000 [0.000000, 0.000000]|+0.522082 [0.319007, 0.734358]|

## Controls and interpretation

NoSharedCorrection sets E=Source and rebuilds uncorrected final geometry, uses the actual saved Source-box ROI features, and recomputes posterior, MAP interval and final DP. Full uses retained exact features actually pooled in shared-corrected E boxes. No old auxiliary-trajectory feature is renamed as direct-ROI appearance. PastOnly removes future anchors; OneWayAffinity replaces the original bidirectional complete-box matching by the original forward term, keeping the other settings fixed. Their effects are assessed against Full in the paired table, not assumed beneficial.

The first auxiliary DP no longer exists in the selected path. AuxIndependent and AuxE_ROI are deliberately redundant identity controls and must be exactly Full; their zero differences are implementation checks, not independent removed-component contributions or statistical equivalence claims. AuxSourceROI changes only the ROI appearance input to actual saved Source-box features while retaining main final geometry. The historical two-DP strict deletion result remains separately archived. The retained final-DP benefit is measured by the separate same-upstream independent-box comparison, not these identity rows.

NoSpatialTrajectory removes the remaining trajectory propagation/candidates and final path DP, using E directly for boxes and ROI; it is not interpolation, greedy replacement or expert overwrite. NoTemporalInference removes visual posterior updating and restores native per-offset MAP/hull, then resolves the retained spatial DP on that interval. Its comparison also changes temporal readout, so it is not solely a posterior-evidence attribution.

The E-fixed factorial uses S0T0=E/q0jointMAP, S1T0=spatial/q0jointMAP, S0T1=E/updated q1 from E ROIs, S1T1=full spatial trajectory/updated q1. Every changed interval resolves the relevant final DP. Interaction is (S11-S10)-(S01-S00), rather than a sum of progressive gains.

Interaction delta t +0.000000 [-0.000000, 0.000000] pp; delta v -0.018463 [-0.055478, 0.017686] pp. The signed interaction and its confidence interval determine support for superadditivity. A zero-crossing interval does not establish a positive or negative interaction and is not an equivalence proof. Separate component effects are reported independently.

## Costs and reproducibility

{
  "current_cached_query_elapsed_wall_seconds_by_PID_start": {
    "3404074:253105204": 1322.3171356040984
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
  "selected_first_auxiliary_DP": false,
  "new_Source": 0,
  "new_expert": 0,
  "scope": "Conditional cached time.perf_counter elapsed wall is not process CPU time and includes provenance checks. For Vid the parent Source was shared with full TTA; do not add TTA Source subtotals again. Original raw-grid acquisition and cached CPU replay reported separately. Parent worker/query/Source/model-load intervals may be nested; do not add them or call replay deployment latency. Shared old raw-grid acquisitions retain original cost provenance."
}

All observed/unobserved fixed-support metrics, boundary errors, signed positives and negatives, severe/rescue counts and full duration CIs remain in report.json, all_paired_tails.json and both signed TSVs. Media, captions, coordinates, raw features and weights stay private.
