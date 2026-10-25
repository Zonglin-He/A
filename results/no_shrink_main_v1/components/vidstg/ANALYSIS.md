# Current no-shrinkage components: vidstg

Complete whole-domain seal → independent official scoring → exact array rederivation and all-mean audit: **10303 queries / 732 parents**, 72121 official dense checks (max discrepancy 3.7747582837255322e-15), 20606 prediction/receipt SHA checks plus bound parent hashes verified during derivation. Zero new Source, expert, optimizer or GT scoring in this root export.

Kappa0 is the directly user-selected current method. Selection followed exposed TA full results; no claim of independent unseen confirmation. Old kappa64 scientific files and outputs remain immutable. Source uses native per-offset MAP and physical hull; SourceJointMAP isolates readout. Every changed posterior uses its own MAP interval and actually re-solves spatial DP.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|31.728454|11.138484|12.831214|3.348539|0/0/0|
|EightReplacement|33.450806|12.227933|14.025041|4.212365|1015/356/1118|
|SharedCorrection|33.450806|13.243519|16.995050|7.366786|1015/1194/1917|
|SpatialInference|33.450806|13.871399|17.790935|7.590022|1015/1109/1837|
|FullMAP|34.927982|14.548371|19.450645|8.492672|1487/1221/2238|
|SourceJointMAP|33.450806|11.641627|13.345627|3.630011|1015/272/1015|
|FinalIndependent|34.927982|13.992207|18.596525|7.638552|1487/1296/2264|

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|EightReplacement-Source|+1.722351 [1.431940, 2.029248]|+1.089449 [0.946177, 1.246573]|1015/356/1118|2395|
|SharedCorrection-EightReplacement|+0.000000 [0.000000, 0.000000]|+1.015587 [0.758144, 1.280449]|0/1187/1187|1979|
|SpatialInference-SharedCorrection|+0.000000 [0.000000, 0.000000]|+0.627880 [0.493177, 0.767989]|0/341/341|774|
|FullMAP-SpatialInference|+1.477176 [1.074696, 1.889017]|+0.676971 [0.472311, 0.885336]|1896/761/1905|2500|
|FullMAP-Source|+3.199528 [2.774632, 3.659093]|+3.409887 [2.994156, 3.838533]|1487/1221/2238|4405|
|FullMAP-SourceJointMAP|+1.477176 [1.074696, 1.889017]|+2.906744 [2.508579, 3.311928]|1896/1282/2643|3944|
|FullMAP-FinalIndependent|+0.000000 [0.000000, 0.000000]|+0.556164 [0.507176, 0.608019]|0/19/19|293|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / EightReplacement-Source|6060|-0.475684 [-0.719901, -0.221092]|+0.256555 [0.117451, 0.405348]|
|short_le5s / SharedCorrection-EightReplacement|6060|+0.000000 [0.000000, 0.000000]|+0.004080 [-0.216243, 0.228558]|
|short_le5s / SpatialInference-SharedCorrection|6060|+0.000000 [0.000000, 0.000000]|+0.701956 [0.545885, 0.860638]|
|short_le5s / FullMAP-SpatialInference|6060|+0.397731 [0.073838, 0.727991]|+0.165978 [0.002471, 0.333395]|
|short_le5s / FullMAP-Source|6060|-0.077952 [-0.381935, 0.234039]|+1.128568 [0.774518, 1.512491]|
|short_le5s / FullMAP-SourceJointMAP|6060|+0.397731 [0.073838, 0.727991]|+1.399819 [1.039258, 1.794239]|
|short_le5s / FullMAP-FinalIndependent|6060|+0.000000 [0.000000, 0.000000]|+0.327101 [0.282989, 0.373181]|
|long_gt5s / EightReplacement-Source|4243|+4.861661 [4.407957, 5.369571]|+2.279018 [2.053177, 2.527708]|
|long_gt5s / SharedCorrection-EightReplacement|4243|+0.000000 [0.000000, 0.000000]|+2.460255 [2.015357, 2.923026]|
|long_gt5s / SpatialInference-SharedCorrection|4243|+0.000000 [0.000000, 0.000000]|+0.522082 [0.319007, 0.734358]|
|long_gt5s / FullMAP-SpatialInference|4243|+3.018877 [2.157248, 3.881630]|+1.406790 [0.990124, 1.839240]|
|long_gt5s / FullMAP-Source|4243|+7.880539 [7.042697, 8.733172]|+6.668145 [5.967036, 7.399667]|
|long_gt5s / FullMAP-SourceJointMAP|4243|+3.018877 [2.157248, 3.881630]|+5.058985 [4.389985, 5.739213]|
|long_gt5s / FullMAP-FinalIndependent|4243|+0.000000 [0.000000, 0.000000]|+0.883319 [0.796041, 0.975637]|

## Fixed controls and interpretation

The five progressive rows use unchanged native Source, eight-only expert replacement, shared initialization **E**, spatial path inference under q0 jointMAP, and current kappa0 FullMAP. The first four rows exactly equal the previous component experiment. SharedCorrection is E, not all observed frames forcibly overwritten by experts. The first native-Source→eight-replacement change also includes q0 jointMAP readout; SourceJointMAP is therefore separate. Signed effects and confidence intervals are dataset-specific and are reported in the complete paired table above; no common sign is assumed across domains. All observed/unobserved outcomes remain in report.json.

FinalDP versus independent selection holds the final interval and all upstream evidence/candidates/unaries fixed. It supports the final trajectory selection only, not an independent contribution of the first auxiliary DP. All improvements, harms, ties and severe losses remain in final_DP_control.json; no claim of universally beneficial DP.

## Costs and evidence

{
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "current_worker_CPU_seconds": {
    "1668005:245631588": 305.5160458246246
  },
  "scope": "Conditional cached-query CPU wall excludes setup/outer-loop, may include shared-host effects. Original Source/expert acquisitions remain nonzero; nested costs must not be added or used as current deployment latency."
}

Confidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means a signed tIoU or vIoU loss >5pp, union counted once; rescues use a >5pp gain. All signed positives and negatives, all means, CIs and tails remain in the public report/TSVs. Private captions/media/coordinates/features/weights are excluded.
