# Current no-shrinkage components: hc2

Complete whole-domain seal → independent official scoring → exact array rederivation and all-mean audit: **1902 queries / 227 parents**, 13314 official dense checks (max discrepancy 1.3322676295501878e-15), 3804 prediction/receipt SHA checks plus bound parent hashes verified during derivation. Zero new Source, expert, optimizer or GT scoring in this root export.

Kappa0 is the directly user-selected current method. Selection followed exposed TA full results; no claim of independent unseen confirmation. Old kappa64 scientific files and outputs remain immutable. Source uses native per-offset MAP and physical hull; SourceJointMAP isolates readout. Every changed posterior uses its own MAP interval and actually re-solves spatial DP.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|40.141475|20.704678|29.022082|8.937960|0/0/0|
|EightReplacement|46.693293|24.709592|36.698212|12.355415|220/125/244|
|SharedCorrection|46.693293|24.868872|37.013670|13.827550|220/171/304|
|SpatialInference|46.693293|25.847764|39.011567|15.036803|220/241/364|
|FullMAP|49.514796|27.214726|41.062040|16.614090|364/305/467|
|SourceJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|FinalIndependent|49.514796|26.165243|38.853838|14.563617|364/332/477|

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|EightReplacement-Source|+6.551818 [5.857056, 7.264109]|+4.004913 [3.573244, 4.454615]|220/125/244|916|
|SharedCorrection-EightReplacement|+0.000000 [0.000000, 0.000000]|+0.159280 [-0.056526, 0.382438]|0/148/148|127|
|SpatialInference-SharedCorrection|+0.000000 [0.000000, 0.000000]|+0.978892 [0.637944, 1.331710]|0/180/180|263|
|FullMAP-SpatialInference|+2.821502 [2.097354, 3.544531]|+1.366962 [0.938541, 1.799398]|468/305/476|705|
|FullMAP-Source|+9.373320 [8.373270, 10.392112]|+6.510048 [5.779450, 7.252761]|364/305/467|1124|
|FullMAP-SourceJointMAP|+2.821502 [2.097354, 3.544531]|+3.175629 [2.559407, 3.810588]|468/374/597|896|
|FullMAP-FinalIndependent|+0.000000 [0.000000, 0.000000]|+1.049483 [0.941888, 1.162109]|0/9/9|103|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / EightReplacement-Source|544|+4.097325 [2.717493, 5.493670]|+2.687126 [1.817025, 3.545493]|
|short_le5s / SharedCorrection-EightReplacement|544|+0.000000 [0.000000, 0.000000]|-0.278307 [-0.641356, 0.061186]|
|short_le5s / SpatialInference-SharedCorrection|544|+0.000000 [0.000000, 0.000000]|+1.293586 [0.676245, 1.908987]|
|short_le5s / FullMAP-SpatialInference|544|-0.511595 [-1.668386, 0.652049]|-0.284473 [-0.970635, 0.407222]|
|short_le5s / FullMAP-Source|544|+3.585731 [1.767606, 5.419920]|+3.417933 [2.090251, 4.723997]|
|short_le5s / FullMAP-SourceJointMAP|544|-0.511595 [-1.668386, 0.652049]|+1.522689 [0.477723, 2.570922]|
|short_le5s / FullMAP-FinalIndependent|544|+0.000000 [0.000000, 0.000000]|+0.862905 [0.692524, 1.030666]|
|long_gt5s / EightReplacement-Source|1358|+7.535061 [6.728364, 8.313545]|+4.532805 [4.015531, 5.063404]|
|long_gt5s / SharedCorrection-EightReplacement|1358|+0.000000 [0.000000, 0.000000]|+0.334573 [0.055359, 0.604667]|
|long_gt5s / SpatialInference-SharedCorrection|1358|+0.000000 [0.000000, 0.000000]|+0.852829 [0.479383, 1.259232]|
|long_gt5s / FullMAP-SpatialInference|1358|+4.156705 [3.289639, 5.042575]|+2.028509 [1.510372, 2.549535]|
|long_gt5s / FullMAP-Source|1358|+11.691766 [10.525444, 12.887558]|+7.748716 [6.815321, 8.703140]|
|long_gt5s / FullMAP-SourceJointMAP|1358|+4.156705 [3.289639, 5.042575]|+3.837779 [3.032479, 4.644635]|
|long_gt5s / FullMAP-FinalIndependent|1358|+0.000000 [0.000000, 0.000000]|+1.124224 [0.990171, 1.260118]|

## Fixed controls and interpretation

The five progressive rows use unchanged native Source, eight-only expert replacement, shared initialization **E**, spatial path inference under q0 jointMAP, and current kappa0 FullMAP. The first four rows exactly equal the previous component experiment. SharedCorrection is E, not all observed frames forcibly overwritten by experts. The first native-Source→eight-replacement change also includes q0 jointMAP readout; SourceJointMAP is therefore separate. Shared−Eight vIoU CI crosses zero and observed-frame sIoU falls, while unobserved sIoU rises. Spatial−Shared vIoU is positive. Full−Spatial vIoU is positive on average, but short-event delta is negative with a CI crossing zero; unobserved spatial-IoU delta also crosses zero. All observed/unobserved outcomes remain in report.json.

FinalDP versus independent selection holds the final interval and all upstream evidence/candidates/unaries fixed. It supports the final trajectory selection only, not an independent contribution of the first auxiliary DP. All improvements, harms, ties and severe losses remain in final_DP_control.json; no claim of universally beneficial DP.

## Costs and evidence

{
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "current_worker_CPU_seconds": {
    "1668004:245631588": 43.81499255076051
  },
  "scope": "Conditional cached-query CPU wall excludes setup/outer-loop, may include shared-host effects. Original Source/expert acquisitions remain nonzero; nested costs must not be added or used as current deployment latency."
}

Confidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means a signed tIoU or vIoU loss >5pp, union counted once; rescues use a >5pp gain. All signed positives and negatives, all means, CIs and tails remain in the public report/TSVs. Private captions/media/coordinates/features/weights are excluded.
