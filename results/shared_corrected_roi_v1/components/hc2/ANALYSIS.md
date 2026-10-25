# Shared-corrected-box ROI main and components: hc2

All 1902 queries / 227 parents sealed before independent official CPU scoring: 15216 dense checks, maximum discrepancy 1.3322676295501878e-15. Separate audit verified 3804 prediction/receipt SHAs, bound parent/compact/capture hashes, every cold-rederived array and all scalar means. This root export reads saved scalar results only, with zero model, Source, expert, optimizer or GT-scoring calls.

The user selected this simplification after both full structural ablations. The same eight expert observations inform shared-corrected per-frame boxes; their ROIs directly pool appearance features. Kappa0/beta1/one prototype update, jointMAP interval and final exact trajectory DP remain. The first auxiliary DP is removed. Old method files and results remain immutable. Full datasets were already exposed; this is not independent unseen confirmation.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|40.141475|20.704678|29.022082|8.937960|0/0/0|
|SourceJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|EightReplacement|46.693293|24.709592|36.698212|12.355415|220/125/244|
|SharedCorrection|46.693293|24.868872|37.013670|13.827550|220/171/304|
|SpatialInference|46.693293|25.847764|39.011567|15.036803|220/241/364|
|HistoricalAuxDP|49.514796|27.214726|41.062040|16.614090|364/305/467|
|FullMAP|49.764548|27.349044|41.219769|16.876972|346/305/458|
|FinalIndependent|49.764548|26.281532|39.484753|15.036803|346/323/459|

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|EightReplacement-Source|+6.551818 [5.857056, 7.264109]|+4.004913 [3.573244, 4.454615]|220/125/244|916|
|SharedCorrection-EightReplacement|+0.000000 [0.000000, 0.000000]|+0.159280 [-0.056526, 0.382438]|0/148/148|127|
|SpatialInference-SharedCorrection|+0.000000 [0.000000, 0.000000]|+0.978892 [0.637944, 1.331710]|0/180/180|263|
|FullMAP-SpatialInference|+3.071255 [2.341218, 3.800142]|+1.501279 [1.059534, 1.941792]|460/292/465|719|
|FullMAP-Source|+9.623073 [8.594747, 10.683117]|+6.644365 [5.902588, 7.408451]|346/305/458|1129|
|FullMAP-SourceJointMAP|+3.071255 [2.341218, 3.800142]|+3.309946 [2.692047, 3.938965]|460/365/588|903|
|FullMAP-FinalIndependent|+0.000000 [0.000000, 0.000000]|+1.067512 [0.958161, 1.183652]|0/9/9|106|
|FullMAP-HistoricalAuxDP|+0.249752 [-0.041523, 0.535223]|+0.134317 [-0.028280, 0.300683]|182/66/183|203|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / EightReplacement-Source|544|+4.097325 [2.717493, 5.493670]|+2.687126 [1.817025, 3.545493]|
|short_le5s / SharedCorrection-EightReplacement|544|+0.000000 [0.000000, 0.000000]|-0.278307 [-0.641356, 0.061186]|
|short_le5s / SpatialInference-SharedCorrection|544|+0.000000 [0.000000, 0.000000]|+1.293586 [0.676245, 1.908987]|
|short_le5s / FullMAP-SpatialInference|544|-0.558067 [-1.779814, 0.646902]|-0.267863 [-0.979443, 0.462360]|
|short_le5s / FullMAP-Source|544|+3.539258 [1.682666, 5.388260]|+3.434542 [2.062777, 4.774484]|
|short_le5s / FullMAP-SourceJointMAP|544|-0.558067 [-1.779814, 0.646902]|+1.539298 [0.450914, 2.615297]|
|short_le5s / FullMAP-FinalIndependent|544|+0.000000 [0.000000, 0.000000]|+0.872858 [0.686613, 1.063996]|
|short_le5s / FullMAP-HistoricalAuxDP|544|-0.046472 [-0.539594, 0.453872]|+0.016610 [-0.273856, 0.319419]|
|long_gt5s / EightReplacement-Source|1358|+7.535061 [6.728364, 8.313545]|+4.532805 [4.015531, 5.063404]|
|long_gt5s / SharedCorrection-EightReplacement|1358|+0.000000 [0.000000, 0.000000]|+0.334573 [0.055359, 0.604667]|
|long_gt5s / SpatialInference-SharedCorrection|1358|+0.000000 [0.000000, 0.000000]|+0.852829 [0.479383, 1.259232]|
|long_gt5s / FullMAP-SpatialInference|1358|+4.525121 [3.666805, 5.403609]|+2.209979 [1.675721, 2.743814]|
|long_gt5s / FullMAP-Source|1358|+12.060182 [10.872964, 13.264188]|+7.930185 [7.010995, 8.893365]|
|long_gt5s / FullMAP-SourceJointMAP|1358|+4.525121 [3.666805, 5.403609]|+4.019249 [3.228219, 4.829502]|
|long_gt5s / FullMAP-FinalIndependent|1358|+0.000000 [0.000000, 0.000000]|+1.145488 [1.009648, 1.284256]|
|long_gt5s / FullMAP-HistoricalAuxDP|1358|+0.368417 [0.003753, 0.707777]|+0.181470 [-0.038542, 0.384981]|

## Interpretation

The first four progressive rows exactly retain their original values. Source uses native per-offset MAP plus physical hull; SourceJointMAP isolates readout. EightReplacement also changes the native readout to q0 jointMAP, so that increment is not solely an expert-box effect. SharedCorrection means globally corrected boxes, not raw expert overwrite. FullMAP-HistoricalAuxDP compares the selected direct-ROI method with the preserved two-DP method. Zero-crossing CIs do not establish equivalence or an average gain. The selection follows the user condition, rather than choosing the highest full-data score.

FullMAP-FinalIndependent fixes the new posterior/interval and all final candidates/unaries. It measures final path selection, not the removed first auxiliary DP. All improvements, harms, ties, short/long results and severe tails are retained. Fixed-support observed/unobserved overlap and boundary errors are in report.json; mean improvements do not guarantee every query improves.

## Costs and remaining scope

{
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "cached_query_CPU_seconds_by_PID_start": {
    "3352382:252416924": 57.750889853574336
  },
  "inherited_paid_acquisition_and_eight_expert_cost_nonzero": true,
  "scope": "Conditional cached CPU replay; excludes original acquisition and setup/outer loop. Timings may be nested and must not be added or used as deployment latency. A separate actual matched deployment experiment is required for this method."
}

CIs are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means signed tIoU or vIoU loss >5pp, union once; rescue means >5pp gain. All signed outcomes remain in public TSVs. Media, captions, coordinates, raw features and weights stay private. Structural/factorial, budget, CG transfer, old corrupt panels, actual timing and qualitative synchronization remain separate tasks; full219690corruption and extra generic TTA are not complete.
