# Shared-corrected-box ROI main and components: vidstg

All 10303 queries / 732 parents sealed before independent official CPU scoring: 82424 dense checks, maximum discrepancy 3.7747582837255322e-15. Separate audit verified 20606 prediction/receipt SHAs, bound parent/compact/capture hashes, every cold-rederived array and all scalar means. This root export reads saved scalar results only, with zero model, Source, expert, optimizer or GT-scoring calls.

The user selected this simplification after both full structural ablations. The same eight expert observations inform shared-corrected per-frame boxes; their ROIs directly pool appearance features. Kappa0/beta1/one prototype update, jointMAP interval and final exact trajectory DP remain. The first auxiliary DP is removed. Old method files and results remain immutable. Full datasets were already exposed; this is not independent unseen confirmation.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|31.728454|11.138484|12.831214|3.348539|0/0/0|
|SourceJointMAP|33.450806|11.641627|13.345627|3.630011|1015/272/1015|
|EightReplacement|33.450806|12.227933|14.025041|4.212365|1015/356/1118|
|SharedCorrection|33.450806|13.243519|16.995050|7.366786|1015/1194/1917|
|SpatialInference|33.450806|13.871399|17.790935|7.590022|1015/1109/1837|
|HistoricalAuxDP|34.927982|14.548371|19.450645|8.492672|1487/1221/2238|
|FullMAP|34.918192|14.558151|19.411822|8.463554|1562/1222/2294|
|FinalIndependent|34.918192|13.997841|18.615937|7.551199|1562/1297/2321|

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|EightReplacement-Source|+1.722351 [1.431940, 2.029248]|+1.089449 [0.946177, 1.246573]|1015/356/1118|2395|
|SharedCorrection-EightReplacement|+0.000000 [0.000000, 0.000000]|+1.015587 [0.758144, 1.280449]|0/1187/1187|1979|
|SpatialInference-SharedCorrection|+0.000000 [0.000000, 0.000000]|+0.627880 [0.493177, 0.767989]|0/341/341|774|
|FullMAP-SpatialInference|+1.467386 [1.057285, 1.888033]|+0.686752 [0.481305, 0.895982]|1947/750/1957|2536|
|FullMAP-Source|+3.189738 [2.753594, 3.658697]|+3.419668 [3.009582, 3.851146]|1562/1222/2294|4460|
|FullMAP-SourceJointMAP|+1.467386 [1.057285, 1.888033]|+2.916524 [2.524561, 3.322600]|1947/1269/2679|3959|
|FullMAP-FinalIndependent|+0.000000 [0.000000, 0.000000]|+0.560311 [0.512204, 0.612344]|0/19/19|303|
|FullMAP-HistoricalAuxDP|-0.009790 [-0.128661, 0.107217]|+0.009781 [-0.042102, 0.060899]|766/207/773|762|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / EightReplacement-Source|6060|-0.475684 [-0.719901, -0.221092]|+0.256555 [0.117451, 0.405348]|
|short_le5s / SharedCorrection-EightReplacement|6060|+0.000000 [0.000000, 0.000000]|+0.004080 [-0.216243, 0.228558]|
|short_le5s / SpatialInference-SharedCorrection|6060|+0.000000 [0.000000, 0.000000]|+0.701956 [0.545885, 0.860638]|
|short_le5s / FullMAP-SpatialInference|6060|+0.346037 [0.006013, 0.687903]|+0.173215 [0.007694, 0.339635]|
|short_le5s / FullMAP-Source|6060|-0.129647 [-0.447674, 0.187245]|+1.135805 [0.785926, 1.511448]|
|short_le5s / FullMAP-SourceJointMAP|6060|+0.346037 [0.006013, 0.687903]|+1.407056 [1.050908, 1.795827]|
|short_le5s / FullMAP-FinalIndependent|6060|+0.000000 [0.000000, 0.000000]|+0.331101 [0.287258, 0.376284]|
|short_le5s / FullMAP-HistoricalAuxDP|6060|-0.051694 [-0.183422, 0.079223]|+0.007237 [-0.044845, 0.056879]|
|long_gt5s / EightReplacement-Source|4243|+4.861661 [4.407957, 5.369571]|+2.279018 [2.053177, 2.527708]|
|long_gt5s / SharedCorrection-EightReplacement|4243|+0.000000 [0.000000, 0.000000]|+2.460255 [2.015357, 2.923026]|
|long_gt5s / SpatialInference-SharedCorrection|4243|+0.000000 [0.000000, 0.000000]|+0.522082 [0.319007, 0.734358]|
|long_gt5s / FullMAP-SpatialInference|4243|+3.068936 [2.217529, 3.932641]|+1.420204 [1.001527, 1.849466]|
|long_gt5s / FullMAP-Source|4243|+7.930598 [7.102139, 8.789204]|+6.681559 [5.972095, 7.420513]|
|long_gt5s / FullMAP-SourceJointMAP|4243|+3.068936 [2.217529, 3.932641]|+5.072398 [4.395900, 5.758100]|
|long_gt5s / FullMAP-FinalIndependent|4243|+0.000000 [0.000000, 0.000000]|+0.887675 [0.800044, 0.980530]|
|long_gt5s / FullMAP-HistoricalAuxDP|4243|+0.050059 [-0.161602, 0.267030]|+0.013414 [-0.091333, 0.117418]|

## Interpretation

The first four progressive rows exactly retain their original values. Source uses native per-offset MAP plus physical hull; SourceJointMAP isolates readout. EightReplacement also changes the native readout to q0 jointMAP, so that increment is not solely an expert-box effect. SharedCorrection means globally corrected boxes, not raw expert overwrite. FullMAP-HistoricalAuxDP compares the selected direct-ROI method with the preserved two-DP method. Zero-crossing CIs do not establish equivalence or an average gain. The selection follows the user condition, rather than choosing the highest full-data score.

FullMAP-FinalIndependent fixes the new posterior/interval and all final candidates/unaries. It measures final path selection, not the removed first auxiliary DP. All improvements, harms, ties, short/long results and severe tails are retained. Fixed-support observed/unobserved overlap and boundary errors are in report.json; mean improvements do not guarantee every query improves.

## Costs and remaining scope

{
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "cached_query_CPU_seconds_by_PID_start": {
    "3352381:252416924": 429.14804867841303
  },
  "inherited_paid_acquisition_and_eight_expert_cost_nonzero": true,
  "scope": "Conditional cached CPU replay; excludes original acquisition and setup/outer loop. Timings may be nested and must not be added or used as deployment latency. A separate actual matched deployment experiment is required for this method."
}

CIs are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means signed tIoU or vIoU loss >5pp, union once; rescue means >5pp gain. All signed outcomes remain in public TSVs. Media, captions, coordinates, raw features and weights stay private. Structural/factorial, budget, CG transfer, old corrupt panels, actual timing and qualitative synchronization remain separate tasks; full219690corruption and extra generic TTA are not complete.
