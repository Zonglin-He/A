# Completed CPU stages of the six-experiment scope: hc2

Full1902 queries; fixed current method/eight observations/parameters. Pure expert interpolation has ONLY original expert boxes/time, nearest endpoint holding, no E/Source fallback, TR or DP. The old hybrid remains separately named. Whole-domain seals preceded new official scoring; separate cold-array/input/prediction/receipt-SHA/means audits passed. Original128 pure matched-time results reproduced/reused exactly; no new model/Source/expert/GPU/optimizer/media. This completes full pure interpolation, three temporal-quality bins and expert-quality quarters. Matching propagation, controlled actual timing, updated distance curves and density experiments remain separately gated/pending.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved actual sIoU %|
|---|---:|---:|---:|---:|---:|
|Source|40.141475|20.704678|29.022082|8.937960|51.049723|
|SourceMatched|49.764548|25.604197|38.801262|14.773922|51.049723|
|PureNative|40.141475|21.905032|31.440589|10.778128|47.262804|
|PureMatched|49.764548|25.177364|38.012618|13.932702|47.262804|
|Current|49.764548|27.349044|41.219769|16.876972|52.942787|
|HybridMatched|49.764548|25.214291|37.749737|14.037855|47.336877|

Source/PureNative use original native Source intervals; SourceMatched/PureMatched/Current/HybridMatched use exactly the same Current interval. SourceMatched is not the native main-table Source. Pure interpolation itself predicts no event interval; the native-interval standalone comparator explicitly inherits the original Source temporal output. No hidden posterior update is granted.

|Paired comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Delta unobserved sIoU pp [95% CI]|Severe t/v/union|Rescue union|
|---|---|---|---|---|---:|
|Current-PureMatched|+0.000000 [+0.000000, +0.000000]|+2.171679 [+1.839911, +2.519693]|+5.679982 [+4.841201, +6.569601]|0/129/129|424|
|Current-PureNative|+9.623073 [+8.594747, +10.683117]|+5.444011 [+4.823284, +6.075176]|+5.679982 [+4.841201, +6.569601]|346/261/398|1063|
|Current-Source|+9.623073 [+8.594747, +10.683117]|+6.644365 [+5.902588, +7.408451]|+1.893064 [+1.164536, +2.632155]|346/305/458|1129|
|Current-SourceMatched|+0.000000 [+0.000000, +0.000000]|+1.744846 [+1.233989, +2.268511]|+1.893064 [+1.164536, +2.632155]|0/224/224|456|
|PureMatched-HybridMatched|+0.000000 [+0.000000, +0.000000]|-0.036926 [-0.108844, +0.030238]|-0.074073 [-0.199102, +0.043915]|0/18/18|15|

## Native Source temporal-quality groups

All groups use ORIGINAL native Source tIoU, not updated scores. Before/after are t-only: whole branch Source→Current, and matched joint-MAP q0→visual update. Fixed tolerance1e-10pp; decline quantiles use positive magnitudes among harmed queries only, linear median/p90. No vIoU mixes spatial paths in this temporal attribution.

|Comparison/group|Queries|Before tIoU %|After tIoU %|Delta t pp [95% CI]|Improve / unchanged / harm %|Decline median / p90 pp|
|---|---:|---:|---:|---|---|---|
|whole_temporal_branch/all|1902|40.141475|49.764548|+9.623073 [+8.594747, +10.683117]|62.828601/10.988433/26.182965|9.777650902401113/27.988809962180227|
|whole_temporal_branch/Source_t_0_03|699|11.474298|26.054570|+14.580273 [+13.188000, +16.066411]|64.949928/24.749642/10.300429|3.4337813296410014/11.048223910247268|
|whole_temporal_branch/Source_t_03_05|485|39.799454|55.749533|+15.950079 [+14.272121, +17.753677]|75.876289/3.711340/20.412371|6.0637133074957825/15.582848027477327|
|whole_temporal_branch/Source_t_05_1|718|68.281082|68.804320|+0.523238 [-0.937757, +1.999998]|51.949861/2.506964/45.543175|12.824662813102112/32.62189244841976|
|visual_update_same_jointMAP/all|1902|46.693293|49.764548|+3.071255 [+2.341218, +3.800142]|50.525762/14.773922/34.700315|8.61734751633295/24.499634953637646|
|visual_update_same_jointMAP/Source_t_0_03|699|20.839377|26.054570|+5.215193 [+4.224294, +6.264625]|49.642346/26.895565/23.462089|5.422647527910687/22.30996055194325|
|visual_update_same_jointMAP/Source_t_03_05|485|49.182009|55.749533|+6.567524 [+5.257730, +7.960386]|63.505155/7.010309/29.484536|9.034749034749034/18.634089419829625|
|visual_update_same_jointMAP/Source_t_05_1|718|70.181957|68.804320|-1.377636 [-2.413482, -0.357925]|42.618384/8.217270/49.164345|9.814563545906823/26.744985541218345|

## Expert-quality quarters

Mean expert IoU uses only actually GT-supported observations. Missing quality remains separate, never0. Own-interval and common-Current-interval rows are different estimands; GT is offline only.

|Support/group|Queries|Source v %|Pure v %|Current v %|Current-Source pp [95% CI]|Current-Pure pp [95% CI]|Improve/harm vs pure %|
|---|---:|---:|---:|---:|---|---|---|
|own_intervals/all|1902|20.704678|21.905032|27.349044|+6.644365 [+5.902588, +7.408451]|+5.444011 [+4.823284, +6.075176]|61.777077/28.706625|
|own_intervals/q0_25|331|11.274322|4.168188|7.217765|-4.056556 [-5.674766, -2.526710]|+3.049578 [+2.363351, +3.810689]|60.422961/26.586103|
|own_intervals/q25_50|272|16.721994|14.412995|19.362081|+2.640087 [+1.049757, +4.235535]|+4.949086 [+3.805973, +6.134779]|66.911765/25.000000|
|own_intervals/q50_75|453|23.380670|26.173519|32.359066|+8.978397 [+7.672254, +10.281964]|+6.185548 [+4.927715, +7.410233]|64.900662/30.242826|
|own_intervals/q75_100|824|24.842148|29.707331|35.840870|+10.998722 [+9.770401, +12.217542]|+6.133539 [+5.048056, +7.264133]|58.859223/30.582524|
|own_intervals/missing|22|1.761027|1.269176|7.762777|+6.001750 [+2.753071, +9.385666]|+6.493600 [+3.300428, +9.927934]|63.636364/4.545455|
|common_Current_interval/all|1902|25.604197|25.177364|27.349044|+1.744846 [+1.233989, +2.268511]|+2.171679 [+1.839911, +2.519693]|56.309148/33.859096|
|common_Current_interval/q0_25|331|14.611719|5.513655|7.217765|-7.393953 [-9.098606, -5.845138]|+1.704111 [+1.165817, +2.297694]|52.265861/34.743202|
|common_Current_interval/q25_50|272|20.292856|16.077722|19.362081|-0.930775 [-2.087076, +0.249017]|+3.284359 [+2.581401, +4.048199]|68.382353/23.161765|
|common_Current_interval/q50_75|453|28.801706|29.802157|32.359066|+3.557360 [+2.766985, +4.401462]|+2.556909 [+1.748329, +3.386049]|60.706402/33.995585|
|common_Current_interval/q75_100|824|30.471881|34.049766|35.840870|+5.368988 [+4.663810, +6.119289]|+1.791103 [+1.334846, +2.270142]|51.820388/37.257282|
|common_Current_interval/missing|22|8.502007|5.991030|7.762777|-0.739231 [-2.408853, +0.551320]|+1.771747 [+0.111024, +3.643650]|45.454545/22.727273|

All signed positive/negative per-query metrics, severe>5pp t/v union-once/rescues and short<=5s/long groups are retained in JSON/TSVs. Queryequal10000paired-parent percentile95% CIs are pointwise, not familywise, equivalence, causal correctness or unknown-validation claims. No parameter/sample choice follows these exposed target scores.

Cached CPU wall includes IO/validation/replay and is not deployment latency; original Source/expert acquisition costs remain nonzero. Stage/PID/start/actual waits are preserved privately and scalar stage costs publicly. The other three numerical studies are NOT complete merely because these CPU stages finished. Paper/current profile and old frozen plans unchanged; private media/captions/GTcoords/features/masks/weights excluded.
