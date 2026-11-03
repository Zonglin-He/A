# Completed CPU stages of the six-experiment scope: vidstg

Full10303 queries; fixed current method/eight observations/parameters. Pure expert interpolation has ONLY original expert boxes/time, nearest endpoint holding, no E/Source fallback, TR or DP. The old hybrid remains separately named. Whole-domain seals preceded new official scoring; separate cold-array/input/prediction/receipt-SHA/means audits passed. Original128 pure matched-time results reproduced/reused exactly; no new model/Source/expert/GPU/optimizer/media. This completes full pure interpolation, three temporal-quality bins and expert-quality quarters. Matching propagation, controlled actual timing, updated distance curves and density experiments remain separately gated/pending.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved actual sIoU %|
|---|---:|---:|---:|---:|---:|
|Source|31.728454|11.138484|12.831214|3.348539|33.025378|
|SourceMatched|34.918192|12.227364|14.966515|4.241483|33.025378|
|PureNative|31.728454|13.549125|17.121227|7.269727|35.542132|
|PureMatched|34.918192|14.704394|19.722411|8.638261|35.542132|
|Current|34.918192|14.558151|19.411822|8.463554|35.785762|
|HybridMatched|34.918192|14.665107|19.596234|8.541202|35.512989|

Source/PureNative use original native Source intervals; SourceMatched/PureMatched/Current/HybridMatched use exactly the same Current interval. SourceMatched is not the native main-table Source. Pure interpolation itself predicts no event interval; the native-interval standalone comparator explicitly inherits the original Source temporal output. No hidden posterior update is granted.

|Paired comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Delta unobserved sIoU pp [95% CI]|Severe t/v/union|Rescue union|
|---|---|---|---|---|---:|
|Current-PureMatched|+0.000000 [+0.000000, +0.000000]|-0.146243 [-0.294717, -0.000487]|+0.243630 [-0.305862, +0.790250]|0/714/714|676|
|Current-PureNative|+3.189738 [+2.753594, +3.658697]|+1.009026 [+0.760203, +1.267281]|+0.243630 [-0.305862, +0.790250]|1562/1054/1979|3489|
|Current-Source|+3.189738 [+2.753594, +3.658697]|+3.419668 [+3.009582, +3.851146]|+2.760384 [+2.092477, +3.439830]|1562/1222/2294|4460|
|Current-SourceMatched|+0.000000 [+0.000000, +0.000000]|+2.330787 [+1.985568, +2.683873]|+2.760384 [+2.092477, +3.439830]|0/1136/1136|2593|
|PureMatched-HybridMatched|+0.000000 [+0.000000, +0.000000]|+0.039287 [+0.012578, +0.068451]|+0.029142 [-0.068634, +0.133767]|0/37/37|72|

## Native Source temporal-quality groups

All groups use ORIGINAL native Source tIoU, not updated scores. Before/after are t-only: whole branch Source→Current, and matched joint-MAP q0→visual update. Fixed tolerance1e-10pp; decline quantiles use positive magnitudes among harmed queries only, linear median/p90. No vIoU mixes spatial paths in this temporal attribution.

|Comparison/group|Queries|Before tIoU %|After tIoU %|Delta t pp [95% CI]|Improve / unchanged / harm %|Decline median / p90 pp|
|---|---:|---:|---:|---|---|---|
|whole_temporal_branch/all|10303|31.728454|34.918192|+3.189738 [+2.753594, +3.658697]|45.938076/25.351839/28.710084|5.458350261499867/17.115504442623166|
|whole_temporal_branch/Source_t_0_03|5558|9.479507|13.612081|+4.132574 [+3.614471, +4.671230]|40.050378/38.736956/21.212666|2.982370366109689/10.647428877900614|
|whole_temporal_branch/Source_t_03_05|1863|39.261188|43.610705|+4.349516 [+3.327250, +5.432286]|59.098229/7.675792/33.225980|6.124681006197596/18.062360078189467|
|whole_temporal_branch/Source_t_05_1|2882|69.766680|70.388426|+0.621747 [-0.120404, +1.355485]|48.785566/10.964608/40.249827|8.175876411170531/21.850372630003907|
|visual_update_same_jointMAP/all|10303|33.450806|34.918192|+1.467386 [+1.057285, +1.888033]|42.007182/24.691837/33.300980|6.000000000000005/18.301741803278688|
|visual_update_same_jointMAP/Source_t_0_03|5558|11.374157|13.612081|+2.237924 [+1.844373, +2.655010]|37.063692/38.467074/24.469234|3.9318941004334262/13.011101999801209|
|visual_update_same_jointMAP/Source_t_03_05|1863|40.765845|43.610705|+2.844860 [+1.908477, +3.864682]|55.555556/9.125067/35.319377|5.575291721148573/17.89475264204642|
|visual_update_same_jointMAP/Source_t_05_1|2882|71.297474|70.388426|-0.909048 [-1.759557, -0.060808]|42.782790/8.188758/49.028452|8.921933085501854/21.155661478379695|

## Expert-quality quarters

Mean expert IoU uses only actually GT-supported observations. Missing quality remains separate, never0. Own-interval and common-Current-interval rows are different estimands; GT is offline only.

|Support/group|Queries|Source v %|Pure v %|Current v %|Current-Source pp [95% CI]|Current-Pure pp [95% CI]|Improve/harm vs pure %|
|---|---:|---:|---:|---:|---|---|---|
|own_intervals/all|10303|11.138484|13.549125|14.558151|+3.419668 [+3.009582, +3.851146]|+1.009026 [+0.760203, +1.267281]|40.958944/31.990682|
|own_intervals/q0_25|3347|6.480859|2.410670|2.807283|-3.673576 [-4.147395, -3.226527]|+0.396613 [+0.243380, +0.538904]|35.882880/30.086645|
|own_intervals/q25_50|1065|13.283024|13.101551|14.128299|+0.845275 [+0.063987, +1.674952]|+1.026748 [+0.520448, +1.534315]|51.643192/39.812207|
|own_intervals/q50_75|1148|16.505930|23.091272|24.826314|+8.320385 [+7.292366, +9.371845]|+1.735042 [+0.975132, +2.529632]|51.306620/40.853659|
|own_intervals/q75_100|2947|19.875044|30.485684|32.187067|+12.312023 [+11.469805, +13.167649]|+1.701383 [+1.087920, +2.319423]|49.304377/40.753308|
|own_intervals/missing|1796|0.780279|0.682004|1.221672|+0.441393 [+0.316308, +0.571150]|+0.539668 [+0.425029, +0.658371]|23.775056/10.857461|
|common_Current_interval/all|10303|12.227364|14.704394|14.558151|+2.330787 [+1.985568, +2.683873]|-0.146243 [-0.294717, -0.000487]|33.203921/39.250704|
|common_Current_interval/q0_25|3347|7.040447|2.666666|2.807283|-4.233164 [-4.728294, -3.769157]|+0.140617 [-0.013578, +0.274646]|31.102480/34.598148|
|common_Current_interval/q25_50|1065|14.361420|14.039352|14.128299|-0.233121 [-1.041958, +0.583518]|+0.088947 [-0.339138, +0.513005]|44.319249/46.384977|
|common_Current_interval/q50_75|1148|17.949482|24.891124|24.826314|+6.876832 [+5.989555, +7.790614]|-0.064810 [-0.578490, +0.451808]|43.989547/47.648084|
|common_Current_interval/q75_100|2947|21.759425|33.005254|32.187067|+10.427642 [+9.739673, +11.140645]|-0.818188 [-1.133296, -0.505111]|35.561588/53.749576|
|common_Current_interval/missing|1796|1.329741|0.991445|1.221672|-0.108069 [-0.189329, -0.027312]|+0.230227 [+0.152635, +0.314482]|19.766147/14.532294|

All signed positive/negative per-query metrics, severe>5pp t/v union-once/rescues and short<=5s/long groups are retained in JSON/TSVs. Queryequal10000paired-parent percentile95% CIs are pointwise, not familywise, equivalence, causal correctness or unknown-validation claims. No parameter/sample choice follows these exposed target scores.

Cached CPU wall includes IO/validation/replay and is not deployment latency; original Source/expert acquisition costs remain nonzero. Stage/PID/start/actual waits are preserved privately and scalar stage costs publicly. The other three numerical studies are NOT complete merely because these CPU stages finished. Paper/current profile and old frozen plans unchanged; private media/captions/GTcoords/features/masks/weights excluded.
