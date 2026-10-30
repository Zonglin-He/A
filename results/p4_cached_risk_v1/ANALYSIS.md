# P4: full-target cached negative-benefit and severe-tail statistics

Direct user authorization on2026-10-30: reuse existing P1/P2; compute P4 on CPU; defer full P2 and make no manuscript changes. Both complete target lists are retained: HC1902queries/227parents and Vid10303queries/732parents. This is report-only processing of previously sealed, officially scored and independently array/input/mean-audited outputs. No Source/expert/model/optimizer/prediction/RGB/GT scoring was performed. P1/P2 controls were not rerun.

Define d=100*(Current vIoU-native Source vIoU), in percentage points. Negative is d<−1e-10pp; |d|<=1e-10 is equal, following the existing reporting tolerance. Severe is exactly d<=−10pp (inclusive predeclared descriptive threshold). q05 is the empirical query-level fifth percentile, linear interpolation. No threshold or sample selection from outcomes.

|Domain|Queries / parents|Mean d pp [95% CI]|Negative n (%)|Severe <=−10pp n (%)|q05 pp|Positive / equal|
|---|---:|---:|---:|---:|---:|---:|
|hc2|1902 / 227|+6.644365 [5.902588, 7.408451]|516 (27.1293%)|175 (9.2008%)|-17.039259|1216 / 170|
|vidstg|10303 / 732|+3.419668 [3.009582, 3.851146]|2821 (27.3804%)|722 (7.0077%)|-14.292504|5015 / 2467|

All pointwise percentile confidence intervals use10000 parent-cluster bootstrap replicates, seed20261022. Each parent draw is applied jointly to both arms; means remain query-equal (sum of deltas divided by sum of query counts), not simple means of parent means. Negative/severe rate CIs use the same cluster draws and query denominators. q05 is a descriptive empirical quantile, without an added CI. Overall mean/CI exactly reproduce the prior main comparison (absolute tolerance1e-12). Group intervals are not familywise corrected, and these previously exposed full target lists are not new unknown validation. Empirical zero events can produce degenerate bootstrap intervals; these are not zero-population-risk guarantees.

## Existing Vid sentence-form labels

Labels are recovered by an exact original-video-ID plus full-caption join to the original catalog captions/questions fields, not punctuation heuristics. All10303 match uniquely:4610declarative and5693interrogative.

|Form|Queries|Mean d pp [95% CI]|Negative %|Severe %|q05 pp|
|---|---:|---:|---:|---:|---:|
|declarative|4610|+6.491476 [5.927217, 7.073940]|19.0672|3.4056|-6.934688|
|interrogative|5693|+0.932220 [0.489692, 1.378552]|34.1121|9.9245|-18.252203|

Interrogative queries have heavier observed negative tails and smaller average gains in this dataset. This is a descriptive association; event length, initial difficulty, expert quality and query form are not randomized, so the table does not identify query form as a unique cause.

## Existing offline expert-quality groups

Quality is the previously audited mean expert-box IoU at actually GT-supported anchor frames. Quarter bins are reused unchanged, the upper bin includes1. Missing/unannotated anchors are never fake zero. Unscorable queries remain a separate group and in every overall denominator.

|Domain / quality|Queries|Mean d pp [95% CI]|Negative %|Severe %|q05 pp|
|---|---:|---:|---:|---:|---:|
|hc2 / IoU_0_25|331|-4.056556 [-5.674766, -2.526710]|46.8278|21.7523|-37.412080|
|hc2 / IoU_25_50|272|+2.640087 [1.049757, 4.235535]|33.4559|11.7647|-21.009742|
|hc2 / IoU_50_75|453|+8.978397 [7.672254, 10.281964]|22.9581|6.1810|-12.005696|
|hc2 / IoU_75_100|824|+10.998722 [9.770401, 12.217542]|19.9029|5.2184|-10.443770|
|hc2 / unscorable_no_GT_anchor|22|+6.001750 [2.753071, 9.385666]|9.0909|0.0000|-0.433531|
|vidstg / IoU_0_25|3347|-3.673576 [-4.147395, -3.226527]|47.6247|15.0583|-24.782836|
|vidstg / IoU_25_50|1065|+0.845275 [0.063987, 1.674952]|39.4366|11.2676|-17.887046|
|vidstg / IoU_50_75|1148|+8.320385 [7.292366, 9.371845]|18.9024|3.8328|-8.091790|
|vidstg / IoU_75_100|2947|+12.312023 [11.469805, 13.167649]|12.2158|1.7645|-4.360769|
|vidstg / unscorable_no_GT_anchor|1796|+0.441393 [0.316308, 0.571150]|12.8062|0.1114|-2.242694|

Both lowest-quality groups have negative mean gains. Risk is associated with observed expert quality, but even the highest-quality groups retain negative and severe outcomes. Expert quality therefore does not explain every failure, and this offline diagnostic does not authorize a GT-based online gate or establish a single causal mechanism. Positive overall means are not per-query reliability guarantees.

## Reuse and figure connections

P1 already covers both full domains with exactly common event intervals and per-query tIoU. Ours-minus-interpolation vHC+2.134753CI[1.814042,2.473029], Vid−.106955CI[−.250579,.036605]. P2 Local/Global already covers192retainedqueries/parents per domain: vHC+.444126CI[−.135099,1.069766], Vid+.000760CI[−.316023,.331518]; these are not full-domain results or evidence of equivalence. Existing native Source and joint-prior MAP readouts are distinct; whole-domain pure-prior and smoothed-prior controls and their paired intervals are reused in reused_results_summary.json. Full P2 remains explicitly deferred.

Four existing failure figures are joined to the new scalar table in existing_failure_case_connections.json. Their original baselines are preserved: Source, same-candidate independent or Source-box temporal appearance. In particular, a large loss against independent decoding or SourceROI is not automatically a <=−10pp loss against native Source. These post-hoc selected figures illustrate mechanisms, not failure frequencies; no new pictures, GT overlays, cases or scores were produced.

## Costs and scope

One sequential CPU consumer, affinity14-15, nice19, idleIO and one BLAS thread; CUDA hidden and the scripts reject GPU-device, model/checkpoint/raw-prediction, video and GT-label file reads. Torch was never imported in the auditor. No GPU process or unrelated service was signaled. Scalar input SHA/provenance, all group counts/linear quantiles and query-weighted parent bootstrap were independently checked in a separate process. Prior Source/expert acquisition costs are inherited and nonzero; these elapsed statistics/checking clocks are not inference or deployment timings. No main-method/default/production/frozen-source/parameter/manuscript changes; P5 and writing tasks excluded.
