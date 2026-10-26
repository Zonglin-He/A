# Current selected method: offline expert quality and distance (hc2)

1902 complete queries, 227 parents. Independent full rederivation of all annotated anchor IoUs, physical distances, fixed-support frame IoUs and input SHA checks passed. No model, Source, expert, optimizer or new prediction.

Quality is exactly the historical mean annotated anchor-IoU definition, now joined to current already-audited main t/v outcomes. Unsupported anchors are missing, never zero. Quality is evaluated offline only; associations do not prove a causal expert-quality effect.

|Quality bin|Queries|Parents|Annotated anchors|Ours minus Source v pp [95% CI]|
|---|---:|---:|---:|---|
|IoU_0_25|331|147|988|{'mean': -4.056556212551351, 'CI95': [-5.674766102632386, -2.526709878349522], 'queries': 331, 'parents': 147, 'improved': 144, 'harmed': 155, 'unchanged': 32, 'declined_over5pp': 99}|
|IoU_25_50|272|134|909|{'mean': 2.640087237026271, 'CI95': [1.0497571013643243, 4.235535196027403], 'queries': 272, 'parents': 134, 'improved': 159, 'harmed': 91, 'unchanged': 22, 'declined_over5pp': 59}|
|IoU_50_75|453|174|1566|{'mean': 8.978396639338474, 'CI95': [7.67225424302721, 10.281963507502986], 'queries': 453, 'parents': 174, 'improved': 327, 'harmed': 104, 'unchanged': 22, 'declined_over5pp': 54}|
|IoU_75_100|824|197|2556|{'mean': 10.998721721128321, 'CI95': [9.77040059544442, 12.217542462788275], 'queries': 824, 'parents': 197, 'improved': 573, 'harmed': 164, 'unchanged': 87, 'declined_over5pp': 93}|
|unscorable_no_GT_anchor|22|18|0|{'mean': 6.001749583805139, 'CI95': [2.7530709975300267, 9.385665564168523], 'queries': 22, 'parents': 18, 'improved': 13, 'harmed': 2, 'unchanged': 7, 'declined_over5pp': 0}|

Distance analysis uses identical actually annotated native sampled unobserved-frame support for every arm, independent of temporal interval. Physical nearest-observation distance/fps; no interpolated GT or extra frames. Query-equal means and parent-cluster10000 bootstrap95% CIs are primary; frame-weighted means are separate descriptive summaries. Pointwise intervals, not familywise. Interpolation keeps its original path and receives no DP.

|Seconds|Frames|Queries|Ours minus E fixed-support sIoU pp [95% CI]|
|---|---:|---:|---|
|[0, 0.5]|11050|1894|{'mean': 1.9451040274576672, 'CI95': [1.3356000546877742, 2.5611382998401235], 'queries': 1894, 'parents': 226, 'improved': 726, 'harmed': 585, 'unchanged': 583, 'declined_over5pp': 269}|
|[0.5, 1]|15154|1894|{'mean': 0.819549622634825, 'CI95': [0.28376896758844955, 1.3906321424848611], 'queries': 1894, 'parents': 226, 'improved': 652, 'harmed': 680, 'unchanged': 562, 'declined_over5pp': 309}|
|[1, 2]|10662|1715|{'mean': 0.09172196399800228, 'CI95': [-0.5611645604253597, 0.7535434740275764], 'queries': 1715, 'parents': 223, 'improved': 431, 'harmed': 521, 'unchanged': 763, 'declined_over5pp': 282}|
|[2, 4]|1829|561|{'mean': -0.26635213312422584, 'CI95': [-1.4482136449817737, 0.8726267367924874], 'queries': 561, 'parents': 180, 'improved': 81, 'harmed': 99, 'unchanged': 381, 'declined_over5pp': 65}|
|[4, None]|0|0|None|

All signed query/bin means and all positive/negative subgroup comparisons are retained. This descriptive sensitivity/quality analysis does not erase prior target-domain development selection or establish independent unseen validation. Inherited Source/expert acquisition cost is nonzero; this CPU/offline work is not deployment timing.
