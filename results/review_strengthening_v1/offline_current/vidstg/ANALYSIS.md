# Current selected method: offline expert quality and distance (vidstg)

10303 complete queries, 732 parents. Independent full rederivation of all annotated anchor IoUs, physical distances, fixed-support frame IoUs and input SHA checks passed. No model, Source, expert, optimizer or new prediction.

Quality is exactly the historical mean annotated anchor-IoU definition, now joined to current already-audited main t/v outcomes. Unsupported anchors are missing, never zero. Quality is evaluated offline only; associations do not prove a causal expert-quality effect.

|Quality bin|Queries|Parents|Annotated anchors|Ours minus Source v pp [95% CI]|
|---|---:|---:|---:|---|
|IoU_0_25|3347|599|11584|{'mean': -3.6735756170042384, 'CI95': [-4.147394830500508, -3.226526650929161], 'queries': 3347, 'parents': 599, 'improved': 906, 'harmed': 1594, 'unchanged': 847, 'declined_over5pp': 767}|
|IoU_25_50|1065|393|4356|{'mean': 0.8452747889078503, 'CI95': [0.06398683432970728, 1.67495213939204], 'queries': 1065, 'parents': 393, 'improved': 553, 'harmed': 420, 'unchanged': 92, 'declined_over5pp': 209}|
|IoU_50_75|1148|424|4810|{'mean': 8.320384722438652, 'CI95': [7.292365696361876, 9.371845429912069], 'queries': 1148, 'parents': 424, 'improved': 841, 'harmed': 217, 'unchanged': 90, 'declined_over5pp': 87}|
|IoU_75_100|2947|625|11561|{'mean': 12.312022636072959, 'CI95': [11.469804537955415, 13.167648811113285], 'queries': 2947, 'parents': 625, 'improved': 2290, 'harmed': 360, 'unchanged': 297, 'declined_over5pp': 136}|
|unscorable_no_GT_anchor|1796|375|0|{'mean': 0.4413932397645951, 'CI95': [0.31630754179851994, 0.5711496047143494], 'queries': 1796, 'parents': 375, 'improved': 425, 'harmed': 230, 'unchanged': 1141, 'declined_over5pp': 23}|

Distance analysis uses identical actually annotated native sampled unobserved-frame support for every arm, independent of temporal interval. Physical nearest-observation distance/fps; no interpolated GT or extra frames. Query-equal means and parent-cluster10000 bootstrap95% CIs are primary; frame-weighted means are separate descriptive summaries. Pointwise intervals, not familywise. Interpolation keeps its original path and receives no DP.

|Seconds|Frames|Queries|Ours minus E fixed-support sIoU pp [95% CI]|
|---|---:|---:|---|
|[0, 0.5]|103389|8991|{'mean': 1.2654054951303602, 'CI95': [0.9667126523591784, 1.5748019379700404], 'queries': 8991, 'parents': 721, 'improved': 2586, 'harmed': 2515, 'unchanged': 3890, 'declined_over5pp': 739}|
|[0.5, 1]|66503|7528|{'mean': 0.3486305072585802, 'CI95': [0.05230623763964024, 0.6575020043441973], 'queries': 7528, 'parents': 685, 'improved': 1630, 'harmed': 2014, 'unchanged': 3884, 'declined_over5pp': 693}|
|[1, 2]|83822|5898|{'mean': 0.025431564204111447, 'CI95': [-0.24482586566169376, 0.300981936804169], 'queries': 5898, 'parents': 610, 'improved': 985, 'harmed': 1429, 'unchanged': 3484, 'declined_over5pp': 445}|
|[2, 4]|69507|3616|{'mean': 0.08211618975080105, 'CI95': [-0.1616823803056595, 0.34411683430399836], 'queries': 3616, 'parents': 464, 'improved': 503, 'harmed': 621, 'unchanged': 2492, 'declined_over5pp': 160}|
|[4, None]|41800|1757|{'mean': 0.04892041044706583, 'CI95': [-0.25239324466714363, 0.3660615036376723], 'queries': 1757, 'parents': 293, 'improved': 194, 'harmed': 246, 'unchanged': 1317, 'declined_over5pp': 58}|

All signed query/bin means and all positive/negative subgroup comparisons are retained. This descriptive sensitivity/quality analysis does not erase prior target-domain development selection or establish independent unseen validation. Inherited Source/expert acquisition cost is nonzero; this CPU/offline work is not deployment timing.
