# hc2: source-only reliability gate, target128 sealed decisions

Selected source-only rule: **combined_50**. Source36 queries/36parents,4parentfold OOF; target128queries/parents. No target configuration selection.

|Policy|tIoU|vIoU|R.3|R.5|Reject to Source|Severe t/v uniononce vs Source|
|---|---:|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|128|0|
|Full|54.968548|31.791298|50.781250|22.656250|0|30|
|RiskGate|49.734178|28.170796|47.656250|18.750000|75|12|
|RandomGate|50.651716|27.916286|45.312500|17.968750|75|14|

RiskGate-Full paired all: {'queries': 128, 'parents': 128, 'statistics': {'delta_v_pp': {'mean': -3.6205022890592815, 'CI95': [-5.845461131078488, -1.4041516116698816], 'queries': 128, 'parents': 128}, 'delta_t_pp': {'mean': -5.234369932127926, 'CI95': [-7.758079900454017, -2.81106874668627], 'queries': 128, 'parents': 128}, 'delta_severe_Source_relative_union_once_pp': {'mean': -14.0625, 'CI95': [-20.3125, -8.59375], 'queries': 128, 'parents': 128}}}
RiskGate-RandomGate paired all: {'queries': 128, 'parents': 128, 'statistics': {'delta_v_pp': {'mean': 0.25451042088682846, 'CI95': [-1.876047173596164, 2.4221126220105798], 'queries': 128, 'parents': 128}, 'delta_t_pp': {'mean': -0.9175378988906309, 'CI95': [-3.667156598424392, 1.8079506130838754], 'queries': 128, 'parents': 128}, 'delta_severe_Source_relative_union_once_pp': {'mean': -1.5625, 'CI95': [-7.03125, 3.90625], 'queries': 128, 'parents': 128}}}
RiskGate retained/avoided/foregone: {'rejected_to_Source': 75, 'harmed_v_vs_Source': 14, 'severe_t_vs_Source': 9, 'severe_v_vs_Source': 7, 'severe_tv_union_once': 12, 'rescue_tv_union_once': 29, 'negative_v_corrections_avoided': 20, 'severe_corrections_averted': 18, 'beneficial_severe_rescues_foregone': 47, 'positive_v_gain_retained_ratio': 0.41321133680613115, 'queryparentCI': {'queries': 128, 'parents': 128, 'statistics': {'vIoU_pp': {'mean': 28.17079605251029, 'CI95': [24.164716451234476, 32.2634434351505], 'queries': 128, 'parents': 128}, 'tIoU_pp': {'mean': 49.73417838348676, 'CI95': [44.66982532728865, 54.74889085779602], 'queries': 128, 'parents': 128}, 'severe_union_percent': {'mean': 9.375, 'CI95': [4.6875, 14.84375], 'queries': 128, 'parents': 128}}}}

All nine prespecified source-cutoff coverage points and all signed short/long/negative/severe/rescue outcomes remain in report.json and scalars. AUROC is offline target evaluation of frozen risk signals, never a threshold selection criterion. An alwaysFull source selection means no gate is deployed; it is retained as a null outcome, not replaced after seeing target scores.

This tests query-level rejection of the whole correction. It does not establish individual anchor filtering, time-only protection, action semantics, or calibrated expert correctness. Source confidence is temporal confidence, not correct boxes. Source data were previously exposed in the opposite transfer direction; target panel is historically exposed. Current softanchors and production unchanged.

New72 source-domain ReferDINO calls use the correct saved Source checkpoint native eight-position policy; no new Source/optimizer. Target study reuses Source/Full arrays and experts. Signal/replay clocks are conditional cached experiment costs, not end-to-end deployment latency or demonstrated savings.
