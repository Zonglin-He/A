# vidstg: source-only reliability gate, target128 sealed decisions

Selected source-only rule: **combined_25**. Source36 queries/34parents,4parentfold OOF; target128queries/parents. No target configuration selection.

|Policy|tIoU|vIoU|R.3|R.5|Reject to Source|Severe t/v uniononce vs Source|
|---|---:|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|128|0|
|Full|35.202435|15.339259|19.531250|10.156250|0|34|
|RiskGate|35.733691|15.575955|20.312500|10.156250|20|26|
|RandomGate|33.622151|14.443802|17.968750|8.593750|20|30|

RiskGate-Full paired all: {'queries': 128, 'parents': 128, 'statistics': {'delta_v_pp': {'mean': 0.2366954773341438, 'CI95': [-0.11948155477884123, 0.6547853091825019], 'queries': 128, 'parents': 128}, 'delta_t_pp': {'mean': 0.5312566701096612, 'CI95': [-0.34196226822645875, 1.3948371285608698], 'queries': 128, 'parents': 128}, 'delta_severe_Source_relative_union_once_pp': {'mean': -6.25, 'CI95': [-10.9375, -2.34375], 'queries': 128, 'parents': 128}}}
RiskGate-RandomGate paired all: {'queries': 128, 'parents': 128, 'statistics': {'delta_v_pp': {'mean': 1.1321526943396987, 'CI95': [-0.09883540088407873, 2.5251211439808303], 'queries': 128, 'parents': 128}, 'delta_t_pp': {'mean': 2.11154063010906, 'CI95': [0.7400487798782972, 3.7559587737065496], 'queries': 128, 'parents': 128}, 'delta_severe_Source_relative_union_once_pp': {'mean': -3.125, 'CI95': [-7.8125, 1.5625], 'queries': 128, 'parents': 128}}}
RiskGate retained/avoided/foregone: {'rejected_to_Source': 20, 'harmed_v_vs_Source': 30, 'severe_t_vs_Source': 13, 'severe_v_vs_Source': 18, 'severe_tv_union_once': 26, 'rescue_tv_union_once': 52, 'negative_v_corrections_avoided': 9, 'severe_corrections_averted': 8, 'beneficial_severe_rescues_foregone': 4, 'positive_v_gain_retained_ratio': 0.9700536812998318, 'queryparentCI': {'queries': 128, 'parents': 128, 'statistics': {'vIoU_pp': {'mean': 15.575954518016013, 'CI95': [12.142977632039244, 19.318632474064014], 'queries': 128, 'parents': 128}, 'tIoU_pp': {'mean': 35.73369140195021, 'CI95': [30.63340165370169, 41.047988121942716], 'queries': 128, 'parents': 128}, 'severe_union_percent': {'mean': 20.3125, 'CI95': [13.28125, 27.34375], 'queries': 128, 'parents': 128}}}}

All nine prespecified source-cutoff coverage points and all signed short/long/negative/severe/rescue outcomes remain in report.json and scalars. AUROC is offline target evaluation of frozen risk signals, never a threshold selection criterion. An alwaysFull source selection means no gate is deployed; it is retained as a null outcome, not replaced after seeing target scores.

This tests query-level rejection of the whole correction. It does not establish individual anchor filtering, time-only protection, action semantics, or calibrated expert correctness. Source confidence is temporal confidence, not correct boxes. Source data were previously exposed in the opposite transfer direction; target panel is historically exposed. Current softanchors and production unchanged.

New72 source-domain ReferDINO calls use the correct saved Source checkpoint native eight-position policy; no new Source/optimizer. Target study reuses Source/Full arrays and experts. Signal/replay clocks are conditional cached experiment costs, not end-to-end deployment latency or demonstrated savings.
