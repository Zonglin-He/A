# Same DINO geometry pool: original selection, R1 selection, and fixed propagation

Same32 exposed parents/128 T2 observation anchors as the prior R1-only geometry screen. D0 is exact stage1 DINO original selection/admission plus unchanged paper OPD. D1 changes only the selector, using the same cached native DINO pool (context3 boxes, fallback10 boxes after the original SpatialExpert coordinate conversion) and original scores for stable ID order, native R1 full context4 plus one anchor-only contact sheet, one fixed Prompt and96-token cap. D2 uses exact D1 sparse references and Source trajectory, interpolating Source-relative center/log-size residuals in physical time within the same T2 interval; no Student parameter update, tracker or extra backbone. D1 retains the original paper episodic Gaussian/response/fullstep rollback/reject-stop; no score mix, optimizer redesign or Prompt/threshold search.

The context interface supplies3 processed boxes per anchor; the original fallback supplies10 raw top boxes, converted with the exact original SpatialExpert float32 xyxy clamp/roundtrip before both selectors. Its admitted Top1 equals sealed D0 bitwise. Three CPU-only schema/conversion preparation failures before any new model/GT are preserved; no pool was chosen by GT. D0 original rejects and D1 selections from the same existing pool can change adaptation opportunity; these are separately counted and not presented as pure selection quality. No-pool positions remain missing. ID1 is original Top1 where admitted; fixed order may bias R1. Localization IoU success changes are not independent identity/role labels. Abstention subgroups are descriptive: mixed-window updates do not identify a causal abstention effect. All32 parents, failures, unknowns and negative tails remain in the paired report. D0/Frozen/T2/current WF are exact old predictions; their cost is not zero. New D1/D2 predictions and roots are sealed before any stage2 GT. No production promotion, confirmation or official fullset.

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|28.116723|50.497117|52.754115|31.250000|25.000000|
|hc2|T2|27.487135|50.497117|53.666875|37.500000|18.750000|
|hc2|Current_WF|28.641217|52.179161|53.666875|37.500000|25.000000|
|hc2|D0|28.440075|51.967586|53.666875|37.500000|25.000000|
|hc2|D1|28.600908|52.314891|53.666875|37.500000|25.000000|
|hc2|D2|32.632171|57.825914|53.666875|43.750000|31.250000|
|vidstg|Frozen|12.185607|32.870500|35.923576|12.500000|0.000000|
|vidstg|T2|12.239461|32.870500|35.680949|12.500000|0.000000|
|vidstg|Current_WF|12.072499|32.573620|35.680949|12.500000|0.000000|
|vidstg|D0|15.368798|33.573892|35.680949|25.000000|6.250000|
|vidstg|D1|16.158575|34.885355|35.680949|25.000000|0.000000|
|vidstg|D2|15.046009|38.012582|35.680949|25.000000|6.250000|

| Domain | Contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |
|---|---|---:|---|---|
|hc2|D0_minus_Current_WF|-0.201142|[-0.9638743111996029, 0.5615901338279239]|7/7/2/0|
|hc2|D1_minus_Current_WF|-0.040309|[-0.8290067040006852, 0.7483884548702167]|10/5/1/0|
|hc2|D1_minus_D0|0.160833|[-0.009778416514285182, 0.33144434475549545]|6/1/9/0|
|hc2|D2_minus_D0|4.192096|[-7.751140858732023, 16.135332468885796]|11/4/1/0|
|vidstg|D0_minus_Current_WF|3.296299|[-0.3587689164264365, 6.9513671501145415]|8/4/4/0|
|vidstg|D1_minus_Current_WF|4.086077|[-0.05885980031878191, 8.231012905716728]|9/3/4/0|
|vidstg|D1_minus_D0|0.789777|[-2.0933481520166337, 3.672903023726475]|3/5/8/0|
|vidstg|D2_minus_D0|-0.322788|[-4.0957935032416355, 3.45021678044449]|5/5/6/0|

All metrics including nonanchor sIoU and every source-paired signed delta are in RESULTS.json. Reference selection/correctness proxies, missing-pool/admission differences and descriptive abstention groups are in SELECTION_READBACK.json. No semantic labels are inferred from these proxies; existing semantic512 panel remains44model-assisted/468pending. This finite complement study does not automatically authorize new modules.
