# Native2D risk calibration v1

All original Source/Global6/eight observations/BoxFixed and MBR fixed. Six shared configurations: kappa16/64 x beta.25/.5/1. No new expert/observation, optimizer, training, extra visual encoder, quality/GT rejection or per-domain parameter selection. Risk is a query with time OR full-v loss >5pp versus Global6, counted once. This is risk calibration of existing evidence, not proof of better event recognition.

| Direction | Candidate | tIoU (%) | Full vIoU (%) | Severe queries / N | Worst10% mean loss (pp) |
|---|---|---:|---:|---:|---:|
|hc2|K16_B25|47.0776|27.0535|3 / 64|4.4768|
|hc2|K16_B50|47.6664|27.1832|4 / 64|7.1734|
|hc2|K16_B100|48.0467|27.1294|12 / 64|12.9508|
|hc2|K64_B25|46.5993|26.8747|1 / 64|3.0793|
|hc2|K64_B50|46.8328|26.9448|1 / 64|3.6172|
|hc2|K64_B100|47.3632|27.1351|4 / 64|6.1877|
|vidstg|K16_B25|42.6953|22.2546|2 / 64|6.4820|
|vidstg|K16_B50|42.9549|22.3664|4 / 64|8.4928|
|vidstg|K16_B100|43.2567|22.7138|7 / 64|11.2257|
|vidstg|K64_B25|42.6301|22.2480|1 / 64|4.2885|
|vidstg|K64_B50|42.7451|22.3164|2 / 64|6.1852|
|vidstg|K64_B100|42.8975|22.3110|4 / 64|8.0940|

Predeclared development winner: kappa64, beta0.25. Risk-first lexicographic objective, with each-direction mean protection. All candidates generated and sealed before independent CPU official-label scoring.

| Confirmation direction | Method | tIoU (%) | Full vIoU (%) | Severe queries /32 vsG6 |
|---|---|---:|---:|---:|
|hc2|Source|53.7894|28.8182|13|
|hc2|NativeG6|54.0152|31.6124|0|
|hc2|CurrentRegion|57.6241|34.6472|3|
|hc2|K16|57.2559|34.2639|5|
|hc2|Selected|54.5272|32.0878|0|
|vidstg|Source|37.9155|14.3481|15|
|vidstg|NativeG6|40.2580|18.2486|0|
|vidstg|CurrentRegion|41.0845|18.5774|3|
|vidstg|K16|40.9400|18.3701|3|
|vidstg|Selected|40.2317|18.0925|1|

Fixed confirmation mean/risk guards: {'hc2': False, 'vidstg': False}. The original scientific decision/fallback is retained verbatim in DECISION.json. NO retrospective pass. AllpairedCI/longshort/boundaries/support/individualsigns/severe cases versus Source/Global6/Native16/current-fused are in reports and TSVs.

Integrity: 1472 official/dense metric checks, max error 1.221e-15; Source/logits/q0/current fused outputs exact on all64 new queries. No GT in inference; complete64 seal before independent CPU labels. Original baseline128/192 cached representations reused without models. Research Source64 wall 274.47s; no teacher/observation/training. Research timing is NOT deployment latency.

v1 selected64/.25 reduces severe5->0 HC and3->1 Vid but loses HCfullv2.1760pp andtime2.7287pp versus Native16. It fails the predeclared mean protection. v2 explicitly reclassifies this now-exposed confirmation as development, keeps the same six-config grid and fixes a final new parent-disjoint confirmation BEFORE scoring. This is an additional disclosed development stage, not reuse of v1 confirmation as unseen validation.
