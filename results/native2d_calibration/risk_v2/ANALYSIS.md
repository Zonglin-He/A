# Native2D risk calibration v2

All original Source/Global6/eight observations/BoxFixed and MBR fixed. Six shared configurations: kappa16/64 x beta.25/.5/1. No new expert/observation, optimizer, training, extra visual encoder, quality/GT rejection or per-domain parameter selection. Risk is a query with time OR full-v loss >5pp versus Global6, counted once. This is risk calibration of existing evidence, not proof of better event recognition.

| Direction | Candidate | tIoU (%) | Full vIoU (%) | Severe queries / N | Worst10% mean loss (pp) |
|---|---|---:|---:|---:|---:|
|hc2|K16_B25|49.8111|28.9244|5 / 96|4.7693|
|hc2|K16_B50|50.4408|29.2268|8 / 96|7.7690|
|hc2|K16_B100|51.1165|29.5075|17 / 96|12.6315|
|hc2|K64_B25|49.2419|28.6124|1 / 96|2.9728|
|hc2|K64_B50|49.6080|28.8676|2 / 96|4.0010|
|hc2|K64_B100|50.1012|29.1093|8 / 96|6.9754|
|vidstg|K16_B25|41.9924|20.8553|4 / 96|6.1000|
|vidstg|K16_B50|42.1990|20.9833|6 / 96|7.7702|
|vidstg|K16_B100|42.4845|21.2659|10 / 96|10.3211|
|vidstg|K64_B25|41.8306|20.8629|2 / 96|4.0997|
|vidstg|K64_B50|42.0357|20.9001|4 / 96|5.9607|
|vidstg|K64_B100|42.1103|20.9060|6 / 96|7.5541|

Predeclared development winner: kappa64, beta1.0. Risk-first lexicographic objective, with each-direction mean protection. All candidates generated and sealed before independent CPU official-label scoring.

| Confirmation direction | Method | tIoU (%) | Full vIoU (%) | Severe queries /32 vsG6 |
|---|---|---:|---:|---:|
|hc2|Source|43.8670|23.3458|16|
|hc2|NativeG6|46.5772|26.5147|0|
|hc2|CurrentRegion|50.3783|28.3260|4|
|hc2|K16|50.4216|28.2444|3|
|hc2|Selected|48.4382|27.3755|1|
|vidstg|Source|37.1626|11.2146|14|
|vidstg|NativeG6|38.3768|17.2057|0|
|vidstg|CurrentRegion|38.8401|17.1108|4|
|vidstg|K16|38.4008|16.9656|6|
|vidstg|Selected|38.5811|16.9631|3|

Fixed confirmation mean/risk guards: {'hc2': False, 'vidstg': True}. The original scientific decision/fallback is retained verbatim in DECISION.json. NO retrospective pass. AllpairedCI/longshort/boundaries/support/individualsigns/severe cases versus Source/Global6/Native16/current-fused are in reports and TSVs.

Integrity: 2048 official/dense metric checks, max error 1.221e-15; Source/logits/q0/current fused outputs exact on all64 new queries. No GT in inference; complete64 seal before independent CPU labels. Original baseline128/192 cached representations reused without models. Research Source64 wall 314.66s; no teacher/observation/training. Research timing is NOT deployment latency.

v2 dev96parents/direction + finalconfirm32/direction, ZERO parent overlap. Previous Source/full/othermethod history exposed, NOT untouchedtest. Winner64/1 reduces finalconfirmation severeHC3->1 andVid6->3; HCt-1.9834/v-.8690pp relativeNative16 fails .5pp guard, Vidt+.1803/v-.0026. Userprioritizes low severe incidence and explicitly selects Native2D: separate MAIN_SELECTION retains the development winner as engineeringmain, without altering failed scientific DECISION/fallback16 or trying another candidate after confirmation. All stop here.

User later adds hard final-full acceptance: each direction queryequal tIoU must exceed Source by >=.9pp. Fixed confirmation selectedHC+4.5712/Vid+1.4185 meets this locally; exposed devHC+.8366 does NOT. Final full is pending: no guarantee or claim of meeting the final target, no full-result retuning.
