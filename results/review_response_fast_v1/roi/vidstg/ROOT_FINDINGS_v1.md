# vidstg: standalone retained192 roi review control

Exactly192queries, 192 parents. This pre-existing retained-raw cohort includes previously exposed development and confirmation sources; it is not the full1902/10303 domain, an unseen test, or a prefix accuracy estimate. All192 prediction configurations sealed before this independent official scoring. Separate actual ROI repooling/cold gamma chain, prediction/input/parent/receipt SHA and all-mean audits passed.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|29.926298|12.612844|14.062500|3.125000|
|CorrectedROI|33.734719|15.350508|20.312500|8.854167|
|SourceROI|34.050322|15.416767|20.833333|8.854167|
|GlobalAppearance|33.887327|15.349748|20.833333|9.895833|
|ShiftedROI|33.975019|15.422917|21.354167|9.895833|
|RandomROI|33.438103|15.141131|20.833333|9.375000|

Parent-paired10000-bootstrap pointwise95% CIs, no familywise significance claim:

- CorrectedROI-Source: t+3.808420pp CI[1.9292801496811613, 5.876613509238303]; v+2.737663pp CI[0.9896791622984895, 4.513635268269065]; severe t/v/union28/25/43, rescues88.
- CorrectedROI-SourceROI: t-0.315604pp CI[-0.8345035565445306, 0.2445655883332711]; v-0.066259pp CI[-0.23602194814301486, 0.0973773806096196]; severe t/v/union16/2/16, rescues7.
- CorrectedROI-GlobalAppearance: t-0.152608pp CI[-0.9316341287558837, 0.646223131652582]; v+0.000760pp CI[-0.31602271015611083, 0.33151822401928105]; severe t/v/union20/5/20, rescues10.
- CorrectedROI-ShiftedROI: t-0.240300pp CI[-1.1005572208207925, 0.554406792398951]; v-0.072410pp CI[-0.3915373605535942, 0.23721069145146553]; severe t/v/union25/8/25, rescues19.
- CorrectedROI-RandomROI: t+0.296616pp CI[-0.5076158097508204, 1.153509029839668]; v+0.209376pp CI[-0.19123474348384426, 0.6181771832983859]; severe t/v/union24/5/24, rescues24.

Short<=5s/long ground-truth-duration strata are offline only:

- short_le5s, n118, CorrectedROI-Source: v+0.959518pp CI[-0.4248373343966502, 2.3936609644564544]; t+0.433597 CI[-1.1546238299874418, 2.0199162033691636].
- short_le5s, n118, CorrectedROI-SourceROI: v+0.075895pp CI[-0.10335712264917037, 0.2617602359538254]; t-0.076036 CI[-0.7244885870028063, 0.6291903893874662].
- short_le5s, n118, CorrectedROI-GlobalAppearance: v+0.117801pp CI[-0.29286635611526873, 0.5398259533854091]; t+0.035794 CI[-0.7108123599940861, 0.784504627632889].
- short_le5s, n118, CorrectedROI-ShiftedROI: v+0.279772pp CI[-0.02675554240554927, 0.6290981023548907]; t+0.286041 CI[-0.4302657993810723, 1.0108036722821032].
- short_le5s, n118, CorrectedROI-RandomROI: v+0.399317pp CI[0.01855190634971032, 0.8138162628774064]; t+0.202350 CI[-0.6071670547158559, 1.074400234296438].
- long_gt5s, n74, CorrectedROI-Source: v+5.573084pp CI[1.7631192478600393, 9.620162332528484]; t+9.189896 CI[5.1297578391782475, 13.686027981386646].
- long_gt5s, n74, CorrectedROI-SourceROI: v-0.292937pp CI[-0.6331312713121322, 0.006415135072594851]; t-0.697617 CI[-1.551322268160282, 0.15614785330484637].
- long_gt5s, n74, CorrectedROI-GlobalAppearance: v-0.185873pp CI[-0.6886989199296856, 0.31071720248409884]; t-0.453033 CI[-2.046589172930604, 1.2808401045744824].
- long_gt5s, n74, CorrectedROI-ShiftedROI: v-0.633996pp CI[-1.2523852609839794, -0.056146838014418596]; t-1.079602 CI[-3.0085378544233605, 0.580763984230072].
- long_gt5s, n74, CorrectedROI-RandomROI: v-0.093502pp CI[-0.9251717316617475, 0.7683100728149336]; t+0.446932 CI[-1.1669416743680767, 2.2174474467806995].

All appearance arms retain identical q0/eight observations, exact MAIN spatial candidate bank and coefficients, centering/unit/prototype/kappa0/beta1. Only temporal pooling changes, each computes own posterior/interval/finalDP. GlobalAppearance pools all valid native grid cells. ShiftedROI translates the corrected trajectory by a predeclared SHA-selected horizontal quarter image; RandomROI fixes a SHA-derived querywise center with original per-frame widths/heights. Context1.5 and original out-of-grid global fallback remain, with counts reported. These counterfactual controls can alter coverage/temporal smoothness and are not semantic ground truth or pure action-feature attribution. Source is untouched native-time Source boxes, distinct from the SourceROI full chain. No control, scale or main method is selected using these scores.

Global-fallback frame counts: {'CorrectedROI': 396, 'SourceROI': 0, 'ShiftedROI': 916, 'RandomROI': 0}.

All signed positive/negative rows, severe tails/rescues and short/long CIs are retained. Conditional replay elapsed clocks include file SHA, resource checks and actual original-grid pooling; not process CPU or complete deployment latency. New model/Source/expert/optimizer calls are zero. Original acquired Source/eight-expert contexts and old gamma preprocessing are inherited nonzero costs, not free observations. Shared original Vid producer/TTA costs must not be added twice.
