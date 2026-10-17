# Original256: Native Temporal B versus matched Frozen and T2

Same original HC128/Vid128 exposed development parents. The current B implementation and all per-query parameters are unchanged. Old32 complete results are reused after exact source/hypothesis/pixel/Teacher/fit/readout identity; only remaining224 acquire new evidence and fit. This is not an independent confirmation or a single-factor I2/Actionness ablation. No Spatial adaptation/Full run, new Source capture, T2 rerun, Prompt/parameter search or CURRENT promotion.

## Complete256 (primary)

|Domain|Arm|vIoU %|tIoU %|fixed dense sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|30.468750|7.812500|
|hc2|T2|26.331915|49.232269|53.809342|41.406250|14.062500|
|hc2|B|21.518514|39.443767|53.809342|29.687500|7.031250|
|vidstg|Frozen|9.843861|32.374371|30.729497|10.156250|1.562500|
|vidstg|T2|10.286840|33.278160|30.729497|10.937500|1.562500|
|vidstg|B|9.779438|32.181130|30.729497|10.156250|1.562500|

|Domain|Contrast|Metric|Delta pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---|---:|---|---|
|hc2|T2_minus_Frozen|m_vIoU|4.680756|[3.1010778893282684, 6.26043341687928]|78/26/24/0|
|hc2|T2_minus_Frozen|m_tIoU|9.685434|[6.878881006782061, 12.491986915261927]|83/23/22/0|
|hc2|B_minus_Frozen|m_vIoU|-0.132646|[-0.4690431617264037, 0.2037516610006923]|18/18/92/0|
|hc2|B_minus_Frozen|m_tIoU|-0.103068|[-0.6490748616619343, 0.4429386587999101]|20/19/89/0|
|hc2|B_minus_T2|m_vIoU|-4.813401|[-6.4074328068151605, -3.2193700001180994]|27/77/24/0|
|hc2|B_minus_T2|m_tIoU|-9.788502|[-12.596384758030961, -6.98061936687505]|25/82/21/0|
|vidstg|T2_minus_Frozen|m_vIoU|0.442980|[-0.02543153251891561, 0.9113907841462057]|48/26/54/0|
|vidstg|T2_minus_Frozen|m_tIoU|0.903789|[-0.14820723156951088, 1.9557853537647212]|53/30/45/0|
|vidstg|B_minus_Frozen|m_vIoU|-0.064423|[-0.16973298851128066, 0.04088775380779079]|8/16/104/0|
|vidstg|B_minus_Frozen|m_tIoU|-0.193240|[-0.44243426237774264, 0.05595374924022692]|8/20/100/0|
|vidstg|B_minus_T2|m_vIoU|-0.507402|[-0.9611010251479032, -0.05370346118287672]|26/46/56/0|
|vidstg|B_minus_T2|m_tIoU|-1.097029|[-2.1415967083224183, -0.05246192701030772]|29/52/47/0|

## prior32 / 32 sources

|Domain|Arm|vIoU %|tIoU %|fixed dense sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|28.116723|52.754115|50.497117|31.250000|25.000000|
|hc2|T2|27.487135|53.666875|50.497117|37.500000|18.750000|
|hc2|B|27.894260|52.126979|50.497117|31.250000|25.000000|
|vidstg|Frozen|12.185607|35.923576|32.870500|12.500000|0.000000|
|vidstg|T2|12.239461|35.680949|32.870500|12.500000|0.000000|
|vidstg|B|12.185607|35.923576|32.870500|12.500000|0.000000|

|Domain|Contrast|Metric|Delta pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---|---:|---|---|
|hc2|T2_minus_Frozen|m_vIoU|-0.629588|[-4.926975025097099, 3.6677983900208284]|7/5/4/0|
|hc2|T2_minus_Frozen|m_tIoU|0.912759|[-5.522021984276886, 7.3475408755921645]|9/4/3/0|
|hc2|B_minus_Frozen|m_vIoU|-0.222463|[-1.6788558692579054, 1.2339293115590102]|2/3/11/0|
|hc2|B_minus_Frozen|m_tIoU|-0.627136|[-2.652363860346468, 1.3980922527298967]|2/4/10/0|
|hc2|B_minus_T2|m_vIoU|0.407125|[-3.2259621342041056, 4.040212211581482]|5/7/4/0|
|hc2|B_minus_T2|m_tIoU|-1.539895|[-7.391303757256484, 4.311513258324634]|4/9/3/0|
|vidstg|T2_minus_Frozen|m_vIoU|0.053854|[-0.6828629886615561, 0.7905703597422037]|4/4/8/0|
|vidstg|T2_minus_Frozen|m_tIoU|-0.242627|[-2.522827393045292, 2.0375737504112035]|5/4/7/0|
|vidstg|B_minus_Frozen|m_vIoU|0.000000|[0.0, 0.0]|0/0/16/0|
|vidstg|B_minus_Frozen|m_tIoU|0.000000|[0.0, 0.0]|0/0/16/0|
|vidstg|B_minus_T2|m_vIoU|-0.053854|[-0.7905703597422037, 0.6828629886615561]|4/4/8/0|
|vidstg|B_minus_T2|m_tIoU|0.242627|[-2.0375737504112035, 2.522827393045292]|4/5/7/0|

## added224 / 224 sources

|Domain|Arm|vIoU %|tIoU %|fixed dense sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|20.727508|37.660081|54.282517|30.357143|5.357143|
|hc2|T2|26.166884|48.598754|54.282517|41.964286|13.392857|
|hc2|B|20.607693|37.631880|54.282517|29.464286|4.464286|
|vidstg|Frozen|9.509326|31.867341|30.423640|9.821429|1.785714|
|vidstg|T2|10.007895|32.934904|30.423640|10.714286|1.785714|
|vidstg|B|9.435700|31.646495|30.423640|9.821429|1.785714|

|Domain|Contrast|Metric|Delta pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---|---:|---|---|
|hc2|T2_minus_Frozen|m_vIoU|5.439376|[3.765755942624316, 7.1129964980523495]|71/21/20/0|
|hc2|T2_minus_Frozen|m_tIoU|10.938673|[7.90670589627965, 13.970640458725583]|74/19/19/0|
|hc2|B_minus_Frozen|m_vIoU|-0.119815|[-0.45588741865917337, 0.21625806892963106]|16/15/81/0|
|hc2|B_minus_Frozen|m_tIoU|-0.028201|[-0.5946683581359568, 0.5382657845245823]|18/15/79/0|
|hc2|B_minus_T2|m_vIoU|-5.559191|[-7.27794101382173, -3.840440776584478]|22/70/20/0|
|hc2|B_minus_T2|m_tIoU|-10.966874|[-14.028528534541008, -7.905220394075597]|21/73/18/0|
|vidstg|T2_minus_Frozen|m_vIoU|0.498569|[-0.028464497902856878, 1.0256025896082388]|44/22/46/0|
|vidstg|T2_minus_Frozen|m_tIoU|1.067563|[-0.09724883368193682, 2.232374350852761]|48/26/38/0|
|vidstg|B_minus_Frozen|m_vIoU|-0.073626|[-0.1941160637185439, 0.046864366914555436]|8/16/88/0|
|vidstg|B_minus_Frozen|m_tIoU|-0.220846|[-0.5058169224399135, 0.06412490742560975]|8/20/84/0|
|vidstg|B_minus_T2|m_vIoU|-0.572195|[-1.08176545851183, -0.06262432999754042]|22/42/48/0|
|vidstg|B_minus_T2|m_tIoU|-1.288409|[-2.443219953724877, -0.13359757846025144]|25/47/40/0|

Different sampled inputs do not establish different event semantics. Nonzero P(A)-P(B) differences and conditional targets are uncalibrated numerical evidence, not verified referent/event labels. Captions never see the original Query; consistency evaluation does. Invalid/truncated/equal/missing-hypothesis cases remain exact Native and stay in all denominators. No third hypothesis or retry.

Native candidate prior is tilted with tau1; zero-init512 endpoint adaptation minimizes joint-state candidate KL and sum offset Native-grid trust KL(lambda1). Original LBFGS50/Armijo remains, and reaching50 iterations is not proof of full convergence. Final intervals come from actual complete native-grid decode. Frozen and T2 are independently rescored from sealed predictions and checked exactly against old scalars. All B Source boxes remain exact. All256 prediction/root barriers precede new GT scoring.

These are exposed development sources, including the previously read32; no unseen confirmation/noninferiority claim follows automatically. The Teacher was trained on target-domain datasets and exact checkpoint-parent overlap is unresolved. No identity/role semantic panel is completed by numerical metrics or engineering pixel inspection. Other-backbone plug-and-play is unmeasured; current native two-offset512 interface and internal frozen gates remain.

Engineering recovery: two original Source prefix cache receipts were absent. The first readout worker stopped after33 new complete readouts, before any new GT scoring. Original failure/log/runtime and those33 outputs are preserved. Two full frozen prefixes were reconstructed from SHA-matched original video pixels and checked exactly against sealed Source logits, hidden states, boxes, records and Native intervals. Remaining191 readouts then continued with unchanged fitted parameters; no Teacher generation or fitting was repeated and the original deadline was retained.


Measured decision: B-minus-Frozen vIoU means are -0.132646pp HC and -0.064423pp Vid; both paired95% intervals cross zero. B-minus-T2 is -4.813401pp HC and -0.507402pp Vid, with both paired intervals entirely below zero. Thus the current Native-only B does not retain old T2 benefit on the original256; the earlier32 subset alone cannot explain the method gap. This is not an isolated Actionness removal or I2-swap experiment, so the gap is not fully attributed to either component. No Full256 effect or universal failure is inferred.
