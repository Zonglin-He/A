# Three-state semantic boundary feedback

Binary semantic prototypes improved exposedHC64 but failed fixedHC32/Vid32. POSTSEAL specificcases show different event-vs-before and event-vs-after rankings and different before/after semanticmeans. This motivates a new hypothesis, not a proved root cause: pooling both outside states can lose boundary information.

ONLYchange is before/event/after statefactorization. Same frozen learnedwordfeatures, nativeq0, oppositeoffset crossfit, two iterations and full-time original spatialprocess. Everycandidate assigns ALLframes to before/event/after; class-summed expectedBernoulli logpseudolikelihood over WHOLEvideo, subtract common before density; anchor each iteration to fixedq0. No duration penalty, class normalization, new gain/threshold or acceptancegate. No new model calls or expert observations. Undefined exactzero seed state uses fit-parity globalmean, recorded; allqueries retained.

Controls isolate additional state capacity from boundary placement: UnorderedBackground uses SAMEthree centroids but equalmix before/after densities on BOTHoutside regions. SwappedBackground exchanges learnedbefore/afterroles. Reverseframeplacement, samehistogramrandomprojection, same-second-evidenceDirect, one round, exactparentBinary andSharpaliases, constant projection exactnative. FiveCPUtests statemembership/fullvideo likelihood/constant/parity+classinvariance/syntheticorder. All128 outputs seal before anynewlabels read; HC64 andfixedHC32/Vid32 are exposeddevelopment panels.

|Panel|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---|---:|---:|---:|---:|
|HC64|NativeG6|29.6952|55.9613|0/0/64|0|
|HC64|ThreeStateLoop2|29.8181|56.4946|28/25/11|14|
|HC64|OneStepThreeState|29.7949|56.3717|27/24/13|10|
|HC64|DirectThreeState|27.1933|51.7773|28/36/0|30|
|HC64|UnorderedBackgroundLoop2|31.5568|60.1468|36/18/10|5|
|HC64|SwappedBackgroundLoop2|29.9350|57.8471|34/27/3|12|
|HC64|ReversedThreeStateLoop2|29.3627|54.2575|23/30/11|17|
|HC64|RandomThreeStateLoop2|28.9752|54.7468|16/31/17|13|
|HC64|BinaryPrototypeLoop2|30.9658|58.8883|31/17/16|6|
|HC64|SharpenOriginal|28.7258|53.7518|16/38/10|20|
|HC64|ConstantProjectionNoOp|29.6952|55.9613|0/0/64|0|
|HC64|Source|26.8938|54.8230|—|—|

HC64: primary−G6 t+0.5332pp CI[-1.120175331719954, 2.3686625096931073]; v+0.1229pp CI[-0.8930403875393811, 1.2124209127082226]. Primary−Source t+1.6715/v+2.9243pp.
Source short-1.8076/long+2.5584tpp; fixed screen=False.
|hc2_recheck|NativeG6|27.8907|51.6010|0/0/32|0|
|hc2_recheck|ThreeStateLoop2|27.2063|50.5794|10/15/7|7|
|hc2_recheck|OneStepThreeState|27.2961|50.6422|10/15/7|5|
|hc2_recheck|DirectThreeState|25.0394|45.0612|15/17/0|16|
|hc2_recheck|UnorderedBackgroundLoop2|25.8321|49.7094|14/14/4|6|
|hc2_recheck|SwappedBackgroundLoop2|22.2852|42.9022|10/19/3|10|
|hc2_recheck|ReversedThreeStateLoop2|27.5198|52.1031|13/9/10|3|
|hc2_recheck|RandomThreeStateLoop2|29.1377|53.7521|16/7/9|2|
|hc2_recheck|BinaryPrototypeLoop2|27.3123|51.9587|15/12/5|5|
|hc2_recheck|SharpenOriginal|27.5315|50.1412|8/19/5|5|
|hc2_recheck|ConstantProjectionNoOp|27.8907|51.6010|0/0/32|0|
|hc2_recheck|Source|23.4655|50.4639|—|—|

hc2_recheck: primary−G6 t-1.0216pp CI[-3.5332294842369962, 1.3725014377103948]; v-0.6844pp CI[-2.4484511878885193, 0.9214200893563212]. Primary−Source t+0.1155/v+3.7408pp.
Source short+2.4504/long-1.1076tpp; fixed screen=False.
|vidstg_recheck|NativeG6|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|ThreeStateLoop2|22.6197|33.2071|7/19/6|7|
|vidstg_recheck|OneStepThreeState|22.6596|33.2835|8/16/8|6|
|vidstg_recheck|DirectThreeState|22.0958|32.4381|10/19/3|15|
|vidstg_recheck|UnorderedBackgroundLoop2|24.0429|34.4828|7/18/7|6|
|vidstg_recheck|SwappedBackgroundLoop2|22.9221|32.8077|7/23/2|11|
|vidstg_recheck|ReversedThreeStateLoop2|22.4558|33.6122|11/15/6|7|
|vidstg_recheck|RandomThreeStateLoop2|22.5848|33.5731|8/14/10|3|
|vidstg_recheck|BinaryPrototypeLoop2|23.6888|35.0400|9/16/7|7|
|vidstg_recheck|SharpenOriginal|23.4813|35.3245|14/11/7|1|
|vidstg_recheck|ConstantProjectionNoOp|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|Source|15.6772|34.0110|—|—|

vidstg_recheck: primary−G6 t-2.4482pp CI[-3.9179373008265546, -1.1018111791848164]; v-1.3379pp CI[-2.2549761108697925, -0.5427815706968923]. Primary−Source t-0.8039/v+6.9425pp.
Source short-1.1126/long+0.2988tpp; fixed screen=False.

Not qualified. No new full/default registration, state weights/iterations/class-count normalization scan or posthoccontrol promotion.
Calls{'new_model_calls': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_fusion': 0, 'new_expert': 0, 'new_observations': 0, 'new_optimizer': 0}; CPU study13.16s includes128allarmreads/DP/I/O, notdeploymentlatency.
Official/dense checks3072 maximumerror1.78e-15.

[All signed outcomes](signed.tsv), [all severe declines](severe_declines.tsv), [prospective protocol](protocol.json).
