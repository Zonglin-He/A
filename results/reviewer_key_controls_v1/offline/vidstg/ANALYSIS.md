# Offline spatial support, expert quality and risk: vidstg

Existing independently audited frame IoUs and scalar outcomes are reused. New event-membership masks, query/form/quality joins, counts, means and paired CIs passed separate-process independent recomputation. No new predictions, model, Source, expert, optimizer, media access or GT metric scoring. GT intervals/anchor quality are offline labels only, never an online gate.

|Support / nearest-observation distance|Queries|Parents|Annotated frames|Ours IoU %|Interpolation IoU %|Paired delta IoU pp [95% CI]|
|---|---:|---:|---:|---:|---:|---|
|all_unobserved/all_distances|10300|732|365021|35.78576151681422|35.51298936543868|+0.272772 [-0.254923, +0.802648]|
|all_unobserved/bin0|8991|721|103389|37.558331738659696|40.14515946675464|-2.586828 [-3.084346, -2.084155]|
|all_unobserved/bin1|7528|685|66503|35.850224353892926|36.18677749848028|-0.336553 [-0.954093, +0.305586]|
|all_unobserved/bin2|5898|610|83822|34.554549538731315|33.02011644142492|+1.534433 [+0.781581, +2.298682]|
|all_unobserved/bin3|3616|464|69507|33.77070588563841|30.05268016258418|+3.718026 [+2.753252, +4.730752]|
|all_unobserved/bin4|1757|293|41800|33.26605888620796|28.626486118392933|+4.639573 [+3.370389, +5.997261]|
|predicted_event/all_distances|8241|710|192395|38.86783419452302|38.61071022419902|+0.257124 [-0.238486, +0.742046]|
|predicted_event/bin0|7650|695|75095|40.217433643201424|41.45366957169638|-1.236236 [-1.691648, -0.781338]|
|predicted_event/bin1|5999|653|42264|39.04873343201389|38.30869387262563|+0.740040 [+0.137754, +1.348143]|
|predicted_event/bin2|4116|563|40529|38.313616743069545|36.01369456211811|+2.299922 [+1.576660, +3.064851]|
|predicted_event/bin3|1987|369|22785|38.81456327058092|35.04258732433539|+3.771976 [+2.756186, +4.817705]|
|predicted_event/bin4|761|197|11722|38.57676620722849|34.7284474107866|+3.848319 [+2.460074, +5.371713]|
|GT_event/all_distances|10300|732|363283|35.81637065425977|35.55364613879496|+0.262725 [-0.265999, +0.793407]|
|GT_event/bin0|8972|721|102581|37.61629792923937|40.21915693615732|-2.602859 [-3.105690, -2.095624]|
|GT_event/bin1|7489|685|66165|35.87368418116394|36.235781107768425|-0.362097 [-0.967998, +0.273666]|
|GT_event/bin2|5874|609|83513|34.5944538404002|33.05921167946761|+1.535242 [+0.797121, +2.312620]|
|GT_event/bin3|3592|463|69323|33.81735196350694|30.113248943802585|+3.704103 [+2.726077, +4.694470]|
|GT_event/bin4|1753|293|41701|33.25025981596237|28.6797475830672|+4.570512 [+3.316459, +5.923789]|

Distance bins: [0,.5), [.5,1), [1,2), [2,4), [4,infinity) seconds. Frames must be actually annotated native sampled UNOBSERVED frames, with physical nearest-anchor distance / fps. Prediction membership uses exactly the Table10 current joint-MAP interval; GT membership is half open as in the official scorer. Empty cells are explicitly unscorable; primary query-equal per-bin means and parent CIs are distinct from descriptive frame-weighted means. This diagnoses where spatial gains occur; it cannot uniquely decompose full dense vIoU or causally explain query form/quality effects.

|Query form / expert-quality third|Queries|Spatial path minus interpolation v pp [95% CI]|Unobserved IoU delta pp [95% CI]|
|---|---:|---|---|
|all/all|10303|-0.106955 [-0.250579, +0.036605]|+0.272772 [-0.254923, +0.802648]|
|all/low|3731|+0.155186 [+0.012837, +0.287952]|+1.258384 [+0.839438, +1.664952]|
|all/medium|1326|+0.018852 [-0.457922, +0.483042]|-0.234350 [-1.344302, +0.870406]|
|all/high|3450|-0.599935 [-0.910924, -0.299904]|-2.947871 [-3.892945, -2.033462]|
|all/unscorable|1796|+0.202572 [+0.126064, +0.284560]|+4.784800 [+3.582553, +6.010854]|
|declarative/all|4610|-0.103775 [-0.333900, +0.120043]|+0.590022 [-0.202154, +1.382338]|
|declarative/low|841|+0.308225 [-0.017660, +0.642646]|+1.859022 [+0.811468, +2.981094]|
|declarative/medium|553|+0.342773 [-0.525122, +1.210776]|+1.002271 [-0.942249, +2.950724]|
|declarative/high|2423|-0.508097 [-0.841980, -0.178508]|-2.480817 [-3.472478, -1.496701]|
|declarative/unscorable|793|+0.383289 [+0.235739, +0.545988]|+8.336280 [+6.264106, +10.468599]|
|interrogative/all|5693|-0.109531 [-0.230538, +0.009802]|+0.015940 [-0.445363, +0.491465]|
|interrogative/low|2890|+0.110651 [-0.008116, +0.230361]|+1.083536 [+0.718568, +1.462032]|
|interrogative/medium|773|-0.212879 [-0.648623, +0.240293]|-1.117423 [-2.285275, +0.054781]|
|interrogative/high|1027|-0.816607 [-1.225773, -0.414596]|-4.049335 [-5.391428, -2.729975]|
|interrogative/unscorable|1003|+0.059691 [-0.008876, +0.132244]|+1.976900 [+0.858821, +3.134439]|

Expert quality is the original mean IoU on GT-supported expert anchors only. Fixed thirds [0,1/3), [1/3,2/3), [2/3,1], no score-selected quantiles; missing remains unscorable. Declarative/interrogative labels are exact catalog joins, not punctuation heuristics.

|Risk group|Queries|Delta v pp|Gain >+1 pp %|Loss <-1 pp %|Loss <=-10 pp %|Any negative %|
|---|---:|---|---|---|---|---|
|all|10303|+3.419668 [+3.009582, +3.851146]|+41.017179 [+39.284653, +42.789238]|+21.255945 [+20.192881, +22.364095]|+7.007668 [+6.363447, +7.676425]|+27.380375 [+26.120642, +28.662492]|
|Source_v_ge_0.5|345|-1.434858 [-3.971558, +1.023191]|+60.000000 [+54.654655, +65.506602]|+34.782609 [+29.559561, +39.850115]|+24.057971 [+19.420290, +28.875380]|+37.391304 [+32.081631, +42.531846]|
|Source_v_ge_0.8|2|+3.085675 [-0.269000, +6.440350]|+50.000000 [+0.000000, +100.000000]|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|+50.000000 [+0.000000, +100.000000]|
|Source_t_ge_0.8|654|+3.783300 [+1.714556, +5.838411]|+54.892966 [+50.551138, +59.084363]|+33.792049 [+30.015077, +37.649102]|+21.406728 [+18.009461, +25.034583]|+38.073394 [+34.161491, +42.192276]|

All thresholds predeclared; strong Source groups are vIoU>=.5, vIoU>=.8 and tIoU>=.8. Query equal, parent10000 pointwise95% CIs, not familywise. No new method adoption or parameter selection. Inherited acquisition costs nonzero, offline cached runtime is not deployment latency.
