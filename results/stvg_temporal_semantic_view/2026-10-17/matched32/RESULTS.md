# Matched32 Native temporal feedback experiment

Fixed exposed HC16/Vid16. All temporal and fresh-time Full predictions sealed before evaluation. No paper text, tuning, new confirmation or method promotion. Original negative results remain.

|Domain|Arm|vIoU %|tIoU %|interval-conditioned sIoU %|common Native-clock sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|---:|
|hc2|Native|28.116723|52.754115|50.497117|50.497117|31.250000|25.000000|
|hc2|SEM_STUDENT|28.116723|52.754115|50.497117|50.497117|31.250000|25.000000|
|hc2|SEM_VIEW|28.304305|52.986786|50.497117|50.497117|31.250000|31.250000|
|hc2|DIRECT_STUDENT|27.937821|52.525177|50.497117|50.497117|31.250000|25.000000|
|hc2|DIRECT_VIEW|27.750886|52.314841|50.497117|50.497117|31.250000|25.000000|
|hc2|PARA_STUDENT|22.275037|43.718849|50.497117|50.497117|18.750000|18.750000|
|hc2|PARA_VIEW|21.163607|42.574428|50.497117|50.497117|18.750000|18.750000|
|hc2|T2|27.487135|53.666875|50.497117|50.497117|37.500000|18.750000|
|hc2|SEM_FULL_DIRECT_OPD|29.034209|52.754115|52.075333|52.075333|31.250000|25.000000|
|hc2|T2_R1_DIRECT_OPD|28.580257|53.666875|52.108541|52.108541|37.500000|25.000000|
|vidstg|Native|12.185607|35.923576|32.870500|32.870500|12.500000|0.000000|
|vidstg|SEM_STUDENT|12.032638|35.727839|32.870500|32.870500|12.500000|0.000000|
|vidstg|SEM_VIEW|11.904988|35.564500|32.870500|32.870500|12.500000|0.000000|
|vidstg|DIRECT_STUDENT|10.490200|33.754155|32.870500|32.870500|12.500000|0.000000|
|vidstg|DIRECT_VIEW|10.490200|33.754155|32.870500|32.870500|12.500000|0.000000|
|vidstg|PARA_STUDENT|12.085089|36.101431|32.870500|32.870500|12.500000|0.000000|
|vidstg|PARA_VIEW|11.759314|35.433872|32.870500|32.870500|12.500000|0.000000|
|vidstg|T2|12.239461|35.680949|32.870500|32.870500|12.500000|0.000000|
|vidstg|SEM_FULL_DIRECT_OPD|14.420672|35.727839|30.870587|30.870587|18.750000|0.000000|
|vidstg|T2_R1_DIRECT_OPD|14.572580|35.680949|31.878854|31.878854|18.750000|0.000000|

|Domain|Contrast|vIoU delta pp|paired t95 pp|+/-/neutral/unknown|
|---|---|---:|---|---|
|hc2|SEM_STUDENT_minus_Native|0.000000|[0.0, 0.0]|0/0/16/0|
|hc2|DIRECT_STUDENT_minus_Native|-0.178903|[-0.5602253261665225, 0.20241957645487707]|0/1/15/0|
|hc2|DIRECT_STUDENT_minus_SEM_STUDENT|-0.178903|[-0.5602253261665225, 0.20241957645487707]|0/1/15/0|
|hc2|DIRECT_STUDENT_minus_SEM_VIEW|-0.366485|[-1.1476284281546305, 0.4146589764947301]|0/1/15/0|
|hc2|PARA_STUDENT_minus_Native|-5.841687|[-14.24146393372272, 2.5580903009470637]|2/5/9/0|
|hc2|PARA_STUDENT_minus_SEM_STUDENT|-5.841687|[-14.24146393372272, 2.5580903009470637]|2/5/9/0|
|hc2|PARA_STUDENT_minus_SEM_VIEW|-6.029269|[-14.72341396757568, 2.664876632851768]|2/5/9/0|
|hc2|PARA_STUDENT_minus_DIRECT_STUDENT|-5.662784|[-13.790208564632739, 2.464640681568725]|2/5/9/0|
|hc2|PARA_STUDENT_minus_DIRECT_VIEW|-5.475849|[-13.32839063183116, 2.3766926117358853]|2/5/9/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_Native|0.917486|[0.056601058653682035, 1.7783710744098438]|10/4/2/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_SEM_STUDENT|0.917486|[0.056601058653682035, 1.7783710744098438]|10/4/2/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_SEM_VIEW|0.729904|[-0.20131569907004787, 1.6611241301853186]|9/5/2/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_DIRECT_STUDENT|1.096389|[0.13786141343127256, 2.0549164693438984]|10/4/2/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_DIRECT_VIEW|1.283324|[0.09374651338209516, 2.4729012323618136]|10/4/2/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_PARA_STUDENT|6.759173|[-1.6161018626469712, 15.134447628486154]|12/3/1/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_PARA_VIEW|7.870603|[-1.3802951012132567, 17.121500988603355]|11/4/1/0|
|hc2|SEM_FULL_DIRECT_OPD_minus_T2|1.547074|[-2.6754602778865153, 5.7696090460263125]|9/5/2/0|
|vidstg|SEM_STUDENT_minus_Native|-0.152969|[-0.4790149657672381, 0.1730767995614154]|0/1/15/0|
|vidstg|DIRECT_STUDENT_minus_Native|-1.695407|[-5.309082537253559, 1.9182678618056883]|0/1/15/0|
|vidstg|DIRECT_STUDENT_minus_SEM_STUDENT|-1.542438|[-4.830067571486321, 1.7451910622442726]|0/1/15/0|
|vidstg|DIRECT_STUDENT_minus_SEM_VIEW|-1.414788|[-4.430337841432282, 1.6007614570930235]|0/1/15/0|
|vidstg|PARA_STUDENT_minus_Native|-0.100518|[-0.3865742278805187, 0.1855377073055246]|1/1/14/0|
|vidstg|PARA_STUDENT_minus_SEM_STUDENT|0.052451|[-0.391900345593171, 0.4968019912239996]|2/1/13/0|
|vidstg|PARA_STUDENT_minus_SEM_VIEW|0.180101|[-0.49567171836090135, 0.8558734888945201]|2/1/13/0|
|vidstg|PARA_STUDENT_minus_DIRECT_STUDENT|1.594889|[-2.044301344419712, 5.234079499292589]|2/1/13/0|
|vidstg|PARA_STUDENT_minus_DIRECT_VIEW|1.594889|[-2.044301344419712, 5.234079499292589]|2/1/13/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_Native|2.235064|[-1.0454474183286075, 5.515576231537464]|7/2/7/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_SEM_STUDENT|2.388033|[-0.882561394006907, 5.658628373421586]|7/2/7/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_SEM_VIEW|2.515684|[-0.7714603301265428, 5.802827434444012]|7/2/7/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_DIRECT_STUDENT|3.930472|[-0.7516359122537462, 8.612579400910473]|7/2/7/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_DIRECT_VIEW|3.930472|[-0.7516359122537462, 8.612579400910473]|7/2/7/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_PARA_STUDENT|2.335583|[-0.9486677265437384, 5.619833060327589]|7/2/7/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_PARA_VIEW|2.661358|[-0.7308523927053554, 6.053567839090755]|7/2/7/0|
|vidstg|SEM_FULL_DIRECT_OPD_minus_T2|2.181211|[-1.611242600582243, 5.9736640427104515]|4/5/7/0|

All32 and common-admission comparisons are separate. DIRECT shares the SEM caption support gate, with an additional valid-boundary requirement; PARA uses its own text-only gate. Full uses SEM_STUDENT time irrespective of measured outcomes, rebuilt observations and fresh Direct references. Same spatial parameters/optimizer/reward/response/acceptance as revised Direct; this is end-to-end temporal-route comparison, not a claim of isolated Teacher compute equivalence.

The constructor attempted an annotation-file open before frozen re-grounding and was blocked by the no-GT audit hook. Established clean loader substitutes empty constructor dictionaries; no annotation data was opened. Two additional CPU-only launcher failures (missing system NumPy and an overbroad failure-path rename) are retained. All32 Teacher outputs are reused exactly, not regenerated; original finite deadline preserved.

Only raw pre-text ResNet/Swin outputs are reused per Query for new-text views; all text-conditioned fusion and decoder outputs recomputed. Per-domain actual fresh/cached Source and new-text qualifiers are recorded. Student uses original-query hidden states, exact512 affine insertion, original native decoding; every temporal-only spatial box is exact Source. Full requires actual combined temporal/spatial native readout.

CPU scoring initially assumed a Native-first prediction dictionary and treated interval-conditioned sIoU as a frozen-box invariant; the asserted check failed before results were saved. Original failures are preserved. Recovery explicitly evaluates Native first and adds a fixed Native-clock geometric comparison, retaining the original official metrics and all sealed predictions. No model/Teacher/fit repeats. Boundary error deltas are seconds (not percentage points). GT subgrouping is descriptive after all prediction seals. Role-swapped/nonexistent-event labels remain unknown; generated support labels are not GT semantic labels. Prior semantic512 panel remains44 model-assisted/468 pending, not completed by this experiment. R1 training uses target-domain datasets; exact checkpoint-to-parent overlap remains unresolved, no claim of globally unseen Teacher training.
