# Native Temporal Hypothesis Augmentation: matched32 experiment

Fixed exposed HC16/Vid16; no paper draft, new confirmation, Prompt/parameter/candidate-rule search or method promotion. A completed and was published before this B group. Gate was locked before A scores: at least one domain A SEM_STUDENT-minus-Native <=0, or exactly zero Teacher KL throughout. A HC0 and Vid negative triggered it. All32 retained, no source-level GT routing.

|Domain|Arm|vIoU %|tIoU %|interval-conditioned sIoU %|common Native-clock sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|---:|
|hc2|Native|28.116723|52.754115|50.497117|50.497117|31.250000|25.000000|
|hc2|T2|27.487135|53.666875|50.497117|50.497117|37.500000|18.750000|
|hc2|SEM_VIEW|28.304305|52.986786|50.497117|50.497117|31.250000|31.250000|
|hc2|SEM_STUDENT|28.116723|52.754115|50.497117|50.497117|31.250000|25.000000|
|hc2|HYP_VIEW|28.037384|52.458105|50.497117|50.497117|31.250000|25.000000|
|hc2|HYP_STUDENT|27.894260|52.126979|50.497117|50.497117|31.250000|25.000000|
|hc2|SEM_FULL_DIRECT_OPD|29.034209|52.754115|52.075333|52.075333|31.250000|25.000000|
|hc2|T2_R1_DIRECT_OPD|28.580257|53.666875|52.108541|52.108541|37.500000|25.000000|
|hc2|HYP_FULL_DIRECT_OPD|28.830244|52.126979|52.011520|52.011520|31.250000|25.000000|
|vidstg|Native|12.185607|35.923576|32.870500|32.870500|12.500000|0.000000|
|vidstg|T2|12.239461|35.680949|32.870500|32.870500|12.500000|0.000000|
|vidstg|SEM_VIEW|11.904988|35.564500|32.870500|32.870500|12.500000|0.000000|
|vidstg|SEM_STUDENT|12.032638|35.727839|32.870500|32.870500|12.500000|0.000000|
|vidstg|HYP_VIEW|12.289683|36.194334|32.870500|32.870500|12.500000|0.000000|
|vidstg|HYP_STUDENT|12.185607|35.923576|32.870500|32.870500|12.500000|0.000000|
|vidstg|SEM_FULL_DIRECT_OPD|14.420672|35.727839|30.870587|30.870587|18.750000|0.000000|
|vidstg|T2_R1_DIRECT_OPD|14.572580|35.680949|31.878854|31.878854|18.750000|0.000000|
|vidstg|HYP_FULL_DIRECT_OPD|14.633156|35.923576|31.152643|31.152643|25.000000|0.000000|

|Domain|Contrast|vIoU delta pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---:|---|---|
|hc2|HYP_STUDENT_minus_Native|-0.222463|[-1.6788558692579054, 1.2339293115590102]|2/3/11/0|
|hc2|HYP_STUDENT_minus_SEM_STUDENT|-0.222463|[-1.6788558692579054, 1.2339293115590102]|2/3/11/0|
|hc2|HYP_STUDENT_minus_SEM_VIEW|-0.410045|[-1.9119301734325886, 1.0918399137854387]|2/4/10/0|
|hc2|HYP_STUDENT_minus_DIRECT_STUDENT|-0.043560|[-1.5570313728608822, 1.4699105648736326]|3/3/10/0|
|hc2|HYP_STUDENT_minus_DIRECT_VIEW|0.143375|[-1.5234821836742516, 1.8102312386557389]|3/3/10/0|
|hc2|HYP_STUDENT_minus_PARA_STUDENT|5.619224|[-2.113702473930844, 13.352149549007606]|6/4/6/0|
|hc2|HYP_STUDENT_minus_PARA_VIEW|6.730654|[-1.8402847781953229, 15.301591974823001]|8/4/4/0|
|hc2|HYP_STUDENT_minus_T2|0.407125|[-3.2259621342041056, 4.040212211581482]|5/7/4/0|
|hc2|HYP_STUDENT_minus_SEM_FULL_DIRECT_OPD|-1.139949|[-2.7451526837093247, 0.46525399294690417]|3/11/2/0|
|hc2|HYP_STUDENT_minus_T2_R1_DIRECT_OPD|-0.685997|[-4.391742395188142, 3.0197481295705395]|5/9/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_Native|0.713520|[-1.0253648141459735, 2.4524056779766883]|10/4/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_SEM_STUDENT|0.713520|[-1.0253648141459735, 2.4524056779766883]|10/4/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_SEM_VIEW|0.525939|[-1.242279032473911, 2.294156194356371]|9/5/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_DIRECT_STUDENT|0.892423|[-0.902981252669957, 2.6878278662123174]|10/4/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_DIRECT_VIEW|1.079358|[-0.8552974882594575, 3.014013964770555]|10/4/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_PARA_STUDENT|6.555207|[-1.2413835915816112, 14.351798088187984]|11/4/1/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_PARA_VIEW|7.666637|[-0.9625535963628771, 16.295828214520164]|11/4/1/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_T2|1.343109|[-2.2831622767530853, 4.969379775660071]|9/5/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_SEM_FULL_DIRECT_OPD|-0.203966|[-1.688213970328947, 1.2802827010961357]|2/3/11/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_T2_R1_DIRECT_OPD|0.249987|[-3.323960178784505, 3.8239333346965125]|6/6/4/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_HYP_STUDENT|0.935984|[0.14148632283310836, 1.7304810986965014]|10/4/2/0|
|hc2|HYP_FULL_DIRECT_OPD_minus_HYP_VIEW|0.792860|[-0.07902365091600116, 1.664744174224061]|9/5/2/0|
|vidstg|HYP_STUDENT_minus_Native|0.000000|[0.0, 0.0]|0/0/16/0|
|vidstg|HYP_STUDENT_minus_SEM_STUDENT|0.152969|[-0.1730767995614154, 0.4790149657672381]|1/0/15/0|
|vidstg|HYP_STUDENT_minus_SEM_VIEW|0.280619|[-0.31750640471266556, 0.8787446958212785]|1/0/15/0|
|vidstg|HYP_STUDENT_minus_DIRECT_STUDENT|1.695407|[-1.9182678618056883, 5.309082537253559]|1/0/15/0|
|vidstg|HYP_STUDENT_minus_DIRECT_VIEW|1.695407|[-1.9182678618056883, 5.309082537253559]|1/0/15/0|
|vidstg|HYP_STUDENT_minus_PARA_STUDENT|0.100518|[-0.1855377073055246, 0.3865742278805187]|1/1/14/0|
|vidstg|HYP_STUDENT_minus_PARA_VIEW|0.426293|[-0.3708667851404673, 1.223453418317011]|3/2/11/0|
|vidstg|HYP_STUDENT_minus_T2|-0.053854|[-0.7905703597422037, 0.6828629886615561]|4/4/8/0|
|vidstg|HYP_STUDENT_minus_SEM_FULL_DIRECT_OPD|-2.235064|[-5.515576231537464, 1.0454474183286075]|2/7/7/0|
|vidstg|HYP_STUDENT_minus_T2_R1_DIRECT_OPD|-2.386973|[-5.161600194082291, 0.3876537729921061]|3/7/6/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_Native|2.447549|[-0.8279610338809995, 5.7230592556234985]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_SEM_STUDENT|2.600518|[-0.7099700489837395, 5.911006436932061]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_SEM_VIEW|2.728168|[-0.6355062835895486, 6.091842796440661]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_DIRECT_STUDENT|4.142956|[-0.8732783006737597, 9.159191197864128]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_DIRECT_VIEW|4.142956|[-0.8732783006737597, 9.159191197864128]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_PARA_STUDENT|2.548067|[-0.729213771840898, 5.8253485141583905]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_PARA_VIEW|2.873842|[-0.5054222259428283, 6.25310708086187]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_T2|2.393695|[-1.4170518557147385, 6.2044427063765895]|4/5/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_SEM_FULL_DIRECT_OPD|0.212485|[-0.24041572208100198, 0.6653851306146443]|1/0/15/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_T2_R1_DIRECT_OPD|0.060576|[-0.9810554740115995, 1.102207274663914]|5/4/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_HYP_STUDENT|2.447549|[-0.8279610338809995, 5.7230592556234985]|7/2/7/0|
|vidstg|HYP_FULL_DIRECT_OPD_minus_HYP_VIEW|2.343473|[-0.9564474158087486, 5.643393315177063]|6/3/7/0|

Teacher signal is sealed and audited before fitting: captions never see the original Query, while the consistency check does. Different observed frames are an input distinction, not evidence of different event meaning. Both finite, unequal rewards required; no semantic-correctness claim follows from that condition or from any numerical preference difference. Actual reward magnitudes, conditional priors/targets and no-op reasons are all retained.

I1 preserves the original two-offset Native decode. I2 is selected on the Native joint state distribution, and retains both latent endpoint pairs. Neither joint-mode merging nor this retained-state conditional is presented as the mode/mass of the marginal distribution over merged intervals. The optimization uses the conditional probability of the two retained joint states plus the exact factorized full-grid trust KL, tau1 and trust1, zero-initialized512 affine endpoint parameters, original LBFGS50/Armijo; Source query and native complete-grid final decode remain. The alternative is not copied to Student as a hard label.

Student then precedes new-time four-anchor R1 Direct/OPD Full, with the same source, original1792 episodic Gaussian/domain settings/response/acceptance/rollback/early stop as A and revised Direct. Exact prior Teacher outputs are reused only after complete input/prompt/model identity, otherwise acquired afresh. Full is not assembled from an unrelated spatial trajectory/time interval: actual combined native readout is checked. No additional full-video context, connections, backtracking or target-GT tuning.

The user accepts a small development score tradeoff for removing the external Actionness/Projection dependency. Portability of the evidence-to-distribution mechanism is a motivation for retaining the route; arbitrary-Backbone plug-and-play and new-Backbone effect have not been measured here. TA-STVG internal frozen action gates remain, and its two-offset/256D/512 interface is disclosed. Native-clock common-support sIoU separates geometry from time-support metric coupling. A/B end-to-end Full contrasts also change observation positions/Teacher evidence and real Teacher compute; they are not single-factor optimizer comparisons.

All signed, neutral, unknown and no-fit outcomes remain. Paired t intervals describe this exposed development cohort, not unseen confirmation. R1 training uses target-domain datasets and exact checkpoint-to-parent overlap is unresolved. Identity/role/nonexistent-event labels remain unknown; the old512 panel remains44 model-assisted/468 pending.
