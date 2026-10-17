# Original256: change only I2 ranking, preserve the no-update control

HC128/Vid128 original exposed development parents; Source, I1, Teacher/Prompt/processor/clock, reward, zero-init512 joint-state target, optimizer/trust/temperature and final Native decoder unchanged. Only I2 physical-interval ranking uses summed Native probability. No external Actionness, Spatial/Full, search or promotion. Native internal frozen gates remain.

The selected physical I2 still retains one representative latent state by the original FP32 joint-state ranking. Marginalizing the training target would be another intervention and is not done. Marginal no-update is a separate readout control; its free decoder gain is not credited to TTA. Frozen/T2/original B are sealed matched baseline predictions, not rerun models.

|Domain|Arm|vIoU %|tIoU %|dense sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|30.468750|7.812500|
|hc2|T2|26.331915|49.232269|53.809342|41.406250|14.062500|
|hc2|B|21.518514|39.443767|53.809342|29.687500|7.031250|
|hc2|MARGINAL_NO_UPDATE|24.945094|46.626294|53.809342|38.281250|13.281250|
|hc2|B_MARGINAL_I2|21.480541|39.520097|53.809342|31.250000|6.250000|
|vidstg|Frozen|9.843861|32.374371|30.729497|10.156250|1.562500|
|vidstg|T2|10.286840|33.278160|30.729497|10.937500|1.562500|
|vidstg|B|9.779438|32.181130|30.729497|10.156250|1.562500|
|vidstg|MARGINAL_NO_UPDATE|10.299485|34.126993|30.729497|10.156250|1.562500|
|vidstg|B_MARGINAL_I2|9.839035|32.234905|30.729497|10.156250|2.343750|

## All matched contrasts and negative outcomes

|Domain|Contrast|Metric|Delta pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---|---:|---|---|
|hc2|B_MARGINAL_I2_minus_Frozen|m_vIoU|-0.170619|[-0.6234566055783847, 0.282218556028465]|11/14/103/0|
|hc2|B_MARGINAL_I2_minus_Frozen|m_tIoU|-0.026739|[-0.7090420299298533, 0.6555643514863151]|13/13/102/0|
|hc2|B_MARGINAL_I2_minus_T2|m_vIoU|-4.851375|[-6.359179365917124, -3.3435699898403444]|25/78/25/0|
|hc2|B_MARGINAL_I2_minus_T2|m_tIoU|-9.712173|[-12.41323946648409, -7.0111061340034375]|23/83/22/0|
|hc2|B_MARGINAL_I2_minus_B|m_vIoU|-0.037973|[-0.4235617041106895, 0.34761515528648124]|18/20/90/0|
|hc2|B_MARGINAL_I2_minus_B|m_tIoU|0.076329|[-0.5530091553043155, 0.7056676797228014]|19/21/88/0|
|hc2|B_MARGINAL_I2_minus_MARGINAL_NO_UPDATE|m_vIoU|-3.464553|[-4.869651563194241, -2.0594550643889247]|23/77/28/0|
|hc2|B_MARGINAL_I2_minus_MARGINAL_NO_UPDATE|m_tIoU|-7.106197|[-9.42482747264955, -4.78756708574461]|23/82/23/0|
|vidstg|B_MARGINAL_I2_minus_Frozen|m_vIoU|-0.004825|[-0.13319684634221704, 0.12354617727866943]|6/12/110/0|
|vidstg|B_MARGINAL_I2_minus_Frozen|m_tIoU|-0.139466|[-0.4110191169594816, 0.13208770844204215]|6/13/109/0|
|vidstg|B_MARGINAL_I2_minus_T2|m_vIoU|-0.447805|[-0.9151350951582136, 0.01952517446737584]|27/47/54/0|
|vidstg|B_MARGINAL_I2_minus_T2|m_tIoU|-1.043255|[-2.1462632512469244, 0.059753720534274635]|30/52/46/0|
|vidstg|B_MARGINAL_I2_minus_B|m_vIoU|0.059597|[-0.06503255943761174, 0.184227125077554]|14/12/102/0|
|vidstg|B_MARGINAL_I2_minus_B|m_tIoU|0.053775|[-0.22611729235907568, 0.33366639697915196]|15/13/100/0|
|vidstg|B_MARGINAL_I2_minus_MARGINAL_NO_UPDATE|m_vIoU|-0.460450|[-1.0374529361796212, 0.11655373324907775]|29/48/51/0|
|vidstg|B_MARGINAL_I2_minus_MARGINAL_NO_UPDATE|m_tIoU|-1.892088|[-3.4091697606481093, -0.3750071246797968]|33/57/38/0|

## Actual execution and evidence scope

All256 I1 inputs exact; 240 I2 inputs changed. Actual new captions 240, support prefills 240, short-prefix qualifications 2; 272 exact old caption/support pairs reused. Reuse requires Source/query/actual pixels/fps/formatted Prompt/clock binding. New I2 feedback is acquired from its actual new video, not substituted from an unrelated caption.

Actual new optimized queries 239; all no-information/invalid/truncated/equal cases stay in the full denominator. All actual integrated original-query head readouts and unchanged Source boxes pass ROOT before new GT evaluation. All baseline and no-update scalar checks: 6144. Source probability/ranking independently reconstructed by CDF over 2560069 bins; all aggregate/per-source signed/unknown/t95 checks: 18000.

Two historically absent Source prefix receipts require fresh frozen forwards from identical RGB; their logits, hidden states, boxes, records and Native intervals are exact. No Teacher/fit repeated for that path. Original cached prefixes remain bound to the correct immutable Source. These prefix rebuilds and all actual phase costs are listed separately.

Thirteen previous B no-information cases are not dropped or hand-corrected; current no-information counts follow the same original admission rules, not GT. Numerical Teacher preference does not establish identity/event semantic correctness. The independent512-window semantic panel is not completed by this experiment or engineering pixel inspection.

These are already exposed development sources; no held-out confirmation, noninferiority, universal failure or arbitrary-backbone plug-and-play conclusion. Teacher training-domain overlap remains unresolved. Limits/negative tails/unknowns and original B/T2 results are retained. No automatic follow-up or CURRENT/PAPER promotion.

## Measured decision

Only changing I2 ranking does not establish improvement over original B: HC -.037973pp, paired t95 [-.423562,.347615]; Vid +.059597pp, t95 [-.065033,.184227]. New B versus Frozen is -.170619pp / -.004825pp, both intervals cross zero. No new independent Temporal gain is established. It remains -4.851375pp / -.447805pp below T2 by point estimate; the HC paired interval is entirely negative, while the Vid interval crosses zero.

The no-update readout remains 24.945094% / 10.299485% vIoU, compared with new B 21.480541% / 9.839035%. Better observable-interval probability readout does not automatically transfer to an unchanged representative-state target/optimizer. This run changed only support ranking, as authorized; it does not evaluate a fully marginal training objective. No post-hoc parameter adjustment or additional model experiment follows.

Representative-state prior gap medians rise from .020551 to .192252 HC and .024054 to .178577 Vid; informative targets favor I2 on 48 to 15 HC and 36 to 14 Vid queries. These are saved-state descriptions, not proof that prior strength alone causes the negative result. Candidate ranking uses merged mass, but the conditional training prior remains exactly the original single-state prior; no marginal-target change was slipped into this test.

