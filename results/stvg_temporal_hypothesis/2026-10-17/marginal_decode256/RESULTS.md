# Original256: final marginal-MAP decoding with exact saved B2 parameters

Same HC128/Vid128 exposed development sources; same cross-domain Source, I1/I2, actual evidence, frozen R1 rewards, prior/target, zero-init512 optimizer/trust/temperature and complete B2 parameter trajectory. Only the final decoder changes. No new Teacher, Student fit, GPU forward, Source capture, pixel decoding, Spatial/Full run, search or CURRENT/PAPER promotion.

## One changed factor

For each offset, normalize legal start<end joint probabilities from the actual integrated FP32 endpoint logits. Use float64 sum/normalization, identical to the Source marginal no-update control. Sum every independent state pair mapping to a physical merged interval, then choose the global physical interval MAP. The search is not restricted to I1/I2. Sorted physical row-major ties are fixed; no clamp/top-k approximation/GT selection.

All256 saved logits are bound by SHA to B2 fitted parameters and the prior bit-exact actual full-head GPU ROOT. This run reuses those verified full outputs and executes new CPU decoding; it does not claim a new model forward or fit. Source spatial boxes are unchanged. Independently reproduce every Source marginal control and audit both Source and adapted full probability tables with CDF/inclusion-exclusion against explicit outer-product sums. Synthetic Python pair enumeration, shared endpoints, ties, zero-change and NaN rejection contracts also pass.

All17 no-information queries retain exact zero parameters. Apply the same decoder to them: their output equals Source marginal no-update, not necessarily old Frozen Native MAP. Any decoder-only changes at zero parameters are not credited to learning.

## Complete256 results

|Domain|Arm|vIoU %|tIoU %|dense sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|30.468750|7.812500|
|hc2|T2|26.331915|49.232269|53.809342|41.406250|14.062500|
|hc2|B0 original|21.518514|39.443767|53.809342|29.687500|7.031250|
|hc2|B1 I2-only ranking|21.480541|39.520097|53.809342|31.250000|6.250000|
|hc2|B2 marginal training / old decoder|21.307892|39.291354|53.809342|31.250000|6.250000|
|hc2|Source marginal MAP / no update|24.945094|46.626294|53.809342|38.281250|13.281250|
|hc2|B3 B2 parameters / marginal MAP|24.832876|46.553780|53.809342|38.281250|12.500000|
|vidstg|Frozen|9.843861|32.374371|30.729497|10.156250|1.562500|
|vidstg|T2|10.286840|33.278160|30.729497|10.937500|1.562500|
|vidstg|B0 original|9.779438|32.181130|30.729497|10.156250|1.562500|
|vidstg|B1 I2-only ranking|9.839035|32.234905|30.729497|10.156250|2.343750|
|vidstg|B2 marginal training / old decoder|9.822672|32.205317|30.729497|10.156250|2.343750|
|vidstg|Source marginal MAP / no update|10.299485|34.126993|30.729497|10.156250|1.562500|
|vidstg|B3 B2 parameters / marginal MAP|10.290674|34.144786|30.729497|10.937500|1.562500|

## Matched contrasts

Primary B3-minus-Source-marginal-no-update measures the incremental effect of adaptation under the same decoder. B3-minus-B2 measures decoder change with theta exact. Other contrasts are secondary; nominal paired source t95 intervals are not multiplicity-adjusted. Every source signed delta and unknown is retained in RESULTS.json.

|Domain|Contrast|Metric|Delta pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---|---:|---|---|
|hc2|B_MARGINAL_DECODE_minus_Frozen|m_vIoU|3.181717|[1.7662366175355657, 4.597197026739087]|77/23/28/0|
|hc2|B_MARGINAL_DECODE_minus_Frozen|m_tIoU|7.006944|[4.690304978670677, 9.323583473098356]|84/21/23/0|
|hc2|B_MARGINAL_DECODE_minus_T2|m_vIoU|-1.499039|[-2.705755728449252, -0.29232193348364377]|44/58/26/0|
|hc2|B_MARGINAL_DECODE_minus_T2|m_tIoU|-2.678490|[-4.762841625761649, -0.5941378445133041]|44/59/25/0|
|hc2|B_MARGINAL_DECODE_minus_B|m_vIoU|3.314363|[1.8813725022501433, 4.747352642750221]|74/24/30/0|
|hc2|B_MARGINAL_DECODE_minus_B|m_tIoU|7.110012|[4.766039261010114, 9.453985393620943]|81/22/25/0|
|hc2|B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE|m_vIoU|-0.112217|[-0.3711482686246771, 0.1467133348660843]|17/12/99/0|
|hc2|B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE|m_tIoU|-0.072514|[-0.5458671753654112, 0.40083874718382206]|17/11/100/0|
|hc2|B_MARGINAL_DECODE_minus_B_MARGINAL_I2|m_vIoU|3.352336|[1.983674100283256, 4.720997593541316]|74/20/34/0|
|hc2|B_MARGINAL_DECODE_minus_B_MARGINAL_I2|m_tIoU|7.033683|[4.782101807762087, 9.285264322450486]|80/19/29/0|
|hc2|B_MARGINAL_DECODE_minus_B_MARGINAL_TRAIN|m_vIoU|3.524984|[2.1395159352134816, 4.910453026718052]|74/20/34/0|
|hc2|B_MARGINAL_DECODE_minus_B_MARGINAL_TRAIN|m_tIoU|7.262425|[4.993113599019866, 9.531736858044743]|80/19/29/0|
|vidstg|B_MARGINAL_DECODE_minus_Frozen|m_vIoU|0.446813|[-0.10989791456910514, 1.0035248276538304]|46/29/53/0|
|vidstg|B_MARGINAL_DECODE_minus_Frozen|m_tIoU|1.770416|[0.30592125157697314, 3.234909840827918]|55/33/40/0|
|vidstg|B_MARGINAL_DECODE_minus_T2|m_vIoU|0.003834|[-0.33973170143771525, 0.3473993628951506]|37/37/54/0|
|vidstg|B_MARGINAL_DECODE_minus_T2|m_tIoU|0.866626|[-0.18101134957657916, 1.9142643197862603]|47/41/40/0|
|vidstg|B_MARGINAL_DECODE_minus_B|m_vIoU|0.511236|[-0.036287002410260505, 1.0587591501984757]|48/27/53/0|
|vidstg|B_MARGINAL_DECODE_minus_B|m_tIoU|1.963656|[0.5095517344960141, 3.4177598710463926]|58/32/38/0|
|vidstg|B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE|m_vIoU|-0.008811|[-0.12590347414325234, 0.1082818533609819]|11/8/109/0|
|vidstg|B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE|m_tIoU|0.017793|[-0.3257302908934231, 0.3613159064878477]|12/9/107/0|
|vidstg|B_MARGINAL_DECODE_minus_B_MARGINAL_I2|m_vIoU|0.451639|[-0.09214016318431661, 0.9954177453325896]|46/26/56/0|
|vidstg|B_MARGINAL_DECODE_minus_B_MARGINAL_I2|m_tIoU|1.909881|[0.3988158920177227, 3.420946608904608]|55/30/43/0|
|vidstg|B_MARGINAL_DECODE_minus_B_MARGINAL_TRAIN|m_vIoU|0.468002|[-0.07764957810463685, 1.0136530378221276]|45/26/57/0|
|vidstg|B_MARGINAL_DECODE_minus_B_MARGINAL_TRAIN|m_tIoU|1.939469|[0.42457856939896255, 3.4543603384689394]|54/30/44/0|

## Validation, cost and scope

All256 new predictions/Source controls and independent probability ROOT sealed before this evaluation; no GT enters decoding, fitting or selection. Independent scalar/source signed/unknown/t95 checks 37744, old six-arm scalar exact checks 9216. Both Source and adapted CDF probability tables cover 5120138 bins; maximum mass error 2.982e-16. Saved B2 fit hashes exact256;17 zero-parameter predictions exactly match the same-decoder no-update control.

Zero new Teacher calls/Student fits/optimizer steps/GPU forwards. Reuse239 previous optimized queries/11904 accepted steps plus17 zero-parameter queries and512 historical caption/support pairs. Historical evidence and fitting were not free. New CPU decode and scoring costs are in COST.json; not a fresh full end-to-end speed comparison.

These are exposed development parents, including previous iterations. No independent confirmation, noninferiority, arbitrary-backbone effectiveness or universal mechanism conclusion follows. Teacher target-training overlap remains unresolved. Numeric semantic rewards and engineering pixel inspection are not event/identity/role labels; the512-window semantic panel remains partial44 model-assisted/468 pending. Source dense/common-Native-time geometry is unchanged. Preserve all earlier negative results and raw artifacts. No automatic follow-up.

## Measured comparisons

hc2 B_MARGINAL_DECODE_minus_Frozen: +3.181717pp, paired source t95 [1.7662366175355657, 4.597197026739087]; 77 positive / 23 negative / 28 neutral / 0 unknown.
hc2 B_MARGINAL_DECODE_minus_T2: -1.499039pp, paired source t95 [-2.705755728449252, -0.29232193348364377]; 44 positive / 58 negative / 26 neutral / 0 unknown.
hc2 B_MARGINAL_DECODE_minus_B: +3.314363pp, paired source t95 [1.8813725022501433, 4.747352642750221]; 74 positive / 24 negative / 30 neutral / 0 unknown.
hc2 B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE: -0.112217pp, paired source t95 [-0.3711482686246771, 0.1467133348660843]; 17 positive / 12 negative / 99 neutral / 0 unknown.
hc2 B_MARGINAL_DECODE_minus_B_MARGINAL_I2: +3.352336pp, paired source t95 [1.983674100283256, 4.720997593541316]; 74 positive / 20 negative / 34 neutral / 0 unknown.
hc2 B_MARGINAL_DECODE_minus_B_MARGINAL_TRAIN: +3.524984pp, paired source t95 [2.1395159352134816, 4.910453026718052]; 74 positive / 20 negative / 34 neutral / 0 unknown.
vidstg B_MARGINAL_DECODE_minus_Frozen: +0.446813pp, paired source t95 [-0.10989791456910514, 1.0035248276538304]; 46 positive / 29 negative / 53 neutral / 0 unknown.
vidstg B_MARGINAL_DECODE_minus_T2: +0.003834pp, paired source t95 [-0.33973170143771525, 0.3473993628951506]; 37 positive / 37 negative / 54 neutral / 0 unknown.
vidstg B_MARGINAL_DECODE_minus_B: +0.511236pp, paired source t95 [-0.036287002410260505, 1.0587591501984757]; 48 positive / 27 negative / 53 neutral / 0 unknown.
vidstg B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE: -0.008811pp, paired source t95 [-0.12590347414325234, 0.1082818533609819]; 11 positive / 8 negative / 109 neutral / 0 unknown.
vidstg B_MARGINAL_DECODE_minus_B_MARGINAL_I2: +0.451639pp, paired source t95 [-0.09214016318431661, 0.9954177453325896]; 46 positive / 26 negative / 56 neutral / 0 unknown.
vidstg B_MARGINAL_DECODE_minus_B_MARGINAL_TRAIN: +0.468002pp, paired source t95 [-0.07764957810463685, 1.0136530378221276]; 45 positive / 26 negative / 57 neutral / 0 unknown.

## Measured conclusion

B3 vIoU is24.832876% HC /10.290674% Vid. Versus unchanged parameters with the old Native decoder (B2), it gains+3.524984pp /+.468002pp; HC paired t95 [2.139516,4.910453], Vid [-.077650,1.013653]. Final readout accounts for a large recovery in point estimate.

The primary matched-decoder adaptation contrast B3-minus-Source-marginal-no-update is-.112217pp HC (t95 [-.371148,.146713]) and-.008811pp Vid ([-.125903,.108282]). Neither establishes incremental parameter-adaptation benefit; neither proves equivalence or universal harm. Do not credit the Source marginal decoding gain to Teacher feedback or parameter learning.

B3-minus-Frozen is+3.181717pp HC (t95 [1.766237,4.597197]) /+.446813pp Vid ([-.109898,1.003525]). B3-minus-T2 is-1.499039pp HC ([-2.705756,-.292322]) /+.003834pp Vid ([-.339732,.347399]). HC still trails T2 in this development cohort; Vid point means are close, without an equivalence/noninferiority claim. No result-driven next mechanism or tuning is launched.

