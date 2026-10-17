# Original256: exact merged-interval training with fixed evidence

Same HC128/Vid128 exposed development parents. B0 is original Native-only B; B1 changes only I2 physical-interval ranking; B2 keeps B1 I1/I2 and every Teacher response byte-exact and changes only candidate training representation. Frozen/T2/B0/B1/no-update are sealed matched baselines, not rerun models. No external Actionness, Spatial/Full, new Prompt, search or method promotion. Internal frozen Source action gates remain.

## Exact objective

P_theta(I) = sum over every latent state pair merging to physical I of the product of its two offset probabilities.
Candidate Student distribution is P_theta(I_i) / sum_j P_theta(I_j). The frozen Teacher target is proportional to P_N(I_i) exp(r_i/tau), tau=1. Minimize KL(q_T || p_theta^C) plus the unchanged sum of full Native-offset grid trust KLs. Target and candidate intervals are frozen; gradients flow through every compatible latent state.

Exact log-sum-exp over four disjoint boundary-supply partitions is independently checked against explicit latent-pair enumeration/Torch autograd and finite differences, including shared endpoints. No subtraction, clamp or representative-state approximation in training. Zero-init512, LBFGS50/Armijo/history10/tolerances/trust1 and admission/no-op rules remain.

The final adapted readout deliberately remains the original actual FP32 offset MAP then merge decoder, not marginal MAP. Separate Source marginal-MAP no-update control is retained, so its free decoding benefit cannot be credited to adaptation.

## Complete256 results

|Domain|Arm|vIoU %|tIoU %|dense sIoU %|R@.3 %|R@.5 %|
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|30.468750|7.812500|
|hc2|T2|26.331915|49.232269|53.809342|41.406250|14.062500|
|hc2|B0 original|21.518514|39.443767|53.809342|29.687500|7.031250|
|hc2|B1 I2 ranking only|21.480541|39.520097|53.809342|31.250000|6.250000|
|hc2|Marginal no update|24.945094|46.626294|53.809342|38.281250|13.281250|
|hc2|B2 marginal training|21.307892|39.291354|53.809342|31.250000|6.250000|
|vidstg|Frozen|9.843861|32.374371|30.729497|10.156250|1.562500|
|vidstg|T2|10.286840|33.278160|30.729497|10.937500|1.562500|
|vidstg|B0 original|9.779438|32.181130|30.729497|10.156250|1.562500|
|vidstg|B1 I2 ranking only|9.839035|32.234905|30.729497|10.156250|2.343750|
|vidstg|Marginal no update|10.299485|34.126993|30.729497|10.156250|1.562500|
|vidstg|B2 marginal training|9.822672|32.205317|30.729497|10.156250|2.343750|

## Matched source contrasts

Primary B2-B1 isolates the change in training probability representation. B2-B0, Frozen, T2 and no-update are secondary matched comparisons. Full per-source signed deltas and unknowns are in RESULTS.json.

|Domain|Contrast|Metric|Delta pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---|---:|---|---|
|hc2|B_MARGINAL_TRAIN_minus_Frozen|m_vIoU|-0.343268|[-0.7976252221291735, 0.11108990447229233]|10/14/104/0|
|hc2|B_MARGINAL_TRAIN_minus_Frozen|m_tIoU|-0.255481|[-0.9506627494611678, 0.43970074416559257]|12/15/101/0|
|hc2|B_MARGINAL_TRAIN_minus_T2|m_vIoU|-5.024023|[-6.5480766367854555, -3.499969987078974]|24/78/26/0|
|hc2|B_MARGINAL_TRAIN_minus_T2|m_tIoU|-9.940915|[-12.654770083336228, -7.227059844003335]|21/83/24/0|
|hc2|B_MARGINAL_TRAIN_minus_B|m_vIoU|-0.210622|[-0.6054831613208542, 0.18423934438968448]|15/24/89/0|
|hc2|B_MARGINAL_TRAIN_minus_B|m_tIoU|-0.152413|[-0.8010119197996187, 0.49618611736606766]|15/25/88/0|
|hc2|B_MARGINAL_TRAIN_minus_MARGINAL_NO_UPDATE|m_vIoU|-3.637202|[-5.058255956570134, -2.2161479391199927]|23/77/28/0|
|hc2|B_MARGINAL_TRAIN_minus_MARGINAL_NO_UPDATE|m_tIoU|-7.334939|[-9.670572172025844, -4.999306713220355]|23/82/23/0|
|hc2|B_MARGINAL_TRAIN_minus_B_MARGINAL_I2|m_vIoU|-0.172649|[-0.333689933446901, -0.011607334660060503]|2/5/121/0|
|hc2|B_MARGINAL_TRAIN_minus_B_MARGINAL_I2|m_tIoU|-0.228742|[-0.4583000347984907, 0.0008157079464536365]|2/5/121/0|
|vidstg|B_MARGINAL_TRAIN_minus_Frozen|m_vIoU|-0.021188|[-0.14459224106694413, 0.10221569443417876]|5/10/113/0|
|vidstg|B_MARGINAL_TRAIN_minus_Frozen|m_tIoU|-0.169054|[-0.43616720792170294, 0.09805939245869241]|5/11/112/0|
|vidstg|B_MARGINAL_TRAIN_minus_T2|m_vIoU|-0.464168|[-0.9296820263620085, 0.001346228101953062]|26/47/55/0|
|vidstg|B_MARGINAL_TRAIN_minus_T2|m_tIoU|-1.072843|[-2.1746584080975864, 0.028972470439365683]|29/52/47/0|
|vidstg|B_MARGINAL_TRAIN_minus_B|m_vIoU|0.043234|[-0.07657327357633371, 0.16304196164705825]|15/12/101/0|
|vidstg|B_MARGINAL_TRAIN_minus_B|m_tIoU|0.024186|[-0.25123223309282083, 0.29960493076732597]|16/14/98/0|
|vidstg|B_MARGINAL_TRAIN_minus_MARGINAL_NO_UPDATE|m_vIoU|-0.476813|[-1.0555902776427009, 0.10196519714293961]|29/48/51/0|
|vidstg|B_MARGINAL_TRAIN_minus_MARGINAL_NO_UPDATE|m_tIoU|-1.921677|[-3.4425889813778263, -0.4007643108956507]|33/57/38/0|
|vidstg|B_MARGINAL_TRAIN_minus_B_MARGINAL_I2|m_vIoU|-0.016363|[-0.05142468984364547, 0.018698812274427715]|2/1/125/0|
|vidstg|B_MARGINAL_TRAIN_minus_B_MARGINAL_I2|m_tIoU|-0.029588|[-0.08222563716319199, 0.023049230217620915]|2/2/124/0|

## Actual execution and validation

All256 fixed I1/I2/video/query/Source/Teacher bindings pass;512 actual input video hashes checked. Zero new Teacher calls;512 saved description/support pairs reused, whose historical acquisition cost is not zero. New optimized queries 239, accepted optimizer steps 11904, exact no-information queries 17. All256 actual integrated heads and complete Source boxes pass ROOT before GT evaluation; no GT affects selection, target, fitting or acceptance.

Independent saved-aggregate/per-source signed/unknown/t95 checks 26976; old five-arm metric checks 7680 exact; CDF probability bins 2560069, maximum interval-mass error 2.116e-16; explicit-pair Torch objective/gradient accepted-path checks 12160. All no-information cases retain exact zero parameters/Native outputs. All original configuration and optimizer hashes retained.

Two historically missing Source prefix receipts require fresh frozen forwards from identical RGB, with complete logits/hidden/boxes/records/Native outputs exact. No new Source state, Teacher or repeated fitting is claimed for these forwards. Phase costs include this work and are recorded in COST.json; reuse is not a fresh end-to-end speed comparison.

Full common-Native-time sIoU is invariant because this experiment does not update space; dense sIoU on changed temporal support can change. This is Temporal-only; there is no new Spatial/Full result. Mathematical interval mass and numerical semantic rewards are not verified event/identity correctness. Actual pixel views are engineering checks, not additional semantic labels; the separate512-window panel remains partial.

All sources were already exposed development data. No independent confirmation, arbitrary-backbone efficacy, universal failure, noninferiority or causal attribution to external Actionness alone follows. Teacher training-domain overlap remains unresolved; all old negative findings and actual denominator/no-op cases remain. Reaching50 optimizer iterations is not proof of convergence. CURRENT/PAPER unchanged; no automatic next experiment.

Measured primary hc2: B2-B1 -0.172649pp, paired t95 [-0.333689933446901, -0.011607334660060503].
Measured primary vidstg: B2-B1 -0.016363pp, paired t95 [-0.05142468984364547, 0.018698812274427715].

## Measured conclusion

This fixed-evidence change does not restore the T2 benefit. Versus B1, HC vIoU declines by .172649pp (paired t95 [-.333690,-.011607]); Vid declines by .016363pp ([-.051425,.018699]). These are predeclared domain-wise paired t intervals, not a multiplicity-adjusted discovery claim. Relative to Frozen, new B is -.343268pp / -.021188pp with both intervals crossing zero. Relative to T2 it is -5.024023pp / -.464168pp; HC is clearly below within this development cohort and Vid interval narrowly crosses zero.

The exact same marginal no-update control remains 24.945094% / 10.299485% vIoU, compared with new B 21.307892% / 9.822672%. Consistent marginal probability training has been tested, but it does not establish semantic adaptation benefit under the unchanged final Native MAP decoder. This experiment does not isolate a cause for that residual gap or test adapted marginal-MAP decoding. No extra hypothesis, reward adjustment, optimizer search or follow-up experiment is triggered.

