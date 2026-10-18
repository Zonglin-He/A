# Teacher-relative correction estimator: original32 re-evaluation

Single preauthorized change: mu_new=mu+mean(y-a). The former uncentered empirical KL optimum is not claimed. Same population mean at fixed current mean under zero-mean noise; finite-sample paths change. No parameter search. Fixed24 and dynamic8x3 teacher feedback were actually recomputed, not spliced from v1. All32 outputs sealed before new GT; old target exposure acknowledged.

| Domain | E+MBR vIoU% | One correction (exact old)% | Fixed24 repaired% | Dynamic3 repaired% | Old dynamic% |
|---|---:|---:|---:|---:|---:|
| hc2 | 26.861045 | 27.015681 | 26.833244 | 26.759125 | 21.651524 |
| vidstg | 28.187972 | 27.946652 | 28.158499 | 27.996763 | 27.601864 |

| Domain | Contrast | Complete vIoU pp [95% CI] | Unobserved actual sIoU pp [95% CI] | v loss >5pp /16 |
|---|---|---:|---:|---:|
| hc2 | On_policy_distill_minus_E_MBR | -0.101920 [-0.240603,+0.016861] | -0.093509 [-0.222536,+0.014172] | 0/16 |
| hc2 | On_policy_distill_minus_Fixed_sample_distill | -0.074119 [-0.164936,+0.003865] | -0.083744 [-0.170465,-0.010128] | 0/16 |
| hc2 | Fixed_sample_distill_minus_E_MBR | -0.027801 [-0.076025,+0.016603] | -0.009766 [-0.058081,+0.037455] | 0/16 |
| vidstg | On_policy_distill_minus_E_MBR | -0.191209 [-0.420645,+0.004821] | -0.233119 [-0.597286,+0.166779] | 0/16 |
| vidstg | On_policy_distill_minus_Fixed_sample_distill | -0.161735 [-0.348771,-0.002788] | -0.171754 [-0.438727,+0.107166] | 0/16 |
| vidstg | Fixed_sample_distill_minus_E_MBR | -0.029474 [-0.080689,+0.011311] | -0.061365 [-0.185431,+0.067628] | 0/16 |

## Feedback coverage

| Domain/arm | Query with feedback | Legal candidates | Candidates with feedback | Accepted anchor feedback | All-identity queries | E-exact outputs |
|---|---:|---:|---:|---:|---:|---:|
| hc2/Fixed_sample_distill | 5/16 | 306/384 | 8 | 9 | 11 | 11/16 |
| hc2/On_policy_distill | 5/16 | 307/384 | 8 | 9 | 11 | 11/16 |
| vidstg/Fixed_sample_distill | 12/16 | 276/288 | 86 | 121 | 0 | 4/16 |
| vidstg/On_policy_distill | 12/16 | 274/288 | 82 | 110 | 0 | 4/16 |

All-identity feedback now leaves every round mean exactly unchanged. CPU contract covers K1/8/24/32 identity,100 identity rounds, partial feedback, different sample means and zero empirical noise mean. Actual saved paths independently verify the same invariant. No-feedback candidates contribute zero; denominator remains all sampled candidates. No clipping, rejection resampling or extra threshold. Unknown GT is not failure.

## Decision

`stop_this_teacher_interface_no_LR_rescue`. No automatic data expansion or teacher-interface tuning. Return near E alone establishes estimator repair, not added visual-correction or on-policy utility. Original E+MBR and one-correction are exact prior baselines. Three-round time intervals/tIoU remain identical. All new output movement must come from actual admitted correction residuals.

This is exposed HC16/Vid16 development, not independent confirmation. Frozen nearest-native-proposal content initialization may produce very little target-qualified feedback; native replay/clone qualification does not establish arbitrary-box grounding reliability. The four image positions are sparse visual evidence even with many sampled states. No new Qwen/SAM, critic, association training, QR/LN/backbone fit, second backbone or historical Before replay.

## Cost and integrity

Actual GPU lease 67.501s; full anchor encode/native-forward 9.829s; additional decoder 2.668s (includes8 clone qualifications). Existing E generation cost excluded. No new Source call.112 new anchor image/text encodings for28 active queries;4 unsupported Vid queries remain no-op. Not complete first online latency.

Old predictions preserved. Fixed24 and dynamic round1 actions and teacher outputs checked exact against v1 before update; later dynamic states used fresh teacher calls. Root audit independently reconstructs every correction, output and official metric, signed bootstrap intervals and all identity no-op states. Raw images, Query/parent text, GT coordinates, weights and tensors stay private.
