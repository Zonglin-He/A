# Measured interpretation: frozen endpoint rule, new-parent32 retest

The frozen 24-state endpoint rule produced positive complete-vIoU means against E on these new method-selection-disjoint parents, but did not establish added value from sampling. The full center query has higher means in both domains. The sampled endpoint and p-scaled center endpoint are close, with paired intervals crossing zero in both domains. The predefined condition for proceeding to dynamic OPD was not met; no dynamic run or further formula search was started.

## Complete tube results

All values below are vIoU%, using exactly the same final-distribution MBR interval.

| Local output | HC16 | Vid16 |
|---|---:|---:|
| E | 35.305477 | 15.755931 |
| Center endpoint, full replacement if admitted | 35.484322 | 15.819496 |
| 24-state endpoint update, denominator24 | 35.400866 | 15.807407 |
| Center endpoint scaled by sampled p | 35.385810 | 15.805803 |

| Paired complete-vIoU difference, pp | HC mean [95% CI] | Vid mean [95% CI] |
|---|---:|---:|
| 24-state minus E | +0.095389 [+0.011352, +0.198365] | +0.051475 [-0.000809, +0.115246] |
| 24-state minus full center | -0.083456 [-0.269968, +0.065613] | -0.012089 [-0.027305, -0.000147] |
| 24-state minus center with p | +0.015056 [-0.009620, +0.052012] | +0.001604 [-0.000064, +0.004240] |

These are paired parent/query bootstrap intervals,10000 replicates,seed20261018. A positive mean is not proof of stable superiority; a crossing interval is not proof of equivalence. The Vid sampled-versus-center difference is small even though its saved interval lies below zero. HC full-center-minus-E is +0.178846pp[-0.015042,+0.431654], Vid +0.063564pp[0,+0.139861]. This small retest does not establish either center rule as a globally superior replacement.

## Identical admission support

Common admission was fixed before GT: the center must be admitted and at least one of24 states must have admitted feedback. The independently reconstructed counts are HC41/Vid22 common nodes, of which HC30/Vid9 have GT; these known common nodes cover12/6 queries respectively. Missing GT remains unknown, not a failure or a correct localization.

On that identical known support, sampled-minus-p-scaled-center sIoU is HC+0.034310pp[-0.080566,+0.143031], Vid+0.035133[-0.011877,+0.084867]. Thus the small full-tube numerical advantage over Center_p is not convincing evidence that24 states yield better teacher endpoints. Sampled-minus-full-center on common support is HC-2.258122pp[-4.644967,-0.181792], Vid-0.211757[-0.378620,-0.050196]. Full and p-scaled updates have different amplitudes: these comparisons do not isolate endpoint quality independently of displacement.

The p-scaled center is a mechanism control requiring sampled admissions, not a low-cost center baseline. Equal p does not mean equal displacement norm, and p is not a calibrated correctness probability. Current results do not establish conservative scaling as beneficial either: center-p has lower complete means than full center in both domains.

## Coverage, no-op cases and harms

All32 stayed in complete metrics. HC has15/16 E-supported queries; Vid11/16. The six unsupported queries remain exact E with no teacher encoding. Center accepted43/64 planned HC nodes and22/64 Vid nodes; sample-any admission covers46/64 and29/64. These planned denominators retain unsupported and illegal/no-feedback positions. Actual center references were56/40; actual sampled reference queries1128/884, of which318/421 were admitted. Whole sampled-state legality is282/384 and221/384 when all16 planned queries are retained; unsupported states were not executed.

Complete-vIoU positive/negative/unchanged query counts for Center_full, Samples24, Center_p respectively: HC8/3/5,9/2/5,9/2/5; Vid4/1/11,5/1/10,4/1/11. None of the three arms loses more than5pp in complete vIoU. This does not mean no harmful local updates: Vid NV014 observed-node sIoU falls by5.836800pp for full/p-scaled center and5.828350pp for24 states. Its complete-vIoU changes are-0.074634/-0.074478/-0.074634pp. HC NH014 is the largest complete negative for all three arms:-0.373644/-0.193600/-0.140479pp. Both signs and all unknown support remain public.

HC NH008 is the largest complete positive: center+1.538001pp,24 states+0.653289pp,center-p+0.679492pp. The saved leave-one summaries remain positive for24-state-minus-E in each domain, but this is post-hoc sensitivity rather than a changed main denominator or independent confirmation.

## Local nodes and benchmark interpolation

Every unobserved actual decoder-node array and its sIoU stay exactly E. For24-state-minus-E, complete-vIoU contributions observed/unobserved-actual/benchmark-interpolated are HC+0.011973/0/+0.083416pp and Vid+0.011711/0/+0.039764pp. Gains at benchmark-interpolated frames follow official coordinate interpolation; they do not show new visual evidence or learned sparse-to-dense propagation. All tIoUs match exactly. Pretrained weights never changed.

## Actual work and cost

New frozen Source forwards32 restored missing time logits; original Source boxes and native intervals were bit-exact with the saved inputs. No historical Before replay or optimization occurred. E reused original cached detector observations. The new conditional experiment nevertheless executed104 real native DINO image/text encodings, shared between center and sampled decoders; it must not be described as zero DINO computation.

Actual conditional reference queries:center96,sampled2012. Center decoder subtotal0.818031s,sampled subtotal0.890059s; the sample references are batched, so query-count ratios are not latency ratios. Shared native encoding/forward subtotal9.799886s. Teacher-worker GPU allocation72.868338s; Source HC/Vid allocations75.222623/72.115188s; total220.206149s,under the prelocked900s cap. This includes worker load/qualification/IO overhead, but reuses historical Source inputs, detector outputs and E calibration. There was no separately timed center-only first-use deployment, so its end-to-end speedup is not measured here.

## Scope and decision

Parents were chosen by prelocked hashes before cache inventory and inference, excluding original tuning parents and recorded method-selection panels including E-expanded256. HC selected16 of32 eligible parents after205 exclusions; Vid16 of458 after274 exclusions. They are disjoint from those recorded selection pools, but have historical official full-result exposure. This is a new-parent retest for the current mechanism, not a globally unseen blind test.

Retain teacher-endpoint/local update as a research candidate and preserve E+MBR as the reference. The added value of24-state sampling remains unestablished, and the predefined condition for dynamic OPD was not met. Do not expand the current sampled rule or promote it to CURRENT based on these results. No new formula, threshold, learning rate, state representation, cohort or dynamic OPD was run.
