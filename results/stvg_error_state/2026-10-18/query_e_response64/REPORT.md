# Query-only E64: actual decoder response and bounded step control

All64 original development queries. Frozen pretrained network, QR256 only, fixed E and final-MBR. No new expert call, source training, cohort, backbone, temporal adaptation or hyperparameter sweep. Historical outputs stay unchanged. This is a matched mechanism experiment, not independent confirmation.

## Complete adaptation results

| Domain | Source | E | Sparse original | Sparse control | Dense original | Dense control | Fixed rollout | Dense Direct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hc2 | 28.004527 | 27.558379 | 25.155821 | 28.106042 | 26.973980 | 28.035710 | 28.037336 | 29.299386 |
| vidstg | 16.616909 | 22.453299 | 16.945818 | 16.700871 | 16.845936 | 16.659394 | 16.678095 | 17.128366 |

All numbers above are full vIoU percent at the predefined last step. Complete t/s, exact unobserved decoder support, positive/negative/neutral and severe-tail counts are in RESULTS_PUBLIC.json. Every path, trial scale, displacement, local factor output and FD value is in PATHS_hc2.json/PATHS_vidstg.json. Unsupported8 Vid queries remain in all denominators.

| Domain / paired comparison | Full vIoU delta pp [parent 95% CI] | Unobserved sIoU delta pp [95% CI] |
|---|---:|---:|
| hc2 / Dense_control-Dense_OPD | +1.061730 [-0.565240, +3.590601] | +1.217866 [-1.488122, +4.959272] |
| hc2 / Dense_control-Source_MBR | +0.031183 [-0.065306, +0.137986] | +0.124625 [-0.044090, +0.339737] |
| hc2 / Dense_control-E_MBR | +0.477331 [-3.273266, +4.513893] | +0.827527 [-5.671342, +8.080351] |
| hc2 / Dense_control-Dense_Direct | -1.263676 [-3.954376, +0.488085] | -2.052578 [-6.156242, +1.440102] |
| hc2 / Dense_control-Fixed_rollout | -0.001626 [-0.088313, +0.090481] | -0.167662 [-0.805092, +0.252753] |
| hc2 / Sparse_control-Sparse_OPD | +2.950221 [-0.298624, +7.669205] | +3.596984 [-0.870148, +9.173655] |
| hc2 / Sparse_control-Source_MBR | +0.101515 [-0.040218, +0.256955] | +0.401869 [-0.044135, +0.987479] |
| vidstg / Dense_control-Dense_OPD | -0.186542 [-0.383961, -0.018225] | -0.096449 [-0.430793, +0.264428] |
| vidstg / Dense_control-Source_MBR | +0.042485 [+0.001548, +0.116854] | +0.029654 [-0.038586, +0.118938] |
| vidstg / Dense_control-E_MBR | -5.793906 [-9.772767, -2.453457] | -9.526070 [-15.376856, -4.403532] |
| vidstg / Dense_control-Dense_Direct | -0.468972 [-0.993124, -0.110151] | -2.065740 [-6.277088, +0.222840] |
| vidstg / Dense_control-Fixed_rollout | -0.018701 [-0.084279, +0.046319] | +0.016325 [-0.111032, +0.147104] |
| vidstg / Sparse_control-Sparse_OPD | -0.244948 [-0.466919, -0.072239] | -0.136990 [-0.454107, +0.215768] |
| vidstg / Sparse_control-Source_MBR | +0.083962 [+0.001702, +0.196076] | +0.126372 [-0.103776, +0.436861] |

## Matched local factorial

192 predetermined Dense states: three nominal early/middle/late pre-rounds per query. Each of eight proposals shares the SAME historical QR/LN state, teacher, actions, weights and original Adam second-moment metric. First compare historical versus current-only first moment and native-detach versus full-chain gradient at equal QR norm; then original versus half norm. Main effects average all other factors, then three states within parent. They describe one-step responses, NOT full adaptation variants. Two extra H008 Sparse states are separate descriptive checks.

| Domain / local factor | vIoU effect pp [parent 95% CI] | E overlap effect pp | Same-rollout surrogate change |
|---|---:|---:|---:|
| hc2 / half_minus_full | -0.003658 [-0.026932, +0.021401] | +0.063451 | -9.242158 |
| hc2 / current_minus_history | -0.006665 [-0.027766, +0.007665] | -0.024596 | -0.016417 |
| hc2 / chain_minus_detach | -0.010175 [-0.035931, +0.009022] | +0.122984 | -8.766801 |
| vidstg / half_minus_full | -0.002015 [-0.028030, +0.033125] | +0.044989 | -0.519276 |
| vidstg / current_minus_history | +0.010684 [-0.002001, +0.033172] | +0.036424 | +0.025925 |
| vidstg / chain_minus_detach | -0.003775 [-0.059057, +0.048008] | +0.202003 | -0.994280 |

The full-chain intervention removes only the internal reference stop-gradient in one decoder instance. Its forward boxes are bit-identical; the vendor source and pretrained tensors are unchanged. Current-only first moment is a direction control: the original second moment and total proposal norm stay fixed, so this is not a new Adam tuning recipe. Factor interactions can remain; no additive causal allocation of all failures is claimed.

## Finite-difference check

| Domain | Direction/h checks | Median absolute native-autograd error | Median absolute chain-autograd error | Native / chain sign disagreements |
|---|---:|---:|---:|---:|
| hc2 | 384 | 0.059604 | 0.000522 | 16 / 2 |
| vidstg | 264 | 0.018813 | 0.000046 | 54 / 2 |

Both fixed central-difference scales (.001 and .0005 QR-unit displacement) are reported, including numerical inconsistency. These are checks of the actual deterministic forward surrogate at that state, not GT gradients or proof that a full-chain optimizer would improve final localization. Sign counts near numerical zero should not be interpreted as substantive errors.

## Bounded native-gradient control

Original Adam proposes the QR step. Up to four actual decoder trials use scales1,.5,.25,.125, accepting the FIRST same-rollout FP64 surrogate nonincrease within tolerance1e-7*(1+abs(before)). No candidate/weight refresh within retries. Accepted scaled steps keep the proposed Adam moments; complete rejection restores QR and all optimizer state. Next original scheduled round resamples around the actual accepted Student. No extra rounds, teacher refresh, GT checkpoint or event-boundary rule. Actual output displacement is measured, not used as an independently tuned quality gate.

| Domain / arm | Accepted | Rejected/restored | Flat reward | Accepted scale counts | Mean final output L2 |
|---|---:|---:|---:|---|---:|
| hc2 / Sparse_control | 663 | 497 | 120 | {"0.125": 90, "0.25": 67, "0.5": 74, "1.0": 432} | 0.390941 |
| hc2 / Dense_control | 294 | 972 | 14 | {"0.125": 59, "0.25": 30, "0.5": 38, "1.0": 167} | 0.027093 |
| vidstg / Sparse_control | 79 | 141 | 20 | {"0.125": 23, "0.25": 14, "0.5": 11, "1.0": 31} | 0.197062 |
| vidstg / Dense_control | 34 | 186 | 20 | {"0.125": 13, "0.25": 13, "0.5": 5, "1.0": 3} | 0.012853 |

## Predeclared cases

| Case | Source | E | Dense original | Dense control | Sparse original | Sparse control |
|---|---:|---:|---:|---:|---:|---:|
| H008 | 60.524579 | 68.840124 | 60.627595 | 60.524579 | 2.238566 | 60.585225 |
| H128 | 44.669483 | 15.142532 | 10.389052 | 44.669483 | 10.439245 | 44.669483 |
| H119 | 28.586077 | 42.340072 | 36.957890 | 28.863489 | 37.158498 | 28.858450 |
| V038 | 56.983348 | 70.356694 | 56.142706 | 57.021167 | 58.468243 | 58.343766 |
| V016 | 18.803103 | 4.450416 | 20.672985 | 18.803103 | 21.627083 | 18.803103 |

H008/H128 test prevention of abrupt failures; H119 is the positive-control preservation check. V038 and V016 retain feedback/task disagreement. A nearly frozen output may protect Source while discarding useful updates: report both the output displacement and task gains, rather than declaring all rejection protective. Proxy improvement and teacher overlap are not sufficient conditions for real task benefit.

## Scope and audit

All64 local/control predictions sealed before new GT evaluation. tIoU is invariant. Independent audit verifies all reported metric states, known unobserved support, original gradients/next outputs, matched QR norms, accepted-step geometry, Adam moment recurrences and exact QR/optimizer restoration. Rejected trial loss records are checked against their rule, not independently re-forwarded on GPU; this limitation is explicit. No new identity/role annotation. The two scientific charts receive actual root review.

This test does not authorize an automatic sweep, full run, new student or teacher. Keep E/MBR as fixed baselines and choose further work from the measured split between direction/response control and feedback alignment; do not promote a development result to CURRENT.

[All64 summary](RESULTS_PUBLIC.json) · [All HC paths](PATHS_hc2.json) · [All Vid paths](PATHS_vidstg.json) · [Fixed cases](CASES.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Controlled paths](CONTROL_PATHS.svg) · [All64 outcomes](ALL64_CONTROL.svg).


## Measured decision

**Do not expand this proxy-nonincrease controller.** It removes the severe HC collapses but mostly suppresses movement: Dense is exactly Source on18/32 HC and24/32 Vid queries (including8 unsupported Vid). Of1486 informative Dense proposals,1158 are rejected/restored and328 accepted. Dense control−Source is only+0.031183/+0.042485pp fullv, versus original Dense−Source−1.030547/+0.229027pp. Dense control−original is+1.061730pp HC (CI[−0.565240,+3.590601]) and−0.186542pp Vid (CI[−0.383961,−0.018225]).

H008 Sparse is protected (2.238566→60.585225%); H128 Dense is protected (10.389052→44.669483%), but H128 accepts0/40 proposals. The positive H119 Dense gain falls from+8.371813pp to+0.277412pp (only1/40 accepted). V016's+1.869882pp gain disappears. Thus a safer output is demonstrated, not an improved ability to use E. HC >5pp v-loss counts versus Source drop Sparse4→0 andDense2→0; Vid had no such cases in either original or controlled arms.

The detach dependency is now an experimentally supported gradient-fidelity issue: median absolute FD/autograd error decreases HC0.059604→0.000522 andVid0.018813→0.000046 with the full chain. This does NOT establish final-task improvement from removing detach: equal-norm local v effects of full-chain, half-step and current-only momentum all have parent intervals crossing zero. They improve different aspects of the local surrogate/derivative, and cannot be turned into an additive causal percentage of failures. Both fixed FD scales and exceptional numerical discrepancies remain public.

No new controller, smaller-scale sweep or full-chain adaptation stream is automatically launched. Keep the original E/MBR/query-only evidence, the Direct and Fixed-rollout baselines, and all negative results. This finite experiment separates local derivative fidelity from finite-step control and real localization; it does not declare the whole OPD route impossible.
