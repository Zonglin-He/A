# Coarse bidirectional region feedback: actual OPD32

Same exposed NH16/NV16 parents, one query per parent, unchanged E, final-MBR, four DINO planned nodes, widths and heights. This round ran96 actual temporary center-state fits/192 rounds: fixed, live on-policy, and shuffled live. No new GPU, Source capture, DINO call, pretrained parameter update, head or model training. Fine layer3 qualification failed its original gate and is not reused here.

Reverse affinities are independently normalized in the reverse direction from the same cached coarse cosine logits, not the transpose of forward attention probabilities. Each current candidate region is projected back to the original anchor grid. Per-anchor reward is sqrt(forward soft-IoU * reverse soft-IoU), then anchors are equally averaged. This finite alternative tests regional mutual support; it does not establish a calibrated target probability or guaranteed correspondence.

Fixed and live share identical first-round candidates/teacher/gradient/state. On the second round live candidates follow the actual updated student; current region masks, backprojections and both-direction rewards are recomputed. Fixed retains E as sampling center. Shuffling moves the target evidence and validity together, keeping anchor identities fixed. Four old forward-only outputs are reused exactly as matched controls.

Original configuration retained: affinity .07, reward temperature .1, Gaussian prior and center scale .25*E width/height, categorical-KL step .5, two rounds,17 candidates per round (center+8 antithetic pairs); final step2 only. State independent for every query/unobserved actual node, reset at query boundary. No-reference/informative-false cases are exact no-op. All nine outputs and paths sealed for all32 before NEW GT. Historical target development exposure informed the hypothesis and is disclosed; this is not independent confirmation.

| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Bidir_readout | 34.901446 | 15.719470 | 55.791271 | 44.626448 |
| Fixed_dense | 35.289817 | 15.829538 | 56.705508 | 45.381590 |
| On_policy_dense | 35.286059 | 15.828910 | 56.697973 | 45.366339 |
| Shuffled_OPD | 35.274275 | 15.738076 | 56.572591 | 45.110875 |
| Forward_readout | 34.933405 | 15.747606 | 55.839923 | 44.746585 |
| Forward_Fixed | 35.326194 | 15.805854 | 56.759813 | 45.319056 |
| Forward_On | 35.325026 | 15.805278 | 56.756321 | 45.303659 |
| Forward_Shuffled | 35.308068 | 15.715043 | 56.603165 | 45.075705 |

## hc2: paired parent differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Bidir_readout_minus_E_MBR | -0.404031 [-0.812486,-0.050916] | -0.869088 [-1.698750,-0.164415] | 4/11/1 | 0 |
| Fixed_dense_minus_E_MBR | -0.015660 [-0.178375,+0.153319] | +0.045149 [-0.363608,+0.435683] | 7/8/1 | 0 |
| On_policy_dense_minus_E_MBR | -0.019418 [-0.189827,+0.157177] | +0.037614 [-0.401021,+0.452265] | 7/8/1 | 0 |
| Shuffled_OPD_minus_E_MBR | -0.031202 [-0.262513,+0.181437] | -0.087768 [-0.590620,+0.376494] | 9/6/1 | 0 |
| Forward_readout_minus_E_MBR | -0.372072 [-0.783913,-0.005178] | -0.820437 [-1.704121,-0.052311] | 6/9/1 | 0 |
| Forward_Fixed_minus_E_MBR | +0.020718 [-0.129296,+0.178776] | +0.099454 [-0.324809,+0.487544] | 8/7/1 | 0 |
| Forward_On_minus_E_MBR | +0.019550 [-0.136579,+0.182447] | +0.095962 [-0.358697,+0.501874] | 8/7/1 | 0 |
| Forward_Shuffled_minus_E_MBR | +0.002591 [-0.221857,+0.206990] | -0.057194 [-0.560983,+0.398310] | 10/5/1 | 0 |
| On_policy_dense_minus_Fixed_dense | -0.003758 [-0.014442,+0.007421] | -0.007535 [-0.045209,+0.020909] | 8/7/1 | 0 |
| On_policy_dense_minus_Shuffled_OPD | +0.011784 [-0.126257,+0.162886] | +0.125382 [-0.104297,+0.406491] | 8/7/1 | 0 |
| On_policy_dense_minus_Bidir_readout | +0.384613 [+0.082617,+0.730035] | +0.906702 [+0.344355,+1.554452] | 10/5/1 | 0 |
| On_policy_dense_minus_Forward_On | -0.038967 [-0.099696,+0.017449] | -0.058348 [-0.198926,+0.085455] | 6/9/1 | 0 |
| On_policy_dense_minus_Forward_Fixed | -0.040135 [-0.102638,+0.019287] | -0.061840 [-0.200842,+0.072825] | 5/10/1 | 0 |
| On_policy_dense_minus_Forward_readout | +0.352654 [+0.030563,+0.719272] | +0.858050 [+0.247640,+1.523624] | 8/7/1 | 0 |

Active parents 15/16; all16 remain in complete-tube denominator. Missing actual-frame GT is unknown. Strata, every signed case and interpolation contributions are in RESULTS_PUBLIC.json.

## vidstg: paired parent differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Bidir_readout_minus_E_MBR | -0.036461 [-0.262901,+0.126754] | -0.822025 [-2.253909,+0.131350] | 4/2/10 | 0 |
| Fixed_dense_minus_E_MBR | +0.073607 [+0.006489,+0.179285] | -0.066884 [-0.723434,+0.377096] | 5/1/10 | 0 |
| On_policy_dense_minus_E_MBR | +0.072978 [+0.006144,+0.176633] | -0.082135 [-0.795012,+0.395809] | 5/1/10 | 0 |
| Shuffled_OPD_minus_E_MBR | -0.017856 [-0.125567,+0.081864] | -0.337598 [-0.953742,+0.104795] | 3/3/10 | 0 |
| Forward_readout_minus_E_MBR | -0.008326 [-0.163984,+0.119465] | -0.701889 [-1.940354,+0.118240] | 3/2/11 | 0 |
| Forward_Fixed_minus_E_MBR | +0.049923 [-0.020813,+0.163895] | -0.129418 [-0.563622,+0.197100] | 3/3/10 | 0 |
| Forward_On_minus_E_MBR | +0.049347 [-0.020258,+0.161412] | -0.144814 [-0.619163,+0.203951] | 3/3/10 | 0 |
| Forward_Shuffled_minus_E_MBR | -0.040889 [-0.154719,+0.067064] | -0.372769 [-0.771688,-0.034628] | 2/4/10 | 0 |
| On_policy_dense_minus_Fixed_dense | -0.000629 [-0.004638,+0.002873] | -0.015251 [-0.074431,+0.023762] | 3/3/10 | 0 |
| On_policy_dense_minus_Shuffled_OPD | +0.090834 [+0.012843,+0.185289] | +0.255464 [+0.018199,+0.513384] | 5/1/10 | 0 |
| On_policy_dense_minus_Bidir_readout | +0.109439 [-0.037437,+0.319609] | +0.739891 [+0.091895,+1.553432] | 5/2/9 | 0 |
| On_policy_dense_minus_Forward_On | +0.023631 [-0.003031,+0.055748] | +0.062680 [-0.216080,+0.292685] | 5/1/10 | 0 |
| On_policy_dense_minus_Forward_Fixed | +0.023055 [-0.004600,+0.056345] | +0.047283 [-0.273168,+0.301973] | 5/1/10 | 0 |
| On_policy_dense_minus_Forward_readout | +0.081304 [-0.036754,+0.227118] | +0.619754 [+0.082942,+1.255928] | 5/1/10 | 0 |

Active parents 11/16; all16 remain in complete-tube denominator. Missing actual-frame GT is unknown. Strata, every signed case and interpolation contributions are in RESULTS_PUBLIC.json.

## Decision and limits

The fixed eligibility rule failed: live-minus-E complete v is negative in HC, and unobserved actual s is negative in Vid. Live-minus-fixed point estimates are negative on all four primary cells, with intervals crossing zero. Stop this exact coarse bidirectional-feedback/two-step policy interface, without temperature/step/layer scanning or expansion. This is an actual OPD result, not a claim that OPD was unimplemented.

Retain positive evidence: Vid live-minus-E complete v is +0.072978pp [+.006144,+.176633]; live-minus-shuffled v and actual s are positive with intervals above zero. Real video evidence has information, but does not establish an increment over E and fixed feedback across both domains. Actual-node v contribution +.018051pp and official-interpolation contribution +.054927pp explain that positive complete-v mean; fixed-support actual s uses different weighting and is still negative. Official interpolation is not independent visual supervision.

Cache diagnostics retain all initial34 exact candidates and old saved candidate GT quality. Forward, reverse and symmetric E-relative ranks, reward contrast/ESS/KL and sigma-normalized actual output movement are in SEALED_CACHE_REVIEW.json. New live second-round candidate GT quality remains unknown; initial labels are never attached to changed candidates. These diagnostics are not new deployed arms. FEEDBACK_AND_RESPONSE field/support corrections are explicitly recorded in ANALYSIS_FIELD_NOTES.json.

Severe signed cases are preserved: NV005 live full-v +.752533pp/actual-s +1.338317pp; NV001 full-v unchanged/actual-s -4.663563pp; NH004 full-v +.765563pp and NH007 -.684193pp. These are geometric outcomes, not new identity annotations. Parent10000 bootstrap seed20261018, positive/negative/neutral and severe tails are retained; no case removed from main means.

CPU independent audit checks reverse affinities, every both-direction reward/posterior,192 actual KL gradient/state/output updates, reused old outputs, fixed sizes/anchors/time, no-reference no-op and1440 official metric scalars. Passed without a GPU/model rerun. A preflight property tolerance adjustment and a post-seal scoring-schema repair are recorded separately: original code/runtime/failures retained, prediction bytes never changed or recomputed. The initial scorer KeyError occurred before new GT access; only CPU scoring resumed. Independent audit sub-ulp comparison tolerance was corrected, original audit retained.

Costs (old E/DINO/features excluded): {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_GPU_seconds': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_pretrained_weights_updated': 0, 'actual_fit_episodes': 96, 'actual_update_rounds': 192, 'CPU_construction_and_three_fits_seconds': 3.7589311241172254, 'fit_seconds': 0.48846366186626256, 'old_Source_E_DINO_feature_generation_cost_excluded': True, 'not_first_deployment_latency': True}

No production promotion. No cross-backbone claim, no new expert/training. A single GPT-6.1 Sol reviewer checked results and historical selection routes; root retains decision authority. General many-to-one matching and OPD are not declared impossible by this configuration.
