# Actual frozen-decoder latent OPD32

Original exposed NH16/NV16; one query per parent, not independent confirmation. E+finalMBR and four cached E-selected DINO references fixed. Original cross-domain EMA TA-STVG weights entirely frozen. No new Source extraction, experts, DINO encoding/conditional decoder, trained modules, network backward or model parameter update. Temporary256D Gaussian mean resets each query. All32 x8 outputs and actual candidate paths sealed before new GT evaluation; every signed result, unknown, missing-reference case and adverse tail retained.

## Question and historical boundary

Old QR pathwise updates could disagree with actual finite decoder response because intermediate references detach, finite steps and optimizer interact. Old output-coordinate OPD also samples boxes that need not be realizable by that decoder. Here every latent draw actually executes all six frozen spatial decoder layers, including reference changes; score-function/likelihood updates avoid the pathwise decoder derivative. This is not the first latent decoder sampling in project history: September19 reconstruction ranking and S1.1 parameter probes remain documented negative/prerequisite results. The new test is this actual256D rollout distribution, two-step mean update and matched fixed/search/mean/shuffle controls.

A Gaussian mean is adapted in standardized query coordinates. sigma=native query_tgt channel RMS/16; a256D draw therefore has norm RMS1/16 of the original query vector norm RMS, not equal norm. Two rounds x16 antithetic draws, original temperature0.1 and fixed step0.5. No scale/temperature/step search. Native zero residual exactly reproduces Source, and the fixed E-retaining center-difference/log-size mapping exactly reproduces E. E remains necessary; this is not evidence that the decoder learned E.

For current mean m and proposal c, base weights v are normalized Gaussian likelihood ratios N(a;m,I)/N(a;c,I); detached teacher weights w are v*exp(R/0.1), normalized. Update m +=0.5*sum((w-v)*a). This finite self-normalized, control-variate estimate is not an exact empirical categorical KL optimum or an unbiased integral estimator. Constant reward/no reference gives exact no-op. Invalid boxes excluded with recorded counts, invalid final means revert only to the prior valid mean; no GT-dependent safety selection.

Reward is mean IoU to the fixed E-selected actual DINO boxes at retained four-frame positions, not target GT and not dense visual feedback. Shuffle cyclically swaps these boxes across their retained positions (fewer than two is explicitly degenerate). Time stays exactly original finalMBR. The decoder can change centers/sizes across all actual frames; this differs from earlier independent-center ROI OPD in both state and feedback, not a single-factor superiority claim.

Fixed and Live have identical first draws and first update. Fixed second proposal remains initial mean0 with the proper current/proposal likelihood ratio; Live second proposal follows its own updated mean and executes fresh actual decoder draws. Deterministic exact state cache hits are reused and physical calls counted. Search evaluates the same32 initial draws plus E, chooses one whole trajectory by teacher reward, E first on ties. MeanControl applies coefficient1 to the same32 initial draws, matching the explicit cumulative2x0.5 coefficient, then actually decodes the new mean; it is not the same optimization path. Endpoint_local reuses the old aligned conditional-DINO center query/local deployment baseline and has a different teacher.

Natural-search and reward-weighted regression are prior decision/update principles, not claimed new algorithms: [NES](https://jmlr.org/papers/v15/wierstra14a.html), [AWR](https://arxiv.org/abs/1910.00177). No new value model or policy network is trained here.

## Actual localization results

| Output | HC full vIoU % | Vid full vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| Source_MBR | 32.385061201 | 11.848986045 | 54.378142928 | 44.792410735 |
| E_MBR | 35.305476650 | 15.755931453 | 56.660359341 | 45.448473353 |
| Endpoint_local | 35.484322238 | 15.819495621 | 56.660359341 | 45.448473353 |
| Latent_search | 35.308116215 | 15.754161660 | 56.669554727 | 45.444533319 |
| Latent_mean_control | 35.305485213 | 15.755930941 | 56.660375567 | 45.448468956 |
| Latent_fixed | 35.305484857 | 15.755930850 | 56.660374078 | 45.448468832 |
| Latent_live | 35.305485101 | 15.755931020 | 56.660375381 | 45.448469070 |
| Latent_shuffle | 35.305491506 | 15.755932281 | 56.660408577 | 45.448470266 |

## hc2: parent paired differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Source_MBR_minus_E_MBR | -2.920415449 [-6.351562506,+1.402472730] | -2.282216413 [-8.218397322,+5.507697890] | 3/12/1 | 7 |
| Endpoint_local_minus_E_MBR | +0.178845587 [-0.015042382,+0.431654025] | +0.000000000 [+0.000000000,+0.000000000] | 8/3/5 | 0 |
| Latent_search_minus_E_MBR | +0.002639565 [-0.007228807,+0.011172149] | +0.009195386 [-0.011486284,+0.035421064] | 9/6/1 | 0 |
| Latent_mean_control_minus_E_MBR | +0.000008562 [-0.000000704,+0.000022231] | +0.000016226 [-0.000001749,+0.000043251] | 10/5/1 | 0 |
| Latent_fixed_minus_E_MBR | +0.000008207 [-0.000000989,+0.000021959] | +0.000014737 [-0.000003620,+0.000042502] | 9/6/1 | 0 |
| Latent_live_minus_E_MBR | +0.000008450 [-0.000000528,+0.000021778] | +0.000016041 [-0.000001993,+0.000042930] | 10/5/1 | 0 |
| Latent_shuffle_minus_E_MBR | +0.000014855 [-0.000000202,+0.000035929] | +0.000049236 [-0.000000827,+0.000132798] | 9/6/1 | 0 |
| Latent_live_minus_Latent_fixed | +0.000000244 [-0.000000544,+0.000000929] | +0.000001304 [-0.000000029,+0.000002840] | 11/4/1 | 0 |
| Latent_live_minus_Latent_shuffle | -0.000006405 [-0.000022691,+0.000005652] | -0.000033196 [-0.000098636,+0.000003837] | 5/10/1 | 0 |
| Latent_live_minus_Latent_search | -0.002631114 [-0.011152351,+0.007227766] | -0.009179346 [-0.035382957,+0.011484273] | 6/9/1 | 0 |
| Latent_live_minus_Latent_mean_control | -0.000000112 [-0.000000905,+0.000000588] | -0.000000185 [-0.000002541,+0.000001560] | 10/5/1 | 0 |
| Latent_live_minus_Endpoint_local | -0.178837137 [-0.431629310,+0.015051017] | +0.000016041 [-0.000001993,+0.000042930] | 6/9/1 | 0 |

## vidstg: parent paired differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Source_MBR_minus_E_MBR | -3.906945408 [-13.362801448,+2.764510670] | -0.656062618 [-15.833502473,+15.255391725] | 2/6/8 | 3 |
| Endpoint_local_minus_E_MBR | +0.063564168 [+0.000000000,+0.139860868] | +0.000000000 [+0.000000000,+0.000000000] | 4/1/11 | 0 |
| Latent_search_minus_E_MBR | -0.001769794 [-0.004731463,+0.000252198] | -0.003940034 [-0.013676600,+0.004437918] | 2/4/10 | 0 |
| Latent_mean_control_minus_E_MBR | -0.000000513 [-0.000001066,-0.000000087] | -0.000004397 [-0.000013169,+0.000001690] | 1/5/10 | 0 |
| Latent_fixed_minus_E_MBR | -0.000000603 [-0.000001210,-0.000000113] | -0.000004521 [-0.000013555,+0.000001894] | 1/5/10 | 0 |
| Latent_live_minus_E_MBR | -0.000000434 [-0.000001073,+0.000000000] | -0.000004283 [-0.000012921,+0.000002094] | 2/4/10 | 0 |
| Latent_shuffle_minus_E_MBR | +0.000000827 [-0.000000311,+0.000002662] | -0.000003087 [-0.000008807,+0.000001567] | 3/3/10 | 0 |
| Latent_live_minus_Latent_fixed | +0.000000169 [-0.000000031,+0.000000413] | +0.000000238 [-0.000000365,+0.000000898] | 4/2/10 | 0 |
| Latent_live_minus_Latent_shuffle | -0.000001261 [-0.000003450,+0.000000162] | -0.000001196 [-0.000008010,+0.000005647] | 2/3/11 | 0 |
| Latent_live_minus_Latent_search | +0.001769360 [-0.000252170,+0.004730861] | +0.003935751 [-0.004438340,+0.013662902] | 4/2/10 | 0 |
| Latent_live_minus_Latent_mean_control | +0.000000079 [-0.000000076,+0.000000230] | +0.000000114 [-0.000000826,+0.000000914] | 4/1/11 | 0 |
| Latent_live_minus_Endpoint_local | -0.063564601 [-0.139861375,+0.000000024] | -0.000004283 [-0.000012921,+0.000002094] | 2/4/10 | 0 |

## Original decision and interpretation

{'status': 'evaluated', 'passed': False, 'rule': 'Both-domain Latent_live minus E complete-v and unobserved actual-s means positive, and Live minus Fixed/Search/MeanControl/Shuffle complete-v positive; CIs retain uncertainty', 'no_network_backward': True}

The prelocked continuation rule fails. Actual rollouts and mean updates executed, but the final policy output is practically unchanged: complete-v changes versus E are about +0.00000845pp HC and -0.000000434pp Vid. These numerical signs are not meaningful task improvement or stability success. No support for on-policy value, superiority to search/mean control or replacing E. The fixed scale and finite reward-tilt combination did not establish effective adaptation. This does not show every latent distribution or OPD implementation is impossible. No posthoc rescue by increasing sigma, temperature contrast, rounds or choosing GT-best steps.

ACTUAL_RESPONSE_PUBLIC.json separately reports the finite initial32 actual-candidate GT diagnostic headroom, whether teacher reward ranks beneficial changes above E, and each saved mean response. That diagnostic is not a global OPD upper bound, and does not choose any deployed step or candidate using GT. AUC and raw reward use their own probability units, not pp; explicit full-v/s differences are pp. Missing annotations remain unknown.

Root independent CPU audit verifies actual cached native-to-E readout algebra, zero exactness, native sigma, every reward and Gaussian importance/control-variate mean, all realized candidate/state cache associations, first-step matching, search/mean controls, no-feedback no-op, independent official interpolation metrics and parent bootstraps. It does not rerun the GPU decoder; the sealed physical-call and frozen-weight receipts cover that execution. Signed counts include float precision sized changes and cannot by themselves establish meaningful benefit.

Two fixed engineering pilots (NH001/NV001) checked replay exactness and nonconstant response without GT before full32. Every true cache miss checked time/GPU limits before and after; full model hashes and pins plus a last guard preceded the global barrier. GPU allowance900s, overnight cap7200s and09:30 cutoff unchanged.

Costs exclude all historical Source/E/expert/cache generation: {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 67.40574708394706, 'fit_episodes': 96, 'round_slots': 192, 'actual_informative_steps': 154, 'actual_unique_trajectory_decoder_calls': 1834, 'actual_PosDecoder_calls': 3668, 'backward_calls': 0, 'old_Source_E_DINO_replay_inputs_cost_excluded': True, 'not_first_online_latency': True}

## All signed full-v changes vs E, pp

| Alias | Domain | Live | Fixed | Search | MeanControl |
|---|---|---:|---:|---:|---:|
| NH001 | hc2 | +0.000001849 | +0.000001858 | +0.007365399 | +0.000003908 |
| NH002 | hc2 | +0.000000804 | +0.000000035 | +0.000363147 | +0.000000035 |
| NH003 | hc2 | -0.000017679 | -0.000018286 | -0.052710008 | -0.000018768 |
| NH004 | hc2 | +0.000089294 | +0.000092342 | +0.039516043 | +0.000091622 |
| NH005 | hc2 | -0.000000607 | -0.000000492 | +0.005125814 | -0.000000768 |
| NH006 | hc2 | +0.000000302 | +0.000000245 | -0.001095184 | +0.000000171 |
| NH007 | hc2 | +0.000001566 | -0.000000745 | +0.005020302 | +0.000000069 |
| NH008 | hc2 | -0.000000856 | -0.000001427 | -0.008659885 | -0.000003212 |
| NH009 | hc2 | +0.000018852 | +0.000018008 | +0.024491662 | +0.000018216 |
| NH010 | hc2 | -0.000002151 | -0.000002228 | -0.002094967 | -0.000002228 |
| NH011 | hc2 | +0.000005713 | +0.000008945 | -0.002928760 | +0.000009802 |
| NH012 | hc2 | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NH013 | hc2 | +0.000031267 | +0.000030721 | +0.018626998 | +0.000032319 |
| NH014 | hc2 | -0.000000972 | -0.000001342 | -0.002815009 | -0.000001342 |
| NH015 | hc2 | +0.000003821 | +0.000001868 | +0.000076117 | +0.000003055 |
| NH016 | hc2 | +0.000004002 | +0.000001802 | +0.011951364 | +0.000004119 |
| NV001 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV002 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV003 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV004 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV005 | vidstg | -0.000000804 | -0.000002285 | -0.020990459 | -0.000001102 |
| NV006 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV007 | vidstg | -0.000001662 | -0.000001884 | -0.002331162 | -0.000001970 |
| NV008 | vidstg | -0.000000398 | -0.000001064 | -0.002451881 | -0.000001064 |
| NV009 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV010 | vidstg | +0.000000192 | +0.000000235 | +0.000499740 | +0.000000192 |
| NV011 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV012 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV013 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV014 | vidstg | -0.000004512 | -0.000004008 | -0.006578358 | -0.000003783 |
| NV015 | vidstg | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| NV016 | vidstg | +0.000000248 | -0.000000640 | +0.003535422 | -0.000000475 |
