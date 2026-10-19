# Persistent four-anchor reference conditioning32

Original exposed NH16/NV16, one query per parent; not independent confirmation. Frozen cross-domain EMA TA-STVG, fixed E+finalMBR, original four cached DINO/E-selected anchors. No new expert/DINO call, Source capture, head/model training, pretrained parameter update or backward. This is a conditional teacher qualification, NOT an OPD score. Seven outputs and full per-layer actual reference/output traces for all32 sealed before NEW GT evaluation; all signed/unknown/no-reference/adverse cases retained.

## New intervention and matched controls

TA-STVG has an absolute bbox head that overwrites reference boxes after each layer. Initial-only reseeding can therefore disappear. Here only the retained anchor positions are replaced just before sine/reference positional embedding in all six frozen layers; the matched control supplies E boxes at the identical positions and layers. Native content, memory, first-pass/routing, query time, learned heads and all unobserved reference updates remain unchanged. Self-attention operates within each offset; no claim of cross-offset propagation. Missing-anchor offsets must have identical DINO/E paths and exactly E readout.

The primary background is box coordinates conditioning positional attention: [DAB-DETR official](https://github.com/IDEA-Research/DAB-DETR), [paper](https://arxiv.org/abs/2201.12329). Its training results or additive bbox-regression behavior are not imported. Current TA-STVG code defines the actual absolute-head intervention. Earlier raw initial-reference32, object-query aggregation and native-constant ClosureReplay derivative controls are different and remain documented; this is not a claim no prior reference interventions existed.

Only two schedules prelocked, initial1 and all6, without intermediate-layer/strength search. For each schedule run DINO and E references, then read out z(E)+z(D_DINO)-z(D_E), z=(cx,cy,logw,logh). Same condition inputs give exactly E. This removes the common redecoding/reseeding response; it does not guarantee nonlinear causal additivity or correct target identity. Main Initial_delta/Persistent_delta/Persistent_shuffle keep all four planned actual nodes exactly E and change only unobserved nodes. Raw predicted boxes are not overwritten by DINO, so this is attention conditioning rather than forced final-anchor accuracy.

Shuffle cyclically rotates the same retained DINO boxes at fixed positions; fewer than2 retained anchors is explicitly degenerate. Anchor_local directly replaces retained E nodes with their selected DINO boxes, a practical sparse control. Persistent_allnodes applies the same paired readout also to planned nodes as a prelocked sensitivity readout; it is excluded from the gate and cannot rescue a failed primary arm. No-reference queries keep every new method E, remain in the full16 domain denominator.

## Results

| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| Source_MBR | 32.385061 | 11.848986 | 54.378143 | 44.792411 |
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Anchor_local | 35.569101 | 15.871525 | 56.660359 | 45.448473 |
| Initial_delta | 35.305358 | 15.755931 | 56.660268 | 45.448472 |
| Persistent_delta | 35.285166 | 15.752841 | 56.613145 | 45.445427 |
| Persistent_shuffle | 35.310244 | 15.752215 | 56.675644 | 45.440464 |
| Persistent_allnodes | 35.221016 | 15.712438 | 56.613145 | 45.445427 |

## hc2: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Source_MBR_minus_E_MBR | -2.920415 [-6.351563,+1.402473] | -2.282216 [-8.218397,+5.507698] | 3/12/1 | 7 |
| Anchor_local_minus_E_MBR | +0.263624 [+0.031822,+0.531398] | +0.000000 [+0.000000,+0.000000] | 9/3/4 | 0 |
| Initial_delta_minus_E_MBR | -0.000119 [-0.000547,+0.000170] | -0.000091 [-0.001117,+0.000826] | 8/7/1 | 0 |
| Persistent_delta_minus_E_MBR | -0.020310 [-0.061080,+0.001754] | -0.047214 [-0.135298,+0.003555] | 7/8/1 | 0 |
| Persistent_shuffle_minus_E_MBR | +0.004768 [-0.014772,+0.032977] | +0.015285 [-0.025274,+0.070868] | 7/8/1 | 0 |
| Persistent_allnodes_minus_E_MBR | -0.084461 [-0.151015,-0.026968] | -0.047214 [-0.135298,+0.003555] | 2/13/1 | 0 |
| Persistent_delta_minus_Initial_delta | -0.020191 [-0.060618,+0.001725] | -0.047123 [-0.134516,+0.003589] | 7/8/1 | 0 |
| Persistent_delta_minus_Persistent_shuffle | -0.025078 [-0.092497,+0.013976] | -0.062499 [-0.204806,+0.026186] | 6/9/1 | 0 |
| Persistent_delta_minus_Anchor_local | -0.283934 [-0.563807,-0.037552] | -0.047214 [-0.135298,+0.003555] | 4/11/1 | 0 |
| Persistent_delta_minus_Persistent_allnodes | +0.064150 [+0.018158,+0.123467] | +0.000000 [+0.000000,+0.000000] | 11/2/3 | 0 |

## vidstg: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Source_MBR_minus_E_MBR | -3.906945 [-13.362801,+2.764511] | -0.656063 [-15.833502,+15.255392] | 2/6/8 | 3 |
| Anchor_local_minus_E_MBR | +0.115594 [+0.001785,+0.285097] | +0.000000 [+0.000000,+0.000000] | 5/1/10 | 0 |
| Initial_delta_minus_E_MBR | -0.000001 [-0.000003,+0.000001] | -0.000002 [-0.000019,+0.000013] | 3/3/10 | 0 |
| Persistent_delta_minus_E_MBR | -0.003091 [-0.009157,+0.000927] | -0.003046 [-0.012690,+0.003033] | 2/4/10 | 0 |
| Persistent_shuffle_minus_E_MBR | -0.003716 [-0.008024,-0.000328] | -0.008009 [-0.021140,+0.001887] | 0/6/10 | 0 |
| Persistent_allnodes_minus_E_MBR | -0.043493 [-0.121786,+0.018222] | -0.003046 [-0.012690,+0.003033] | 2/4/10 | 0 |
| Persistent_delta_minus_Initial_delta | -0.003090 [-0.009158,+0.000928] | -0.003044 [-0.012688,+0.003029] | 2/4/10 | 0 |
| Persistent_delta_minus_Persistent_shuffle | +0.000625 [-0.003371,+0.005575] | +0.004963 [-0.005292,+0.020180] | 3/3/10 | 0 |
| Persistent_delta_minus_Anchor_local | -0.118685 [-0.290790,-0.001839] | -0.003046 [-0.012690,+0.003033] | 1/5/10 | 0 |
| Persistent_delta_minus_Persistent_allnodes | +0.040402 [-0.020293,+0.114812] | +0.000000 [+0.000000,+0.000000] | 4/2/10 | 0 |

## Original qualification decision

{'status': 'evaluated', 'passed': False, 'rule': 'Both-domain Persistent_delta minus E complete-v and unobserved actual-s positive, Persistent_delta minus Initial_delta and Persistent_shuffle complete-v positive; retain CI and tails, no automatic OPD', 'OPD_not_run': True}

Measured Persistent_delta minus E complete-v is HC-0.020310pp/Vid-0.003091pp, unobserved actual-s-0.047214/-0.003046pp; CIs cross0. Initial-only paired changes are nearly zero. Persistent_allnodes full-v is-0.084461/-0.043493pp, with HC CI strictly negative; it cannot replace the failed primary arm. Direct selected-anchor replacement yields+0.263624/+0.115594pp, with unobserved actual nodes exactly unchanged. These findings separate useful local evidence from an unsupported conditional propagation/readout.

The unchanged prelocked gate determines only whether this specific conditional teacher may enter a separately locked OPD test. It does not prove any OPD gain, even if qualified. If not qualified, retain every result and stop this exact schedule/readout interface without layer/blend/threshold search. Broader frozen conditioning and OPD are not universally disproved. An attention path existing or creating larger movements does not mean the transmitted information is correct.

LAYER_RESPONSE_PUBLIC.json is no-GT arithmetic on saved layer outputs, reporting raw coordinate differences between the paired DINO/E paths by layer. It is not an accuracy measure or a basis for selecting an intermediate layer. Official interpolation is retained in full tube v; interpolated video frames are not additional independent supervision. Missing target annotations remain unknown. All temporal intervals and tIoU are identical.

Root CPU audit checks actual prescribed reference-slot/schedule replacement, nonanchor own-path carry, actual no-injection all-six-layer native equality, raw final assembly, independent center/logsize paired readout, planned-node invariance, no-reference/missing-offset exactness, physical calls, independent official metrics and parent bootstraps. It does not rerun GPU decoder; frozen weights and input hashes plus saved physical execution receipts cover that boundary.

Two fixed engineering pilot aliases checked exactness and response/time without GT. Per-pair time/resources checked before/after; modelhash/fullpins plus final guard precede32barrier. Exception cleanup independently attempts partial/cost persistence and always releases shared GPU lease. No signals to unrelated services.

Incremental costs excluding old Source/E/expert/cache generation: {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 36.945801373105496, 'actual_trajectory_decoder_calls': 196, 'actual_PosDecoder_calls': 392, 'OPD_fit_episodes': 0, 'backward_calls': 0, 'old_Source_E_DINO_cache_cost_excluded': True, 'not_first_online_latency': True}

## Every signed full-v change vs E, pp

| Alias | Domain | Persistent | Initial | Shuffle | Anchor-local | All-nodes auxiliary |
|---|---|---:|---:|---:|---:|---:|
| NH001 | hc2 | -0.313643 | -0.002863 | +0.184815 | +0.654163 | -0.328972 |
| NH002 | hc2 | +0.000612 | +0.000011 | +0.000709 | +0.081737 | -0.027404 |
| NH003 | hc2 | -0.001823 | -0.000324 | +0.004021 | +0.714483 | +0.003901 |
| NH004 | hc2 | -0.003930 | -0.000138 | -0.004874 | +0.571667 | -0.049903 |
| NH005 | hc2 | -0.000315 | +0.000002 | -0.003130 | +0.000000 | -0.000315 |
| NH006 | hc2 | -0.007685 | -0.000073 | -0.007581 | +0.000000 | -0.005155 |
| NH007 | hc2 | +0.010764 | +0.000040 | +0.011342 | -0.200347 | -0.136923 |
| NH008 | hc2 | -0.023224 | +0.000853 | -0.018311 | +1.522025 | -0.405592 |
| NH009 | hc2 | +0.002276 | -0.000043 | +0.002638 | +0.000000 | +0.002276 |
| NH010 | hc2 | +0.002377 | +0.000036 | -0.020871 | -0.181379 | -0.027979 |
| NH011 | hc2 | +0.004222 | -0.000039 | -0.001175 | +1.140116 | -0.272639 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.006657 | +0.000539 | +0.008229 | +0.063870 | -0.007769 |
| NH014 | hc2 | +0.002384 | -0.000037 | -0.008774 | -0.451039 | -0.002354 |
| NH015 | hc2 | -0.003296 | +0.000032 | +0.000299 | +0.255415 | -0.058805 |
| NH016 | hc2 | -0.000340 | +0.000101 | -0.071056 | +0.047272 | -0.033740 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | -0.016431 | +0.000008 | -0.008301 | +1.184186 | +0.164350 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | -0.000004 | -0.000000 | -0.000077 | +0.060330 | +0.000597 |
| NV008 | vidstg | +0.007469 | +0.000002 | -0.023941 | +0.299587 | -0.430129 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | -0.000430 | +0.000000 | -0.000422 | +0.014280 | -0.001145 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.000332 | -0.000013 | -0.002628 | -0.075246 | -0.037143 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | -0.040391 | -0.000010 | -0.024089 | +0.366366 | -0.392424 |
