# Trained cross-attention anchor-content dense OPD32

Actual160 external-state fit episodes/320 round slots, five branches, all32 x8outputs sealed before NEW GT. Original exposed NH16/NV16, one query per parent, not independent confirmation. Original frozen cross-domain EMA TA-STVG, E+finalMBR, four cached E-selected DINO references; no new Source/expert/DINO/model/head training/backbone/weight update or network backward. Student is disposable per-actual-frame external2D center state; widths and four planned nodes remain E.

## Feedback mechanism and history

Old native CA attention retention and attention-density/query-negative experiments remain negative evidence, including a same-class Vid distractor subset with native L6 density win0%. This is not native attention integration renamed. Here the round8 all-layer anchor-reference paths separately provide DINO/E-conditioned final-layer query CONTENT. Transfer each retained anchor content to every actual target frame own learned visual KEY content. Split the first half of EACH concatenated head, not the first half of flattened Q/K. Native concat-head scaling sqrt(head_dim), no L2 or new affinity temperature; softmax valid visual keys separately for each anchor/head, then head mean. Explicit anchor position term not transferred; upstream content still includes temporal, position, language and visual history, not pure identity.

Original custom CA has no extra Q/K in-projection; projections belong to the existing frozen trained model. Complete native Q/K with text and padding rebuild actual returned head-mean attention as an engineering control. Derived fields explicitly exclude text/padding; DINO/E and native K must be identical. Query transfer may cross offsets, which is different from within-offset decoder self-attention. Hook capture preserves all32 Source and both E/DINO six-layer outputs exactly versus round8. No new feature encoding. Implementation authority is the [original TA-STVG decoder](https://github.com/HengLan/TA-STVG/blob/main/models/grounding_model/query_decoder.py), not a newly trained critic.

For each candidate, map original normalized coordinates to the valid image-feature rectangle and compute exact fractional-cell coverage. L_a = ROI attention mass / uniform visible ROI mass, then equal arithmetic anchor mean L. Teacher posterior = normalized geometryprior*L, not exp(L/tau). L is an attention-density enrichment working likelihood, not IoU, full-object coverage or identity probability. Uniform/no-reference/all-invisible cases strictly no-op. Partial visibility can favor small high-density visible regions; visibility fractions are retained rather than introducing posthoc filters.

Matched E-anchor query uses exactly the same slots/K/probability/update. OldCosEnrichment uses original coarse cosine propagation with the same full selected anchor set, normalizes each old map to a probability and uses identical enrichment/prior/update; zero-mass old map is neutral uniform. It is not old soft-IoU cosine replication nor a strict one-weight-only comparison, since field construction also differs. The DINO/E-query comparison isolates anchor conditioning more strictly.

Two17 antithetic batches including zero, original alias noise, .25*Esize exploration and .5 per-frame categorical-KL block step, final2, no search. Fixed second batch remains initialE; Live follows actual state and rescores at new positions. Teacher content/keys stay frozen: this is on-policy sampling of a new fixed field, not fresh DINO/query generation each round. Shuffle reorders only unobserved target K/mask fields with fixed anchor queries. Direct pools34initial candidates using the same field/posterior, preserving duplicate zero slots, no GT selection. All five branches actually run, without a target-GT qualification gate skipping OPD.

## Localization results

| Output | HC full vIoU % | Vid full vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Dino_direct | 35.058459 | 15.711696 | 56.071553 | 45.225475 |
| Equery_direct | 35.058459 | 15.711696 | 56.085342 | 45.225475 |
| Dino_fixed | 35.152536 | 15.720173 | 56.379020 | 45.222019 |
| Dino_live | 35.120459 | 15.711156 | 56.301150 | 45.177964 |
| Dino_shuffle | 35.137013 | 15.705187 | 56.359856 | 45.063649 |
| Equery_live | 35.120533 | 15.711114 | 56.300250 | 45.177954 |
| Cosine_like_live | 35.267965 | 15.771820 | 56.635565 | 45.169490 |

## hc2: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Dino_direct_minus_E_MBR | -0.247018 [-0.447888,-0.083854] | -0.588806 [-1.203008,-0.121122] | 2/9/5 | 0 |
| Equery_direct_minus_E_MBR | -0.247018 [-0.447888,-0.083854] | -0.575017 [-1.145868,-0.131254] | 2/9/5 | 0 |
| Dino_fixed_minus_E_MBR | -0.152941 [-0.335853,-0.013055] | -0.281339 [-0.599309,-0.009088] | 6/9/1 | 0 |
| Dino_live_minus_E_MBR | -0.185017 [-0.393486,-0.028157] | -0.359210 [-0.737306,-0.035874] | 6/9/1 | 0 |
| Dino_shuffle_minus_E_MBR | -0.168464 [-0.372614,-0.012977] | -0.300503 [-0.689207,+0.031763] | 6/9/1 | 0 |
| Equery_live_minus_E_MBR | -0.184943 [-0.392988,-0.028258] | -0.360109 [-0.737117,-0.036948] | 6/9/1 | 0 |
| Cosine_like_live_minus_E_MBR | -0.037511 [-0.186773,+0.096369] | -0.024795 [-0.354261,+0.251091] | 8/7/1 | 0 |
| Dino_live_minus_Dino_fixed | -0.032076 [-0.057637,-0.012141] | -0.077870 [-0.144248,-0.024854] | 4/11/1 | 0 |
| Dino_live_minus_Dino_direct | +0.062001 [-0.093106,+0.207997] | +0.229597 [-0.111270,+0.620294] | 10/5/1 | 0 |
| Dino_live_minus_Dino_shuffle | -0.016553 [-0.039930,+0.011241] | -0.058707 [-0.106457,-0.006064] | 6/9/1 | 0 |
| Dino_live_minus_Equery_live | -0.000074 [-0.001106,+0.001338] | +0.000899 [-0.001499,+0.003924] | 6/9/1 | 0 |
| Dino_live_minus_Cosine_like_live | -0.147506 [-0.388868,+0.076015] | -0.334415 [-0.767457,+0.079619] | 3/12/1 | 0 |

## vidstg: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Dino_direct_minus_E_MBR | -0.044236 [-0.135356,+0.007640] | -0.222998 [-0.469433,-0.021363] | 4/3/9 | 0 |
| Equery_direct_minus_E_MBR | -0.044236 [-0.135356,+0.007640] | -0.222998 [-0.469433,-0.021363] | 4/3/9 | 0 |
| Dino_fixed_minus_E_MBR | -0.035759 [-0.095627,+0.009022] | -0.226454 [-0.585599,+0.027229] | 4/3/9 | 0 |
| Dino_live_minus_E_MBR | -0.044775 [-0.115771,+0.008386] | -0.270509 [-0.693310,+0.024923] | 4/3/9 | 0 |
| Dino_shuffle_minus_E_MBR | -0.050745 [-0.135588,+0.012875] | -0.384824 [-0.856290,-0.033892] | 3/4/9 | 0 |
| Equery_live_minus_E_MBR | -0.044818 [-0.116046,+0.008803] | -0.270519 [-0.692789,+0.025690] | 4/3/9 | 0 |
| Cosine_like_live_minus_E_MBR | +0.015889 [-0.028683,+0.064716] | -0.278983 [-1.027621,+0.233500] | 4/2/10 | 0 |
| Dino_live_minus_Dino_fixed | -0.009016 [-0.020986,-0.000064] | -0.044055 [-0.108341,-0.001230] | 3/4/9 | 0 |
| Dino_live_minus_Dino_direct | -0.000539 [-0.079187,+0.074621] | -0.047511 [-0.526831,+0.367525] | 4/3/9 | 0 |
| Dino_live_minus_Dino_shuffle | +0.005970 [-0.009590,+0.023558] | +0.114315 [-0.037824,+0.340690] | 3/4/9 | 0 |
| Dino_live_minus_Equery_live | +0.000043 [-0.000551,+0.000697] | +0.000010 [-0.002275,+0.002155] | 5/2/9 | 0 |
| Dino_live_minus_Cosine_like_live | -0.060664 [-0.154452,+0.016844] | +0.008474 [-0.656221,+0.728602] | 3/4/9 | 0 |

## Unchanged decision

{'status': 'evaluated', 'passed': False, 'rule': 'Both-domain Dino_live minus E full-v and unobserved actual-s positive; Live full-v above Fixed/Direct/Shuffle/EqueryLive/CosineLikeLive; CIs/tails retained', 'actual_OPD': True}

Dino_live minus E full-v is HC-0.185017pp/Vid-0.044775pp; unobserved actual-s-0.359210/-0.270509pp. HC paired intervals are below0, Vid remains uncertain. Live minus Fixed full-v-0.032076/-0.009016pp, both paired intervals below0. This configuration does not support useful dense OPD or on-policy refresh. There are measurable changes, not absent conditioning: supported-parent max content difference averages .22034/.21386 and field max difference .004144/.002474; Live per-node center L2 averages .007070/.006335. DINO minus E-query full-v differences are only-0.000074/+0.000043pp, with intervals crossing0: this interface has not established added DINO-conditioned target information. It does not prove all trained Q/K or attention cannot work.

Matched cosine enrichment minus E full-v-0.037511/+0.015889pp, actual-s-0.024795/-0.278983pp, all uncertain; changing the likelihood/readout alone has not established reliable benefit. Direct DINO readout also fails to improve E. Stop this exact ref-conditioned content-transfer field; no temperature, prior, step or head/layer rescue based on target outcomes.

E-relative initial34candidate diagnostics report good-versus-harmful AUC and whether modifications better than E outrank E. DINO raw AUC HC0.631360/Vid0.614818 (15/9 defined parents), E-query0.631527/0.614244, oldcos0.770802/0.575131. For strong>=5IoUpp candidate pools with teacher, no beneficial posterior candidate outranks E at query-mean rates92.873677%/97.857143%; these are conditional rates over12/10defined parents, not overall query failure rates. Initial candidate GT quality is not applied to moved second-round candidates. AUC improvement alone never substitutes full-tube gain.

Root audit reconstructs actual native full attention, independent per-head content fields and oldcos normalization, ROI likelihood/prior/student categorical probabilities, every320 actual gradient/state/output round, direct controls, widths/planned/no-ref/first-step exactness, official interpolation metrics and parent bootstraps. It does not rerun GPU decoder. Missing GT remains unknown, all samples retained; interpolated frames are not extra visual evidence. Raw attention/feature/qk/target-coordinate caches remain private.

One CPU audit-only recovery is preserved: initial independent oldcos probability check abs<3e-7 failed at largest3.348819e-7. Independent NumPy recipe retained, probability sum promoted to double to match the documented normalizer, and FP32 comparison made scale-aware32eps*maxprob+1e-12, with additional bit-exact regeneration of original cached CPU execution recipe. No model/candidate/prediction/GT metric/decision change. Per-alias maximum error, scale, threshold, ratio and original abs-threshold failure are retained. ROOT_AUDIT_RECOVERY.json records original source hash, failure and limitation; this is not a proof of arbitrary transcendental roundoff.

Incremental costs excluding historical Source/E/DINO/features: {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 37.63177254702896, 'actual_trajectory_decoder_calls': 84, 'actual_PosDecoder_calls': 168, 'actual_fit_episodes': 160, 'round_slots': 320, 'informative_node_rounds': 18940, 'CPU_adapt_seconds': 4.48854617215693, 'backward_calls': 0, 'old_Source_E_DINO_and_feature_cost_excluded': True, 'not_first_online_latency': True}

## Every signed full-v change vs E, pp

| Alias | Target | Live | Fixed | Direct | E-query Live | Cosine-like Live |
|---|---|---:|---:|---:|---:|---:|
| NH001 | hc2 | -0.047003 | -0.038859 | -0.329579 | -0.043748 | +0.026921 |
| NH002 | hc2 | -0.208658 | -0.178040 | -0.602142 | -0.208700 | +0.116308 |
| NH003 | hc2 | -0.153847 | -0.136261 | -0.268518 | -0.153898 | +0.162328 |
| NH004 | hc2 | +0.164872 | +0.150600 | +0.049420 | +0.165895 | +0.479973 |
| NH005 | hc2 | +0.010082 | +0.009332 | +0.000000 | +0.010125 | +0.053110 |
| NH006 | hc2 | +0.039084 | +0.036865 | +0.000000 | +0.030255 | +0.120393 |
| NH007 | hc2 | -0.143813 | -0.126500 | +0.000000 | -0.143957 | -0.654226 |
| NH008 | hc2 | -0.699984 | -0.595688 | -0.012061 | -0.698219 | -0.343782 |
| NH009 | hc2 | -0.288342 | -0.245186 | -0.885149 | -0.287846 | -0.007421 |
| NH010 | hc2 | -1.413357 | -1.230465 | -1.250521 | -1.411247 | -0.012904 |
| NH011 | hc2 | -0.194983 | -0.170737 | -0.152971 | -0.195837 | -0.065374 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | -0.298090 | -0.242217 | +0.088408 | -0.296084 | +0.191814 |
| NH014 | hc2 | +0.227968 | +0.251813 | +0.000000 | +0.228046 | -0.704872 |
| NH015 | hc2 | +0.010912 | +0.036007 | -0.505362 | +0.010161 | +0.103084 |
| NH016 | hc2 | +0.034884 | +0.032281 | -0.083810 | +0.035966 | -0.065533 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +0.015317 | +0.023230 | +0.006037 | +0.015838 | -0.218896 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | -0.125244 | -0.105643 | -0.098614 | -0.125521 | -0.005612 |
| NV008 | vidstg | +0.070254 | +0.067969 | +0.027594 | +0.073240 | +0.281372 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.033343 | +0.029915 | +0.040746 | +0.033214 | +0.000412 |
| NV011 | vidstg | +0.015035 | +0.010513 | +0.002368 | +0.014963 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | -0.441083 | -0.372012 | -0.004966 | -0.441098 | +0.100314 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | -0.284021 | -0.226115 | -0.680939 | -0.287723 | +0.096636 |
