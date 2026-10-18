# Cached cosine-feedback output-distribution OPD32

Actual five-output experiment on exposed NH16/NV16; one query/parent. Independent of the failed ridge teacher; its NO-GO remains unchanged. All3 external-center update branches truly executed, no ranking pre-gate used. No new pretrained model/backbone/DINO call, no historical Before, no QR/LN update.

Fixed E/MBR, only independent per-frame center-state means change outside four planned DINO nodes. E widths/heights remain exact. Same original two17-candidate antithetic noise batches, sigma0.25*E size, feedback tau0.1, per-frame KL step0.5,2 rounds/readstep2. Teacher = Gaussian sampling-prior tilt by the equally averaged all-E-anchor cosine soft-IoU rewards. No ridge probability mixture. Soft evidence is frozen and non-calibrated.

Fixed samples both batches about E. On-policy has EXACT first step with Fixed; second batch follows its updated state. Shuffled_OPD uses the same policy/update but cyclically deranges only unobserved target-frame evidence. Raw-feature-before-matching shuffle equivalence checked on a fixed example/domain, no model rerun. Direct readout picks an actual candidate under the original34-state distribution. No averaging endpoints or post-fit E.

| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Cosine_readout | 34.933405 | 15.747606 | 55.839923 | 44.746585 |
| Fixed_dense | 35.326194 | 15.805854 | 56.759813 | 45.319056 |
| On_policy_dense | 35.325026 | 15.805278 | 56.756321 | 45.303659 |
| Shuffled_OPD | 35.308068 | 15.715043 | 56.603165 | 45.075705 |

## hc2: paired query differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/unchanged | v losses below -5pp |
|---|---:|---:|---:|---:|
| Cosine_readout_minus_E_MBR | -0.372072 [-0.783913,-0.005178] | -0.820437 [-1.704121,-0.052311] | 6/9/1 | 0 |
| Fixed_dense_minus_E_MBR | +0.020718 [-0.129296,+0.178776] | +0.099454 [-0.324809,+0.487544] | 8/7/1 | 0 |
| On_policy_dense_minus_E_MBR | +0.019550 [-0.136579,+0.182447] | +0.095962 [-0.358697,+0.501874] | 8/7/1 | 0 |
| Shuffled_OPD_minus_E_MBR | +0.002591 [-0.221857,+0.206990] | -0.057194 [-0.560983,+0.398310] | 10/5/1 | 0 |
| On_policy_dense_minus_Fixed_dense | -0.001168 [-0.011561,+0.008933] | -0.003492 [-0.036479,+0.020848] | 8/7/1 | 0 |
| On_policy_dense_minus_Shuffled_OPD | +0.016959 [-0.139349,+0.184405] | +0.153156 [-0.103425,+0.465127] | 9/6/1 | 0 |
| On_policy_dense_minus_Cosine_readout | +0.391621 [+0.060429,+0.769291] | +0.916399 [+0.258590,+1.647323] | 8/7/1 | 0 |

Active all-E queries 15/16; all16 remain. tIoU is EXACT fixed MBR for all outputs. Unknown actual-node support stays null.

Complete v contributions by actual observed / actual unobserved / official-interpolated frame support: {'Cosine_readout': {'observed_actual': 0.0, 'unobserved_actual': -0.06654942056088689, 'benchmark_interpolated': -0.3055222150551863}, 'Fixed_dense': {'observed_actual': 0.0, 'unobserved_actual': 0.006616247670802054, 'benchmark_interpolated': 0.014101397629924786}, 'On_policy_dense': {'observed_actual': 0.0, 'unobserved_actual': 0.006627285197194504, 'benchmark_interpolated': 0.012922248086429501}, 'Shuffled_OPD': {'observed_actual': 0.0, 'unobserved_actual': -0.001945719248001584, 'benchmark_interpolated': 0.004536590343373739}}

## vidstg: paired query differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/unchanged | v losses below -5pp |
|---|---:|---:|---:|---:|
| Cosine_readout_minus_E_MBR | -0.008326 [-0.163984,+0.119465] | -0.701889 [-1.940354,+0.118240] | 3/2/11 | 0 |
| Fixed_dense_minus_E_MBR | +0.049923 [-0.020813,+0.163895] | -0.129418 [-0.563622,+0.197100] | 3/3/10 | 0 |
| On_policy_dense_minus_E_MBR | +0.049347 [-0.020258,+0.161412] | -0.144814 [-0.619163,+0.203951] | 3/3/10 | 0 |
| Shuffled_OPD_minus_E_MBR | -0.040889 [-0.154719,+0.067064] | -0.372769 [-0.771688,-0.034628] | 2/4/10 | 0 |
| On_policy_dense_minus_Fixed_dense | -0.000576 [-0.003253,+0.001863] | -0.015397 [-0.058949,+0.012717] | 3/3/10 | 0 |
| On_policy_dense_minus_Shuffled_OPD | +0.090236 [+0.009454,+0.185278] | +0.227954 [+0.000277,+0.480948] | 6/0/10 | 0 |
| On_policy_dense_minus_Cosine_readout | +0.057673 [-0.068064,+0.211329] | +0.557075 [-0.084499,+1.411060] | 3/3/10 | 0 |

Active all-E queries 11/16; all16 remain. tIoU is EXACT fixed MBR for all outputs. Unknown actual-node support stays null.

Complete v contributions by actual observed / actual unobserved / official-interpolated frame support: {'Cosine_readout': {'observed_actual': 0.0, 'unobserved_actual': -0.0027529791928226745, 'benchmark_interpolated': -0.0055727629619469286}, 'Fixed_dense': {'observed_actual': 0.0, 'unobserved_actual': 0.013214277998690856, 'benchmark_interpolated': 0.036708681945276066}, 'On_policy_dense': {'observed_actual': 0.0, 'unobserved_actual': 0.01337187619809242, 'benchmark_interpolated': 0.03597508297667218}, 'Shuffled_OPD': {'observed_actual': 0.0, 'unobserved_actual': -0.007652992599629422, 'benchmark_interpolated': -0.03323584582910396}}

## All signed case changes versus E, pp

| Alias | Domain | Cosine readout | Fixed | On-policy | Shuffled |
|---|---|---:|---:|---:|---:|
| NH001 | hc2 | -0.224444 | +0.110986 | +0.115222 | +0.041571 |
| NH002 | hc2 | -1.729115 | +0.215639 | +0.216875 | +0.181677 |
| NH003 | hc2 | -0.779397 | +0.223956 | +0.224227 | +0.222873 |
| NH004 | hc2 | +0.739808 | +0.610823 | +0.653256 | +0.628715 |
| NH005 | hc2 | +0.155350 | +0.078066 | +0.086377 | +0.203501 |
| NH006 | hc2 | +0.274934 | +0.226510 | +0.236913 | +0.347196 |
| NH007 | hc2 | -2.437831 | -0.436245 | -0.449417 | -0.434250 |
| NH008 | hc2 | +0.234871 | -0.098317 | -0.099810 | -0.919561 |
| NH009 | hc2 | -0.135385 | -0.179356 | -0.197777 | -0.223998 |
| NH010 | hc2 | -1.221404 | -0.308631 | -0.281466 | +0.309192 |
| NH011 | hc2 | +0.201204 | -0.250905 | -0.277526 | -0.866989 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | -0.213402 | -0.271793 | -0.324697 | -0.327653 |
| NH014 | hc2 | +0.037785 | +0.071240 | +0.067206 | +0.073587 |
| NH015 | hc2 | -0.097962 | +0.703907 | +0.711839 | +0.631438 |
| NH016 | hc2 | -0.758159 | -0.364398 | -0.368430 | +0.174156 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +0.192041 | +0.835959 | +0.821782 | +0.529400 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | -0.001608 | -0.126402 | -0.115594 | -0.191189 |
| NV008 | vidstg | +0.573669 | +0.098809 | +0.100922 | -0.391254 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.040746 | +0.075804 | +0.075385 | +0.075313 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.000000 | -0.032790 | -0.029544 | -0.061095 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | -0.938060 | -0.052613 | -0.063399 | -0.615396 |

## Offline common-bin strata

hc2 target_area_bin=Q1: n=2; on-policy−E v -0.026650 [-0.277526,+0.224227]; unobserved actual s +0.399198 [+0.167646,+0.630750].
hc2 target_area_bin=Q2: n=3; on-policy−E v +0.309571 [+0.000000,+0.711839]; unobserved actual s +0.564180 [+0.000000,+1.583901].
hc2 target_area_bin=Q3: n=6; on-policy−E v -0.051492 [-0.336161,+0.274803]; unobserved actual s -0.103312 [-0.606603,+0.429260].
hc2 target_area_bin=Q4: n=5; on-policy−E v -0.050734 [-0.217098,+0.094081]; unobserved actual s -0.067134 [-1.248129,+0.878184].
hc2 motion_bin=Q2: n=8; on-policy−E v -0.032180 [-0.255494,+0.224258]; unobserved actual s -0.180400 [-0.904343,+0.500271].
hc2 motion_bin=Q3: n=3; on-policy−E v -0.077109 [-0.368430,+0.236913]; unobserved actual s -0.087037 [-0.708367,+0.562736].
hc2 motion_bin=Q4: n=5; on-policy−E v +0.160311 [-0.104395,+0.438043]; unobserved actual s +0.647940 [+0.288383,+1.033399].
hc2 active=False: n=1; on-policy−E v +0.000000 [+0.000000,+0.000000]; unobserved actual s +0.000000 [+0.000000,+0.000000].
hc2 active=True: n=15; on-policy−E v +0.020853 [-0.144268,+0.196295]; unobserved actual s +0.102359 [-0.381480,+0.533432].
vidstg target_area_bin=Q1: n=2; on-policy−E v +0.000000 [+0.000000,+0.000000]; unobserved actual s +0.000000 [+0.000000,+0.000000].
vidstg target_area_bin=Q2: n=4; on-policy−E v +0.208442 [-0.031700,+0.616336]; unobserved actual s +0.373011 [-0.032370,+0.919952].
vidstg target_area_bin=Q3: n=5; on-policy−E v +0.000000 [+0.000000,+0.000000]; unobserved actual s -0.727467 [-1.958342,-0.006617].
vidstg target_area_bin=Q4: n=5; on-policy−E v -0.008843 [-0.069356,+0.054644]; unobserved actual s -0.034348 [-0.246108,+0.191098].
vidstg motion_bin=Q1: n=7; on-policy−E v +0.010197 [-0.012662,+0.043252]; unobserved actual s +0.033833 [-0.050945,+0.153134].
vidstg motion_bin=Q2: n=4; on-policy−E v +0.189596 [-0.047549,+0.616336]; unobserved actual s -0.482967 [-2.320297,+0.903767].
vidstg motion_bin=Q3: n=2; on-policy−E v +0.000000 [+0.000000,+0.000000]; unobserved actual s -0.016542 [-0.033083,+0.000000].
vidstg motion_bin=Q4: n=3; on-policy−E v -0.013403 [-0.115594,+0.075385]; unobserved actual s -0.196302 [-0.510524,+0.330185].
vidstg active=False: n=5; on-policy−E v +0.000000 [+0.000000,+0.000000]; unobserved actual s +0.000000 [+0.000000,+0.000000].
vidstg active=True: n=11; on-policy−E v +0.071777 [-0.028954,+0.235483]; unobserved actual s -0.210639 [-0.873999,+0.296772].

All32 five-output predictions/paths sealed before NEW GT evaluation. Cohort and teacher choice have previous target-GT development exposure; not independent confirmation. No GT in candidate construction, feedback, updates, fit-step selection or readout. All32 no-ref/no-update/failures retained. Paired query=parent bootstrap10000 seed20261018, not candidate-level independence.

Per-frame categorical Gaussian KL gradients are independently verified against actual state/output changes. This is output-space on-policy distillation, not pretrained-weight adaptation or an autoregressive GKD replication. Dense visual evidence covers existing actual decoder nodes; interpolated benchmark frames are not additional expert observations.

Cosine gave positive weighted ranking on the previous perturbed pool; that was NOT an E gain. This run measures the actual two-step outputs, not the mean candidate IoU. Sampling-prior and feedback scales are fixed, no sweep/safety gate/third round/second backbone is added. Future expansion is not automatic.

Actual added work: {'new_GPU_seconds': 0, 'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_DINO_calls': 0, 'CPU_construction_and_three_fits_seconds': 3.7611798450816423, 'three_fit_seconds': 0.27550109289586544, 'fit_episodes': 96, 'scheduled_rounds': 192, 'old_Source_E_DINO_and_map_cost_excluded': True}

Existing source/expert/correspondence generation is excluded; CPU cached runtime is not first-use deployment latency.
