# Ordered ROI layout dense OPD32

Historically exposed NH16/NV16; one query per parent. Original frozen cross-domain EMA TA-STVG, E-selected four-frame DINO cached references and final-MBR. Original coarse input_proj features only. No new experts, Source/model forward, learned head, training or GPU.128 actual fits and256 actual updates ran before all32 outputs were sealed and NEW GT evaluation.

Fixed3x3 corresponding-bin cosine versus matched unordered all9x9-bin cosine; per-patch L2 then equal anchor means. Padding activations are masked before sampling; valid rectangle mapping and partial valid-bin mass saved. Boundary normalization supports differ for the two aggregations, a material limitation. There is no new cross-model embedding. SiamFC motivates structured comparison but its trained embedding is not imported: [paper](https://arxiv.org/abs/1606.09549), [official code](https://github.com/bertinetto/siamese-fc). Single-anchor fixed-size old soft-IoU is monotonic in ROI integral; its denominator is not established as a defect.

Actual current-query center state resets to zero; original sizes, four planned locations and temporal intervals remain unchanged. Two17-proposal rounds, .25 E-size scale, .1 reward temperature, .5 categorical-KL step, final step2; no target parameter search. Fixed reuses initial proposal center; Live second round follows updated state; Shuffle permutes actual target maps/valid masks with anchor templates fixed. Unordered Live is a matched alternative teacher. Old Forward outputs are reused, not rerun.

| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Layout_readout | 35.225311 | 15.831327 | 56.435078 | 43.247752 |
| Unordered_readout | 34.515136 | 15.251503 | 54.592460 | 41.767896 |
| Fixed_dense | 35.382708 | 15.866958 | 56.813242 | 44.852369 |
| On_policy_dense | 35.384641 | 15.871998 | 56.815738 | 44.782052 |
| Shuffled_OPD | 35.336017 | 15.772594 | 56.611797 | 44.762434 |
| Unordered_OPD | 34.953348 | 15.725093 | 55.896830 | 44.784534 |
| Forward_readout | 34.933405 | 15.747606 | 55.839923 | 44.746585 |
| Forward_Fixed | 35.326194 | 15.805854 | 56.759813 | 45.319056 |
| Forward_On | 35.325026 | 15.805278 | 56.756321 | 45.303659 |
| Forward_Shuffled | 35.308068 | 15.715043 | 56.603165 | 45.075705 |

## hc2: parent paired differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Layout_readout_minus_E_MBR | -0.080166 [-0.308277,+0.158312] | -0.225281 [-0.862380,+0.431250] | 4/11/1 | 0 |
| Unordered_readout_minus_E_MBR | -0.790341 [-1.322574,-0.315797] | -2.067900 [-3.250295,-1.015651] | 3/12/1 | 0 |
| Fixed_dense_minus_E_MBR | +0.077231 [-0.048743,+0.215945] | +0.152883 [-0.145704,+0.458606] | 10/5/1 | 0 |
| On_policy_dense_minus_E_MBR | +0.079164 [-0.052002,+0.224827] | +0.155379 [-0.163941,+0.481179] | 10/5/1 | 0 |
| Shuffled_OPD_minus_E_MBR | +0.030540 [-0.174619,+0.222824] | -0.048562 [-0.473775,+0.356443] | 10/5/1 | 0 |
| Unordered_OPD_minus_E_MBR | -0.352128 [-0.759312,-0.016499] | -0.763530 [-1.509402,-0.107330] | 5/10/1 | 0 |
| Forward_readout_minus_E_MBR | -0.372072 [-0.783913,-0.005178] | -0.820437 [-1.704121,-0.052311] | 6/9/1 | 0 |
| Forward_Fixed_minus_E_MBR | +0.020718 [-0.129296,+0.178776] | +0.099454 [-0.324809,+0.487544] | 8/7/1 | 0 |
| Forward_On_minus_E_MBR | +0.019550 [-0.136579,+0.182447] | +0.095962 [-0.358697,+0.501874] | 8/7/1 | 0 |
| Forward_Shuffled_minus_E_MBR | +0.002591 [-0.221857,+0.206990] | -0.057194 [-0.560983,+0.398310] | 10/5/1 | 0 |
| On_policy_dense_minus_Fixed_dense | +0.001933 [-0.007039,+0.011495] | +0.002496 [-0.025196,+0.029771] | 7/8/1 | 0 |
| On_policy_dense_minus_Shuffled_OPD | +0.048624 [-0.086530,+0.220538] | +0.203941 [-0.033778,+0.507674] | 8/7/1 | 0 |
| On_policy_dense_minus_Unordered_OPD | +0.431293 [+0.152743,+0.810076] | +0.918909 [+0.382402,+1.543444] | 12/3/1 | 0 |
| On_policy_dense_minus_Layout_readout | +0.159330 [-0.023503,+0.340963] | +0.380660 [-0.094084,+0.819663] | 11/4/1 | 0 |
| On_policy_dense_minus_Forward_On | +0.059615 [-0.146580,+0.228997] | +0.059417 [-0.259840,+0.345112] | 10/5/1 | 0 |
| On_policy_dense_minus_Forward_Fixed | +0.058447 [-0.143538,+0.223373] | +0.055925 [-0.254477,+0.328165] | 10/5/1 | 0 |
| On_policy_dense_minus_Forward_readout | +0.451236 [+0.123404,+0.840865] | +0.975816 [+0.320998,+1.740910] | 11/4/1 | 0 |
| Layout_readout_minus_Unordered_readout | +0.710175 [+0.323795,+1.137913] | +1.842618 [+0.916578,+2.868969] | 14/1/1 | 0 |

## vidstg: parent paired differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Layout_readout_minus_E_MBR | +0.075396 [-0.036415,+0.222393] | -2.200722 [-5.000516,+0.000003] | 3/1/12 | 0 |
| Unordered_readout_minus_E_MBR | -0.504428 [-1.113697,-0.061903] | -3.680577 [-6.826953,-1.087590] | 1/6/9 | 0 |
| Fixed_dense_minus_E_MBR | +0.111027 [-0.016475,+0.301425] | -0.596105 [-1.864817,+0.353069] | 4/2/10 | 0 |
| On_policy_dense_minus_E_MBR | +0.116067 [-0.016349,+0.311585] | -0.666421 [-2.060654,+0.375009] | 4/2/10 | 0 |
| Shuffled_OPD_minus_E_MBR | +0.016663 [-0.124943,+0.184777] | -0.686040 [-1.852323,+0.140387] | 2/4/10 | 0 |
| Unordered_OPD_minus_E_MBR | -0.030839 [-0.131302,+0.049513] | -0.663939 [-1.506517,-0.023822] | 2/4/10 | 0 |
| Forward_readout_minus_E_MBR | -0.008326 [-0.163984,+0.119465] | -0.701889 [-1.940354,+0.118240] | 3/2/11 | 0 |
| Forward_Fixed_minus_E_MBR | +0.049923 [-0.020813,+0.163895] | -0.129418 [-0.563622,+0.197100] | 3/3/10 | 0 |
| Forward_On_minus_E_MBR | +0.049347 [-0.020258,+0.161412] | -0.144814 [-0.619163,+0.203951] | 3/3/10 | 0 |
| Forward_Shuffled_minus_E_MBR | -0.040889 [-0.154719,+0.067064] | -0.372769 [-0.771688,-0.034628] | 2/4/10 | 0 |
| On_policy_dense_minus_Fixed_dense | +0.005040 [+0.000225,+0.011037] | -0.070316 [-0.202401,+0.027928] | 5/1/10 | 0 |
| On_policy_dense_minus_Shuffled_OPD | +0.099404 [+0.018285,+0.198765] | +0.019619 [-0.868211,+0.641537] | 6/0/10 | 0 |
| On_policy_dense_minus_Unordered_OPD | +0.146905 [+0.022036,+0.303078] | -0.002482 [-0.664059,+0.550970] | 5/1/10 | 0 |
| On_policy_dense_minus_Layout_readout | +0.040671 [-0.013489,+0.111992] | +1.534301 [+0.298460,+3.045496] | 4/2/10 | 0 |
| On_policy_dense_minus_Forward_On | +0.066720 [-0.012037,+0.160509] | -0.521607 [-1.534592,+0.249259] | 4/2/10 | 0 |
| On_policy_dense_minus_Forward_Fixed | +0.066144 [-0.013385,+0.159364] | -0.537003 [-1.587145,+0.261504] | 4/2/10 | 0 |
| On_policy_dense_minus_Forward_readout | +0.124392 [-0.006940,+0.305495] | +0.035468 [-0.775749,+0.675936] | 4/2/10 | 0 |
| Layout_readout_minus_Unordered_readout | +0.579824 [+0.117453,+1.158150] | +1.479856 [-0.644475,+4.405500] | 6/1/9 | 0 |

## Decision and limits

Original locked gate is false. Live minus E complete-v means are small positive but neither CI excludes zero; Vid unobserved actual-s declines. Spatial pairing has positive matched evidence; this is not dual-domain improvement over E or demonstrated on-policy superiority. All no-reference/failure/negative cases stay in denominators; absent GT remains unknown, not failure or success. No production registry changes.

Official full-v gains mostly arise at interpolated positions, not independent additional visual feedback: HC actual unobserved contribution+.014429/interpolated+.064736pp; Vid+.027084/+.088983pp. Postseal scope partition in POST_SEAL_ANALYSIS.json: Vid all-unobserved actual-s contributions inside MBR+.243003/outside-.909424pp; interpolation-relevant+.163168/irrelevant-.829589pp. These sum to-.666421. Interpolation-relevant conditional per-query mean differs because different numbers of known nodes contribute. No query or frame is removed from main evaluation. Zero temporal overlap remains a temporal limitation; it does not excuse spatial harms.

No-feedback and constant-feature no-op, exact first Fixed/Live state, spatial permutation positive control and padding invariance passed before target scoring. Root independent NumPy bilinear/ROI/normalization and likelihood audit passed1,395,360 rewards,256 state updates,1,760 official metric scalars,180 bootstraps; max reward discrepancy2.8531e-7, unchanged probability tolerance. Original fits and outputs never rewritten.

Actual cost excluding existing Source/E/DINO/features: {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 0, 'actual_fits': 128, 'actual_rounds': 256, 'CPU_prediction_and_pin_verification_seconds': 8.145313101122156, 'four_fit_seconds': 3.3597840177826583, 'old_Source_E_DINO_feature_cost_excluded': True, 'not_first_online_latency': True}

This is development evidence, not independent confirmation. GT is only postseal analysis, never candidate construction, anchor selection, feedback or checkpoint selection. Width/height/time and original E baseline exact across arms. Severe tails and strata are retained in RESULTS_PUBLIC.json.

## Every signed full-v change vs E, pp

| Alias | Target | Layout Live | Layout Fixed | Unordered Live |
|---|---|---:|---:|---:|
| NH001 | hc2 | +0.314395 | +0.318439 | -0.073425 |
| NH002 | hc2 | +0.015477 | +0.003965 | -0.086071 |
| NH003 | hc2 | +0.182911 | +0.170635 | +0.165817 |
| NH004 | hc2 | +0.779427 | +0.741588 | +0.659331 |
| NH005 | hc2 | -0.072379 | -0.064954 | -0.015049 |
| NH006 | hc2 | +0.345893 | +0.318926 | +0.124488 |
| NH007 | hc2 | -0.022407 | -0.021738 | -0.693196 |
| NH008 | hc2 | +0.109665 | +0.107324 | -0.831615 |
| NH009 | hc2 | -0.114056 | -0.095481 | -0.310415 |
| NH010 | hc2 | -0.256344 | -0.285565 | -0.626504 |
| NH011 | hc2 | +0.156460 | +0.172056 | +0.243230 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.296954 | +0.287216 | -0.403786 |
| NH014 | hc2 | +0.045711 | +0.050442 | -2.795314 |
| NH015 | hc2 | -0.521334 | -0.481148 | -1.153087 |
| NH016 | hc2 | +0.006256 | +0.013990 | +0.161542 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +1.325017 | +1.287981 | +0.375094 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.111263 | +0.106847 | -0.005119 |
| NV008 | vidstg | +0.531536 | +0.511074 | -0.165268 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | -0.106579 | -0.111465 | +0.000128 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.155523 | +0.134131 | -0.079272 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | -0.159694 | -0.152136 | -0.618982 |
