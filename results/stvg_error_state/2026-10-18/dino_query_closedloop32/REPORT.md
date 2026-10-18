# E-retaining query-only conditional-feedback closed loop32

Exact exposed HC16/Vid16 from endpoint confirmation32. Same original four DINO frames, E/final-MBR, .35 and source scales. Three new student branches update QR256 only, reset every query/branch. All STVG/DINO/LN weights frozen. Two scheduled Adam steps; final step2, no GT selection, LR HC.01/Vid.03 unchanged from prior QR configuration. No historical Before replay or parameter search.

Gaussian endpoint forward KL with covariance diag(source_scale²) and fixed16 normalization is used in all three branches. It is equivalent to source-scaled endpoint squared error; it is not the original IoU-reward OPD objective, nor a direct reproduction of autoregressive GKD. Testing teacher refresh does not by itself establish a novel loss or general OPD advantage. No-feedback entries contribute zero and all-no-feedback skips Adam, including momentum-only updates.

Fixed_center repeats cached aligned T(E) four times per anchor in both rounds. Fixed_sample reuses initial-E candidate feedback, first4 then next4 of prior sealed noise. On_policy has bit-exact first update with Fixed_sample; second round uses the same next4 noises centered on the updated actual E-corrected student, with genuinely refreshed aligned DINO decoding. Sampling is four complete shared-correction trajectories, teacher supervision remains four sparse anchors. It is not independent dense reward.

Output retains fixed E correction for every actual decoder node. At QR0, every output equals E bit-exact. Both original MBR intervals and tIoU remain unchanged.

| Output | HC complete vIoU % | Vid complete vIoU % |
|---|---:|---:|
| E_MBR | 35.305477 | 15.755931 |
| Center_full_local | 35.484322 | 15.819496 |
| Fixed_center_KL | 35.280435 | 15.798980 |
| Fixed_sample_KL | 35.318092 | 12.252162 |
| On_policy_KL | 35.281913 | 12.259547 |

## hc2: paired differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | Observed actual sIoU [95% CI] |
|---|---:|---:|---:|
| Fixed_center_KL_minus_E_MBR | -0.025042 [-0.211443, +0.147554], n=16 | -0.036063 [-0.315698, +0.202384], n=16 | +0.002608 [-0.213746, +0.226132], n=15 |
| Fixed_center_KL_minus_Center_full_local | -0.203887 [-0.533797, +0.071200], n=16 | -0.036063 [-0.315698, +0.202384], n=16 | -1.059407 [-2.709758, +0.422230], n=15 |
| Fixed_sample_KL_minus_E_MBR | +0.012615 [-0.090165, +0.110111], n=16 | +0.011364 [-0.168308, +0.154863], n=16 | +0.049878 [-0.098663, +0.232507], n=15 |
| Fixed_sample_KL_minus_Center_full_local | -0.166230 [-0.431204, +0.038082], n=16 | +0.011364 [-0.168308, +0.154863], n=16 | -1.012137 [-2.613147, +0.422153], n=15 |
| On_policy_KL_minus_E_MBR | -0.023563 [-0.155253, +0.096960], n=16 | -0.045082 [-0.243116, +0.123285], n=16 | +0.045037 [-0.101543, +0.227807], n=15 |
| On_policy_KL_minus_Center_full_local | -0.202409 [-0.468321, +0.014179], n=16 | -0.045082 [-0.243116, +0.123285], n=16 | -1.016978 [-2.618648, +0.415287], n=15 |
| On_policy_KL_minus_Fixed_center_KL | +0.001479 [-0.116927, +0.128965], n=16 | -0.009019 [-0.155535, +0.140745], n=16 | +0.042429 [-0.045911, +0.158347], n=15 |
| On_policy_KL_minus_Fixed_sample_KL | -0.036178 [-0.109805, +0.001393], n=16 | -0.056446 [-0.175110, +0.005123], n=16 | -0.004841 [-0.011901, +0.000191], n=15 |

E-supported queries 15/16; all16 remain in complete means. Unknown GT support stays null.

| Branch minus E | Observed actual v contribution | Unobserved actual v contribution | Official interpolated v contribution | Positive/negative/unchanged | Complete losses below -5pp |
|---|---:|---:|---:|---:|---:|
| Fixed_center_KL | -0.000524 | -0.005279 | -0.019239 | 8/6/2 | 0 |
| Fixed_sample_KL | +0.000413 | +0.000166 | +0.012037 | 6/6/4 | 0 |
| On_policy_KL | +0.000367 | -0.004242 | -0.019688 | 6/6/4 | 0 |

## vidstg: paired differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | Observed actual sIoU [95% CI] |
|---|---:|---:|---:|
| Fixed_center_KL_minus_E_MBR | +0.043049 [-0.021115, +0.136919], n=16 | +0.015471 [-0.061458, +0.102049], n=16 | +0.002379 [-0.104130, +0.128997], n=12 |
| Fixed_center_KL_minus_Center_full_local | -0.020515 [-0.078546, +0.024756], n=16 | +0.015471 [-0.061458, +0.102049], n=16 | -0.540434 [-2.047017, +1.115700], n=12 |
| Fixed_sample_KL_minus_E_MBR | -3.503770 [-10.558220, +0.059229], n=16 | -2.817432 [-8.501056, +0.083662], n=16 | -3.470040 [-10.516516, +0.131494], n=12 |
| Fixed_sample_KL_minus_Center_full_local | -3.567334 [-10.669140, +0.005033], n=16 | -2.817432 [-8.501056, +0.083662], n=16 | -4.012854 [-11.925679, +0.851309], n=12 |
| On_policy_KL_minus_E_MBR | -3.496385 [-10.535778, +0.059125], n=16 | -2.805761 [-8.467765, +0.084681], n=16 | -3.370926 [-10.206244, +0.126689], n=12 |
| On_policy_KL_minus_Center_full_local | -3.559949 [-10.646643, +0.005060], n=16 | -2.805761 [-8.467765, +0.084681], n=16 | -3.913740 [-11.622117, +0.844468], n=12 |
| On_policy_KL_minus_Fixed_center_KL | -3.539434 [-10.636219, +0.018505], n=16 | -2.821231 [-8.488840, +0.030442], n=16 | -3.373306 [-10.164300, +0.043980], n=12 |
| On_policy_KL_minus_Fixed_sample_KL | +0.007385 [-0.000488, +0.022647], n=16 | +0.011671 [+0.000019, +0.033945], n=16 | +0.099114 [-0.016068, +0.313999], n=12 |

E-supported queries 11/16; all16 remain in complete means. Unknown GT support stays null.

| Branch minus E | Observed actual v contribution | Unobserved actual v contribution | Official interpolated v contribution | Positive/negative/unchanged | Complete losses below -5pp |
|---|---:|---:|---:|---:|---:|
| Fixed_center_KL | -0.001132 | +0.009587 | +0.034593 | 2/3/11 | 0 |
| Fixed_sample_KL | -0.088235 | -0.376787 | -3.038748 | 1/5/10 | 1 |
| On_policy_KL | -0.085633 | -0.374960 | -3.035792 | 1/5/10 | 1 |

## All signed query changes versus E, pp

| Alias | Domain | Center local | Fixed center KL | Fixed sample KL | On-policy KL | On-policy minus fixed sample |
|---|---|---:|---:|---:|---:|---:|
| NH001 | hc2 | +0.000000 | -0.991676 | -0.560338 | -0.560338 | +0.000000 |
| NH002 | hc2 | +0.090425 | +0.008065 | +0.010039 | +0.010077 | +0.000038 |
| NH003 | hc2 | +0.383746 | +0.725163 | +0.387750 | +0.387750 | +0.000000 |
| NH004 | hc2 | +1.000429 | -0.643846 | +0.000000 | +0.000000 | +0.000000 |
| NH005 | hc2 | +0.000000 | +0.031057 | +0.054808 | +0.054804 | -0.000003 |
| NH006 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH007 | hc2 | -0.181400 | +0.120915 | +0.000000 | +0.000000 | +0.000000 |
| NH008 | hc2 | +1.538001 | -0.065666 | -0.084528 | -0.078105 | +0.006424 |
| NH009 | hc2 | +0.000000 | -0.079651 | -0.043634 | -0.626884 | -0.583251 |
| NH010 | hc2 | -0.172690 | -0.078510 | -0.067602 | -0.076247 | -0.008645 |
| NH011 | hc2 | +0.105836 | -0.029775 | -0.011126 | -0.010986 | +0.000140 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.296953 | +0.417396 | +0.400299 | +0.400817 | +0.000518 |
| NH014 | hc2 | -0.373644 | +0.019193 | -0.043401 | -0.043190 | +0.000212 |
| NH015 | hc2 | +0.119991 | +0.044123 | +0.038681 | +0.044353 | +0.005672 |
| NH016 | hc2 | +0.053882 | +0.122543 | +0.120896 | +0.120936 | +0.000040 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +0.427866 | +0.601262 | -56.242105 | -56.121437 | +0.120668 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.060417 | -0.046245 | -0.048065 | -0.050934 | -0.002869 |
| NV008 | vidstg | +0.238363 | -0.139185 | -0.137123 | -0.137157 | -0.000034 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.000000 | +0.000000 | -0.009980 | -0.009999 | -0.000019 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | -0.074634 | -0.013221 | -0.011950 | -0.011974 | -0.000023 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | +0.365015 | +0.286173 | +0.388909 | +0.389348 | +0.000440 |

## Actual work

GPU allocation including loading, current Source restoration, new teacher encodings and fits: 385.777968s, prelocked cap900s. Not first-use E/DINO/source-calibration cost.

| Work | HC | Vid |
|---|---:|---:|
| new_Source_captures | 16 | 16 |
| Source_capture_seconds | 50.53612570557743 | 56.66565610701218 |
| DINO_encoding_frames | 60 | 44 |
| new_conditional_queries | 348 | 308 |
| new_conditional_decoder_seconds | 0.8814439037814736 | 0.7451615950558335 |
| Fixed_center_KL fit seconds | 4.520463 | 4.244536 |
| Fixed_sample_KL fit seconds | 5.686260 | 5.534147 |
| On_policy_KL fit seconds | 5.692843 | 5.502022 |

Original Source outputs and logits are exact. New frozen Source captures only restore unsaved backward interfaces, which are now retained privately. Cached center/first-round feedback is reused. Fixed-sample round2 feedback is obtained once at the unchanged initial references, using the same four-candidate layout as on-policy round2. Actual measured costs are reported. Both second-round conditions make new conditional calls with the same four-candidate layout; fixed sampling keeps initial E references, on-policy changes their center. All3 share update loss, LR, scales, number of steps and candidate identifiers; admission/no-op counts and actual update magnitudes can differ and are saved.

All32 five-arm predictions sealed before GT scoring. Paired query=parent bootstrap10000 seed20261018. This is development mechanism validation, not independent confirmation or official full results. Actual-node changes must be separated from official interpolation; four-frame feedback is not event-boundary evidence. Existing QR response/detach limitations remain disclosed. No extra rounds, new backbone, temporal adaptation or automatic expansion.


## Engineering and actual-cost qualification

Final worker251.102898s plus conservative prior engineering allocation bound134.675070s gives total385.777968s under900s. The total is an upper bound including controller/startup/stop delay. Original13 pre-GT partial queries are preserved; main evaluation uses only the matched-layout final32. Completed work totals45 Source captures and152 DINO encodings across both workers, plus possible unreceipted interrupted work covered by the time bound. See COST_BREAKDOWN.json and MEASURED_INTERPRETATION.md. The audit distinguishes192 scheduled rounds from118 actual Adam steps.
