# True finer-feature ordered ROI scoped dense OPD32

Explicitly renewed finite experiment after human pause. Original overnight heartbeat and all other branches remain paused. Same repeatedly exposed NH16/NV16, one query per parent; frozen cross-domain EMA TA-STVG/E+finalMBR/four cached E-selected DINO references. No new expert, DINO/Source/backbone/feature/model forward, trained head/model or pretrained weight update. Actual96fit episodes/192round slots completed; not a hard-selection qualification test. All32 x9outputs and feedback/update paths sealed before NEW GT evaluation.

## What is newly tested

First Fine experiment used real layer3 stride16/1024 but only propagated softmap/direct argmax, zero OPD. Original position-preserving ROI actual OPD used coarse input_proj stride32/256 only. This combination uses existing RAW layer3 maps/valid with the ORIGINAL ROI build/patches/feedback: bilinear3x3 at corresponding relative bin centers, masked padding, per-patch channel-L2, common-valid-bin weighted cosine, equal retained-anchor mean. It does not reuse Fine propagation softmaps or fit a projection. Prior negative gates remain unchanged. Finer versus coarse also changes layer semantics/channels/projection, so this comparison does not identify resolution alone.

Teacher q=normalize(originalGaussian prior*exp(ROIreward/.1)). This is constructed evidence, not calibrated DINO probability or a perfect foreground mask. Original per-query independent2D center state starts0; two17antithetic batches, .25 E-size exploration, .5 original categorical-KL block step, fixed final2. Width/height and four planned nodes remain E. Fixed second proposal center0; Live follows first actual state and recomputes candidate ROI feedback; Shuffle synchronously permutes target maps/valid with anchor templates unchanged. First Fixed/Live candidates/feedback/update exact. No new temperature, scale, loss or parameter search.

## Pure prediction-defined application scope

Allowed nodes are the conservative adjacent-ID interpolation support intersecting fixed predicted [MBRstart,MBRend), excluding planned anchors: right_neighbor>=start AND left_neighbor<end. Retains outside-interval neighbors needed by interpolation, including zero-weight boundary extras. No GT visibility/event interval enters this mask. No triangle/tent multiplier or altered within-scope candidate amplitude. Fine scope-out gradients/state are zero at every round and output exactly E.

Coarse Direct/Fixed/Live/Shuffle are original sealed ROI outputs restored to E outside that same mask, not new fits. Retained blocks equal old outputs; every integer predicted-interval interpolated coordinate is exactly unchanged, and official full-v equals the original arm on every query. All-unobserved actual-frame sIoU still uses the original known-node denominator, including scope-out nodes restored to E. This protection is not additional visual ability or an increase in full-v. Tent Independent_scoped changed amplitudes and is not the comparator.

| Arm | HC full vIoU % | Vid full vIoU % | HC all-unobserved actual sIoU % | Vid all-unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Coarse_direct | 35.225311 | 15.831327 | 56.652287 | 45.440672 |
| Coarse_fixed | 35.382708 | 15.866958 | 56.805367 | 45.597672 |
| Coarse_live | 35.384641 | 15.871998 | 56.809897 | 45.611641 |
| Coarse_shuffle | 35.336017 | 15.772594 | 56.651327 | 45.573044 |
| Fine_direct | 35.426659 | 15.900873 | 56.885089 | 45.514285 |
| Fine_fixed | 35.504561 | 15.897704 | 57.029568 | 45.728795 |
| Fine_live | 35.513048 | 15.901544 | 57.042259 | 45.740623 |
| Fine_shuffle | 35.437894 | 15.794559 | 56.874315 | 45.599169 |

## hc2: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Coarse_direct_minus_E_MBR | -0.080166 [-0.308277,+0.158312] | -0.008072 [-0.436412,+0.544622] | 4/11/1 | 0 |
| Coarse_fixed_minus_E_MBR | +0.077231 [-0.048743,+0.215945] | +0.145007 [-0.040116,+0.349678] | 10/5/1 | 0 |
| Coarse_live_minus_E_MBR | +0.079164 [-0.052002,+0.224827] | +0.149538 [-0.050270,+0.369894] | 10/5/1 | 0 |
| Coarse_shuffle_minus_E_MBR | +0.030540 [-0.174619,+0.222824] | -0.009032 [-0.390001,+0.336815] | 10/5/1 | 0 |
| Fine_direct_minus_E_MBR | +0.121182 [-0.233092,+0.484816] | +0.224729 [-0.382569,+0.866078] | 6/9/1 | 0 |
| Fine_fixed_minus_E_MBR | +0.199084 [+0.087166,+0.334958] | +0.369208 [+0.130442,+0.641480] | 12/3/1 | 0 |
| Fine_live_minus_E_MBR | +0.207571 [+0.092080,+0.349837] | +0.381899 [+0.133463,+0.664725] | 12/3/1 | 0 |
| Fine_shuffle_minus_E_MBR | +0.132418 [-0.002383,+0.253189] | +0.213956 [-0.025483,+0.432662] | 11/4/1 | 0 |
| Fine_live_minus_Coarse_live | +0.128407 [-0.008830,+0.288195] | +0.232361 [-0.025799,+0.527549] | 10/5/1 | 0 |
| Fine_live_minus_Fine_fixed | +0.008487 [-0.001298,+0.019388] | +0.012691 [-0.006107,+0.031227] | 8/7/1 | 0 |
| Fine_live_minus_Fine_shuffle | +0.075153 [-0.051914,+0.231671] | +0.167943 [-0.102564,+0.501752] | 8/7/1 | 0 |
| Fine_live_minus_Fine_direct | +0.086389 [-0.193987,+0.407650] | +0.157170 [-0.383474,+0.739890] | 9/6/1 | 0 |
| Fine_direct_minus_Coarse_direct | +0.201348 [-0.139621,+0.556000] | +0.232801 [-0.326089,+0.802280] | 11/3/2 | 0 |
| Fine_fixed_minus_Coarse_fixed | +0.121853 [-0.009837,+0.274477] | +0.224201 [-0.017675,+0.501321] | 10/5/1 | 0 |
| Fine_shuffle_minus_Coarse_shuffle | +0.101877 [-0.037846,+0.260675] | +0.222988 [-0.027511,+0.511757] | 8/7/1 | 0 |

## vidstg: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Coarse_direct_minus_E_MBR | +0.075396 [-0.036415,+0.222393] | -0.007801 [-0.299594,+0.333879] | 3/1/12 | 0 |
| Coarse_fixed_minus_E_MBR | +0.111027 [-0.016475,+0.301425] | +0.149199 [-0.175421,+0.497951] | 4/2/10 | 0 |
| Coarse_live_minus_E_MBR | +0.116067 [-0.016349,+0.311585] | +0.163168 [-0.187867,+0.531450] | 4/2/10 | 0 |
| Coarse_shuffle_minus_E_MBR | +0.016663 [-0.124943,+0.184777] | +0.124571 [-0.204610,+0.493802] | 2/4/10 | 0 |
| Fine_direct_minus_E_MBR | +0.144941 [-0.018661,+0.386091] | +0.065812 [-0.371667,+0.470355] | 3/2/11 | 0 |
| Fine_fixed_minus_E_MBR | +0.141773 [+0.012933,+0.329884] | +0.280322 [+0.022670,+0.595251] | 5/1/10 | 0 |
| Fine_live_minus_E_MBR | +0.145613 [+0.015699,+0.335593] | +0.292150 [+0.028585,+0.610493] | 5/1/10 | 0 |
| Fine_shuffle_minus_E_MBR | +0.038628 [-0.049072,+0.148661] | +0.150695 [-0.072699,+0.390668] | 4/2/10 | 0 |
| Fine_live_minus_Coarse_live | +0.029546 [-0.008553,+0.088827] | +0.128982 [-0.022276,+0.306238] | 5/1/10 | 0 |
| Fine_live_minus_Fine_fixed | +0.003840 [+0.000416,+0.007852] | +0.011828 [-0.002959,+0.030726] | 4/2/10 | 0 |
| Fine_live_minus_Fine_shuffle | +0.106985 [+0.009158,+0.220612] | +0.141455 [-0.084520,+0.368287] | 5/1/10 | 0 |
| Fine_live_minus_Fine_direct | +0.000671 [-0.058950,+0.065010] | +0.226338 [-0.045431,+0.670758] | 3/3/10 | 0 |
| Fine_direct_minus_Coarse_direct | +0.069546 [-0.033967,+0.209232] | +0.073613 [-0.261282,+0.375727] | 2/2/12 | 0 |
| Fine_fixed_minus_Coarse_fixed | +0.030746 [-0.006354,+0.088249] | +0.131123 [-0.004897,+0.289834] | 5/1/10 | 0 |
| Fine_shuffle_minus_Coarse_shuffle | +0.021965 [-0.039057,+0.084303] | +0.026124 [-0.223006,+0.238237] | 4/2/10 | 0 |

## Measured interpretation

Fine Live minus E full-v HC+0.207571pp CI[+0.092080,+0.349837], Vid+0.145613 [+0.015699,+0.335593]; ALL-unobserved actual-s+0.381899 [+0.133463,+0.664725]/+0.292150 [+0.028585,+0.610493]. These are repeatedly exposed development results, unadjusted descriptive parent intervals, not independent confirmation or a correction for the many earlier research choices. Main full-v positive/negative/unchanged is12/3/1 and5/1/10; no main full-v loss below-5pp. Missing/no-ref/unchanged parents remain.

Fine Live minus same-scope Coarse Live full-v+0.128407/+0.029546pp and actual-s+0.232361/+0.128982pp; all four intervals cross0. Vid Fine-minus-Coarse total full-v gain is about90percent due toNV016; removing it is sensitivity only, mean+0.003150pp. Fine-Live-minus-E leave-one-out full-v remains positive in both domains, so that main contrast is not solely one positive case.

Fixed Fine already improves E full-v+0.199084/+0.141773pp. Live adds only+0.008487/+0.003840pp, about4.1percent/2.6percent of its total mean gain; HC Live-Fixed v interval crosses0, Vid v interval[+0.000416,+0.007852]is positive but tiny, actual-s intervals both cross0. No established practically meaningful on-policy contribution. Live minus Shuffle full-v+0.075153/+0.106985; HC interval crosses0, Vid positive; actual-s intervals both cross0.

Live minus same34candidate Direct full-v+0.086389/+0.000671pp, both intervals cross0. HC removeNH013 reverses this contrast mean to-0.036302pp; Vid3positive/3negative/10unchanged. Do not claim superiority to direct candidate search. These postseal sensitivities do not remove any sample or change the deployment readout.

Pure support restoration alone leaves Coarse full-v exactly unchanged on every query and changes original all-unobserved actual-s+0.155379/-0.666421 to+0.149538/+0.163168pp. It avoids irrelevant-node damage; it does not improve complete-tube geometry or add new visual information. Fine-vs-Coarse with SAME support is the new-feature comparison.

## Original decision and boundaries

{'status': 'evaluated', 'passed': True, 'rule': 'Both-domain Fine_live−E and Fine_live−Coarse_live full-v/ALL unobserved actual-s positive; Fine_live full-v above Fixed/Shuffle/Direct. Confidence intervals/tails retained; no auto expansion.', 'actual_OPD': True}

The fixed gate is a resource/development criterion, not a universal theorem. All intervals, no-reference cases and signed tails remain. Fine gains without Live-over-Fixed support feedback but not unique on-policy benefit. No automatic expansion, no layer/temperature/step rescue, no deployment promotion. Same exposed target parents, not independent confirmation.

POST_SEAL_FEEDBACK reports E-relative beneficial/harmful ranks on identical initial34candidates, separately original all-known unobserved nodes and prediction-scope subset. Scope-conditioned diagnostics do not replace the main denominator. In prediction-scope nodes Fine raw AUC .855/.749 vsCoarse .837/.737 is a numerical increase, but Vid has only6defined parents and no paired-difference CI here. Vid strong-headroom pools with no beneficial posterior candidate outranking E remain78.57percent versus Coarse71.875percent; this is not universal improvement in E-relative ranking. Candidate GT quality is reused only after global output seal and does not evaluate moved second-round candidates. Unknown GT is null, never imputed as success/failure. Interpolated benchmark frames have no additional independent visual supervision.

Independent root NumPy audit reconstructs actual Fine ROI bilinear templates/rewards, saved categorical probabilities, every scoped gradient/state/output, exact Coarse interval interpolation, widths/planned/time/no-reference controls, official metrics and paired bootstraps. This audit has no GPU/model rerun. Existing GPT-6.1 Sol/max separately reviews history, implementation and sealed results.

Incremental computation excluding old Source/E/DINO/Fine capture: {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 0, 'actual_fit_episodes': 96, 'round_slots': 192, 'CPU_prediction_and_pins_seconds': 38.25690550706349, 'three_fit_seconds': 25.828842848772183, 'old_Source_E_DINO_fine_feature_cost_excluded': True, 'old_Coarse_fits_reused_without_reexecution': True, 'not_first_online_latency': True}

## Every signed full-v difference vs E, pp

| Alias | Target | Fine Live | Fine Fixed | Fine Shuffle | Fine Direct | Coarse Live |
|---|---|---:|---:|---:|---:|---:|
| NH001 | hc2 | +0.203457 | +0.195310 | +0.083944 | +0.037207 | +0.314395 |
| NH002 | hc2 | +0.061424 | +0.048690 | +0.110066 | -0.006401 | +0.015477 |
| NH003 | hc2 | +0.169177 | +0.152754 | +0.009974 | -0.132967 | +0.182911 |
| NH004 | hc2 | +0.542646 | +0.501253 | +0.597972 | +1.080855 | +0.779427 |
| NH005 | hc2 | -0.069082 | -0.062036 | -0.005414 | -0.039516 | -0.072379 |
| NH006 | hc2 | +0.180524 | +0.163454 | +0.392716 | +1.122675 | +0.345893 |
| NH007 | hc2 | -0.008978 | -0.006758 | -0.056679 | +0.029711 | -0.022407 |
| NH008 | hc2 | +0.376596 | +0.358172 | -0.544725 | -0.002072 | +0.109665 |
| NH009 | hc2 | -0.090018 | -0.083338 | -0.109445 | -0.290618 | -0.114056 |
| NH010 | hc2 | +0.092577 | +0.040811 | +0.390188 | +0.315610 | -0.256344 |
| NH011 | hc2 | +1.031776 | +0.990885 | +0.395208 | +1.771171 | +0.156460 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.245103 | +0.253003 | +0.264928 | -1.681652 | +0.296954 |
| NH014 | hc2 | +0.111341 | +0.126087 | +0.092592 | -0.055354 | +0.045711 |
| NH015 | hc2 | +0.291420 | +0.304008 | +0.231460 | -0.194084 | -0.521334 |
| NH016 | hc2 | +0.183178 | +0.203053 | +0.265899 | -0.015654 | +0.006256 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +1.327438 | +1.306247 | +0.721867 | +1.631965 | +1.325017 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.126980 | +0.108153 | +0.049784 | -0.239906 | +0.111263 |
| NV008 | vidstg | +0.552806 | +0.556641 | +0.163168 | +0.640779 | +0.531536 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | -0.014609 | -0.013094 | -0.006748 | -0.058667 | -0.106579 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.071391 | +0.055933 | +0.037000 | +0.000000 | +0.155523 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | +0.265794 | +0.254484 | -0.347029 | +0.344892 | -0.159694 |
