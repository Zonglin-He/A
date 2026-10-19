# Equal per-anchor probability mixture Fine Center OPD32

Explicit separately authorized finite experiment. Same repeatedly exposed NH16/NV16, one query/parent, frozen cross-domain EMA TA-STVG/E+finalMBR/four selected cached DINO references/true layer3 stride16/1024 and ordered3x3 ROI. Other branches and heartbeat remain paused. No new model/expert/DINO/Source/features/GPU/head/training or network weight update. All32 outputs and paths sealed before NEWGT. E and original CenterLive exact reuse. No hard-readout qualification, all actual fits executed.

Only new fusion: qa=softmax(logprior+anchorROI/.1), qmix=equal mean of qa. Original geometric pooling is normalized geometric mean on SAME support. Save each anchor score/probability, raw mixture, old geometric posterior and applied distribution. All original admitted anchors remain equally counted; zero common-bin reward remains0, not a new filter or separate per-candidate probability replacement. Candidate support/prior and original mean-score-range>1e-6 informative gate unchanged. Gate false still prior/no-update; this is the old-gate constrained mixture, not every possible mixture scheme. Raw mixture versus applied diagnostics identify suppression; no meaningful suppressed node observed in these paths. A0 strict no-op, A1 entire Fixed/Live/Shuffle paths bitexact original.

Original per-query independent2D center states, exact old17noise/.25 Ewh/tau.1/step.5/final2; widths/planned4/MBR fixed. Pure prediction-interpolation neighbor scope, no tent scaling. Fixed second candidates remainE, Live around actual new state, Shuffle target maps AND valid jointly while anchor templates fixed. Gate rule identical on same inputs; second-round validity/gate sets may naturally differ after paths diverge. No edge update/MPO/anchor weights.

Measured: MixLive−Center fullv−.016243/−.003554pp; all-unobserved actuals−.031343/−.010101. HC actuals interval[-.068888,-.001205]negative, other three comparisons uncertain. Original mixture resource gate=false. Mix still improves E fullv+.191328/+.142058 and actuals+.350557/+.282049, all four intervals positive; do not label allOPD failure. Live−Fixed fullv+.007219/+.004129, only Vid tiny interval positive; actuals both uncertain.

Initial same-candidate matched posterior expectation Mix−Geometric is−.238047/−.119469IoUpp among known predicted-scope visual nodes, parent aggregation. Teacher||prior remains nonzero and anchor JS .038510/.023195nats; raw mixture-geometric TV .014762/.008901. This is a real changed feedback, not a disconnected gate. Mean distributions need not improve E-relative quality. It does not establish multi-view conflict/background/identity as causal error. Close THIS equal mixture, no weights/temperature rescue, keep original score-mean Center baseline.

## Main results

| Arm | HC full vIoU % | Vid full vIoU % | HC ALL unobserved actual sIoU % | Vid ALL unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Center_live | 35.513048 | 15.901544 | 57.042259 | 45.740623 |
| Mix_fixed | 35.489585 | 15.893861 | 57.001056 | 45.719994 |
| Mix_live | 35.496805 | 15.897990 | 57.010916 | 45.730523 |
| Mix_shuffle | 35.430454 | 15.792938 | 56.869320 | 45.590160 |

## hc2: all paired parent differences, pp

| Contrast | Full vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below−5pp |
|---|---|---|---|---|
| Center_live_minus_E_MBR | +0.207571 [+0.092080,+0.349837] | +0.381899 [+0.133463,+0.664725] | 12/3/1 | 0 |
| Mix_fixed_minus_E_MBR | +0.184109 [+0.079418,+0.320754] | +0.340697 [+0.116531,+0.605561] | 12/3/1 | 0 |
| Mix_live_minus_E_MBR | +0.191328 [+0.082371,+0.334043] | +0.350557 [+0.117967,+0.626557] | 12/3/1 | 0 |
| Mix_shuffle_minus_E_MBR | +0.124978 [-0.014226,+0.248708] | +0.208961 [-0.025770,+0.420742] | 11/4/1 | 0 |
| Mix_live_minus_Center_live | -0.016243 [-0.039249,+0.003686] | -0.031343 [-0.068888,-0.001205] | 6/9/1 | 0 |
| Mix_live_minus_Mix_fixed | +0.007219 [-0.003233,+0.019317] | +0.009860 [-0.009474,+0.029764] | 8/7/1 | 0 |
| Mix_live_minus_Mix_shuffle | +0.066350 [-0.057611,+0.217929] | +0.141596 [-0.118785,+0.456040] | 8/7/1 | 0 |

## vidstg: all paired parent differences, pp

| Contrast | Full vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below−5pp |
|---|---|---|---|---|
| Center_live_minus_E_MBR | +0.145613 [+0.015699,+0.335593] | +0.292150 [+0.028585,+0.610493] | 5/1/10 | 0 |
| Mix_fixed_minus_E_MBR | +0.137930 [+0.009497,+0.329674] | +0.271521 [+0.003480,+0.607090] | 5/1/10 | 0 |
| Mix_live_minus_E_MBR | +0.142058 [+0.011540,+0.336235] | +0.282049 [+0.009975,+0.617411] | 5/1/10 | 0 |
| Mix_shuffle_minus_E_MBR | +0.037007 [-0.060722,+0.153976] | +0.141686 [-0.105421,+0.403384] | 4/2/10 | 0 |
| Mix_live_minus_Center_live | -0.003554 [-0.017944,+0.007532] | -0.010101 [-0.041497,+0.019952] | 3/3/10 | 0 |
| Mix_live_minus_Mix_fixed | +0.004129 [+0.000705,+0.008195] | +0.010528 [-0.002992,+0.028108] | 4/2/10 | 0 |
| Mix_live_minus_Mix_shuffle | +0.105052 [+0.009897,+0.217685] | +0.140363 [-0.082420,+0.363840] | 5/1/10 | 0 |

## Fixed-support optimization diagnostics

Native units: KL/JS in nats, displacement norm after dividing center change by fixedEwh; not percentages. These use ALL predicted allowed nodes, no GT mask. GT quality checks use separate known scope nodes and never replace main denominators. Anchor distributions are constructed matching evidence, not calibrated identity probabilities.

| Domain | Arm | Round | KL(q||prior) | KL before | KL after | Mean normalized step |
|---|---|---:|---:|---:|---:|---:|
| hc2 | Center_live | 1 | 0.30383925 | 0.30383925 | 0.29793449 | 0.01095428 |
| hc2 | Center_live | 2 | 0.30108257 | 0.30108257 | 0.29688775 | 0.00920985 |
| hc2 | Mix_fixed | 1 | 0.28624195 | 0.28624195 | 0.28043568 | 0.01094762 |
| hc2 | Mix_fixed | 2 | 0.28204617 | 0.27804844 | 0.27450490 | 0.00866256 |
| hc2 | Mix_live | 1 | 0.28624195 | 0.28624195 | 0.28043568 | 0.01094762 |
| hc2 | Mix_live | 2 | 0.28336740 | 0.28336740 | 0.27921196 | 0.00923971 |
| hc2 | Mix_shuffle | 1 | 0.27752579 | 0.27752579 | 0.26705705 | 0.01390726 |
| hc2 | Mix_shuffle | 2 | 0.27486081 | 0.27486081 | 0.26833238 | 0.01128217 |
| vidstg | Center_live | 1 | 0.24776007 | 0.24776007 | 0.24111863 | 0.00883639 |
| vidstg | Center_live | 2 | 0.25521715 | 0.25521715 | 0.24998600 | 0.00790356 |
| vidstg | Mix_fixed | 1 | 0.23932841 | 0.23932841 | 0.23250062 | 0.00907669 |
| vidstg | Mix_fixed | 2 | 0.24500028 | 0.23920689 | 0.23451673 | 0.00778632 |
| vidstg | Mix_live | 1 | 0.23932841 | 0.23932841 | 0.23250062 | 0.00907669 |
| vidstg | Mix_live | 2 | 0.24521740 | 0.24521740 | 0.23976587 | 0.00818496 |
| vidstg | Mix_shuffle | 1 | 0.24372612 | 0.24372612 | 0.23356111 | 0.01172153 |
| vidstg | Mix_shuffle | 2 | 0.24693126 | 0.24693126 | 0.23954853 | 0.00996865 |

## Decision, validity and cost

{'status': 'evaluated', 'mix_route_passed': False, 'rule': 'Both-domain Mix_live−E and Mix_live−Center_live full-v/ALL unobserved actual-s positive; paired intervals/tails and Live−Fixed/Shuffle separately reviewed, no auto expansion.', 'onpolicy_direction': True, 'no_auto_expansion': True}

All parents including no-reference and unknown kept; fixed temporal outputs/tiou unchanged. Original ALL-known-unobserved actual denominators, not scoped conditional averages. Small exposed development panel, repeated method selection; intervals are unadjusted descriptive paired-parent bootstraps, not independent confirmation. Every signed outcome/stratum retained.

Independent root NumPy checks ROI bilinear rewards and separate actual categorical probabilities/gradients/state/readout, reuse/scope/width/zero-reference controls, official metrics and parent bootstraps. Existing GPT-6.1 Sol/max separately reviews code/history/sealed results. No audit inference rerun.

{'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 0, 'actual_fit_episodes': 96, 'round_slots': 192, 'CPU_prediction_and_pins_wall_seconds': 19.42959611583501, 'old_Source_E_DINO_fine_feature_cost_excluded': True, 'Center_fit_reused_without_reexecution': True, 'extra_diagnostic_ROI_evaluations': 0, 'not_first_online_latency': True}

## Every signed full-v difference vs E, pp

| Alias | Domain | Center_live | Mix_fixed | Mix_live | Mix_shuffle |
|---|---|---:|---:|---:|---:|
| NH001 | hc2 | +0.203457 | +0.189348 | +0.196879 | +0.075976 |
| NH002 | hc2 | +0.061424 | +0.043863 | +0.056069 | +0.105041 |
| NH003 | hc2 | +0.169177 | +0.186628 | +0.204796 | +0.061108 |
| NH004 | hc2 | +0.542646 | +0.472666 | +0.515103 | +0.584108 |
| NH005 | hc2 | -0.069082 | -0.062193 | -0.069284 | -0.004877 |
| NH006 | hc2 | +0.180524 | +0.163376 | +0.180452 | +0.392636 |
| NH007 | hc2 | -0.008978 | -0.005142 | -0.008056 | -0.055509 |
| NH008 | hc2 | +0.376596 | +0.255352 | +0.262370 | -0.604703 |
| NH009 | hc2 | -0.090018 | -0.083808 | -0.090847 | -0.110345 |
| NH010 | hc2 | +0.092577 | +0.047248 | +0.104010 | +0.406546 |
| NH011 | hc2 | +1.031776 | +1.000773 | +1.042596 | +0.404294 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.245103 | +0.160981 | +0.143233 | +0.205906 |
| NH014 | hc2 | +0.111341 | +0.151199 | +0.139964 | +0.115412 |
| NH015 | hc2 | +0.291420 | +0.220169 | +0.197949 | +0.158200 |
| NH016 | hc2 | +0.183178 | +0.205276 | +0.186015 | +0.265849 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +1.327438 | +1.321279 | +1.344938 | +0.736206 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.126980 | +0.113761 | +0.130680 | +0.046363 |
| NV008 | vidstg | +0.552806 | +0.594004 | +0.593713 | +0.222996 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | -0.014609 | -0.024318 | -0.026098 | -0.015824 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.071391 | +0.038191 | +0.053966 | +0.023914 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | +0.265794 | +0.163961 | +0.175736 | -0.421551 |
