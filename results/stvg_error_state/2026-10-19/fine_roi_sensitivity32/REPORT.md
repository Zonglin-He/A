# Four predeclared Fine Center temperature/step sensitivity checks32

Explicit separately authorized finite experiment. Same repeatedly exposed NH16/NV16, one query/parent, frozen cross-domain EMA TA-STVG/E+finalMBR/four selected cached DINO references/true layer3 stride16/1024 and ordered3x3 ROI. Other branches and heartbeat remain paused. No new model/expert/DINO/Source/features/GPU/head/training or network weight update. All32 outputs and paths sealed before NEWGT. E and original CenterLive exact reuse. No hard-readout qualification, all actual fits executed.

Original mean-score anchor fusion and update implementation reused without modification. Four separate Live arms only: Tau_low(.05,.50), Tau_high(.20,.50), Step_low(.10,.25), Step_high(.10,1.00). Original(.10,.50)Center reused; unchanged noise/.25Ewh/2round17/scope/width/planned/MBR/final2. Every branch second round follows ITS actual state; no initial-candidate quality copied into moved round2. PLAN no_parameter_search means no extra/adaptive/best-config search; these four explicit hyperparameter checks ARE varied hyperparameters.

Measured direction: Step_high−Center fullv+.041681/+.072486pp and actuals+.058666/+.138609; Vid fullv interval[+.011045,+.148511]positive, HC fullv/actuals and Vid actuals intervals cross0. This is a promising common point-mean direction, not established double-domain superiority or a newly selected configuration. Tau_low fullv+.048723/+.011573, actuals+.076149/−.016999, all difference intervals cross0. Lower temperature does not uniformly help both endpoints.

Tau_high and Step_low reduce both endpoints versus original in both domains; all eight corresponding intervals negative. This weakens “current updates are generally too strong” on this panel, but does not prove the opposite for every query. Larger step and lower temperature cannot be conflated. No tau×step combination, scale change or additional round run. Keep default(.1,.5) locked until a separate decision and confirmation; no target-specific winners. Step_high HC full-v gain is sensitive: removingNH011 gives remaining15 mean−.004926pp vsCenter; NH013 loses−.827730pp, NH011 gains+.740788. Vid leave-one-out full-v remains positive[+.048678,+.078409]. These are postseal sensitivities only; all16 stay in main metrics. Tau_low HC/Vid full-v leave-one-out means positive but its Vid actual-s mean is negative, so no jointly favorable four-endpoint conclusion.

Fixed-target optimization diagnostics are computed on the SAME saved candidates/valid mask and SAME detached q, using actual eta_before/eta_after likelihoods, not the next refreshed teacher. KL(q||prior), before/after KL and center displacement/Ewh are per-node mean then per-parent. First-round original KL .303839/.247760; step1.0 keeps same teacher but reduces KL to.294169/.236728 versus original.297934/.241119 (exact baseline file authoritative). It moves twice as much on first round; GT benefit remains uncertain in HC. KL improvement is not localization proof; a Gaussian-mean family need not represent an arbitrary17way q, so remaining KL does not by itself prove insufficient optimization or that extra steps would help.

Same-state, same-support Tau_low first-round teacher expected candidate IoU versus originaltemperature increases HC+2.998618/Vid+1.535292pp, both paired intervals positive; actual second-round candidate comparison also positive. This is candidate weighting quality, not actual final tube benefit. Vid unobserved actual-s still declines versus original, including all leave-one-out means. Independent optimization diagnostic audit additionally reports per-node KL decrease/increase/neutral counts; no actual allowed-node increase above1e-12 observed across these paths.

## Main results

| Arm | HC full vIoU % | Vid full vIoU % | HC ALL unobserved actual sIoU % | Vid ALL unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Center_live | 35.513048 | 15.901544 | 57.042259 | 45.740623 |
| Tau_low | 35.561771 | 15.913117 | 57.118407 | 45.723624 |
| Tau_high | 35.452549 | 15.865329 | 56.942261 | 45.672305 |
| Step_low | 35.443718 | 15.842222 | 56.921893 | 45.630403 |
| Step_high | 35.554729 | 15.974030 | 57.100924 | 45.879232 |

## hc2: all paired parent differences, pp

| Contrast | Full vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below−5pp |
|---|---|---|---|---|
| Center_live_minus_E_MBR | +0.207571 [+0.092080,+0.349837] | +0.381899 [+0.133463,+0.664725] | 12/3/1 | 0 |
| Tau_low_minus_E_MBR | +0.256294 [+0.095692,+0.462684] | +0.458048 [+0.129574,+0.852900] | 11/4/1 | 0 |
| Tau_high_minus_E_MBR | +0.147072 [+0.065964,+0.239408] | +0.281902 [+0.106771,+0.475614] | 12/3/1 | 0 |
| Step_low_minus_E_MBR | +0.138242 [+0.065550,+0.220087] | +0.261534 [+0.108147,+0.429300] | 12/3/1 | 0 |
| Step_high_minus_E_MBR | +0.249252 [+0.018279,+0.528445] | +0.440565 [+0.010807,+0.961029] | 10/5/1 | 0 |
| Tau_low_minus_Center_live | +0.048723 [-0.024030,+0.122994] | +0.076149 [-0.041898,+0.209572] | 10/5/1 | 0 |
| Tau_high_minus_Center_live | -0.060499 [-0.124293,-0.006541] | -0.099997 [-0.205807,-0.010821] | 4/11/1 | 0 |
| Step_low_minus_Center_live | -0.069329 [-0.144592,-0.007434] | -0.120365 [-0.246554,-0.012533] | 4/11/1 | 0 |
| Step_high_minus_Center_live | +0.041681 [-0.117883,+0.192425] | +0.058666 [-0.184298,+0.321055] | 9/6/1 | 0 |

## vidstg: all paired parent differences, pp

| Contrast | Full vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below−5pp |
|---|---|---|---|---|
| Center_live_minus_E_MBR | +0.145613 [+0.015699,+0.335593] | +0.292150 [+0.028585,+0.610493] | 5/1/10 | 0 |
| Tau_low_minus_E_MBR | +0.157186 [+0.019956,+0.364276] | +0.275151 [-0.001223,+0.601512] | 5/1/10 | 0 |
| Tau_high_minus_E_MBR | +0.109397 [+0.009072,+0.262892] | +0.223831 [+0.017121,+0.493669] | 5/1/10 | 0 |
| Step_low_minus_E_MBR | +0.086291 [+0.008653,+0.203411] | +0.181930 [+0.017213,+0.390166] | 5/1/10 | 0 |
| Step_high_minus_E_MBR | +0.218098 [+0.025799,+0.480653] | +0.430758 [+0.036363,+0.912421] | 5/1/10 | 0 |
| Tau_low_minus_Center_live | +0.011573 [-0.011849,+0.037908] | -0.016999 [-0.073658,+0.031446] | 4/2/10 | 0 |
| Tau_high_minus_Center_live | -0.036215 [-0.073301,-0.006445] | -0.068319 [-0.146810,-0.000925] | 1/5/10 | 0 |
| Step_low_minus_Center_live | -0.059322 [-0.131886,-0.006950] | -0.110220 [-0.232309,-0.010344] | 1/5/10 | 0 |
| Step_high_minus_Center_live | +0.072486 [+0.011045,+0.148511] | +0.138609 [-0.000593,+0.327080] | 5/1/10 | 0 |

## Fixed-support optimization diagnostics

Native units: KL/JS in nats, displacement norm after dividing center change by fixedEwh; not percentages. These use ALL predicted allowed nodes, no GT mask. GT quality checks use separate known scope nodes and never replace main denominators. Anchor distributions are constructed matching evidence, not calibrated identity probabilities.

| Domain | Arm | Round | KL(q||prior) | KL before | KL after | Mean normalized step |
|---|---|---:|---:|---:|---:|---:|
| hc2 | Center_live | 1 | 0.30383925 | 0.30383925 | 0.29793449 | 0.01095428 |
| hc2 | Center_live | 2 | 0.30108257 | 0.30108257 | 0.29688775 | 0.00920985 |
| hc2 | Tau_low | 1 | 0.65291016 | 0.65291016 | 0.63742250 | 0.01641701 |
| hc2 | Tau_low | 2 | 0.64846494 | 0.64846494 | 0.63894524 | 0.01281069 |
| hc2 | Tau_high | 1 | 0.09643251 | 0.09643251 | 0.09448289 | 0.00653296 |
| hc2 | Tau_high | 2 | 0.09739736 | 0.09739736 | 0.09582729 | 0.00584271 |
| hc2 | Step_low | 1 | 0.30383925 | 0.30383925 | 0.30061796 | 0.00547714 |
| hc2 | Step_low | 2 | 0.30028254 | 0.30028254 | 0.29770126 | 0.00494110 |
| hc2 | Step_high | 1 | 0.30383925 | 0.30383925 | 0.29416885 | 0.02190855 |
| hc2 | Step_high | 2 | 0.30189734 | 0.30189734 | 0.29639375 | 0.01623795 |
| vidstg | Center_live | 1 | 0.24776007 | 0.24776007 | 0.24111863 | 0.00883639 |
| vidstg | Center_live | 2 | 0.25521715 | 0.25521715 | 0.24998600 | 0.00790356 |
| vidstg | Tau_low | 1 | 0.55545184 | 0.55545184 | 0.53781379 | 0.01222121 |
| vidstg | Tau_low | 2 | 0.55708178 | 0.55708178 | 0.54600377 | 0.01019231 |
| vidstg | Tau_high | 1 | 0.08112038 | 0.08112038 | 0.07897392 | 0.00541076 |
| vidstg | Tau_high | 2 | 0.08398245 | 0.08398245 | 0.08207373 | 0.00514005 |
| vidstg | Step_low | 1 | 0.24776007 | 0.24776007 | 0.24415634 | 0.00441819 |
| vidstg | Step_low | 2 | 0.25487216 | 0.25487216 | 0.25161531 | 0.00429944 |
| vidstg | Step_high | 1 | 0.24776007 | 0.24776007 | 0.23672760 | 0.01767277 |
| vidstg | Step_high | 2 | 0.25424344 | 0.25424344 | 0.24781357 | 0.01307172 |

## Decision, validity and cost

{'status': 'evaluated', 'rule': 'Report every predeclared configuration relative E and original Center by both domains, CI and tails; no best-of selection, no domain winners, no auto change to locked baseline.', 'best_configuration_selected': False, 'no_auto_expansion': True}

All parents including no-reference and unknown kept; fixed temporal outputs/tiou unchanged. Original ALL-known-unobserved actual denominators, not scoped conditional averages. Small exposed development panel, repeated method selection; intervals are unadjusted descriptive paired-parent bootstraps, not independent confirmation. Every signed outcome/stratum retained.

Independent root NumPy checks ROI bilinear rewards and separate actual categorical probabilities/gradients/state/readout, reuse/scope/width/zero-reference controls, official metrics and parent bootstraps. Existing GPT-6.1 Sol/max separately reviews code/history/sealed results. No audit inference rerun.

{'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 0, 'actual_fit_episodes': 128, 'round_slots': 256, 'CPU_prediction_and_pins_wall_seconds': 26.780172809958458, 'old_Source_E_DINO_fine_feature_cost_excluded': True, 'Center_fit_reused_without_reexecution': True, 'extra_diagnostic_ROI_evaluations': 0, 'not_first_online_latency': True}

## Every signed full-v difference vs E, pp

| Alias | Domain | Center_live | Tau_low | Tau_high | Step_low | Step_high |
|---|---|---:|---:|---:|---:|---:|
| NH001 | hc2 | +0.203457 | +0.324294 | +0.109274 | +0.107724 | +0.346822 |
| NH002 | hc2 | +0.061424 | +0.082894 | +0.024507 | +0.040346 | +0.094002 |
| NH003 | hc2 | +0.169177 | +0.233643 | +0.105080 | +0.089262 | +0.287156 |
| NH004 | hc2 | +0.542646 | +0.717093 | +0.345863 | +0.299041 | +0.944075 |
| NH005 | hc2 | -0.069082 | -0.147757 | -0.033856 | -0.036172 | -0.131521 |
| NH006 | hc2 | +0.180524 | +0.335998 | +0.089634 | +0.086980 | +0.392717 |
| NH007 | hc2 | -0.008978 | -0.010788 | -0.012985 | -0.003287 | -0.034247 |
| NH008 | hc2 | +0.376596 | +0.435498 | +0.236821 | +0.204384 | +0.612654 |
| NH009 | hc2 | -0.090018 | -0.114032 | -0.051117 | -0.037801 | -0.243473 |
| NH010 | hc2 | +0.092577 | +0.268031 | +0.028414 | +0.085323 | -0.050545 |
| NH011 | hc2 | +1.031776 | +1.442793 | +0.618220 | +0.544604 | +1.772563 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.245103 | -0.076280 | +0.399215 | +0.404204 | -0.582627 |
| NH014 | hc2 | +0.111341 | +0.049036 | +0.124801 | +0.074088 | +0.132383 |
| NH015 | hc2 | +0.291420 | +0.335120 | +0.272645 | +0.247281 | +0.181055 |
| NH016 | hc2 | +0.183178 | +0.225162 | +0.096636 | +0.105892 | +0.267025 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +1.327438 | +1.483061 | +1.098235 | +0.839985 | +1.757034 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.126980 | +0.191439 | +0.073398 | +0.069221 | +0.231696 |
| NV008 | vidstg | +0.552806 | +0.458010 | +0.392411 | +0.293056 | +0.966853 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | -0.014609 | -0.011274 | -0.010572 | -0.007139 | -0.030968 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.071391 | +0.046425 | +0.051738 | +0.038399 | +0.121209 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | +0.265794 | +0.347310 | +0.145148 | +0.147127 | +0.443748 |
