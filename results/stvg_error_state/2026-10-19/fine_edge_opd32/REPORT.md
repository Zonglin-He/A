# Frozen Fine-ROI four-edge conditional distribution OPD32

One user-authorized finite experiment. Other routes and overnight heartbeat stay paused. Original exposed NH16/NV16, one query per parent, original cross-domain frozen EMA TA-STVG/E+finalMBR/four E-selected DINO references/raw layer3 maps and valid masks. No new network, head, expert/DINO, Source/model/feature forward, GPU, pretrained weight update or network backward. Three actual Edge branches96episodes192roundslots; original Center Live reused exactly. All32x5 outputs and actual candidate/distribution/gradient/state paths sealed before NEW GT.

## Actual edge student and teacher

Four unit-variance latent Gaussian means per actual node, reset0. Physical per-edge displacement=.5Espan*tanh(a/2), Ewidth for L/R and Eheight for T/B. Readout by residual sum/difference added to E guarantees zero exactE, local derivative original.25Espan and maximum correction half original span; finite nonsaturated bounds imply width/height in(0,2Esize). This bound changes physical exploration away from zero. It is explicit implementation geometry, not calibrated uncertainty or identity knowledge. Final g(mu) is not E[g(a)].

Two rounds,17 rollout slots per node: shared zero plus4 single-edge perturbations for each edge. Four Gaussian quartiles with open uniform draws in first two, mirrored antithetically into other two; fixed SHA256 alias seed and same draws for all Edge arms. Zero is shared across four5-point conditional supports, making20 conditional slots per round, not17-way joint posterior. Each edge q=softmax(-eps²/2+originalROIscore/.1), p_mu=softmax(-(a-mu)²/2), detachedq, gradient sum(p-q)*a, original step.5, final2. No tanh chain derivative or physical-density Jacobian is part of this finite latent categorical objective. Equal anchor score mean remains; no anchor probability mixing or MPO added.

Fixed second proposal and other three edges remain initialE, while student likelihood uses its updated mu. Live second shared box is current entire composed output, each candidate changes one edge only. Four mean updates are simultaneous; full composed box may have worse score. Shuffle original target maps AND valid masks, fixed templates. Predicted interpolation-neighbor scope and four planned nodes unchanged. Width/height may now change only on allowed nodes.

This borrows the per-edge supervision unit from [D-FINE](https://arxiv.org/html/2410.13842v1); it does not load its trained distribution head,33bin integrator or localization self-distillation, and is not full FDR. Center comparison changes action dimension, four5-point versus17two-dimensional supports, stratified noise and bounded parameterization, not a single-factor width/height ablation.

## Main localization results

| Arm | HC full vIoU % | Vid full vIoU % | HC ALL unobserved actual sIoU % | Vid ALL unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Center_live | 35.513048 | 15.901544 | 57.042259 | 45.740623 |
| Edge_fixed | 35.472904 | 15.926260 | 56.979256 | 45.861881 |
| Edge_live | 35.476533 | 15.926332 | 56.990640 | 45.857103 |
| Edge_shuffle | 35.421797 | 15.797696 | 56.856879 | 45.550473 |

## hc2: paired parent changes, pp

| Contrast | Complete vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Center_live_minus_E_MBR | +0.207571 [+0.092080,+0.349837] | +0.381899 [+0.133463,+0.664725] | 12/3/1 | 0 |
| Edge_fixed_minus_E_MBR | +0.167427 [+0.051712,+0.295413] | +0.318897 [+0.064375,+0.581761] | 11/4/1 | 0 |
| Edge_live_minus_E_MBR | +0.171057 [+0.047192,+0.308022] | +0.330280 [+0.058328,+0.612288] | 11/4/1 | 0 |
| Edge_shuffle_minus_E_MBR | +0.116321 [-0.031011,+0.250781] | +0.196520 [-0.053141,+0.414619] | 11/4/1 | 0 |
| Edge_live_minus_Center_live | -0.036515 [-0.106535,+0.036074] | -0.051619 [-0.186236,+0.079334] | 7/8/1 | 0 |
| Edge_live_minus_Edge_fixed | +0.003629 [-0.014797,+0.021712] | +0.011384 [-0.017158,+0.038249] | 9/6/1 | 0 |
| Edge_live_minus_Edge_shuffle | +0.054736 [-0.071311,+0.223054] | +0.133761 [-0.120421,+0.468935] | 7/8/1 | 0 |

## vidstg: paired parent changes, pp

| Contrast | Complete vIoU [95%CI] | ALL unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Center_live_minus_E_MBR | +0.145613 [+0.015699,+0.335593] | +0.292150 [+0.028585,+0.610493] | 5/1/10 | 0 |
| Edge_fixed_minus_E_MBR | +0.170329 [+0.018634,+0.383406] | +0.413408 [+0.102526,+0.802932] | 5/1/10 | 0 |
| Edge_live_minus_E_MBR | +0.170401 [+0.018439,+0.385407] | +0.408630 [+0.106877,+0.786945] | 5/1/10 | 0 |
| Edge_shuffle_minus_E_MBR | +0.041764 [-0.056438,+0.182787] | +0.102000 [-0.110980,+0.340159] | 4/2/10 | 0 |
| Edge_live_minus_Center_live | +0.024788 [+0.000909,+0.053508] | +0.116480 [+0.014321,+0.238746] | 4/2/10 | 0 |
| Edge_live_minus_Edge_fixed | +0.000072 [-0.005866,+0.005008] | -0.004778 [-0.034824,+0.016757] | 3/3/10 | 0 |
| Edge_live_minus_Edge_shuffle | +0.128636 [+0.010636,+0.263534] | +0.306629 [+0.047436,+0.624918] | 5/1/10 | 0 |

## Measured decision

{'status': 'evaluated', 'edge_route_passed': False, 'rule': 'Both-domain Edge_live−E and Edge_live−Center_live full-v/ALL unobserved actual-s positive; onpolicy evidence additionally Live over Fixed/Shuffle with intervals/tails. No auto expansion.', 'onpolicy_direction': True, 'no_auto_expansion': True}

EdgeLive−E complete-v+.171057/+.170401pp and actual-s+.330280/+.408630, paired intervals all positive in this exposed panel. EdgeLive−Center complete-v−.036515/+.024788, actual-s−.051619/+.116480. HC intervals cross0 and point estimates decline; Vid intervals are positive. Thus the original both-domain improvement-over-Center gate fails. Retain Center as current common research baseline; do not choose methods separately by target domain from this panel. This is a mixed-domain result, not all edge distributions failing.

Live−Fixed complete-v+.003629/+.000072pp and actual-s+.011384/−.004778, all intervals cross0. No established on-policy refresh benefit. Live−Shuffle complete-v+.054736/+.128636 and actual-s+.133761/+.306629, HC intervals cross0/Vid intervals positive: limited domain-specific visual-binding evidence, not both-domain qualification. All main metrics use original GT support and ALL known unobserved actual-node denominators. No sample removed, unknown GT null. These are unadjusted descriptive development intervals after repeated target exposure, not independent confirmation.

## Postseal explanatory checks

Fixed-E-size free-center per-frame geometric ceiling: unclipped maximum intersection=min(Ew,GTw)*min(Eh,GTh), IoU=intersection/(Earea+GTarea-intersection). For HC official lower-coordinate clipping, enumerate full fixed-width/height intervals plus origin-clipped spans at GT edge breakpoints, and maximize intersection-over-union. Clipping can effectively shrink size, so the unclipped analytic ceiling is not falsely called an upper bound for HC official scoring. This uses GT only offline, unconstrained center movement, no teacher/exploration/identity constraints; not a deployable result or a complete-tube oracle.

| Scope known actual nodes, parent mean | HC | Vid |
|---|---:|---:|
| E_IoU | 0.573483 | 0.414563 |
| official_clip_aware_free_center_IoU | 0.737860 | 0.569041 |
| recoverable_by_center_pp | 16.437721 | 15.447783 |
| unrecoverable_with_fixed_size_pp | 26.214010 | 43.095941 |

POST_SEAL_DIAGNOSTICS retains both-round actual per-edge candidate quality, reward/posterior good-vs-harmful rankings relative GT E quality, teacher-minus-prior good mass, candidate headroom and complete-box reward after mean composition. Candidate shared base equalsE only in first round or Fixed; Live second shared base is the current student, so ranking fields explicitly say shared_base. Missing visual supports are counted separately, not assigned artificial ranking success/failure. Every query remains in primary metrics.

Complete composed-box scoring is postseal diagnostic only,2 extra ROI evaluations per node/branch (both intermediate/final mean boxes), never used for step acceptance, rollback, sampling or deployment. Live second shared-zero already reads its first composed full box. No mean ROI score gate. Extra diagnostics are separately accounted, not hidden inside the17rollout budget.

## Actual response and edge interaction

Edge Live changed size at393 HC/372 Vid actual nodes, with0illegal training samples. Complete composed-box reward improved29/30 supported HC query-steps and22/22Vid, mean+.004279/+.005214 in raw reward units, not IoU pp. Thus absent size connection or universal composition-destroyed feedback is not supported as an explanation. Scoring better does not imply better localization.

HC Edge−Center leave-one-out mean full-v stays negative[-.056585,-.019799]pp and actual-s[-.083612,-.012001], while original paired intervals cross0; no single-case explanation or confirmed universal decline. Vid has positive matching intervals and positive leave-one-out means; preserve that domain result without domain-specific method switching. HC NH007 Edge−E complete-v−.283269pp/actual-s−.748953pp remains; worst HC complete-v vsCenter NH016−.287248pp.

Initial per-edge pool best GT gain vsE2.564/3.303pp, second actual Live pool2.994/3.630; not an official full-tube oracle. Initial E-beneficial candidates above shared-base posterior20.42%/10.57%; strong pools no beneficial candidate outranks shared-base66.43%/84.15%. Second-round GT gains remain E-relative while shared-base score comparison uses current box. No assertion that AUC increase across different pools proves Live over Fixed.

## Verification and costs

Independent root NumPy audit reconstructs stratified noise using NormalDist rather than SciPy ndtri, bounded physical candidates, fine ROI bilinear rewards, four separate conditional probabilities/gradients, every actual mean/state/output, exact Center reuse/scope/planned/time, official metrics and parent bootstrap. GPT-6.1 Sol/max separately reviews source/history and sealed evidence.

Pre-first-measurement correction preserved original core/runner/protocol/runtime: nonfinite proposals fail before ROI, finite mean guard, evaluation slots not falsely called globally unique across rounds. No noise/configuration/teacher/GT rule change. Postseal explanatory-analysis-only recovery preserved original script/failure: empty5candidate visual support caused a generic max-over-empty error; now report missing support and geometric headroom separately. No prediction, main metric, denominator or decision changed.

{'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 0, 'actual_fit_episodes': 96, 'round_slots': 192, 'CPU_prediction_and_pins_wall_seconds': 19.614079653983936, 'training_candidate_evaluation_slots_per_node_per_arm': 34, 'per_edge_conditional_slots_per_node_per_arm': 40, 'extra_postseal_complete_box_diagnostic_ROI_evaluations': 15252, 'illegal_training_candidates': 0, 'old_Source_E_DINO_fine_feature_cost_excluded': True, 'Center_fit_reused_without_reexecution': True, 'not_first_online_latency': True}

No new trials, anchor mixture or stronger optimizer launched based on this result. No production promotion.

## Every signed complete-v change vs E, pp

| Alias | Target | Center | Edge Fixed | Edge Live | Edge Shuffle |
|---|---|---:|---:|---:|---:|
| NH001 | hc2 | +0.203457 | +0.221230 | +0.235247 | +0.144248 |
| NH002 | hc2 | +0.061424 | +0.249343 | +0.256064 | +0.333534 |
| NH003 | hc2 | +0.169177 | +0.102479 | +0.123349 | +0.085883 |
| NH004 | hc2 | +0.542646 | +0.542233 | +0.548877 | +0.501480 |
| NH005 | hc2 | -0.069082 | -0.061768 | -0.067419 | -0.014030 |
| NH006 | hc2 | +0.180524 | +0.113936 | +0.117782 | +0.187220 |
| NH007 | hc2 | -0.008978 | -0.246121 | -0.283269 | -0.360603 |
| NH008 | hc2 | +0.376596 | +0.422843 | +0.440458 | -0.614710 |
| NH009 | hc2 | -0.090018 | -0.049513 | -0.051139 | -0.022612 |
| NH010 | hc2 | +0.092577 | +0.275031 | +0.357114 | +0.568505 |
| NH011 | hc2 | +1.031776 | +0.761505 | +0.807385 | +0.351840 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.245103 | +0.241841 | +0.153280 | +0.262869 |
| NH014 | hc2 | +0.111341 | +0.012212 | +0.000887 | +0.051474 |
| NH015 | hc2 | +0.291420 | +0.162136 | +0.202362 | +0.158338 |
| NH016 | hc2 | +0.183178 | -0.068548 | -0.104070 | +0.227697 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +1.327438 | +1.461836 | +1.483740 | +0.958751 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.126980 | +0.081649 | +0.080417 | +0.042781 |
| NV008 | vidstg | +0.552806 | +0.712810 | +0.677646 | -0.061100 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | -0.014609 | -0.016451 | -0.021578 | +0.005563 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.071391 | +0.137907 | +0.146432 | +0.051370 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | +0.265794 | +0.347505 | +0.359753 | -0.329136 |
