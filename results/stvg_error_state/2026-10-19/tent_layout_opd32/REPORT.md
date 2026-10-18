# Fixed tent joint Layout OPD32

Same historically exposed NH16/NV16, one query per parent, frozen opposite-source EMA TA-STVG, E+finalMBR, four cached DINO observations, coarse input_proj features. No new Source/model forward, experts, training, learned heads or GPU. All128fits256rounds and eight outputs globally sealed before new GT evaluation.

Three fixed physical-frame-time triangular basis functions between four planned anchors, zero at anchors/outside anchor span, further restricted to predictedMBR plus conservative official interpolation-neighbor support. Each active segment has two center coordinates; inactive axes excluded from sample/prior/gradient. Width/height/time unchanged. State resets per query. This is an E-centered output state, not network weight training. The hypothesis tests time-coupled use of existing visual Layout feedback; high-frequency damage is not an established cause.

Old smooth identity-trajectory OPD was negative (multi_tube minus multi_frame fullv HC-1.374823/Vid-4.736027pp). New writable6D E-centered single-distribution state and dense Layout feedback differ from its identity mixtures/1792network carrier; previous negative evidence remains. This is not the first trajectory or smooth OPD experiment.

Two17-proposal rounds, .25Ewh perturbations, .1temperature, .5categoricalKLstep, finalround2. Joint reward averages exactly the same eligible positive-basis nodes for every candidate, including zero-support ROI rewards. Direct chooses one complete34-pool trajectory, never stitches per-frame bests. Fixed proposes from initial state, Live from current state, Shuffle permutes target maps/valid with anchors fixed. Matched independent centers have identical scope and pointwise perturbation amplitude. Two pinv projections of matched-independent and previous unscoped Layout results control smoothing/shrinkage without GT. Six-dimensional joint and two-dimensional independent priors have different effective strength; recorded, never target-calibrated.

| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Independent_scoped | 35.324528 | 15.825195 | 56.687934 | 45.656168 |
| Independent_projection | 35.318339 | 15.799571 | 56.647782 | 45.568352 |
| Previous_projection | 35.321897 | 15.809150 | 56.650194 | 45.586728 |
| Joint_direct | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Joint_fixed | 35.309540 | 15.763287 | 56.660571 | 45.468670 |
| Joint_live | 35.310266 | 15.764109 | 56.660923 | 45.470117 |
| Joint_shuffle | 35.308880 | 15.752510 | 56.643807 | 45.447140 |

## hc2: parent paired differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Independent_scoped_minus_E_MBR | +0.019051 [-0.018145,+0.058438] | +0.027574 [-0.015776,+0.076113] | 7/5/4 | 0 |
| Independent_projection_minus_E_MBR | +0.012862 [-0.019817,+0.044917] | -0.012578 [-0.065974,+0.039084] | 7/5/4 | 0 |
| Previous_projection_minus_E_MBR | +0.016420 [-0.027520,+0.058198] | -0.010165 [-0.066210,+0.048180] | 7/5/4 | 0 |
| Joint_direct_minus_E_MBR | +0.000000 [+0.000000,+0.000000] | +0.000000 [+0.000000,+0.000000] | 0/0/16 | 0 |
| Joint_fixed_minus_E_MBR | +0.004064 [-0.006007,+0.014558] | +0.000212 [-0.019113,+0.018193] | 7/5/4 | 0 |
| Joint_live_minus_E_MBR | +0.004790 [-0.005897,+0.015959] | +0.000564 [-0.019742,+0.019704] | 7/5/4 | 0 |
| Joint_shuffle_minus_E_MBR | +0.003403 [-0.013313,+0.019290] | -0.016552 [-0.049655,+0.011899] | 7/5/4 | 0 |
| Joint_live_minus_Independent_scoped | -0.014261 [-0.051997,+0.021808] | -0.027011 [-0.072279,+0.009047] | 6/6/4 | 0 |
| Joint_live_minus_Independent_projection | -0.008073 [-0.037701,+0.020563] | +0.013141 [-0.030366,+0.055714] | 5/7/4 | 0 |
| Joint_live_minus_Previous_projection | -0.011630 [-0.049786,+0.029558] | +0.010729 [-0.039602,+0.060715] | 4/8/4 | 0 |
| Joint_live_minus_Joint_direct | +0.004790 [-0.005897,+0.015959] | +0.000564 [-0.019742,+0.019704] | 7/5/4 | 0 |
| Joint_live_minus_Joint_fixed | +0.000726 [-0.000261,+0.001762] | +0.000352 [-0.001363,+0.002097] | 7/5/4 | 0 |
| Joint_live_minus_Joint_shuffle | +0.001387 [-0.013856,+0.020232] | +0.017116 [-0.003626,+0.049349] | 6/6/4 | 0 |

## vidstg: parent paired differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Independent_scoped_minus_E_MBR | +0.069264 [+0.000245,+0.157666] | +0.207695 [+0.018792,+0.456746] | 4/1/11 | 0 |
| Independent_projection_minus_E_MBR | +0.043639 [-0.002781,+0.103986] | +0.119878 [+0.000389,+0.280636] | 4/1/11 | 0 |
| Previous_projection_minus_E_MBR | +0.053219 [+0.003460,+0.122296] | +0.138255 [+0.009885,+0.322858] | 4/1/11 | 0 |
| Joint_direct_minus_E_MBR | +0.000000 [+0.000000,+0.000000] | +0.000000 [+0.000000,+0.000000] | 0/0/16 | 0 |
| Joint_fixed_minus_E_MBR | +0.007355 [-0.004203,+0.020892] | +0.020197 [-0.005708,+0.052948] | 4/1/11 | 0 |
| Joint_live_minus_E_MBR | +0.008177 [-0.003280,+0.021943] | +0.021643 [-0.004243,+0.053835] | 4/1/11 | 0 |
| Joint_shuffle_minus_E_MBR | -0.003422 [-0.015880,+0.004913] | -0.001333 [-0.026059,+0.017496] | 3/2/11 | 0 |
| Joint_live_minus_Independent_scoped | -0.061087 [-0.136207,-0.002656] | -0.186051 [-0.400929,-0.018717] | 1/4/11 | 0 |
| Joint_live_minus_Independent_projection | -0.035462 [-0.084337,+0.000618] | -0.098235 [-0.226890,-0.002946] | 1/4/11 | 0 |
| Joint_live_minus_Previous_projection | -0.045041 [-0.102955,-0.005167] | -0.116612 [-0.270887,-0.010744] | 0/5/11 | 0 |
| Joint_live_minus_Joint_direct | +0.008177 [-0.003280,+0.021943] | +0.021643 [-0.004243,+0.053835] | 4/1/11 | 0 |
| Joint_live_minus_Joint_fixed | +0.000822 [+0.000023,+0.001874] | +0.001446 [+0.000280,+0.003034] | 4/1/11 | 0 |
| Joint_live_minus_Joint_shuffle | +0.011599 [-0.000158,+0.024910] | +0.022977 [-0.000195,+0.053590] | 3/2/11 | 0 |

## Decision and measured limits

Original locked eligibility is false. JointLive minus E fullv+.004790/+.008177pp and unobservedactuals+.000564/+.021643pp; all CIs cross zero. It is weaker than matched Independent_scoped: fullv-.014261/-.061087pp, actuals-.027011/-.186051pp (Vid both intervals negative). It is also below the same-basis projection controls in fullv. Tiny Live-minus-Fixed effects do not establish useful OPD over E or simpler alternatives. Stop this fixed joint representation/feedback combination, no knot/scale/temperature/step search. Do not call tiny motion successful stability.

Joint_direct is E exactly for every query: center candidate won its declared Gaussian-prior-plus-reward argmax, or no informative feedback left E. This is not zero candidate headroom proof. STATE_RESPONSE_PUBLIC.json retains effective rank, prior/teacherESS, center-candidate mass, candidate visibility, actual gradients/state/output motion and normalized residual first/second differences. Dimensions/priors affect effective movement; no post-GT amplitude matching.

No-reference and missing GT retained; unknown GT is never treated as failure or correctness. All temporal intervals identical. Official interpolation does not count as extra independent observations. Development exposure and previous GT-informed mechanism selection are disclosed; not independent confirmation.

Root independent NumPy physical-time basis/ROI bilinear rewards, Gaussian/categorical probabilities, actual192joint+64independent updates,64pinv projections,1280metric scalars and130paired bootstraps passed. Max independent reward error2.0783e-7. Frozen original source/configuration pins checked, no audit model rerun.

Engineering recovery: original scorer failed only while auxiliary moved-node logging compared a torch tensor to NumPy E and obtained a scalar bool. Original runner/failure preserved. Separate score_stvg_tent_layout_opd32_v1.py explicitly casts E for that log. No fits/predictions/GT metrics were regenerated or changed; complete32 global prediction barrier predates scoring. Engineering failure is distinct from scientific negative result.

Actual incremental costs, old Source/E/DINO/feature generation excluded: {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 0, 'actual_fits': 128, 'actual_rounds': 256, 'CPU_prediction_and_full_pin_verification_seconds': 7.797481788089499, 'four_fit_seconds': 3.700925539014861, 'old_Source_E_DINO_feature_cost_excluded': True, 'not_first_online_latency': True, 'scorer_only_recovery_new_fits': 0}

## Every signed full-v change vs E, pp

| Alias | Target | JointLive | Independent | Independent projection |
|---|---|---:|---:|---:|
| NH001 | hc2 | +0.021978 | +0.106431 | +0.053231 |
| NH002 | hc2 | +0.013892 | +0.053184 | +0.100664 |
| NH003 | hc2 | +0.003778 | +0.046577 | +0.052847 |
| NH004 | hc2 | +0.023551 | +0.182653 | +0.039343 |
| NH005 | hc2 | -0.001306 | -0.004195 | -0.003130 |
| NH006 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH007 | hc2 | -0.017259 | -0.042989 | -0.037157 |
| NH008 | hc2 | +0.025640 | +0.025951 | +0.052698 |
| NH009 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH010 | hc2 | -0.032811 | -0.048619 | -0.118137 |
| NH011 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.060216 | +0.027855 | +0.068276 |
| NH014 | hc2 | -0.009293 | +0.148988 | +0.138703 |
| NH015 | hc2 | +0.018672 | -0.155586 | -0.106106 |
| NH016 | hc2 | -0.030420 | -0.035433 | -0.035433 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +0.081847 | +0.430442 | +0.236444 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.011482 | +0.137445 | +0.104946 |
| NV008 | vidstg | +0.057234 | +0.523198 | +0.386879 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.013872 | +0.092466 | +0.044678 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | -0.033600 | -0.075331 | -0.074719 |
