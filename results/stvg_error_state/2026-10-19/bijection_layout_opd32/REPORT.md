# Nine-patch bijection dense OPD32

Same historically exposed NH16/NV16; one query per parent. Original frozen cross-domain EMA TA-STVG, E+finalMBR, four cached DINO observations, original coarse input_proj. No new experts, model/head training, Source/model forward, GPU or pretrained parameter updates.128actualfits256updates and all32×9outputs sealed before new GT evaluation. All signs, unknowns, no-reference cases and adverse tails retained; not independent confirmation.

Mechanism: compare normalized3x3 ROI patch descriptors through an exact9-slot bijection. Each target slot and anchor slot used once; local descriptors may permute. Pair cost cos*targetvalidweight*anchorvalidweight; fixed denominator9 even for partial/invalid slots, then equal E-selected-anchor mean. Matched diagonal uses same graph, weights and denominator. This differs from earlier common-valid-mass normalized Layout, retained as Previous_independent. No learned projection, classifier, association model, new feature encoder or pseudo-negative training.

The local transport analogy comes from [DeepEMD](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.html), whose embedding and element weights were trained. None are imported, and the present fixed-uniform bijection is not a DeepEMD reproduction. Old cross-frame bbox Hungarian and many-to-many region softmax do not test the same mechanism. Bijection score>=diagonal is a mathematical feasible-set property, never evidence of target correctness. Same descriptor set in another instance can still receive a high score.

Synthetic positive controls retain a permutation of9orthogonal descriptors, penalize nine repeats of one descriptor to1/9, verify matched-diagonal inclusion, no-reference/constant-in-view no-op and same first Fixed/Live update. Partial visibility changes fixed9slot support and can change even a constant-feature score near borders; this is disclosed, not a global flat-map invariance claim. Entirely invisible candidates invalid; all-invisible safe fallback and no informative update.

External independent2D center state per actual node; fifth-round physical-time tent marginal amplitude and predictedMBR+interpolation-neighbor scope fixed. Original E widths, planned anchors and time unchanged. Same old two17antithetic noises, .25E-size/.1temperature/.5categoricalKLstep/final2. BijectionFixed holds initial proposal center; Live second round follows actual updated state and recomputes patch assignment; Shuffle moves target feature/valid frames with anchor descriptors fixed. DiagonalLive shares everything except matching.34-pool Direct has same visual candidate budget as2x17adaptation. Previous fifth-round Independent and Joint predictions reused byte-exact.

| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Bijection_readout | 35.245086 | 15.772761 | 56.587826 | 45.460591 |
| Diagonal_readout | 35.246846 | 15.816905 | 56.581360 | 45.608826 |
| Bijection_fixed | 35.295080 | 15.790276 | 56.644377 | 45.404907 |
| Bijection_live | 35.289987 | 15.791264 | 56.638982 | 45.402148 |
| Bijection_shuffle | 35.278379 | 15.736010 | 56.614608 | 45.298566 |
| Diagonal_live | 35.312826 | 15.822616 | 56.676533 | 45.519379 |
| Previous_independent | 35.324528 | 15.825195 | 56.687934 | 45.656168 |
| Previous_joint | 35.310266 | 15.764109 | 56.660923 | 45.470117 |

## hc2: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Bijection_readout_minus_E_MBR | -0.060391 [-0.142926,+0.018636] | -0.072533 [-0.183857,+0.039779] | 4/8/4 | 0 |
| Diagonal_readout_minus_E_MBR | -0.058631 [-0.164384,+0.042457] | -0.079000 [-0.224677,+0.088532] | 3/9/4 | 0 |
| Bijection_fixed_minus_E_MBR | -0.010396 [-0.056856,+0.033335] | -0.015982 [-0.081661,+0.041918] | 7/5/4 | 0 |
| Bijection_live_minus_E_MBR | -0.015490 [-0.064825,+0.028335] | -0.021378 [-0.090123,+0.039448] | 7/5/4 | 0 |
| Bijection_shuffle_minus_E_MBR | -0.027097 [-0.087754,+0.024736] | -0.045752 [-0.161749,+0.047829] | 7/5/4 | 0 |
| Diagonal_live_minus_E_MBR | +0.007349 [-0.032179,+0.050899] | +0.016173 [-0.039277,+0.069386] | 6/6/4 | 0 |
| Previous_independent_minus_E_MBR | +0.019051 [-0.018145,+0.058438] | +0.027574 [-0.015776,+0.076113] | 7/5/4 | 0 |
| Previous_joint_minus_E_MBR | +0.004790 [-0.005897,+0.015959] | +0.000564 [-0.019742,+0.019704] | 7/5/4 | 0 |
| Bijection_live_minus_Bijection_readout | +0.044901 [-0.036876,+0.133904] | +0.051155 [-0.076194,+0.193817] | 7/5/4 | 0 |
| Bijection_live_minus_Diagonal_readout | +0.043141 [-0.060011,+0.154191] | +0.057622 [-0.120716,+0.232817] | 7/5/4 | 0 |
| Bijection_live_minus_Bijection_fixed | -0.005093 [-0.011904,+0.000455] | -0.005396 [-0.019894,+0.004626] | 5/7/4 | 0 |
| Bijection_live_minus_Bijection_shuffle | +0.011608 [-0.025500,+0.068991] | +0.024374 [-0.059802,+0.159074] | 3/9/4 | 0 |
| Bijection_live_minus_Diagonal_live | -0.022839 [-0.048519,-0.001486] | -0.037551 [-0.077140,-0.005483] | 4/8/4 | 0 |
| Bijection_live_minus_Previous_independent | -0.034541 [-0.085064,+0.007147] | -0.048952 [-0.113850,+0.005970] | 6/6/4 | 0 |
| Bijection_live_minus_Previous_joint | -0.020279 [-0.074545,+0.024760] | -0.021941 [-0.089601,+0.033678] | 7/5/4 | 0 |

## vidstg: paired parent differences, pp

| Contrast | Complete vIoU [95%CI] | Unobserved actual sIoU [95%CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Bijection_readout_minus_E_MBR | +0.016830 [-0.035218,+0.083016] | +0.012117 [-0.139023,+0.202396] | 2/2/12 | 0 |
| Diagonal_readout_minus_E_MBR | +0.060974 [-0.033229,+0.211174] | +0.160353 [-0.131344,+0.640195] | 3/1/12 | 0 |
| Bijection_fixed_minus_E_MBR | +0.034345 [-0.040575,+0.142907] | -0.043566 [-0.262466,+0.105775] | 3/2/11 | 0 |
| Bijection_live_minus_E_MBR | +0.035332 [-0.043427,+0.148150] | -0.046326 [-0.285735,+0.116517] | 3/2/11 | 0 |
| Bijection_shuffle_minus_E_MBR | -0.019921 [-0.133232,+0.105077] | -0.149907 [-0.406154,+0.048482] | 1/4/11 | 0 |
| Diagonal_live_minus_E_MBR | +0.066684 [-0.024422,+0.191527] | +0.070906 [-0.174556,+0.320608] | 3/2/11 | 0 |
| Previous_independent_minus_E_MBR | +0.069264 [+0.000245,+0.157666] | +0.207695 [+0.018792,+0.456746] | 4/1/11 | 0 |
| Previous_joint_minus_E_MBR | +0.008177 [-0.003280,+0.021943] | +0.021643 [-0.004243,+0.053835] | 4/1/11 | 0 |
| Bijection_live_minus_Bijection_readout | +0.018503 [-0.069964,+0.131085] | -0.058443 [-0.333958,+0.175678] | 3/2/11 | 0 |
| Bijection_live_minus_Diagonal_readout | -0.025641 [-0.194676,+0.119882] | -0.206678 [-0.690884,+0.155296] | 2/3/11 | 0 |
| Bijection_live_minus_Bijection_fixed | +0.000988 [-0.003378,+0.005312] | -0.002759 [-0.023859,+0.011434] | 3/2/11 | 0 |
| Bijection_live_minus_Bijection_shuffle | +0.055254 [+0.002327,+0.129056] | +0.103582 [-0.001391,+0.254930] | 5/0/11 | 0 |
| Bijection_live_minus_Diagonal_live | -0.031352 [-0.076410,-0.002398] | -0.117231 [-0.267680,-0.003855] | 0/5/11 | 0 |
| Bijection_live_minus_Previous_independent | -0.033931 [-0.113730,+0.041657] | -0.254020 [-0.562358,-0.017184] | 1/4/11 | 0 |
| Bijection_live_minus_Previous_joint | +0.027155 [-0.047419,+0.130609] | -0.067969 [-0.305960,+0.081551] | 3/2/11 | 0 |

## Prelocked decision

{'status': 'evaluated', 'passed': False, 'rule': 'Expand only if both-domain Bijection_live minus E complete-v and unobserved actual-s positive, and Bijection_live minus Fixed/Shuffle/Direct/matchedDiagonal complete-v positive; CIs remain uncertainty', 'all128fits256rounds_actually_ran': True, 'decision_scope': 'nine-patch bijective visual feedback with fixed-scoped independent center categorical OPD', 'independent_confirmation': False}

Bijection Live fails to improve both domains over E and is lower than matched DiagonalLive in both complete-v means. Allowing one-to-one spatial rearrangement has not added useful localization correction in this fixed feature/state setting. Do not enlarge, scan assignment penalties/patchgrids/temperature/rounds, or rewrite prior gates. This is not a claim all optimal transport, frozen features or OPD are impossible. Larger numerical assignment score alone is specifically excluded as a success criterion.

FEEDBACK_DIAGNOSTICS_PUBLIC.json reports E-relative raw/posterior beneficial-candidate misses and AUC at the same sealed34candidate locations, plus patch reassignment/visibility and actual state movement. This GT analysis occurs only after prediction seal and creates no new outputs, filters, acceptance rules or checkpoints. Missing GT stays unknown. Official interpolated frames are not additional independent supervision.

Independent root audit uses its own NumPy bilinear sampler/normalization and a subset-bitmask dynamic program to verify assignment optimum, not the experimental SciPy solver. It also verifies each permutation uses9slots once, matching>=diagonal, all saved categorical gradients/state/output steps, unchanged anchors/scope/sizes/time, both reused predictions, independent official metrics and parent paired bootstraps. No model/GPU audit rerun.

Incremental costs excluding old Source/E/DINO/features: {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_GPU_seconds': 0, 'actual_fits': 128, 'actual_rounds': 256, 'CPU_prediction_and_full_pin_verification_seconds': 11.087205891031772, 'four_fit_seconds': 5.584419196704403, 'old_Source_E_DINO_feature_cost_excluded': True, 'not_first_online_latency': True}

## Every signed full-v change vs E, pp

| Alias | Target | BijectionLive | DiagonalLive | Fixed |
|---|---|---:|---:|---:|
| NH001 | hc2 | +0.007644 | +0.077809 | +0.003438 |
| NH002 | hc2 | +0.035826 | +0.053149 | +0.033388 |
| NH003 | hc2 | +0.055530 | +0.046606 | +0.052287 |
| NH004 | hc2 | +0.033553 | +0.050530 | +0.032653 |
| NH005 | hc2 | +0.001794 | -0.002560 | +0.001922 |
| NH006 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH007 | hc2 | -0.141588 | -0.122470 | -0.134308 |
| NH008 | hc2 | +0.080299 | +0.094788 | +0.069505 |
| NH009 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH010 | hc2 | -0.125451 | -0.059737 | -0.115158 |
| NH011 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | -0.284963 | -0.119992 | -0.253935 |
| NH014 | hc2 | +0.152649 | +0.227984 | +0.191864 |
| NH015 | hc2 | -0.037598 | -0.088857 | -0.024282 |
| NH016 | hc2 | -0.025529 | -0.039666 | -0.023712 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +0.774635 | +0.807384 | +0.751802 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.025203 | +0.102472 | +0.019853 |
| NV008 | vidstg | +0.107556 | +0.437333 | +0.094348 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | -0.255805 | -0.195376 | -0.233388 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | -0.086270 | -0.084865 | -0.083096 |
