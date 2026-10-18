# E-relative candidate headroom and cosine ranking diagnosis32

Only read existing commit30746bcc initial34-state candidate grids, rewards/posteriors and actual update paths. Same exposed NH16/NV16 one-query/parent. No new network/DINO call, candidate generation, student update, sweep, dataset expansion, or change to original decisions. Ground truth is explicitly used ONLY for this offline diagnosis.

E included as a valid baseline even if its own ROI has no grid support; alternatives require original saved visibility. Planned four nodes fixed E; widths/height/MBR unchanged. Missing actual-node GT stays unknown. Two center duplicates remain in34 states; do not treat candidates as independent data.

Two oracle concepts are different: actual-node per-frame GT best maximizes known-node sIoU; exact finite-product DP maximizes complete official vIoU including interpolation. The former is NOT an upper bound on interpolated complete v. The latter is an upper bound only within this original finite product, may switch identity/produce implausible trajectories, and uses GT at benchmark frames. Neither is deployable or a measured TTA gain.

| Diagnostic / existing output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Actual_node_oracle | 37.934525 | 18.417441 | 64.196362 | 53.870697 |
| Finite_pool_full_v_oracle | 38.146517 | 18.696316 | 61.576131 | 51.777919 |
| On_policy_dense | 35.325026 | 15.805278 | 56.756321 | 45.303659 |
| Cosine_readout | 34.933405 | 15.747606 | 55.839923 | 44.746585 |

## hc2 finite-pool ability

Queries=16; teacher=15; known unobserved actual nodes=325; unknown=651; counts={'meaningful_headroom': 121, 'meaningful_headroom_and_teacher': 102}.
Actual_node_oracle−E: v +2.629049 [+1.284051, +4.288725], n=16; unobserved actual s +7.536002 [+3.486832, +12.341831], n=16.
Finite_pool_full_v_oracle−E: v +2.841041 [+1.476950, +4.502415], n=16; unobserved actual s +4.915771 [+1.807440, +8.917590], n=16.
On_policy_dense−E: v +0.019550 [-0.136579, +0.182447], n=16; unobserved actual s +0.095962 [-0.358697, +0.501874], n=16.
Cosine_readout−E: v -0.372072 [-0.783913, -0.005178], n=16; unobserved actual s -0.820437 [-1.704121, -0.052311], n=16.

## hc2 E-relative ranking

Each statistic averages eligible frames within each query, then averages queries; below n is query/parent support, never frame-pair or candidate count. Good/harm strict difference1e-10; meaningful groups +/-5IoU pp. Raw reward and posterior evaluated separately, score ties1e-12. Posterior includes geometric proximity prior.

| Quantity | All known unobserved actual nodes, query-balanced | Only within fixed MBR |
|---|---:|---:|
| best_gain_pp | +7.536002 [+3.486832, +12.341831], n=16 | +6.931239 [+3.307698, +11.184688], n=16 |
| has_gain_1pp | +0.486865 [+0.338686, +0.638712], n=16 | +0.501723 [+0.368587, +0.637830], n=16 |
| has_gain_5pp | +0.332888 [+0.177635, +0.503183], n=16 | +0.330551 [+0.181009, +0.497636], n=16 |
| has_gain_10pp | +0.254000 [+0.114227, +0.415062], n=16 | +0.230845 [+0.102926, +0.375175], n=16 |
| raw_auc | +0.862272 [+0.800638, +0.918184], n=15 | +0.861866 [+0.796292, +0.919848], n=15 |
| posterior_auc | +0.826794 [+0.762448, +0.888825], n=15 | +0.831545 [+0.768976, +0.892315], n=15 |
| meaningful_raw_auc | +0.904679 [+0.815444, +0.974345], n=12 | +0.903445 [+0.810658, +0.976626], n=12 |
| meaningful_posterior_auc | +0.840370 [+0.759289, +0.919051], n=12 | +0.852940 [+0.774207, +0.927612], n=12 |
| strong_pool_raw_top_good | +0.607529 [+0.401620, +0.805452], n=12 | +0.586279 [+0.389310, +0.782407], n=12 |
| strong_pool_posterior_top_good | +0.472718 [+0.304878, +0.648148], n=12 | +0.472222 [+0.291667, +0.664141], n=12 |
| strong_pool_raw_top_harmful | +0.198909 [+0.050926, +0.378307], n=12 | +0.201073 [+0.058712, +0.376263], n=12 |
| strong_pool_posterior_top_harmful | +0.192626 [+0.054233, +0.370370], n=12 | +0.206124 [+0.063763, +0.382576], n=12 |
| strong_pool_no_good_outranks_E_raw | +0.359182 [+0.167814, +0.564062], n=12 | +0.398569 [+0.194844, +0.603956], n=12 |
| strong_pool_no_good_outranks_E_posterior | +0.496032 [+0.324735, +0.667328], n=12 | +0.500000 [+0.309312, +0.680556], n=12 |
| posterior_good_but_On_nonpositive | +0.047454 [+0.000000, +0.111111], n=12 | +0.037037 [+0.000000, +0.101852], n=12 |

Fractions/AUC are in [0,1], not pp; best_gain_pp is IoU percentage points. Supports vary when no good/bad pair, no teacher or no >=5pp-headroom frame exists. For teacher-only headroom, see RESULTS_PUBLIC.teacher_available_groups; no-teacher queries remain in all-query capacity/full-metric denominator.

Zero-full-v cases: []

## vidstg finite-pool ability

Queries=16; teacher=11; known unobserved actual nodes=542; unknown=896; counts={'meaningful_headroom': 137, 'meaningful_headroom_and_teacher': 106}.
Actual_node_oracle−E: v +2.661509 [+0.974833, +4.818561], n=16; unobserved actual s +8.422224 [+4.315999, +13.439028], n=16.
Finite_pool_full_v_oracle−E: v +2.940385 [+1.131704, +5.235998], n=16; unobserved actual s +6.329446 [+2.125667, +11.634118], n=16.
On_policy_dense−E: v +0.049347 [-0.020258, +0.161412], n=16; unobserved actual s -0.144814 [-0.619163, +0.203951], n=16.
Cosine_readout−E: v -0.008326 [-0.163984, +0.119465], n=16; unobserved actual s -0.701889 [-1.940354, +0.118240], n=16.

## vidstg E-relative ranking

Each statistic averages eligible frames within each query, then averages queries; below n is query/parent support, never frame-pair or candidate count. Good/harm strict difference1e-10; meaningful groups +/-5IoU pp. Raw reward and posterior evaluated separately, score ties1e-12. Posterior includes geometric proximity prior.

| Quantity | All known unobserved actual nodes, query-balanced | Only within fixed MBR |
|---|---:|---:|
| best_gain_pp | +8.422224 [+4.315999, +13.439028], n=16 | +9.143124 [+4.154128, +15.197542], n=13 |
| has_gain_1pp | +0.612185 [+0.460362, +0.761290], n=16 | +0.641996 [+0.453979, +0.820235], n=13 |
| has_gain_5pp | +0.400634 [+0.243498, +0.575506], n=16 | +0.431205 [+0.236330, +0.646305], n=13 |
| has_gain_10pp | +0.309297 [+0.150963, +0.494318], n=16 | +0.329610 [+0.140310, +0.557929], n=13 |
| raw_auc | +0.716901 [+0.579578, +0.839531], n=9 | +0.841701 [+0.796357, +0.890331], n=5 |
| posterior_auc | +0.795777 [+0.688312, +0.869733], n=9 | +0.833612 [+0.784047, +0.884768], n=5 |
| meaningful_raw_auc | +0.823420 [+0.687035, +0.929623], n=8 | +0.933422 [+0.894899, +0.971945], n=5 |
| meaningful_posterior_auc | +0.879225 [+0.845279, +0.909780], n=8 | +0.902134 [+0.874977, +0.928823], n=5 |
| strong_pool_raw_top_good | +0.251449 [+0.052899, +0.451449], n=10 | +0.285714 [+0.071429, +0.571429], n=7 |
| strong_pool_posterior_top_good | +0.210021 [+0.051449, +0.391449], n=10 | +0.219388 [+0.040816, +0.433673], n=7 |
| strong_pool_raw_top_harmful | +0.215735 [+0.004348, +0.501449], n=10 | +0.020408 [+0.000000, +0.061224], n=7 |
| strong_pool_posterior_top_harmful | +0.107143 [+0.000000, +0.307143], n=10 | +0.010204 [+0.000000, +0.030612], n=7 |
| strong_pool_no_good_outranks_E_raw | +0.697101 [+0.497101, +0.894203], n=10 | +0.714286 [+0.428571, +0.928571], n=7 |
| strong_pool_no_good_outranks_E_posterior | +0.789979 [+0.608551, +0.948551], n=10 | +0.780612 [+0.566327, +0.959184], n=7 |
| posterior_good_but_On_nonpositive | +0.008592 [+0.000000, +0.022878], n=10 | +0.010204 [+0.000000, +0.030612], n=7 |

Fractions/AUC are in [0,1], not pp; best_gain_pp is IoU percentage points. Supports vary when no good/bad pair, no teacher or no >=5pp-headroom frame exists. For teacher-only headroom, see RESULTS_PUBLIC.teacher_available_groups; no-teacher queries remain in all-query capacity/full-metric denominator.

Zero-full-v cases: [{'alias': 'NV001', 'no_teacher': False, 'no_temporal_overlap': True, 'actual_center_changed': True}, {'alias': 'NV002', 'no_teacher': False, 'no_temporal_overlap': True, 'actual_center_changed': True}, {'alias': 'NV004', 'no_teacher': False, 'no_temporal_overlap': True, 'actual_center_changed': True}, {'alias': 'NV011', 'no_teacher': False, 'no_temporal_overlap': False, 'actual_center_changed': True}, {'alias': 'NV012', 'no_teacher': False, 'no_temporal_overlap': False, 'actual_center_changed': True}]

## All query diagnostic and actual changes versus E, pp

| Alias | Domain | Exact full-v oracle Δv | Actual-node oracle Δactual-s | On Δv | On Δactual-s |
|---|---|---:|---:|---:|---:|
| NH001 | hc2 | +3.268349 | +14.654519 | +0.115222 | +1.362845 |
| NH002 | hc2 | +0.102312 | +0.516377 | +0.216875 | +0.108639 |
| NH003 | hc2 | +2.511069 | +8.656645 | +0.224227 | +0.630750 |
| NH004 | hc2 | +4.943945 | +7.609727 | +0.653256 | +0.840522 |
| NH005 | hc2 | +1.575675 | +5.360133 | +0.086377 | +0.237937 |
| NH006 | hc2 | +9.238946 | +26.419990 | +0.236913 | +0.562736 |
| NH007 | hc2 | +0.029711 | +0.001799 | -0.449417 | -0.995964 |
| NH008 | hc2 | +2.602477 | +2.621495 | -0.099810 | -0.115479 |
| NH009 | hc2 | +0.082026 | +0.312016 | -0.197777 | -2.238840 |
| NH010 | hc2 | +1.545841 | +2.021794 | -0.281466 | -0.203321 |
| NH011 | hc2 | +4.421961 | +27.669194 | -0.277526 | +0.167646 |
| NH012 | hc2 | +11.120746 | +19.801034 | +0.000000 | +0.000000 |
| NH013 | hc2 | +1.076160 | +1.154399 | -0.324697 | -0.066954 |
| NH014 | hc2 | +0.661926 | +0.593761 | +0.067206 | +0.369340 |
| NH015 | hc2 | +0.943513 | +1.378261 | +0.711839 | +1.583901 |
| NH016 | hc2 | +1.331994 | +1.804893 | -0.368430 | -0.708367 |
| NV001 | vidstg | +0.000000 | +5.323884 | +0.000000 | -3.093729 |
| NV002 | vidstg | +0.000000 | +3.860760 | +0.000000 | -0.033083 |
| NV003 | vidstg | +0.800629 | +1.333973 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +10.441168 | +0.000000 | -0.510524 |
| NV005 | vidstg | +4.196527 | +4.367100 | +0.821782 | +1.226602 |
| NV006 | vidstg | +9.481953 | +22.599829 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.540229 | +2.602450 | -0.115594 | -0.408565 |
| NV008 | vidstg | +4.107331 | +9.071880 | +0.100922 | +0.358121 |
| NV009 | vidstg | +15.522745 | +26.805001 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.372219 | +1.765900 | +0.075385 | +0.330185 |
| NV011 | vidstg | +5.213856 | +30.893364 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.732290 | +5.077462 | +0.000000 | -0.002422 |
| NV013 | vidstg | +0.342535 | +2.735278 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.008769 | +0.000000 | -0.029544 | -0.118871 |
| NV015 | vidstg | +0.023484 | +0.144606 | +0.000000 | +0.000000 |
| NV016 | vidstg | +5.703590 | +7.732921 | -0.063399 | -0.064741 |

Not a claim that a GT oracle provides usable visual evidence. Raw reward AUC near chance suggests feedback limits in this pool; posterior can additionally favor E because of its sampling prior. Teacher top being useful and actual KL output not useful is a descriptive conversion discrepancy, not proof that the optimizer is broken or that argmax would be a good deployment rule.

Previous ridge/actual OPD failed/inconclusive decisions preserved. No automated next experiment is authorized by this report. CPU cached diagnosis excludes historical model/expert/feature cost.
