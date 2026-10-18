# Cached diagnosis of24-state versus center feedback

Original new-parent HC16/Vid16, frozen TA-STVG Source/E, aligned content-mask, .35, four original anchors, denominator24, final-MBR. Read existing prediction traces and already sealed query metrics only. No model/weights/media/GT read, new prediction/scoring, amplitude search, sample expansion or dynamic OPD. Historical official full-result exposure remains.

## Exact accounting

With e=z(E), c=accepted center endpoint, t=mean accepted sampled endpoint, p=n/24 and center mask M: Uc=M(c-e), Up=pUc, Us=p(t-e). Us-Uc=(Up-Uc)+(Us-Up). On common-admitted nodes the terms are (p-1)(c-e) and p(t-c); sampled-only nodes add admission-support differences to the second term. Same p is not equal displacement norm. Geometry and nonlinear task-metric differences are not interchangeable.

The saved complete-score path is exact: Samples24-Center_full=(Center_p-Center_full)+(Samples24-Center_p). It is a path-specific arithmetic accounting, not a unique causal attribution without interactions.

## hc2: complete tube, pp

| Saved contrast | Mean [95% paired interval] |
|---|---:|
| total_gap | -0.083456 [-0.269968, +0.065613], n=16 |
| p_scaling_path | -0.098512 [-0.275253, +0.040218], n=16 |
| endpoint_and_support_path | +0.015056 [-0.009620, +0.052012], n=16 |
| center_gain | +0.178846 [-0.015042, +0.431654], n=16 |
| sample_gain | +0.095389 [+0.011352, +0.198365], n=16 |

| Common-admitted saved sIoU contrast | Mean [95% paired interval] |
|---|---:|
| total_gap | -2.258122 [-4.644967, -0.181792], n=12 |
| p_scaling_path | -2.292432 [-4.661616, -0.218568], n=12 |
| endpoint_path | +0.034310 [-0.080566, +0.143031], n=12 |

## Admission and endpoint geometry

Active 15/16; all64 planned-node support counts: {'neither': 16, 'sample_only': 5, 'common': 41, 'center_only': 2}. Counts keep no-support/illegal/no-feedback nodes.

| Admission quantity | Value |
|---|---:|
| all64_mean | 0.207031250 |
| sample_supported_mean | 0.288043478 |
| common_mean | 0.304878049 |
| common_median | 0.291666667 |
| active_legal_fraction | 0.783333333 |
| accepted_given_legal_all_active | 0.278340650 |
| center_only_nodes | 2.000000000 |
| sample_only_nodes | 5.000000000 |

p=(legal_states/24)*(accepted/ legal_states). Whole-state legality is shared across four anchors: an invalid anchor can remove a sampled state from every anchor. No legal-state resampling was performed. Admission fraction is not a correctness probability.

| Common-admitted endpoint geometry, no GT | Mean |
|---|---:|
| endpoint_mean_center_IoU | 0.978333807 |
| endpoint_mean_center_source_scaled_distance | 0.095013530 |
| endpoint_direction_cosine | 0.911020861 |
| endpoint_unshrunk_norm_ratio | 0.997350660 |
| amplitude_term_norm | 0.303500904 |
| endpoint_and_support_term_norm | 0.027802295 |

| Within-admitted-node dispersion | Value |
|---|---:|
| nodes_with_two_or_more | 42.000000000 |
| mean_teacher_over_input_RMS | 0.110737540 |
| median_teacher_over_input_RMS | 0.057886583 |

Endpoint IoU compares sampled encoded-mean endpoint to the center teacher endpoint, not to GT. Dispersion compares accepted teacher outputs against their exact cached input references in source-scaled coordinates. Concentrated outputs do not prove correctness or identity. Raw per-state GT IoU and the unshrunk mean endpoint GT score were not cached here and remain unmeasured.

| Actual displacement, all64 planned nodes | Source-scaled4D norm | Center image norm | Log-size norm |
|---|---:|---:|---:|
| Center_full_local | 0.315319049 | 0.012067707 | 0.048480670 |
| Samples24_endpoint | 0.098543123 | 0.004390483 | 0.014401812 |
| Center_p_local | 0.086850992 | 0.003409792 | 0.013359890 |

| Actual displacement, common-admitted nodes | Source-scaled4D norm | Center image norm | Log-size norm |
|---|---:|---:|---:|
| Center_full_local | 0.439073183 | 0.017016870 | 0.069154992 |
| Samples24_endpoint | 0.134382352 | 0.005616879 | 0.020823053 |
| Center_p_local | 0.135572280 | 0.005322602 | 0.020854462 |

## Descriptive groups by already saved center-minus-E result

| Center effect | Queries | Mean p on center-admitted nodes | p-scaling contribution to full16 mean | Endpoint/support contribution to full16 mean |
|---|---:|---:|---:|---:|
| benefit | 8 | 0.325757576 | -0.136193 | +0.017154 |
| harm | 3 | 0.263888889 | +0.037680 | -0.002097 |
| unchanged | 5 | 0.240740741 | +0.000000 | +0.000000 |

These query-level groups are explanatory, not node identity/accuracy labels or deployment gates. All contributions retain the original full16 denominator.

Mechanically selected lowest/highest3 Samples24-Center_full queries: {'lowest3': ['NH004', 'NH008', 'NH003'], 'highest3': ['NH014', 'NH007', 'NH011']}. All signed rows below remain primary, including unchanged and unknown common-support outcomes.

## vidstg: complete tube, pp

| Saved contrast | Mean [95% paired interval] |
|---|---:|
| total_gap | -0.012089 [-0.027305, -0.000147], n=16 |
| p_scaling_path | -0.013693 [-0.031357, -0.000138], n=16 |
| endpoint_and_support_path | +0.001604 [-0.000064, +0.004240], n=16 |
| center_gain | +0.063564 [+0.000000, +0.139861], n=16 |
| sample_gain | +0.051475 [-0.000809, +0.115246], n=16 |

| Common-admitted saved sIoU contrast | Mean [95% paired interval] |
|---|---:|
| total_gap | -0.211757 [-0.378620, -0.050196], n=6 |
| p_scaling_path | -0.246890 [-0.436377, -0.072645], n=6 |
| endpoint_path | +0.035133 [-0.011877, +0.084867], n=6 |

## Admission and endpoint geometry

Active 11/16; all64 planned-node support counts: {'neither': 35, 'common': 22, 'sample_only': 7}. Counts keep no-support/illegal/no-feedback nodes.

| Admission quantity | Value |
|---|---:|
| all64_mean | 0.274088542 |
| sample_supported_mean | 0.604885057 |
| common_mean | 0.765151515 |
| common_median | 0.708333333 |
| active_legal_fraction | 0.837121212 |
| accepted_given_legal_all_active | 0.535984848 |
| center_only_nodes | 0.000000000 |
| sample_only_nodes | 7.000000000 |

p=(legal_states/24)*(accepted/ legal_states). Whole-state legality is shared across four anchors: an invalid anchor can remove a sampled state from every anchor. No legal-state resampling was performed. Admission fraction is not a correctness probability.

| Common-admitted endpoint geometry, no GT | Mean |
|---|---:|
| endpoint_mean_center_IoU | 0.998090117 |
| endpoint_mean_center_source_scaled_distance | 0.032712299 |
| endpoint_direction_cosine | 0.999694120 |
| endpoint_unshrunk_norm_ratio | 1.003754765 |
| amplitude_term_norm | 0.407177181 |
| endpoint_and_support_term_norm | 0.022863709 |

| Within-admitted-node dispersion | Value |
|---|---:|
| nodes_with_two_or_more | 28.000000000 |
| mean_teacher_over_input_RMS | 0.031250066 |
| median_teacher_over_input_RMS | 0.024426729 |

Endpoint IoU compares sampled encoded-mean endpoint to the center teacher endpoint, not to GT. Dispersion compares accepted teacher outputs against their exact cached input references in source-scaled coordinates. Concentrated outputs do not prove correctness or identity. Raw per-state GT IoU and the unshrunk mean endpoint GT score were not cached here and remain unmeasured.

| Actual displacement, all64 planned nodes | Source-scaled4D norm | Center image norm | Log-size norm |
|---|---:|---:|---:|
| Center_full_local | 0.574644423 | 0.005014014 | 0.026410641 |
| Samples24_endpoint | 0.475273908 | 0.004202413 | 0.021349146 |
| Center_p_local | 0.434677267 | 0.003875002 | 0.019785141 |

| Actual displacement, common-admitted nodes | Source-scaled4D norm | Center image norm | Log-size norm |
|---|---:|---:|---:|
| Center_full_local | 1.671692867 | 0.014586224 | 0.076830957 |
| Samples24_endpoint | 1.269403292 | 0.011316558 | 0.057700316 |
| Center_p_local | 1.264515686 | 0.011272733 | 0.057556774 |

## Descriptive groups by already saved center-minus-E result

| Center effect | Queries | Mean p on center-admitted nodes | p-scaling contribution to full16 mean | Endpoint/support contribution to full16 mean |
|---|---:|---:|---:|---:|
| benefit | 4 | 0.810606061 | -0.013693 | +0.001471 |
| harm | 1 | 0.989583333 | +0.000000 | +0.000010 |
| unchanged | 11 | 0.565476190 | +0.000000 | +0.000123 |

These query-level groups are explanatory, not node identity/accuracy labels or deployment gates. All contributions retain the original full16 denominator.

Mechanically selected lowest/highest3 Samples24-Center_full queries: {'lowest3': ['NV005', 'NV008', 'NV016'], 'highest3': ['NV015', 'NV014', 'NV010']}. All signed rows below remain primary, including unchanged and unknown common-support outcomes.

## All signed query paths, pp

| Query | Domain | Samples24 minus center | p scaling path | Endpoint/support path | Center minus E | Samples24 minus E |
|---|---|---:|---:|---:|---:|---:|
| NH001 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH002 | hc2 | -0.068796 | -0.060831 | -0.007964 | +0.090425 | +0.021629 |
| NH003 | hc2 | -0.219606 | -0.245978 | +0.026372 | +0.383746 | +0.164140 |
| NH004 | hc2 | -0.938850 | -0.922312 | -0.016538 | +1.000429 | +0.061579 |
| NH005 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH006 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH007 | hc2 | +0.217139 | +0.213465 | +0.003674 | -0.181400 | +0.035740 |
| NH008 | hc2 | -0.884712 | -0.858509 | -0.026203 | +1.538001 | +0.653289 |
| NH009 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH010 | hc2 | +0.172143 | +0.156254 | +0.015889 | -0.172690 | -0.000547 |
| NH011 | hc2 | +0.330469 | +0.083815 | +0.246654 | +0.105836 | +0.436306 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | -0.174771 | -0.154360 | -0.020410 | +0.296953 | +0.122182 |
| NH014 | hc2 | +0.180044 | +0.233165 | -0.053121 | -0.373644 | -0.193600 |
| NH015 | hc2 | -0.073868 | -0.068249 | -0.005619 | +0.119991 | +0.046123 |
| NH016 | hc2 | +0.125507 | +0.047343 | +0.078164 | +0.053882 | +0.179389 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | -0.096684 | -0.102745 | +0.006060 | +0.427866 | +0.331182 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | -0.002827 | -0.002211 | -0.000616 | +0.060417 | +0.057590 |
| NV008 | vidstg | -0.072080 | -0.090735 | +0.018655 | +0.238363 | +0.166282 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.001974 | +0.000000 | +0.001974 | +0.000000 | +0.001974 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | +0.000156 | +0.000000 | +0.000156 | -0.074634 | -0.074478 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | -0.023959 | -0.023393 | -0.000566 | +0.365015 | +0.341056 |

## Scope and resource decision

Paired10000 bootstrap, seed20261018, query=parent unit. Candidate/node repetitions are not independent samples. No GT-based new method selection, no new endpoint variant or matching-norm intervention. All unobserved actual decoder nodes and MBR intervals remain the original predictions. Benchmark-interpolated effects use previously saved metrics, not new visual feedback.

Actual CPU analysis 0.241047s (excludes later audit/report/publication); GPU initialized false, new model/GT/media/weights/scoring/predictions/samples all0. Original predictions, raw traces, scores and CURRENT are hash preserved. Current results do not authorize extra rounds, threshold changes or dynamic OPD.
