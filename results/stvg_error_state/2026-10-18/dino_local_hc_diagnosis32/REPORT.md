# HC local residual decline: cached teacher, aggregation and amplitude diagnosis

This analysis uses the same exposed HC16/Vid16. No new prediction, GT read/scoring, model/expert call, training, parameter search or amplitude trial. All numbers derive from saved 505103c1 feedback/IoU scalars and d9f0c700 output metrics.

## Teacher feedback

| Query-mean gated quality difference, pp | HC (16 known-support queries) | Vid (10 active known-support queries) |
|---|---:|---:|
| teacher_minus_input_gated_pp | +1.244129 [+0.447901, +2.162485] | +5.621234 [+2.800526, +8.666055] |
| teacher_minus_E_gated_pp | +0.018568 [-0.240032, +0.229764] | +1.070208 [+0.141757, +2.062478] |
| E_minus_input_gated_pp | +1.225561 [+0.412137, +2.183618] | +4.551026 [+2.056222, +7.422675] |

At fixed known nodes, teacher-minus-input = teacher-minus-E + E-minus-input exactly. HC +1.244129 = +0.018568 +1.225561 pp. Thus positive feedback relative to sampled inputs is mostly a restoration of perturbation damage in this arithmetic comparison, not established additional quality above E. This is not a causal percentage or an estimate of new deployment gain.

HC has 133 accepted feedback instances with saved GT quality: 64 improve over E, 66 worsen and 3 tie. Another42 accepted instances have no saved GT quality and stay unknown. These are repeated candidate-anchor instances, not independent observations or identity labels.

## Actual residual aggregation

For each node, r = sum(m*(T-E))/24 - sum(m*(input-E))/24, in center/log-size coordinates. Both terms are measured exactly. Dividing by24 does not cancel the admitted inputs: admission varies with sampled state and the finite admitted subset need not have zero mean. No-feedback nodes still have exactly zero residual; this is not the old no-feedback random-mean update bug.

| Finite-set property | HC | Vid |
|---|---:|---:|
| nodes_with_feedback | 33 | 33 |
| accepted_total | 175 | 630 |
| input_norm_larger_than_teacher | 27 | 3 |
| applied_opposite_teacher | 8 | 1 |
| admission_contrast_larger_than_all24_mean | 30 | 12 |
| mean_coherence_query_mean | 0.7229755197044263 | 0.5400357615927048 |
| leave_one_can_reverse_direction | 0 | 0 |

Vector norms and cosines use the previously frozen source scales. HC input-displacement norm exceeds the teacher-displacement norm in27/33 supported nodes, versus3/33 in Vid. The actual HC residual points opposite the mean teacher-displacement term in8/33 such nodes (1/33 Vid). This explains why applying the average teacher correction at E is not equivalent to adopting the teacher output. These are geometrical facts, not GT directional derivatives.

For the admitted-input term, p*mean(all24 input displacement) and p*(mean(admitted)-mean(all24)) sum exactly. The latter has larger norm in30/33 HC supported nodes. This is a finite-set admission contrast; it does not establish a population bias or isolate admission causality.

HC corrections have higher mean coherence than Vid, not lower. Fixed-denominator leave-one-term analysis produces no genuine direction reversal in either domain after roundoff guards. A single term can exceed the net HC residual magnitude at one node, but this alone does not explain task loss. The evidence does not support a generic claim that averaging24 samples destroys the signal through cancellation.

## Applied amplitude

| Geometric amplitude / response | HC | Vid |
|---|---:|---:|
| median_accepted_per_supported_node | 6.000000000 | 22.000000000 |
| mean_center_relative_E_size_all64 | 0.006862771 | 0.013268768 |
| max_center_relative_E_size | 0.054751814 | 0.087930013 |
| max_log_size_norm | 0.070413734 | 0.128868592 |
| median_teacher_input_dispersion_ratio | 0.065841443 | 0.025654777 |
| median_correction_on_input_slope | -0.986997360 | -0.990564758 |

The center is normalized by the E box width/height. HC mean center change over all64 planned nodes is about0.006863 in these units; the maximum is0.054752. Its maximum log-size change norm is0.070414. HC changes are not globally larger than Vid. Original d9f0c700 mean image-center/log-size displacement is HC0.002837/0.014676 versus Vid0.004007/0.020298. This does not prove every HC step is safe: a small change can have the wrong direction or affect important interpolated frames.

Across the29 HC supported nodes with at least two accepted samples, the median teacher-to-input dispersion ratio is0.065841 and the correction-on-input slope is -0.986997. In these admitted same-frame states the teacher mostly returns nearby outputs while correcting input variation. This does not prove it ignores reference boxes or establish identity correctness. The corresponding Vid values are0.025655 and -0.990565, so convergence to similar outputs is not by itself a domain-specific failure; the value of those outputs relative to E differs.

This update is an external closed-form coordinate operation, not a QR/Adam/decoder-backprop step. The applied residual exactly matches the saved arithmetic. Old decoder-response failures are not evidence that this output-layer update overshoots.

No alternative amplitude, GT derivative or checkpoint was evaluated. Therefore this study cannot determine whether shrinking the same direction would rescue each case, and it does not select a learning rate. The priority supported by these data is distinguishing useful E-improvement feedback from corrections of sampled-input perturbations, not assuming all failures are overshoot.

## Per-query HC accounting

| Alias | Teacher-input gated pp | Teacher-E gated pp | Actual observed sIoU delta pp | Complete vIoU delta pp | Interpolated v contribution pp |
|---|---:|---:|---:|---:|---:|
| H007 | -0.035288 | -0.026166 | -0.078016 | -0.008113 | -0.007182 |
| H008 | +1.024935 | -0.104320 | +0.524448 | +0.056928 | +0.050095 |
| H010 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| H018 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| H022 | +3.475384 | +0.201224 | -1.110093 | -0.121757 | -0.095482 |
| H023 | +0.000000 | +0.000000 | +0.000000 | -0.003461 | -0.003461 |
| H031 | +4.080471 | +0.803507 | -1.125214 | -0.058906 | -0.050926 |
| H032 | +1.951401 | +0.695483 | -0.018543 | -0.161724 | -0.160997 |
| H035 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| H037 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| H043 | +2.714620 | -1.503746 | +0.035135 | +0.016504 | +0.015352 |
| H045 | +1.011705 | +0.460734 | -1.445571 | -0.033748 | -0.030125 |
| H050 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| H058 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| H060 | +0.000000 | +0.000000 | +0.000000 | +0.001324 | +0.001324 |
| H061 | +5.682835 | -0.229635 | -1.438159 | -0.171277 | -0.150584 |

H061: teacher repairs sampled inputs by+5.682835pp but is -0.229635pp below E in gated quality. All four supported nodes have an admitted-input norm larger than the teacher-displacement norm; actual observed sIoU loses1.438159pp and complete vIoU loses0.171277pp.

H031: teacher quality exceeds E by+0.803507pp in this accounting, yet the actual local update loses1.125214pp observed sIoU. At the second supported node, the teacher is near E (gated+0.118338pp) while the admitted-input term is much larger. Positive average teacher quality alone does not validate transporting its correction to E.

H045: the only known-GT accepted correction is one sample at the fourth node. That teacher is about11.058pp above E (accepted-only, before dividing24), but its correction applied at E points opposite the E-to-teacher mean in source-scaled coordinates (cosine about -0.637). No new alternate prediction is scored.

H032: observed sIoU changes only -0.018543pp, while complete vIoU loses0.161724pp;0.160997pp of the latter comes from official interpolation. This is the saved endpoint decomposition, not evidence that actual unobserved decoder nodes changed.

Keep positive controls: H008 gains0.056928pp complete vIoU despite its gated teacher-minus-E being -0.104320pp; H043 gains0.016504pp while that quantity is -1.503746pp. The residual transport can move differently from the teacher endpoint. These counterexamples rule out explaining all cases by a single teacher-better/worse threshold.

## Scope and decision

Saved sampled-GT quality is available for43/64 HC planned nodes; actual teacher feedback occurs at25 known and8 unknown nodes. Unknown feedback is not counted as failure. All16 HC queries remain in complete/observed endpoints; zero gated contribution means no observed accepted contribution at known nodes, not a correctness label for unknown feedback. Vid teacher-quality comparison uses10 active known queries; all16 remain in complete vIoU.

The previously measured HC decline remains -0.030264pp complete vIoU and -0.291001pp observed actual sIoU; unobserved actual nodes remain exact E. Teacher added value above E is not established. Finite admitted-input subtraction materially changes the average correction; simple direction cancellation is not the dominant evidence, and overshoot remains unisolated. Do not adjust threshold, candidate count or step size from these diagnostic GT values. No dynamic OPD, expanded cohort or method promotion.

Engineering recovery records are retained locally: JSON conversion for NumPy scalars and roundoff guards for deleting the sole residual term. All original inputs, predictions and GT scores are unchanged; no model or scientific prediction was rerun.
