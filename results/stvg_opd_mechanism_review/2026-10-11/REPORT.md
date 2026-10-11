# Reviewer mechanism evidence: completed CPU diagnosis and queued transfer control

The primary method, old fits, admission rule, temporal interval, scientific parameters, and all running jobs remain unchanged. This package reads already globally sealed P2 Full/Direct fits. It performs no new model inference or adaptation.

Actual scope: 1,792 matched arrival cells, 3,584 complete saved fits, 3,553 expert-frame cells with an official annotation, 413,427 annotated event-frame cells, and 48,384 independent native/oracle scalar comparisons. The root independently checked 1,682,153 scalar aggregations/comparisons. Exact DINO reconstruction matches 512 original clean arrival inputs and 171,032 DINO coordinates against the sealed P1 stateless comparator.

Frames are averaged within arrival/condition/order, then conditions/orders within parent, then parents equally. Intervals are 10,000 paired-parent bootstrap draws, seed 20261011, unadjusted and descriptive. Sources have historical evaluation exposure; excluding the original development parents does not make them fresh. Missing frame groups stay missing.

## 1. Expert quality and finite-sample actionability

Q is the admitted expert-box IoU with the official annotation. The operational strata Q >= 0.5 and Q < 0.5 describe geometric agreement, not verified referent identity. A is the maximum **actual saved GPU IoU reward** in the first round. A == 0 means all sampled actions received zero reward in that observation/round, not mathematically unreachable Gaussian support. The complete continuous observations and every-round zero fraction remain in QUALITY_ROWS.jsonl.gz.

Primary results exclude original development parents. Full-current and Direct-current each use their own final-minus-before frame IoU. Their inherited Before states can differ; the paired algorithm contrast is not an isolated common-initial-state update experiment.

| Setting | Expert quality / first-round reward | Parents | Expert-frame cells | Full-current − Direct-current (pp, 95% CI) | Full harm | Direct harm |
|---|---|---:|---:|---|---:|---:|
| hc2/cross_clean | incorrect/zero | 22 | 72 | +11.520 [+4.066, +19.682] | 31.82% | 60.23% |
| hc2/cross_clean | incorrect/positive | 48 | 144 | +3.174 [+0.546, +6.738] | 44.27% | 48.44% |
| hc2/cross_clean | correct/zero | 5 | 24 | -29.329 [-38.573, -19.680] | 0.00% | 0.00% |
| hc2/cross_clean | correct/positive | 76 | 340 | -0.284 [-1.381, +0.876] | 10.64% | 15.62% |
| hc2/same_corruption | incorrect/zero | 32 | 159 | +9.751 [+1.474, +18.744] | 29.06% | 47.50% |
| hc2/same_corruption | incorrect/positive | 62 | 340 | +0.368 [-1.716, +2.496] | 43.68% | 48.08% |
| hc2/same_corruption | correct/zero | 11 | 57 | -40.875 [-61.903, -19.817] | 0.00% | 0.00% |
| hc2/same_corruption | correct/positive | 95 | 862 | -0.218 [-2.285, +1.384] | 18.07% | 18.70% |
| vidstg/cross_clean | incorrect/zero | 3 | 9 | +7.195 [-0.246, +19.636] | 41.67% | 91.67% |
| vidstg/cross_clean | incorrect/positive | 27 | 101 | +0.943 [-1.884, +4.082] | 34.88% | 39.20% |
| vidstg/cross_clean | correct/zero | 2 | 4 | -12.393 [-22.693, -2.092] | 0.00% | 0.00% |
| vidstg/cross_clean | correct/positive | 61 | 310 | -3.269 [-5.974, -0.948] | 14.00% | 10.79% |
| vidstg/same_corruption | incorrect/zero | 12 | 61 | +10.438 [-2.065, +26.085] | 42.50% | 58.33% |
| vidstg/same_corruption | incorrect/positive | 32 | 249 | -0.183 [-2.058, +2.042] | 49.27% | 48.85% |
| vidstg/same_corruption | correct/zero | 5 | 32 | -8.674 [-27.633, +4.020] | 0.00% | 0.00% |
| vidstg/same_corruption | correct/positive | 66 | 789 | -1.917 [-3.660, -0.133] | 10.10% | 9.39% |

Harm is a per-frame current change below −0.05, reduced with the same parent weighting; these percentages are not the number of parents that ever experienced a bad frame. Separate parent negative tails and all final-output/Before-history contrasts are in SUMMARY.json.

The HC low-Q/all-zero strata favor Full over Direct. However, the high-Q/all-zero strata favor Direct, often strongly, on small parent denominators. Zero reward is not an automatic safety guarantee: other observations, shared decoder parameters, inherited state, and optimizer history may still move that frame. The corresponding Vid low-Q strata have wide intervals. These findings support a conditional conservative-update hypothesis rather than universal OPD superiority.

### Conditional reward pilot

The resource trigger was fixed before this diagnostic readback: at least 10 affected parents, at least 25% of high-Q observations with all-zero first-round reward, and a positive paired Direct-current minus Full-current interval. This threshold controls whether to spend resources on one pilot; it is not evidence that rarer zero-reward failures are unimportant.

| Setting | High-Q observations | All-zero high-Q observations | Affected parents | Zero fraction | Trigger |
|---|---:|---:|---:|---:|---|
| hc2/cross_clean | 364 | 24 | 5 | 6.593% | False |
| hc2/same_corruption | 919 | 57 | 11 | 6.202% | False |
| vidstg/cross_clean | 314 | 4 | 2 | 1.274% | False |
| vidstg/same_corruption | 821 | 32 | 5 | 3.898% | False |

**Decision: not_triggered.** No GIoU-reward pilot was launched; no primary reward or parameter changed.

## 2. Real frame propagation

The four disjoint groups are admitted expert frames inside Native WHEN; unobserved actual decoder-output frames inside Native; interpolation-only frames inside Native; and annotated event frames outside Native. Interpolation is not a new decoder inference. DINO Refine uses the original absolute cxcywh anchor interpolation with nearest endpoint extension, and the same admitted references. It replaces the whole trajectory according to that original rule.

| Setting | Frame group | Eligible parents | Frame cells | After − Before (pp, 95% CI) | After − DINO (pp, 95% CI) |
|---|---|---:|---:|---|---|
| hc2/cross_clean | observed | 103 | 578 | +0.821 [-2.336, +3.755] | +1.629 [-2.521, +5.536] |
| hc2/cross_clean | unobserved_decoder | 112 | 2656 | -0.581 [-3.238, +1.865] | +4.394 [+0.193, +8.450] |
| hc2/cross_clean | interpolated | 112 | 22824 | -0.097 [-2.774, +2.394] | +3.606 [-0.431, +7.504] |
| hc2/cross_clean | outside_native | 111 | 21784 | -2.346 [-5.192, +0.200] | +15.543 [+10.288, +20.855] |
| hc2/same_corruption | observed | 121 | 1416 | +0.394 [-2.486, +3.105] | +3.345 [-0.504, +6.936] |
| hc2/same_corruption | unobserved_decoder | 126 | 11107 | -0.672 [-3.069, +1.536] | +6.595 [+3.002, +10.171] |
| hc2/same_corruption | interpolated | 126 | 89653 | -0.545 [-2.990, +1.723] | +6.129 [+2.616, +9.626] |
| hc2/same_corruption | outside_native | 90 | 17429 | -2.698 [-5.938, +0.278] | +13.705 [+7.389, +19.951] |
| vidstg/cross_clean | observed | 80 | 424 | +9.078 [+4.893, +13.301] | -4.367 [-8.764, +0.251] |
| vidstg/cross_clean | unobserved_decoder | 108 | 4406 | +5.419 [+2.594, +8.278] | +2.647 [-0.146, +5.777] |
| vidstg/cross_clean | interpolated | 108 | 24200 | +5.989 [+3.082, +8.910] | +2.046 [-0.742, +5.149] |
| vidstg/cross_clean | outside_native | 106 | 41250 | +2.533 [-0.763, +5.732] | +8.578 [+4.930, +12.402] |
| vidstg/same_corruption | observed | 88 | 1131 | +4.871 [-0.289, +9.993] | +0.332 [-3.684, +4.638] |
| vidstg/same_corruption | unobserved_decoder | 112 | 16926 | +0.729 [-2.956, +4.247] | +6.734 [+3.713, +9.908] |
| vidstg/same_corruption | interpolated | 112 | 97102 | +1.010 [-2.710, +4.707] | +5.772 [+2.676, +9.009] |
| vidstg/same_corruption | outside_native | 113 | 60541 | -0.666 [-3.428, +1.942] | +12.100 [+8.247, +15.931] |

After−Before is the current-query effect; After−DINO compares complete algorithm histories against parameter-free refinement. They answer different questions. The Vid cross-domain current effect is positive on real unobserved decoder frames; the HC current effect is not positive by this test. Do not attribute the complete inherited-state gain to the current sparse update.

### Common evaluable parents

| Setting | Unobserved group | Common observed/comparison parents | After − Before on comparison (pp, CI) | After − DINO on comparison (pp, CI) |
|---|---|---:|---|---|
| hc2/cross_clean | unobserved_decoder | 103 | -0.721 [-3.629, +1.959] | +4.129 [-0.382, +8.561] |
| hc2/cross_clean | interpolated | 103 | -0.174 [-3.209, +2.525] | +3.330 [-1.006, +7.483] |
| hc2/cross_clean | outside_native | 87 | -1.929 [-5.132, +0.931] | +15.042 [+8.772, +21.081] |
| hc2/same_corruption | unobserved_decoder | 121 | -0.675 [-3.094, +1.616] | +6.570 [+2.763, +10.312] |
| hc2/same_corruption | interpolated | 121 | -0.541 [-2.980, +1.789] | +6.094 [+2.372, +9.772] |
| hc2/same_corruption | outside_native | 85 | -2.664 [-6.107, +0.520] | +13.250 [+6.546, +19.802] |
| vidstg/cross_clean | unobserved_decoder | 80 | +7.021 [+3.450, +10.534] | +3.436 [-0.287, +7.546] |
| vidstg/cross_clean | interpolated | 80 | +7.762 [+4.056, +11.389] | +2.587 [-1.140, +6.708] |
| vidstg/cross_clean | outside_native | 65 | +4.948 [+0.514, +9.331] | +9.597 [+4.611, +14.885] |
| vidstg/same_corruption | unobserved_decoder | 88 | +1.566 [-2.968, +6.007] | +6.551 [+2.908, +10.431] |
| vidstg/same_corruption | interpolated | 88 | +1.929 [-2.776, +6.487] | +5.223 [+1.496, +9.040] |
| vidstg/same_corruption | outside_native | 78 | -1.020 [-4.701, +2.503] | +12.895 [+7.884, +17.897] |

Outside Native WHEN, the spatial prediction can change geometrically, but task-gated coverage stays zero for every arm because the interval is unchanged. These frames do not become recovered event support. SUMMARY.json retains observed-group results on each identical parent intersection.

## 3. Offline spatial and temporal oracle headroom

SpatialOracle uses GT space on its real annotated support with Native WHEN. TemporalOracle uses GT WHEN with original native box interpolation and zero contribution outside saved decoder support. Missing spatial predictions are never filled with GT in TemporalOracle. Oracles are offline interventions and are not deployable baselines or additive error decomposition.

| Setting | Parents | Native mvIoU (%) | SpatialOracle (%) | TemporalOracle (%) | Spatial headroom (pp, CI) | Temporal headroom (pp, CI) | Native event coverage (%) |
|---|---:|---:|---:|---:|---|---|---:|
| hc2/cross_clean | 128 | 22.156 | 42.239 | 51.417 | +20.083 [+17.139, +23.257] | +29.261 [+25.614, +32.921] | 55.834 |
| hc2/same_corruption | 128 | 29.094 | 58.236 | 49.184 | +29.143 [+26.247, +32.177] | +20.090 [+17.495, +22.720] | 86.466 |
| vidstg/cross_clean | 128 | 14.466 | 36.614 | 38.213 | +22.148 [+18.808, +25.523] | +23.748 [+20.427, +27.267] | 56.898 |
| vidstg/same_corruption | 128 | 23.594 | 48.411 | 48.072 | +24.817 [+20.957, +28.843] | +24.478 [+20.701, +28.394] | 64.329 |

Cross-domain temporal headroom exceeds spatial headroom in both directions; a universal spatial-dominant-bottleneck claim is unsupported. Native event coverage is approximately 56%, and the fixed interval remains an explicit limitation. Unsupported saved-decoder event support is 0% for HC and approximately 0.404% for cross-domain Vid; both the ideal joint oracle and support-aware joint oracle are reported separately.

## 4. Cross-parent transfer: incomplete GPU work

The three independently realized 128-parent permutations already provide 2,304 Full/Reset/Full-Carry fits. Only LN-only is missing: 768 new formal fits on the exact three original orders, with 24 complete old/new GPU qualification fits at 12 first-two-arrival cells. The CPU contracts and four saved historical LN-only fit math dictionaries passed; these are not GPU qualifications.

The unique CPU dependency waiter is launched. It waits for the running calibrated baseline and audited-cost/root jobs and the previously authorized expanded 905-parent state/timing jobs to complete and exit before acquiring the same single GPU. No new LN-only qualification, formal prediction, or GT score exists yet. Full−Reset, Full−Full-Carry and Full−LN-only must be reported per real permutation and pooled after all new formal predictions seal. Bootstrap draws do not count as arrival-order repetitions.

The larger 905-parent/5,430-fit Full/alpha0/alpha1 comparison and the 128-path Full/Frozen uncached deployment timing remain separate queued studies. End-to-end efficiency and stable carryover value are not newly established by this completed CPU diagnostic.

## 5. Evidence and manuscript consequences

- Preserve the full-query system improvement (+2.566 pp HC, +2.968 pp Vid) and restore the actual DINO Refine row. DINO slightly exceeds Full on Vid R@0.3; best-value formatting must reflect it.
- Restore Direct in the matched 128-parent P2 table. Never mix that small-cohort contrast into the full-query main table as if evaluated on every official query.
- Explain fixed-covariance, detached-weight OPD as weighted regression in mean/logit space, with resampling as the distinctive update process; the frozen expert supplies one box, not independent semantic judgments for each action.
- Report score threshold 0.35, distinct-candidate margin 0.05, duplicate NMS IoU 0.5, and the no-admitted-observation identity After=Before (Before need not be Source Frozen).
- Keep Native WHEN fixed, separate real decoder propagation from interpolation and geometric outside-interval changes, and retain both positive and negative conditional results.
- Report a spatial-adaptation framework with a conditional actionability tradeoff; stable advantage over Direct and stable carryover over complete reset still require their direct comparisons.

All three final scalar PDF pages were actually rendered and visually checked after a display-only layout correction; original renders and receipt remain preserved. No RGB, query text, official GT box geometry, source weights, raw actions, or saved fit payload belongs in the public export. Anonymous continuous IoU/reward/delta rows do.
