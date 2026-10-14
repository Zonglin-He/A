# Matched-support spatial references: candidate, selection and transfer diagnosis

## Scope and evidence

Eight historically exposed sources, four per direction; 32 old QA centers. Nineteen centers have all displayed candidate boxes, Source, a real annotation, and a valid non-abstaining sealed R1 choice. These 19 centers form identical ordered support for all four reference roles. Thirteen centers are excluded by availability, never by the sign or size of IoU. HC01 has zero support and exact F/T reuse. No missing annotation is filled or interpolated into supervision.

Source and R1 are separately fitted, including R1's Source choices (17/19 anchors). Candidate-Pool Oracle selects the largest center IoU among already displayed candidates; GT Box Oracle uses the true sparse annotation. Both Oracles are explicitly label-assisted offline diagnostics. GT availability is also used to define this diagnostic support. These are not deployable results or independent confirmation. The pool maximum at sparse centers is not an Oracle for full-Tube vIoU after adaptation.

All nonempty roles use the same source checkpoint, identical initial 1792 Query/LN coordinates, original domain config, same Gaussian key, five original rounds and fresh Adam. All 28 fits, 140 actual updates and 28 final full-model redecodes completed. Eight source captures completed in the successful run; one additional prefit capture belongs to a preserved engineering stop. No new teacher call, candidate generation or temporal training was used in this comparison. F and old T are exactly reused. The five cases with identical Source/R1 anchors have separately executed fits whose saved dynamics and output match exactly.

## Full-Tube results

Domain means include all four sources, including HC01 no-update. T+S combines the newly decoded spatial boxes with the exact old T interval. The two columns have different paired baselines and must not be interchanged.

| Domain | Reference | Spatial delta vs Frozen (pp) | T+S delta vs T (pp) | S positive / negative / unchanged |
|---|---|---:|---:|---:|
| hc2 | source | -0.060362 | -0.324697 | 1/2/1 |
| hc2 | r1 | +0.012791 | -0.344224 | 1/2/1 |
| hc2 | pool_oracle | +0.622908 | +0.198296 | 1/2/1 |
| hc2 | gt_oracle | +3.997274 | +5.938989 | 3/0/1 |
| vidstg | source | -0.658057 | -0.849075 | 0/4/0 |
| vidstg | r1 | +0.894899 | +0.793474 | 1/3/0 |
| vidstg | pool_oracle | +0.846732 | +0.560697 | 1/3/0 |
| vidstg | gt_oracle | +8.354085 | +10.926908 | 4/0/0 |

Per-source Spatial-only changes from Frozen (pp):

| Historical source | Support | Source ref. | R1 ref. | Pool Oracle | GT Box Oracle |
|---|---:|---:|---:|---:|---:|
| HC01 | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| HC02 | 4 | -0.379690 | -0.379690 | -0.379690 | +7.347173 |
| HC03 | 3 | -0.554597 | -0.554597 | -0.170052 | +3.450835 |
| HC04 | 3 | +0.692840 | +0.985450 | +3.041376 | +5.191088 |
| V01 | 3 | -0.920704 | -0.920704 | -0.920704 | +13.053712 |
| V02 | 4 | -0.238659 | -0.238659 | -0.431327 | +1.741117 |
| V03 | 1 | -0.170155 | -0.170155 | -0.170155 | +4.445068 |
| V04 | 1 | -1.302708 | +4.909115 | +4.909115 | +14.176443 |

## What the controlled comparison establishes

GT Box Oracle improves all seven nonempty sources. The same update interface therefore has demonstrated capacity under correct sparse geometry on this fixed panel. Candidate-Pool Oracle improves only two sources and degrades five; its domain gains are much smaller. This supports a candidate-reference quality/transfer bottleneck relative to the GT control, not a claim that OPD can never work.

Pool selection loses available benefit to R1 in HC03/HC04, but geometry and transfer also matter. V02's pool reference is fitted more closely (about +20.10 pp mean reference IoU at supervised positions), while its full-Tube score changes by about -0.43 pp. Better fitting of a candidate target is therefore not task improvement. The locally highest-IoU candidate is not guaranteed to be the most useful sparse target for the full trajectory. V04 remains a positive real-reference example.

Source-reference self-distillation itself moves parameters and is not a zero-update baseline. Its average losses must remain visible. GT Oracle fitting and complete-Tube gain do not establish a generally correct gradient for an arbitrary pseudo-reference. No learning-rate, temperature, amplitude, best step or prompt was selected from these labels.

The old V01/V03 harmful anchors had no GT and are excluded from this common-support comparison. Their new Source/R1 equality does not prove that the old anchors were correct or incorrect. Sparse reference IoU, reference-fitting progress, supervised/unsupervised geometry and full-Tube vIoU are reported separately. Window counts are not independent sample counts.

## One conditional candidate test

The GT-versus-pool gap supports the already proposed single local visual-motion region test. It is a new candidate-construction mechanism only: camera-compensated residual motion, connected regions and reciprocal short-window matching on the existing cache. It uses no new detector/segmenter network, no copied non-Source grid seeds and no new frames. Candidate outputs must be sealed before the fixed offline coverage comparison. R1/OPD is conditional on actual improvement, and independent >=32/source-per-direction confirmation remains untriggered until a fixed GT-free spatial rule is supported. No fourth tracker or parameter sweep is authorized by this report.

## Readback, cost and limits

The CPU reader independently checks source/config/hash/frame identity, matched support, samples, rewards, likelihood, distillation weights, stored gradients, Adam arithmetic and final cached-versus-full decode. It does not independently run decoder backward. Metrics are independently recomputed with separate scalar arithmetic using the same official label loader. All 80 metrics, 20 groups, 76 sparse-reference qualities, 28 movement/fit-progress records and all 32 support decisions are preserved, including negative, neutral and unknown entries.

The worker directly opens no annotation file but consumes GT-derived reference locks; the overall experiment is GT-assisted. Successful worker wall time is 124.746990 seconds (includes source capture, fitting and saved IO, excludes independent review/publication and the earlier failed launch). This is diagnostic cost, not production end-to-end TTA latency. Preparation receipt-schema issues and the first prefit pixel-hash scope mismatch were preserved and resolved without changing support, references or science. CURRENT and stopped historical queues are unchanged.
