# Fixed v0: accepted replacement and feedback audit

All64 sealed stage-one queries are evaluated without changing candidates, histories, parameters, predictions or the failed development gate. No OPD/KD/Direct adaptation is run. The primary denominator below is actual saved non-E replacements at nodes with GT; it is not the full candidate pair pool.

| Saved control | Group | Replacements with GT | Harmed | Harm % | Drop >5 pp | Mean ΔIoU pp | Mean harmed ΔIoU pp |
|---|---|---:|---:|---:|---:|---:|---:|
| MemoryGreedy | all | 782 | 486 | 62.1483 | 427 | -6.6365 | -21.9286 |
| MemoryGreedy | observed | 114 | 77 | 67.5439 | 67 | -8.2402 | -23.2798 |
| MemoryGreedy | unobserved | 668 | 409 | 61.2275 | 360 | -6.3628 | -21.6742 |
| MemoryGreedy | near | 367 | 235 | 64.0327 | 202 | -7.6795 | -21.7489 |
| MemoryGreedy | middle | 245 | 142 | 57.9592 | 128 | -3.9042 | -19.8019 |
| MemoryGreedy | far | 56 | 32 | 57.1429 | 30 | -8.4901 | -29.4347 |
| NoMemoryGreedy | all | 791 | 489 | 61.8205 | 429 | -6.1970 | -21.7924 |
| NoMemoryGreedy | observed | 117 | 79 | 67.5214 | 69 | -8.7443 | -23.9693 |
| NoMemoryGreedy | unobserved | 674 | 410 | 60.8309 | 360 | -5.7549 | -21.3729 |
| NoMemoryGreedy | near | 372 | 238 | 63.9785 | 205 | -7.5072 | -21.5276 |
| NoMemoryGreedy | middle | 247 | 142 | 57.4899 | 127 | -3.3497 | -19.5887 |
| NoMemoryGreedy | far | 55 | 30 | 54.5455 | 28 | -4.7039 | -28.5912 |
| EvidenceDP | all | 746 | 456 | 61.1260 | 397 | -5.2015 | -21.0363 |
| EvidenceDP | observed | 119 | 81 | 68.0672 | 71 | -8.9185 | -23.8493 |
| EvidenceDP | unobserved | 627 | 375 | 59.8086 | 326 | -4.4961 | -20.4287 |
| EvidenceDP | near | 348 | 225 | 64.6552 | 191 | -6.5875 | -20.5186 |
| EvidenceDP | middle | 230 | 128 | 55.6522 | 115 | -2.4183 | -19.0744 |
| EvidenceDP | far | 49 | 22 | 44.8980 | 20 | 0.6047 | -27.3886 |

Observation/near/middle/far are mutually exclusive. The latter bins are fixed nearest-anchor physical distance divided by the median anchor gap: (0,.25], (.25,.5], and >.5. All replacement cases without GT remain counted separately, not treated as correct or incorrect.

## Same-state feedback and the fixed path score

`r(a)−r(E) = Δu + .25 Δincoming_link + .1 Δmemory_cos − .1 ΔG` is reconstructed at the saved memory and previous action. No alternate history is executed. The single-replacement fixed-score difference includes its unary, incoming edge and outgoing edge, and is independently checked against two complete fixed-table score evaluations. Both all-E-neighbor and saved-path-neighbor contexts are retained. Single-node differences must not be summed to infer the full path difference because edges overlap.

| Saved state subset | Comparisons | Teacher prefers alternative | Harmed preferred | Harm % among preferred |
|---|---:|---:|---:|---:|
| all_valid_alternatives | 3062 | 2025 | 1218 | 60.1481 |
| selected_initial_student | 595 | 423 | 261 | 61.7021 |
| unobserved_valid_alternatives | 2638 | 1701 | 1024 | 60.1999 |

## Observation-frame geometry

Native ReferDINO boxes are evaluated at all eight saved positions even when E-NMS removed them or the node was inactive. Components are recovered from the saved maps using the exact strict >.5-of-own-peak four-connected rule, deduplicating identical two-direction observation components. They are audit geometry, not new method outputs. The best component uses GT and is explicitly an oracle; top-mass selection uses no GT.

| Observation subset | GT nodes | Source IoU % | E IoU % | Native IoU % | Top-mass component IoU % | Best component GT IoU % |
|---|---:|---:|---:|---:|---:|---:|
| all | 219 | 50.1768 | 53.2302 | 57.6332 | 45.0919 | 45.1153 |
| retained | 91 | 36.2758 | 38.5423 | 44.7294 | 30.6849 | 30.6849 |
| removed_by_E_NMS | 108 | 63.0989 | 66.9403 | 70.1147 | 58.0303 | 58.0777 |
| inactive | 20 | 43.6465 | 46.0259 | 48.9453 | 40.7768 | 40.7768 |

All64 anonymous per-query signed results and the largest observed harms are retained in JSON. Node-level private details remain in outputs/score_audit; no GT coordinates, raw masks/features or private dataset identifiers are exported. A positive proxy score is not a claim of GT improvement.
