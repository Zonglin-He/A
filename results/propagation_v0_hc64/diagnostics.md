# Propagation v0: pre-adaptation HC64 evidence gate

Stage one is complete and sealed. No matching-weight adaptation has been run in this diagnostic branch. The same fixed 64 development parents, one query per parent, are retained. GT is read only after every stage-one prediction, input, bank, evidence and initial path has passed its hash and schema checks.

**Gate: stop_no_adaptation.** The predeclared point-estimate criteria are applied once to the whole panel; confidence-interval signs are reported but are not extra pass criteria. No per-query selection is made.

| Criterion | Point estimate | Pass |
|---|---:|:---:|
| Candidate mean unobserved oracle IoU improvement | 0.05483327 | True |
| Same-state feedback macro non-GT-tie pair accuracy | 0.54974724 | True |
| Same-state feedback macro Spearman | 0.06137425 | True |
| MemoryGreedy − NoMemoryGreedy full vIoU | -0.00118064 | False |
| MemoryGreedy − NoMemoryGreedy unobserved sIoU | -0.00372716 | False |

## Saved control predictions

| Method | Full vIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|
| ERefStrong | 31.0918 | 52.2680 | 50.7055 |
| GeometryOPD | 31.3868 | 53.1654 | 51.2469 |
| MemoryGreedy | 28.8461 | 48.1802 | 47.9424 |
| NoMemoryGreedy | 28.9642 | 47.5745 | 48.3152 |
| EvidenceDP | 29.3919 | 47.4528 | 49.1405 |

| Memory comparison | Delta pp | 95% parent interval pp |
|---|---:|---:|
| v | -0.118064 | [-0.355410, 0.060352] |
| unobserved_actual_sIoU | -0.372716 | [-0.975451, 0.042374] |

## Candidate and feedback checks

Every fixed unobserved node with GT remains in the oracle denominator, including E-only fallback. GT chooses the best valid candidate independently at those nodes; all other nodes retain E. The resulting interpolated full vIoU is an offline readout, not a method and not the exact global trajectory upper bound.

| Feedback quantity | Macro estimate [95% CI] | Micro estimate [95% CI] |
|---|---:|---:|
| r pair accuracy | 0.549747 [0.494629, 0.605565] | 0.579672 [0.506215, 0.652798] |
| r Spearman | 0.061374 [-0.055921, 0.184487] | 0.112113 [-0.031575, 0.258541] |
| teacherq pair accuracy | 0.549747 [0.494629, 0.605565] | 0.579672 [0.506215, 0.652798] |
| teacherq Spearman | 0.061374 [-0.055921, 0.184487] | 0.112113 [-0.031575, 0.258541] |

Each query averages its identifiable node/path Spearman; pair accuracy pools its same-state non-GT-tie pairs; then queries receive equal weight
Correlation of concatenated within-state centered standardized average ranks; equivalent to candidate-count weighted within-state Spearman
E-only/constant states remain in coverage counts and oracle denominators; unidentified ranking metrics are None, not a fabricated 0.5

## Fixed physical-distance bins

Distance is to the nearest expert observation, divided by the median adjacent gap of the sorted eight physical anchor frames. The bins were fixed before GT: (0,.25], (.25,.5], and >.5. Observation frames are separate and are excluded here.

| Distance bin | Fixed GT nodes | Valid queries | Memory − NoMemory sIoU pp [95% CI] |
|---|---:|---:|---:|
| (0,.25] | 693 | 64 | -0.298842 [-1.406403, 0.364860] |
| (.25,.5] | 515 | 61 | -0.332475 [-0.911326, 0.046272] |
| >.5 | 157 | 40 | -0.976207 [-2.570776, -0.043570] |

All positive, negative, unchanged, E-only fallback and unavailable-ranking queries remain in the anonymous JSON. GT selects candidates only for the stated oracle diagnostic, never for an online prediction or memory update. Subsequent adaptation, if the gate passes, remains a development experiment on the exposed panel. A passed gate does not establish OPD superiority over Direct.
