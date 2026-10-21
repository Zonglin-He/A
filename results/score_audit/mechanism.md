# Fixed v0: geometric mechanism stratification

This is an offline stratification of the unchanged saved replacements, not a new method or score experiment. Native quality means box IoU relative to E at the same labeled observation. These measurements cannot establish whether the teacher mask, propagated region or selected object has the correct semantic identity.

| Control | Harmed observation replacements | Native better than E | Native worse than E | Native tied with E |
|---|---:|---:|---:|---:|
| MemoryGreedy | 77 | 38 | 39 | 0 |
| NoMemoryGreedy | 79 | 39 | 40 | 0 |
| EvidenceDP | 81 | 40 | 41 | 0 |

A chosen box that is worse than E while the native box is better demonstrates a lost available geometric correction; it does not isolate whether candidate suppression, mask-to-box conversion, scoring or history caused that loss. A native box already worse than E is compatible with teacher error, without proving that every related component is also wrong. Native E-NMS removal is retained as a separate saved status.

| Control | Harmed unobserved replacements | Nearest anchor better | Worse | Tied | No anchor GT |
|---|---:|---:|---:|---:|---:|
| MemoryGreedy | 409 | 243 | 131 | 14 | 21 |
| NoMemoryGreedy | 410 | 244 | 130 | 14 | 22 |
| EvidenceDP | 375 | 224 | 119 | 12 | 20 |

The nearest anchor is the saved label-free physical nearest-anchor index. If that exact anchor has no GT, quality is unknown; another anchor is never substituted using GT. Near/middle/far cells and every quality stratum, including zero-count groups, are retained in JSON.

## Component size and native agreement

Selected component areas are compared separately with the native box and with the saved GT box. Larger/smaller means an area ratio strictly above/below one, not a semantic or causal classification. Aspect ratio, translation, occlusion and annotation convention can each reduce IoU even at a similar area.

| Control | Harmed component replacements at observations | Smaller than GT | Larger than GT | Smaller than native | Larger than native |
|---|---:|---:|---:|---:|---:|
| MemoryGreedy | 65 | 59 | 6 | 56 | 9 |
| NoMemoryGreedy | 68 | 62 | 6 | 59 | 9 |
| EvidenceDP | 70 | 63 | 7 | 61 | 9 |

## Existing cross-anchor result

CrossAnchor − Projection full vIoU: +0.0627 pp; unobserved sIoU: -0.1943 pp.
This separate probe starts from raw Source with seven-mask proposal evidence per held-out fold, not the fixed E-initialized v0 bank. Eight folds are averaged within parent; fold-nodes are repeated observations.
Held-anchor readout has a small full-vIoU difference from projection but worse unobserved sIoU and does not lower the overall accepted harm fraction. It does not establish semantic identity recovery or isolate the causal mechanism in fixed v0. Cached teacher outputs were temporally coupled across eight frames.

The fixed audit JSON, its private node table, the stage-one seal, the failed gate, original sources, formal pause and registered configuration are unchanged. No new predictions or adaptations were run.
