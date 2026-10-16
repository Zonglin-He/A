# Finite night study: ablation_noCoverage

32 fixed parents, 16 per domain, one query per parent. Complete all-arm Student roots and prediction seals preceded GT access. All positive/negative/neutral and unknown cases are retained. Development global groups are exact core256 subsets; confirmation128 excludes all core256 parents and prioritizes no registered past selection, but is not claimed globally unseen.

Original acceptance, rejection, rollback, precision, per-domain parameters and final full-frame Student readout remain fixed. These groups use the original four-frame context with global64 + local96 tokens and the original clock, not the separate 32-frame/global512 intervention. Semantic identity/role/full-extent pixel labels remain pending; Student IoU cannot substitute for those labels.

This fixed deletion removes only the local candidate-priority coverage loop. Root recomputation of all128 actual windows verified unchanged global indices, total token budget, each frame/role quota, ID/context protection, candidate geometry, pixels, Prompt and Student. Missing-caption fallback remains. Candidate minus LocalB is +0.002029 / +0.578743 pp over all16/domain; the common-caption HC13/Vid12 secondary is +0.002498 / +0.771657 pp. This small development deletion is not promoted after failed confirmation; no further tuning or search follows.

| Arm | HC vIoU % | VidSTG vIoU % |
|---|---:|---:|
| Frozen | 28.116723 | 12.185607 |
| Temporal | 26.625819 | 12.052470 |
| LocalB | 27.181000 | 14.046676 |
| LocalB_Spatial | 28.561318 | 14.201387 |
| Candidate | 27.183029 | 14.625419 |
| Candidate_Spatial | 28.565897 | 14.749611 |

| Domain | Contrast | Mean delta pp | Positive / negative / neutral / unknown |
|---|---|---:|---|
| hc2 | Candidate_minus_Frozen | -0.933694 | 7 / 4 / 5 / 0 |
| hc2 | Candidate_minus_Temporal | +0.557210 | 8 / 3 / 5 / 0 |
| hc2 | Candidate_minus_LocalB | +0.002029 | 3 / 3 / 10 / 0 |
| vidstg | Candidate_minus_Frozen | +2.439812 | 7 / 0 / 9 / 0 |
| vidstg | Candidate_minus_Temporal | +2.572949 | 6 / 0 / 10 / 0 |
| vidstg | Candidate_minus_LocalB | +0.578743 | 5 / 1 / 10 / 0 |

All public files exclude media, annotations/GT coordinates, weights, raw tensors/optimizer state, actual query-caption prompts, raw Teacher responses, personal conversations and credentials. This group publication is not whole-study FINAL. No Bootstrap, online carry, CURRENT promotion or official all-query execution.
