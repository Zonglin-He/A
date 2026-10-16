# Finite night study: ablation_noCaption

32 fixed parents, 16 per domain, one query per parent. Complete all-arm Student roots and prediction seals preceded GT access. All positive/negative/neutral and unknown cases are retained. Development global groups are exact core256 subsets; confirmation128 excludes all core256 parents and prioritizes no registered past selection, but is not claimed globally unseen.

Original acceptance, rejection, rollback, precision, per-domain parameters and final full-frame Student readout remain fixed. These groups use the original four-frame context with global64 + local96 tokens and the original clock, not the separate 32-frame/global512 intervention. Semantic identity/role/full-extent pixel labels remain pending; Student IoU cannot substitute for those labels.

This fixed deletion removes caption consensus/difference evidence and keeps query relevance. Frame/role quotas and ID/context protection are unchanged. It also restores WHO eligibility for 7 caption-unavailable sources (28 new calls), whereas LocalB retains its pre-existing fallback. Thus this is whole-caption-selector deletion, not a score-only perturbation. On the common-caption secondary subset HC13/Vid12, Candidate minus LocalB is -0.054972 / -2.553279 pp, compared with +0.152285 / -1.894123 pp over all 16/domain. All negative tails remain, including the large Vid decline. No new recipe is selected from this ablation.

| Arm | HC vIoU % | VidSTG vIoU % |
|---|---:|---:|
| Frozen | 28.116723 | 12.185607 |
| Temporal | 26.625819 | 12.052470 |
| LocalB | 27.181000 | 14.046676 |
| LocalB_Spatial | 28.561318 | 14.201387 |
| Candidate | 27.333285 | 12.152553 |
| Candidate_Spatial | 28.821409 | 12.296738 |

| Domain | Contrast | Mean delta pp | Positive / negative / neutral / unknown |
|---|---|---:|---|
| hc2 | Candidate_minus_Frozen | -0.783439 | 8 / 5 / 3 / 0 |
| hc2 | Candidate_minus_Temporal | +0.707466 | 11 / 2 / 3 / 0 |
| hc2 | Candidate_minus_LocalB | +0.152285 | 5 / 4 / 7 / 0 |
| vidstg | Candidate_minus_Frozen | -0.033054 | 7 / 4 / 5 / 0 |
| vidstg | Candidate_minus_Temporal | +0.100083 | 7 / 3 / 6 / 0 |
| vidstg | Candidate_minus_LocalB | -1.894123 | 6 / 4 / 6 / 0 |

All public files exclude media, annotations/GT coordinates, weights, raw tensors/optimizer state, actual query-caption prompts, raw Teacher responses, personal conversations and credentials. This group publication is not whole-study FINAL. No Bootstrap, online carry, CURRENT promotion or official all-query execution.
