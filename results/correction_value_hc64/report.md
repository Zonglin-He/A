# HC64: value of one teacher-directed correction

The student is frozen at each query’s previously sealed final B_OPD weights. An independently seeded sampled student path defines at most four preselected disagreement points. Each point branches independently into the original Student action, the Teacher-Q4 argmax, or a uniformly sampled legal non-Student Random action. All later decisions use the same frozen student and the same Gumbel numbers, with each branch updating its own memory. No optimization or model call is performed.

**Primary support:** strictly later than the forced point, active, unobserved sampled nodes with GT. The forced point is excluded. Each query first averages all its preselected interventions equally. A point with no scored tail contributes zero delta; a query with no point or scored tail contributes zero to the all64 mean. Eligible-only results retain these same query deltas and only remove queries with no scored point. Absolute IoU can be undefined.

| Tail support | Comparison | All64 delta pp [95% CI] | Eligible queries | Eligible-only delta pp [95% CI] | Improve / decline / same |
|---|---|---:|---:|---:|---:|
| active_unobserved_tail | Teacher − Student | +0.0079 [0.0000, 0.0236] | 60 | +0.0084 [0.0000, 0.0252] | 1 / 0 / 63 |
| active_unobserved_tail | Random − Student | +0.0717 [0.0000, 0.2071] | 60 | +0.0764 [0.0000, 0.2209] | 2 / 0 / 62 |
| active_unobserved_tail | Teacher − Random | -0.0638 [-0.1914, 0.0000] | 60 | -0.0680 [-0.2041, 0.0000] | 0 / 1 / 63 |
| all_unobserved_tail | Teacher − Student | +0.0079 [0.0000, 0.0236] | 62 | +0.0081 [0.0000, 0.0244] | 1 / 0 / 63 |
| all_unobserved_tail | Random − Student | +0.0462 [0.0000, 0.1306] | 62 | +0.0476 [0.0000, 0.1348] | 2 / 0 / 62 |
| all_unobserved_tail | Teacher − Random | -0.0383 [-0.1148, 0.0000] | 62 | -0.0395 [-0.1185, 0.0000] | 0 / 1 / 63 |
| near3_active_unobserved | Teacher − Student | +0.0236 [0.0000, 0.0709] | 60 | +0.0252 [0.0000, 0.0757] | 1 / 0 / 63 |
| near3_active_unobserved | Random − Student | +0.1193 [0.0000, 0.3343] | 60 | +0.1273 [0.0000, 0.3566] | 2 / 0 / 62 |
| near3_active_unobserved | Teacher − Random | -0.0957 [-0.2870, 0.0000] | 60 | -0.1021 [-0.3062, 0.0000] | 0 / 1 / 63 |
| far_active_unobserved | Teacher − Student | +0.0000 [0.0000, 0.0000] | 59 | +0.0000 [0.0000, 0.0000] | 0 / 0 / 64 |
| far_active_unobserved | Random − Student | +0.0000 [0.0000, 0.0000] | 59 | +0.0000 [0.0000, 0.0000] | 0 / 0 / 64 |
| far_active_unobserved | Teacher − Random | +0.0000 [0.0000, 0.0000] | 59 | +0.0000 [0.0000, 0.0000] | 0 / 0 / 64 |

Near3 means sampled nodes t+1 through t+3; far means t+4 onward. Both retain the primary active/unobserved/GT mask. All-unobserved-tail additionally includes inactive later nodes. Intervals use10000 paired query draws, one query per parent.

## Disagreement and action-support strata

| Stratum | Selected interventions | Teacher − Student all64 pp [95% CI] | Teacher − Random all64 pp [95% CI] |
|---|---:|---:|---:|
| two_actions | 93 | +0.0105 [0.0000, 0.0315] | +0.0000 [0.0000, 0.0000] |
| three_actions | 144 | +0.0000 [0.0000, 0.0000] | -0.0957 [-0.2870, 0.0000] |
| policy_disagreement | 101 | +0.0000 [0.0000, 0.0000] | -0.0957 [-0.2870, 0.0000] |
| noise_only_disagreement | 136 | +0.0105 [0.0000, 0.0315] | +0.0000 [0.0000, 0.0000] |

237 interventions across 62/64 queries; 167 have the same Teacher and Random action. With exactly two legal actions this equality is required, so that stratum cannot independently distinguish Teacher from Random. Policy-disagreement means Teacher differs from the student’s noise-free argmax; noise-only disagreement means Teacher agrees with that argmax but differs from the sampled action.

## Damage, recovery and gate

| Branch | Initially Student-IoU≥0.5 node-evaluations | Damaged below0.5 | Initially below0.5 | Recovered to≥0.5 | Tail harm >5 pp |
|---|---:|---:|---:|---:|---:|
| Teacher | 1154 | 0 | 714 | 0 | 0 |
| Random | 1154 | 0 | 714 | 0 | 0 |

Counts are node-evaluations across independent interventions and can revisit the same physical frame. Damage/recovery labels describe offline final branch boxes; GT never selects points or branch actions.

Across all later active unobserved nodes, including nodes without GT, Teacher versus Student changes 3 / 2814 actions; the directly forced node is excluded.

**Decision: inconclusive; passed=False.** Both all64 primary active-unobserved-tail sIoU95% query-bootstrap CI lower bounds must be strictly above0. This is a predeclared resource-allocation gate for entering a new distillation experiment, not a universal theoretical claim about OPD.

Complete-video vIoU is auxiliary because it includes the forced point. Its gains cannot establish a tail-propagation benefit. All positive, negative, unchanged and empty-query primary changes are retained in report.json and signed_changes.tsv. All64 remains the primary exposed-development panel; no query is removed by outcome.

Cached lightweight-policy intervention cost only; existing Source/ReferDINO evidence is reused. Not full deployed inference latency.
