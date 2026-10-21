# HC64: committed-source policy adaptation

The same64 development parents and existing full-box candidate bank are retained. Nine route states mean E plus eight observation sources; the box bank still has three candidate slots per sampled node. All three optimizers execute on the complete panel before GT. No diagnostic gate selects whether these fits run.

| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|---:|---:|
| ERefStrong | 31.0918 | 58.6043 | 52.2680 | 50.7055 |
| GeometryOPD | 31.3868 | 58.6043 | 53.1654 | 51.2469 |
| BoxFixed | 31.3985 | 58.6043 | 55.3363 | 50.9672 |
| B_OPD | 31.1227 | 58.6043 | 54.6227 | 50.7581 |
| CommitInitial | 31.0918 | 58.6043 | 52.2680 | 50.7055 |
| CommitDP | 31.5294 | 58.6043 | 55.9339 | 51.2172 |
| CommitOPD | 31.0126 | 58.6043 | 52.2680 | 50.5983 |
| CommitFixedKD | 31.0615 | 58.6043 | 52.2680 | 50.6437 |
| CommitDirect | 31.0615 | 58.6043 | 52.2680 | 50.6437 |

| Comparison | vIoU delta pp [95% CI] | Improve / decline / same | Drops >5 pp | Unobserved sIoU delta pp [95% CI] |
|---|---:|---:|---:|---:|
| CommitOPD − ERefStrong | -0.0792 [-0.2072, 0.0000] | 0 / 2 / 62 | 0 | -0.1072 [-0.2762, 0.0000] |
| CommitOPD − GeometryOPD | -0.3742 [-0.7535, -0.0660] | 15 / 48 / 1 | 1 | -0.6486 [-1.1850, -0.1957] |
| CommitOPD − BoxFixed | -0.3859 [-2.1603, 1.4500] | 29 / 33 / 2 | 8 | -0.3689 [-3.1028, 2.0572] |
| CommitOPD − B_OPD | -0.1101 [-1.7017, 1.5133] | 20 / 27 / 17 | 7 | -0.1597 [-2.7974, 2.1253] |
| CommitOPD − CommitInitial | -0.0792 [-0.2072, 0.0000] | 0 / 2 / 62 | 0 | -0.1072 [-0.2762, 0.0000] |
| CommitOPD − CommitDP | -0.5168 [-2.3012, 1.3706] | 22 / 35 / 7 | 8 | -0.6189 [-3.3925, 1.8732] |
| CommitFixedKD − ERefStrong | -0.0302 [-0.0907, 0.0000] | 0 / 1 / 63 | 0 | -0.0618 [-0.1854, 0.0000] |
| CommitFixedKD − GeometryOPD | -0.3252 [-0.7002, -0.0335] | 16 / 47 / 1 | 1 | -0.6032 [-1.1329, -0.1564] |
| CommitFixedKD − BoxFixed | -0.3370 [-2.1237, 1.4950] | 29 / 33 / 2 | 8 | -0.3235 [-3.0642, 2.1107] |
| CommitFixedKD − B_OPD | -0.0612 [-1.6441, 1.5634] | 20 / 27 / 17 | 7 | -0.1144 [-2.7520, 2.1758] |
| CommitFixedKD − CommitInitial | -0.0302 [-0.0907, 0.0000] | 0 / 1 / 63 | 0 | -0.0618 [-0.1854, 0.0000] |
| CommitFixedKD − CommitDP | -0.4679 [-2.2634, 1.4196] | 22 / 35 / 7 | 8 | -0.5735 [-3.3433, 1.9253] |
| CommitDirect − ERefStrong | -0.0302 [-0.0907, 0.0000] | 0 / 1 / 63 | 0 | -0.0618 [-0.1854, 0.0000] |
| CommitDirect − GeometryOPD | -0.3252 [-0.7002, -0.0335] | 16 / 47 / 1 | 1 | -0.6032 [-1.1329, -0.1564] |
| CommitDirect − BoxFixed | -0.3370 [-2.1237, 1.4950] | 29 / 33 / 2 | 8 | -0.3235 [-3.0642, 2.1107] |
| CommitDirect − B_OPD | -0.0612 [-1.6441, 1.5634] | 20 / 27 / 17 | 7 | -0.1144 [-2.7520, 2.1758] |
| CommitDirect − CommitInitial | -0.0302 [-0.0907, 0.0000] | 0 / 1 / 63 | 0 | -0.0618 [-0.1854, 0.0000] |
| CommitDirect − CommitDP | -0.4679 [-2.2634, 1.4196] | 22 / 35 / 7 | 8 | -0.5735 [-3.3433, 1.9253] |
| CommitOPD − CommitDirect | -0.0489 [-0.1467, 0.0000] | 0 / 1 / 63 | 0 | -0.0454 [-0.1361, 0.0000] |
| CommitOPD − CommitFixedKD | -0.0489 [-0.1467, 0.0000] | 0 / 1 / 63 | 0 | -0.0454 [-0.1361, 0.0000] |
| CommitDP − CommitInitial | +0.4376 [-1.4521, 2.2374] | 35 / 22 / 7 | 6 | +0.5117 [-1.9908, 3.2845] |

Intervals use10000 paired parent-query draws. All signed positive, negative, unchanged and severe changes remain in report.json and signed_changes.tsv. Spatial IoU uses actual sampled nodes with GT; all eight observation positions are excluded from unobserved IoU, independently of temporal truncation. All methods use the exact saved Strong interval.

## Actual commitment, optimization and replacements

| Method | Backwards/query | Accepted outer/query | Rejected outer/query | Unobserved commit / switch / reset events | Unobserved replacements / nodes |
|---|---:|---:|---:|---:|---:|
| CommitOPD | 30.0 | 2.953 | 0.047 | 24 / 3 / 0 | 29 / 3648 |
| CommitFixedKD | 30.0 | 2.922 | 0.078 | 24 / 3 / 0 | 27 / 3648 |
| CommitDirect | 30.0 | 2.953 | 0.047 | 24 / 3 / 0 | 27 / 3648 |

Cached leave-self-out meanq=0.220217; 316/512 anchors pass the quality-and-witness condition. Commit permission covers 1489/2800 active valid alternative slots; 620 slots are suppressed by source quality or witnesses. q is leave-self-out cached consistency, not GT quality or semantic-identity correctness. Commit permission additionally requires two other witnesses, positive external support delta and unchanged bank validity/activity.

CommitInitial uses four effective weights initialized to1, so it is a separate no-adaptation control and is not assumed equal to E. Each optimizer receives3 outer rounds×2 histories×10 inner backwards. Rejected outer proposals restore the raw weights and Adam, then the next predeclared outer round still runs. All quantities above come from saved real paths and update receipts.

## Teacher suggestions at the final visited state

| Method | Active unobserved suggestion−student mean IoU pp | Suggestion−E mean IoU pp | Suggested replacements | Harmful / suggested |
|---|---:|---:|---:|---:|
| CommitOPD | +0.7896 | +0.6255 | 25 | 4 / 25 |
| CommitFixedKD | +0.7440 | +0.6657 | 25 | 5 / 25 |
| CommitDirect | +0.7440 | +0.6657 | 25 | 5 / 25 |

These teacher proposals are evaluated only at the already saved final student histories. Their post-hoc GT IoU does not select any online action, configuration, stopping round or query. Different histories versus the previous B_OPD run prevent interpreting a teacher-quality difference as an isolated causal reliability gain. Actual net benefit is judged by the paired complete-query outcomes above.

Observed incremental trial costs with existing Source/teacher caches; not standalone end-to-end inference latency
No RGB, Source, ReferDINO, decoder or neural-weight update occurs in this cached-policy trial.
