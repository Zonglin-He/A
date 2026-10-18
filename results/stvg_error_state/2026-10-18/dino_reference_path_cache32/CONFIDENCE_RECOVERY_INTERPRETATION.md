# Confidence recovery controls and attention-context confounding

Restrict to native target score>=.35, then inspect conditional final overlap>=.9 with the matched native final candidate geometry. This avoids calling a naturally subthreshold native query a confidence-collapse case. These are geometric proxy controls, not GT or a new gate.

| Domain | High native score, recovered geometry >=.9 | Admitted | Rejected |
|---|---:|---:|---:|
| hc2 | 62 | 9 | 53 |
| vidstg | 481 | 114 | 367 |

All existing admitted feedback uses the same mask slot as high content. Changing student reference also changes the nearest native slot and hence the original exclusion mask across samples. The within-frame/content/mask analysis controls this for its comparisons. Cross-mask statistics are descriptive: they cannot prove mask effects, and it is not valid to attribute every confidence difference purely to physical distance. A matched same-reference alternate-mask response is not cached.

The highest-overlap rejected examples in the JSON are selected by this disclosed deterministic rule, without GT, to demonstrate that final geometric recovery is not sufficient for score recovery. They do not establish event/identity correctness, and repeated candidates share the same query. No new layer/reference/mask experiment was run.
