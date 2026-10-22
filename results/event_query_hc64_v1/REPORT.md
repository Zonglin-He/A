# HC64: old event feedback and learned temporal query

Exposed development; one locked configuration, no GT-selected state. Spatial geometry and DP unchanged. Source/Full native controls retained. Raw/Predictive Direct and Query share exactly the same target.

| Method | vIoU % | tIoU % | Unobserved sIoU % | Boundary MAE s |
|---|---:|---:|---:|---:|
| NativeG6 | 29.6952 | 55.9613 | 50.5762 | 2.6034 |
| LegacyContrast | 21.2846 | 38.8586 | 50.3086 | 4.3853 |
| RawDirect | 28.5724 | 55.3419 | 50.3106 | 2.7501 |
| RawQuery | 29.6081 | 55.7427 | 50.5984 | 2.6255 |
| PredictiveDirect | 29.1233 | 54.8013 | 50.2062 | 2.7274 |
| PredictiveQuery | 29.5028 | 55.5441 | 50.6405 | 2.6318 |
| Source | 26.8938 | 54.8230 | 47.6990 | 2.6821 |

Full signed outcomes and parent-cluster intervals are in report.json. Historical token extraction/query reconstruction is not a new contribution. LOO predictive gain is a working score, not calibrated event probability; teacher observations share context. No automatic full resume.
