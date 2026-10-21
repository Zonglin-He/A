# HC64 cached propagation and sequence matching

VidSTG → HC2: the same first 64 development parents, one query each, seed 41. All configurations are frozen before scoring. Every method uses the identical saved Strong interval; the network, RGB decoder and expert are not executed in this CPU experiment.

| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|---:|
| ERefStrong | 31.0918 | 58.6043 | 52.2680 | 50.7055 |
| GeometryOPD | 31.3868 | 58.6043 | 53.1654 | 51.2469 |
| FixedPropagation | 31.1687 | 58.6043 | 51.0311 | 51.3409 |
| OPD | 31.2165 | 58.6043 | 51.5204 | 51.5249 |
| FixedKD | 31.2716 | 58.6043 | 51.5204 | 51.5584 |
| Direct | 31.2826 | 58.6043 | 51.4505 | 51.6015 |

Observed and unobserved sIoU use the fixed actual sampled nodes with GT, independently of time truncation. All eight expert observations are excluded from unobserved sIoU.

| Comparison | vIoU delta pp [95% CI] | Improved / declined / unchanged | Drops >5 pp | Unobserved sIoU delta pp [95% CI] |
|---|---:|---:|---:|---:|
| FixedPropagation − ERefStrong | +0.0770 [-0.5584, 0.7573] | 9 / 14 / 41 | 1 | +0.6354 [-0.2794, 1.9584] |
| OPD − ERefStrong | +0.1248 [-0.5233, 0.8400] | 10 / 15 / 39 | 1 | +0.8194 [-0.3242, 2.3955] |
| FixedKD − ERefStrong | +0.1799 [-0.5172, 0.9359] | 10 / 15 / 39 | 1 | +0.8529 [-0.3321, 2.4584] |
| Direct − ERefStrong | +0.1909 [-0.5044, 0.9525] | 10 / 15 / 39 | 1 | +0.8960 [-0.2869, 2.5153] |
| OPD − GeometryOPD | -0.1702 [-0.8355, 0.5407] | 20 / 43 / 1 | 2 | +0.2780 [-0.9141, 1.8230] |
| OPD − FixedPropagation | +0.0478 [-0.2197, 0.3466] | 9 / 11 / 44 | 0 | +0.1840 [-0.2677, 0.7446] |
| OPD − FixedKD | -0.0551 [-0.2112, 0.0416] | 2 / 2 / 60 | 0 | -0.0334 [-0.1865, 0.0743] |
| OPD − Direct | -0.0661 [-0.2151, 0.0204] | 1 / 4 / 59 | 0 | -0.0765 [-0.2152, 0.0000] |

All positive, negative and unchanged per-query results are retained in `hc64.json`. `signed_changes.tsv` contains every signed change for all four new methods against both ERef and Geometry, including every declining and severe-decline sample. Confidence intervals use 10,000 paired parent draws.

## Offline candidate coverage

GT selects a candidate independently at scored unobserved nodes; all other nodes retain E. This is an offline oracle diagnostic, not a method or online selection. Its dense interpolated vIoU is not the exact global maximum over all candidate trajectories.

| Diagnostic | Value |
|---|---:|
| Fixed unobserved GT nodes | 1365 |
| Nodes with a candidate >5 pp better than E | 189 |
| Query count with at least one such node | 45 / 64 |
| Micro node proportion (%) | 13.8462 |
| Macro available node IoU gain (pp) | 3.5275 |
| GT nodewise candidate readout full vIoU (%) | 32.9824 |

## Recorded CPU costs

Observed incremental trial costs with existing Source/teacher caches; not standalone end-to-end inference latency

| Method | Seconds/query | Updates/query |
|---|---:|---:|
| FixedPropagation | 0.005399 | 0.000000 |
| OPD | 0.062064 | 3.000000 |
| FixedKD | 0.019528 | 3.000000 |
| Direct | 0.067439 | 3.000000 |

Evidence construction and candidate-bank construction are recorded separately per query. These are incremental CPU costs with pre-existing Source/ReferDINO caches, not full deployed inference latency. FixedPropagation performs no parameter update; the geometric baseline is reused from the exact saved run. The registered method and the paused full VidSTG evaluation are unchanged.

| Shared CPU cost | Seconds/query | Total seconds |
|---|---:|---:|
| evidence_seconds | 0.098425 | 6.299216 |
| bank_seconds | 0.033460 | 2.141462 |
| total_seconds | 0.337920 | 21.626850 |

Label-free bank support: active unobserved nodes 1003/3648; valid slot counts [4160, 1260, 624, 293]; unobserved nodes with a non-E candidate 1003.
