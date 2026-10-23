# TA-STVG: SimplePath + native temporal prior + all-video Native2D

Source and imported Global6 reproduce the prior official evaluation. Full and all auxiliary arms are newly computed for the fixed SimplePath + one all-video Native2D prototype system, with native last-layer temporal prior. Same inputs/checkpoints/eight observations, no full-result parameter selection. Historical development/full exposure is disclosed.

| Method | Expert observations | HC2 m_vIoU | R@.3 | R@.5 | Vid m_vIoU | R@.3 | R@.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 0 | 24.9948 | 37.8023 | 12.8812 | 11.7777 | 13.6368 | 3.5621 |
| Interpolation | 8 | 25.1065 | 38.2755 | 13.9853 | 14.2289 | 18.4801 | 7.9297 |
| OursS | 8 | 26.8118 | 40.7992 | 15.8780 | 14.0305 | 18.2277 | 7.8230 |
| Full | 8 | 27.2181 | 41.9558 | 17.1399 | 14.3833 | 19.0527 | 8.1433 |
| Historical project OPD (not third-party generic TTA) | 8 | 26.8784 | 40.8517 | 14.5636 | — | — | — |

Third-party generic TTA was not verified. The historical project OPD row is separately labeled; no new OPD run was performed and incomplete Vid8405 is not used.

## Four module ablations

All rows retain shared initialization. Two spatial ablations keep Full time/anchors; Native temporal readout is exactly OursS and runs unchanged spatial inference on the native interval.

| Target | Configuration | Full m_vIoU% | Δ versus Full pp | Unobserved sIoU% | tIoU% |
|---|---|---:|---:|---:|---:|
| hc2 | Full | 27.2181 | +0.0000 | 53.0351 | 49.5768 |
| hc2 | NoLocal | 26.7799 | -0.4382 | 51.9662 | 49.5768 |
| hc2 | Independent | 26.1986 | -1.0195 | 51.0123 | 49.5768 |
| hc2 | Native temporal readout | 26.8118 | -0.4063 | 53.0696 | 48.6473 |
| vidstg | Full | 14.3833 | +0.0000 | 35.7685 | 34.6044 |
| vidstg | NoLocal | 14.0856 | -0.2977 | 35.0008 | 34.6044 |
| vidstg | Independent | 13.8266 | -0.5567 | 34.5344 | 34.6044 |
| vidstg | Native temporal readout | 14.0305 | -0.3528 | 35.7815 | 33.8046 |

## Spatial stages on common support

Actual sampled GT-supported nodes; common8 observed positions, independent of each predicted interval. Query means then equal-query means; effective query counts shown. Shared initialization is an intermediate result, not a deletion ablation.

| Target | Spatial output | Observed sIoU% | n | Unobserved sIoU% | n |
|---|---|---:|---:|---:|---:|
| hc2 | Source | 51.4714 | 1880 | 51.0497 | 1902 |
| hc2 | SharedInit | 52.7022 | 1880 | 51.9662 | 1902 |
| hc2 | OursS | 56.3960 | 1880 | 53.0696 | 1902 |
| vidstg | Source | 33.5366 | 8507 | 33.0254 | 10300 |
| vidstg | SharedInit | 37.1508 | 8507 | 35.0008 | 10300 |
| vidstg | OursS | 41.9298 | 8507 | 35.7815 | 10300 |

## Complete extent versus mask-derived extent

Same anchor and direction; no GT-selected direction. All-slot IoU zero-fills missing candidates; jointly-available paired IoU and missing rate reported separately.

| Target | Candidate | All-slot unobserved IoU% | Both available IoU% | Missing% |
|---|---|---:|---:|---:|
| hc2 | complete | 41.2467 | 48.8996 | 19.4880 |
| hc2 | mask | 45.3733 | 47.3532 | 0.0000 |
| vidstg | complete | 26.6559 | 36.0735 | 35.0484 |
| vidstg | mask | 30.7042 | 33.6198 | 0.0000 |

## Temporal module

| Target | Readout | tIoU% | Boundary MAE s | Zero-overlap% | Mean interval s | Full vIoU% |
|---|---|---:|---:|---:|---:|---:|
| hc2 | OursS | 48.6473 | 3.2907 | 7.6236 | 8.7294 | 26.8118 |
| hc2 | Full | 49.5768 | 3.2572 | 7.8339 | 8.8653 | 27.2181 |
| vidstg | OursS | 33.8046 | 7.6332 | 20.4601 | 10.5494 | 14.0305 |
| vidstg | Full | 34.6044 | 7.5626 | 19.8583 | 11.2330 | 14.3833 |

All core paired increments and parent-cluster95% intervals, distance bins with node/query counts, natural cache costs segmented by worker, signed changes and every severe decline are in [HC2](hc2/report.json) and [VidSTG](vidstg/report.json).

## Qualitative status

Continuous sampled-frame sheets are private local assets, pending root visual review for the four requested categories. Raw dataset frames, captions, GT coordinates and features are not exported to this repository. See [status](qualitative_status.json).
