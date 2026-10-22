# Fixed single320 screen: rejected

Previously exposed HC63 development; fixed320 before its first predictions/labels, no resolution/weight sweep.

| Method | Full vIoU % | tIoU % | Mean interval s | Boundary MAE s |
|---|---:|---:|---:|---:|
| Source | 27.1951 | 55.1576 | 8.5069 | 2.6737 |
| Global6 | 30.0946 | 56.2981 | 8.6358 | 2.5960 |
| Strong | 31.8466 | 59.1483 | 9.0850 | 2.4704 |
| Single448_0 | 31.2180 | 58.2852 | 9.0922 | 2.5304 |
| Single448_1 | 31.7769 | 59.0460 | 8.9160 | 2.4248 |
| Single320 | 29.7743 | 55.3051 | 8.7128 | 2.6407 |

| Comparison | Full vIoU pp,95%CI | tIoU pp,95%CI | Severe v loss >5pp |
|---|---:|---:|---:|
| Single320-Strong | -2.0723 [-3.3411,-0.8683] | -3.8432 [-5.8318,-1.9291] | 14 |
| Single320-Global6 | -0.3203 [-1.4485,0.8295] | -0.9930 [-2.9984,1.0044] | 9 |
| Single320-Single448_0 | -1.4437 [-2.7861,-0.1829] | -2.9801 [-5.1292,-0.8955] | 11 |
| Single320-Single448_1 | -2.0026 [-3.3113,-0.7185] | -3.7410 [-5.9909,-1.5599] | 16 |
| Single320-Source | +2.5792 [0.2874,4.8853] | +0.1475 [-1.9323,2.2188] | 12 |
| Single448_0-Strong | -0.6285 [-1.1717,-0.1007] | -0.8631 [-1.7293,0.0149] | 2 |
| Single448_1-Strong | -0.0697 [-0.6362,0.4729] | -0.1022 [-1.1470,0.9041] | 2 |

## Actual paired extra-branch cost

Eight fixed queries, counterbalanced order, resident SAME frozen model; required preprocessing/transfers/two time passes/copies/posterior included. No per-module timing synchronization; only outer boundaries. Model startup/decode/parse separately. Not whole-method end-to-end; separate32x2 stage measures actual independent complete pipelines.

| Actual branch | Mean s | Median s |
|---|---:|---:|
| 320_0 | 1.8733 | 1.8458 |
| 448_0 | 2.2484 | 2.2180 |
| 448_01 | 3.5134 | 3.4560 |

Extra-branch reduction 46.68%; Strong full-v loss 2.0723pp. Gate FAIL: only the speed criterion passes. No320 HC32/Vid32 run, no288/352/384 or weight scan; separately authorized448 recheck proceeds.

The candidate retains full-frame context, original offset0, same-view hull and .5Global6. Its failure is not explained by losing the hull or by crop-induced motion changes. A middle resolution is not guaranteed to interpolate accuracy between224 and448: its native features, discrete temporal selection and boundary posterior can change non-monotonically. This experiment establishes the fixed320 result, not the causal share of training-scale mismatch, visual detail and query changes.

All63 outputs sealed before locked-label scoring. Complete actual trajectories and a fixed common Strong spatial trajectory are separately evaluated. Thresholds, observed/unobserved spatial metrics, interval length, boundary errors and all signed losses are in report.json/signed_queries.tsv. No deployment change or full resume.

The matched single448-to-single320 branch reduction is only16.68% (2.2484s to1.8733s), while complete vIoU loses1.4437pp and tIoU2.9801pp under the SAME offset0/self-hull/global fusion. The nominal51% input-pixel ratio does not imply51% runtime: temporal replay, text processing, transfers and module copies remain. These timings include required native implementation costs and are not whole-method speedups.
