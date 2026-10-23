# Joint expert budget confirmation: complete two-domain evaluation

The fixed confirmation contains128 parent-disjoint queries per direction, one query per parent, distinct from screening parents. Both official target sets were historically exposed: this is a pre-fixed confirmation of new contrasts, not an unseen dataset. Ten configurations advanced by predeclared implementation completeness and expert GPU cost only. All2560 predictions were sealed before independent CPU labels; no failures, no parameter reselection. Original Hybrid8 remains the main method.

## HC2 —128 parents

Source tIoU/vIoU: 51.718761% / 28.044780%. Means are query-equal; expert seconds are measured additional forward only. Severe union counts a query once if tIoU or vIoU drops by more than5pp relative to Source.

| Configuration | tIoU (%) | vIoU (%) | Δt vsSource (pp) | Δv vsSource (pp) | Severe t / v / union | Expert GPU s/query |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid2 | 53.0265 | 27.3444 | +1.3077 | -0.7004 | 10 / 25 / 32 | 0.06406 |
| Uniform2 | 52.9680 | 28.1104 | +1.2492 | +0.0656 | 10 / 27 / 33 | 0.05947 |
| Hybrid4 | 52.9614 | 29.4579 | +1.2427 | +1.4132 | 12 / 19 / 28 | 0.10552 |
| Uniform4 | 52.6656 | 27.0864 | +0.9469 | -0.9584 | 14 / 27 / 35 | 0.10231 |
| Hybrid8 | 53.0477 | 30.6899 | +1.3289 | +2.6451 | 11 / 18 / 25 | 0.19362 |
| Uniform8 | 52.9270 | 29.9071 | +1.2082 | +1.8624 | 12 / 22 / 29 | 0.19071 |
| Hybrid16 | 53.0280 | 31.7531 | +1.3092 | +3.7083 | 10 / 14 / 21 | 0.37832 |
| Uniform16 | 53.1776 | 29.9773 | +1.4588 | +1.9325 | 8 / 23 / 28 | 0.37548 |
| Random8 | 53.1871 | 29.6261 | +1.4683 | +1.5813 | 9 / 21 / 28 | 0.19146 |
| SourceOnly8 | 52.9293 | 31.4496 | +1.2105 | +3.4048 | 10 / 16 / 23 | 0.19137 |

Hybrid8−Source paired95% parent-cluster CI: tIoU [0.5383, 2.1474]pp; vIoU [0.4428, 4.9629]pp.

Hybrid16−Hybrid8: vIoU +1.063259pp, CI [0.180874, 1.980673]; expert forward cost 1.9539×. This descriptive contrast is retained without selecting a new main configuration.

## VIDSTG —128 parents

Source tIoU/vIoU: 34.598917% / 14.102136%. Means are query-equal; expert seconds are measured additional forward only. Severe union counts a query once if tIoU or vIoU drops by more than5pp relative to Source.

| Configuration | tIoU (%) | vIoU (%) | Δt vsSource (pp) | Δv vsSource (pp) | Severe t / v / union | Expert GPU s/query |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid2 | 35.0404 | 12.8767 | +0.4415 | -1.2255 | 12 / 23 / 32 | 0.05667 |
| Uniform2 | 35.2030 | 11.9627 | +0.6041 | -2.1394 | 9 / 31 / 36 | 0.05689 |
| Hybrid4 | 35.1895 | 14.1657 | +0.5906 | +0.0636 | 11 / 23 / 30 | 0.09821 |
| Uniform4 | 34.9334 | 13.3496 | +0.3345 | -0.7525 | 14 / 23 / 31 | 0.09798 |
| Hybrid8 | 34.9822 | 15.2054 | +0.3832 | +1.1033 | 13 / 18 / 28 | 0.18196 |
| Uniform8 | 34.9617 | 14.3092 | +0.3628 | +0.2070 | 12 / 20 / 30 | 0.18177 |
| Hybrid16 | 34.9879 | 15.3226 | +0.3889 | +1.2205 | 11 / 19 / 27 | 0.36070 |
| Uniform16 | 35.1940 | 15.1356 | +0.5951 | +1.0334 | 10 / 20 / 28 | 0.36074 |
| Random8 | 35.1761 | 14.5994 | +0.5772 | +0.4973 | 13 / 19 / 29 | 0.18243 |
| SourceOnly8 | 35.1798 | 15.1567 | +0.5809 | +1.0546 | 12 / 19 / 28 | 0.18238 |

Hybrid8−Source paired95% parent-cluster CI: tIoU [-0.8796, 1.8038]pp; vIoU [-1.0173, 3.2023]pp.

Hybrid16−Hybrid8: vIoU +0.117184pp, CI [-0.814463, 0.999887]; expert forward cost 1.9823×. This descriptive contrast is retained without selecting a new main configuration.

## Interpretation and full record

K2 produces negative complete-vIoU change in both domains for Hybrid, and Uniform2 is negative in Vid. Reducing observations is not free. K16 improves HC vIoU over K8 by1.063259pp with a positive paired CI, while the Vid difference is0.117184pp with CI crossing zero, at about twice expert-forward cost. K16 severe Source-relative union is21HC/27Vid versus K8 25HC/28Vid. These positive K16 outcomes remain visible; they do not retroactively change the frozen production K8. Time quality is not monotonic in K.

Hybrid8 exceeds Uniform8 in vIoU mean in both confirmation domains; screening had Uniform8 above Hybrid8 in HC. The original screen and confirm are both retained. Confirmation is not used for another tuning round. All10 arms versus the original main, including every negative value and paired CI, appear in each confirm/paired_main.json and paired_main_signed.tsv.

Confidence intervals are pointwise per contrast, without family-wise multiplicity correction; paired control comparisons remain descriptive and do not select parameters.

No K32 or DenseNative confirmation was executed. They remain not-advanced configurations with planned confirmation queries0, rather than fabricated missing predictions. Their full/partial screening results, fixed-search failures and cost rejection remain in [screening](SCREEN_ANALYSIS.md).

Every domain/confirm directory includes original official report.json and full-precision signed.tsv, all_signed_severe_timegroups.tsv, tails_timegroups_and_plot_data.json, actual_observation_ratios.tsv, paired_main.json and paired_main_signed.tsv. Short/long subgroups use the original <=5s rule and all signed tails, not only mean gains.

## Actual-data figures

- [Measured expert cost versus vIoU](figures/confirm_expert_cost_viou.svg)
- [RequestedK versus tIoU](figures/confirm_K_tiou.svg)
- [Actual observed-frame K/T distributions](figures/confirm_actual_K_over_T.svg)

All three PNG previews were inspected. Bars are95% parent-bootstrap CIs, with explicit per-domain ranges and no hidden outliers or truncated CIs. Each K/schedule uses its own native joint context of the same ReferDINO, not a subset of the original8 outputs. Additional expert-forward timing is not complete deployment latency; Source/model initialization/preprocessing/geometry/pooling/prototype/DP are separate and nested totals must not be summed. Cache hits retain original measured evidence cost.

## Execution and integrity

2560official dense checks maxerror1.5543122344752192e-15; all2560predictionSHA matched sealed records. Original Hybrid8 Source/evidence/initialization/features/time/boxes bitwise assertions remain enforced. Budget controller/predictor/scorer exited successfully; unchanged controlled-RGB shifts started afterward. CG-STVG remains LAST and has not constructed a model. Fig1 and production parameters remain unchanged.
