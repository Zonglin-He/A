# Actual joint expert budgets: completed screening

Each direction contains64 fixed independent parents, one query each. Thirteen configurations were fixed before inference. Whole128-query stage sealed before independent CPU official labels;1660 valid predictions and4 explicit resource failures. Full target sets were historically exposed; this is screening, not the final128-parent confirmation per direction. Production settings remain unchanged.

Every K/schedule invokes the same frozen ReferDINO on its complete selected RGB context. Smaller K is not a subset of old8 predictions. Hybrid8 reproduces the original Full evidence, initialization, pooled features, interval and boxes bitwise. DenseNative is raw teacher boxes on all sampled frames plus fixed Source MBR time; it does not include the sparse correction/DP/prototype pipeline.

## HC2 —64 parents

Source tIoU/vIoU: 49.558010% / 24.415350%. Query-equal means; severe union counts each query once if tIoU or vIoU drops by more than5pp against Source.

| Configuration | Coverage | tIoU (%) | vIoU (%) | Δt (pp) | Δv (pp) | Severe union | Expert GPU s/query | /Hybrid8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid2 | 64/64 | 51.7336 | 25.7503 | +2.1756 | +1.3350 | 14/64 | 0.0664 | 0.35× |
| Uniform2 | 64/64 | 51.8821 | 25.0267 | +2.3241 | +0.6114 | 20/64 | 0.0598 | 0.31× |
| Hybrid4 | 64/64 | 52.0051 | 25.1165 | +2.4471 | +0.7011 | 15/64 | 0.1060 | 0.55× |
| Uniform4 | 64/64 | 51.7850 | 25.2266 | +2.2270 | +0.8113 | 17/64 | 0.1017 | 0.53× |
| Hybrid8 | 64/64 | 51.9834 | 27.3781 | +2.4254 | +2.9628 | 14/64 | 0.1916 | 1.00× |
| Uniform8 | 64/64 | 52.0023 | 28.5284 | +2.4443 | +4.1131 | 12/64 | 0.1887 | 0.98× |
| Hybrid16 | 64/64 | 52.0881 | 27.9049 | +2.5300 | +3.4896 | 11/64 | 0.3747 | 1.95× |
| Uniform16 | 64/64 | 52.2202 | 28.0152 | +2.6622 | +3.5998 | 12/64 | 0.3715 | 1.94× |
| Hybrid32 | 64/64 | 52.0594 | 28.0364 | +2.5013 | +3.6211 | 10/64 | 0.6992 | 3.65× |
| Uniform32 | 64/64 | 52.3017 | 28.7655 | +2.7437 | +4.3501 | 11/64 | 0.6964 | 3.63× |
| Random8 | 64/64 | 51.7825 | 26.0488 | +2.2245 | +1.6335 | 17/64 | 0.1895 | 0.99× |
| SourceOnly8 | 64/64 | 51.8151 | 26.5050 | +2.2571 | +2.0896 | 14/64 | 0.1887 | 0.98× |
| DenseNative | 64/64 | 49.5580 | 27.8517 | +0.0000 | +3.4364 | 6/64 | 1.4108 | 7.36× |

Incomplete K32 means use available62 cases only and their matched Source values. They are not full64-case estimates and are not plotted as complete Vid curves. Both missing queries/config are retained in the failure list.

## VIDSTG —64 parents

Source tIoU/vIoU: 27.039828% / 11.739054%. Query-equal means; severe union counts each query once if tIoU or vIoU drops by more than5pp against Source.

| Configuration | Coverage | tIoU (%) | vIoU (%) | Δt (pp) | Δv (pp) | Severe union | Expert GPU s/query | /Hybrid8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid2 | 64/64 | 28.6788 | 11.1796 | +1.6390 | -0.5594 | 11/64 | 0.0600 | 0.32× |
| Uniform2 | 64/64 | 29.0993 | 12.1534 | +2.0595 | +0.4144 | 9/64 | 0.0590 | 0.31× |
| Hybrid4 | 64/64 | 29.2772 | 13.0969 | +2.2374 | +1.3578 | 10/64 | 0.1030 | 0.54× |
| Uniform4 | 64/64 | 29.7208 | 13.7076 | +2.6809 | +1.9685 | 11/64 | 0.1014 | 0.54× |
| Hybrid8 | 64/64 | 29.6971 | 15.0840 | +2.6573 | +3.3450 | 8/64 | 0.1892 | 1.00× |
| Uniform8 | 64/64 | 29.5861 | 13.9172 | +2.5463 | +2.1782 | 11/64 | 0.1890 | 1.00× |
| Hybrid16 | 64/64 | 29.3009 | 15.6869 | +2.2611 | +3.9478 | 8/64 | 0.3712 | 1.96× |
| Uniform16 | 64/64 | 29.1870 | 15.6493 | +2.1471 | +3.9103 | 8/64 | 0.3688 | 1.95× |
| Hybrid32 | 62/64 | 28.5632 | 15.1882 | +2.1321 | +3.9440 | 7/62 | 0.6749 | 3.57× |
| Uniform32 | 62/64 | 28.6226 | 15.8900 | +2.4654 | +5.0856 | 6/62 | 0.6776 | 3.58× |
| Random8 | 64/64 | 29.7606 | 13.0649 | +2.7208 | +1.3258 | 9/64 | 0.1884 | 1.00× |
| SourceOnly8 | 64/64 | 29.5520 | 16.2482 | +2.5122 | +4.5091 | 7/64 | 0.1880 | 0.99× |
| DenseNative | 64/64 | 27.0398 | 15.5814 | +0.0000 | +3.8423 | 5/64 | 2.5266 | 13.36× |

Incomplete K32 means use available62 cases only and their matched Source values. They are not full64-case estimates and are not plotted as complete Vid curves. Both missing queries/config are retained in the failure list.

## Fixed advancement decision

Ten configurations advance: Hybrid2, Uniform2, Hybrid4, Uniform4, Hybrid8, Uniform8, Hybrid16, Uniform16, Random8, SourceOnly8. Hybrid32 and Uniform32 fail two-domain implementation completeness. DenseNative exceeds10×Hybrid8 mean expert-forward cost in Vid. Accuracy is not part of qualification; no parameters change.

DenseNative additional expert forward costs approximately7.36×Hybrid8 in HC and13.36× in Vid. These are expert-forward components, not complete deployment latency. Source extraction, initialization/loading, expert preprocessing/postprocessing, geometry, prototypes and DP remain separate costs. Cache hits retain original measured forward cost. Failed-search expert/processing costs appear separately in screen_failures.json; the first original failure has verified expert cost but no exact total processing-time scalar, marked unknown rather than zero.

Screening does not establish a unique optimal K or schedule. Uniform8 has higher HC vIoU than Hybrid8, while Hybrid8 has higher Vid vIoU. Fixed confirmation evaluates all eligible schedules without accuracy selection. Every severe outcome and short/long paired CI remains in full exports.

## Figures and full data

- [Expert cost versus vIoU](figures/screen_expert_cost_viou.svg):95% parent-bootstrap intervals and complete per-domain configurations.
- [K versus tIoU](figures/screen_K_tiou.svg):native joint Hybrid/Uniform contexts.
- [Actual K/T distributions](figures/screen_actual_K_over_T.svg):all outliers retained.
- Each domain/screen folder: report.json, signed.tsv, all_signed_severe_timegroups.tsv, actual_observation_ratios.tsv, tails_timegroups_and_plot_data.json. TSV retains full precision.

## Engineering preservation

Originalscreen scoring computed both domain reports but qualification read obsolete budget/ reports. Frozen source, failure logs and all1660 prediction hashes were preserved. An independent CPU adapter changes that single path literal; reverse edits recover original code byte-for-byte. Both domain report/TSV bytes match before and after repair.1660 official dense checks maxerror1.7763568394002505e-15; no predictions or expert calls rerun for this repair. Newqueue continues original fixed confirmation, then unchanged controlled shifts.

CG-STVG v3 remains CPU-only and LAST: only its screen receipt references and actual TA-owner names are corrected. v1/v2 preparation retained. FiveCPU groups passed,0CG models/forwards/experts/realRGB/GT. True checkpoint no-op and cross-backbone results remain pending.
