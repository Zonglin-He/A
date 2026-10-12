# P7-A: expanded fixed-roster temporal × spatial study

This study includes the original 128 historical non-development parents per target, two original orders, 512 clean cells and 4,096 sealed eight-way readouts. Complete P6 temporal records are reused for the old32 per target; the added96 per target have192 new frozen capture forwards, after8 independent qualification captures. No spatial/temporal optimizer, new DINO or trainable parameter was used in P7-A.

Temporal Native T0 and original Actionness Projection T1 use identical original P6 recipe. Native is frozen source space; OPD, Direct and output-capped Direct share the saved original Full initial state/history. Captures are reused between orders only under exact source-state/pixels/IDs/input equality. Thus this is an exposed matched replication, not untouched confirmation or eight new independent online streams.

| Target | Subset | Parents | Paired comparison | Mean (pp) | 95% CI (pp) | >5 pp harm parents | >20 pp harm parents |
|---|---|---:|---|---:|---|---:|---:|
| hc2 | all128 | 128 | delta_T_OPD_v | +4.560 | [+2.790, +6.343] | 10 | 3 |
| hc2 | all128 | 128 | interaction_OPD_Native_v | +0.000 | [-0.557, +0.511] | 6 | 0 |
| hc2 | all128 | 128 | OPD_minus_Direct_T1_v | +0.741 | [-0.308, +1.866] | 5 | 1 |
| hc2 | all128 | 128 | OPD_minus_Capped_Direct_T1_v | +0.312 | [-0.203, +0.862] | 4 | 0 |
| hc2 | previous_P6_32 | 32 | delta_T_OPD_v | +3.816 | [+0.346, +7.162] | 4 | 1 |
| hc2 | previous_P6_32 | 32 | interaction_OPD_Native_v | +0.062 | [-0.800, +0.900] | 1 | 0 |
| hc2 | previous_P6_32 | 32 | OPD_minus_Direct_T1_v | +2.182 | [-0.186, +5.103] | 2 | 0 |
| hc2 | previous_P6_32 | 32 | OPD_minus_Capped_Direct_T1_v | -0.054 | [-1.236, +0.992] | 3 | 0 |
| hc2 | added96 | 96 | delta_T_OPD_v | +4.808 | [+2.709, +6.912] | 6 | 2 |
| hc2 | added96 | 96 | interaction_OPD_Native_v | -0.020 | [-0.697, +0.580] | 5 | 0 |
| hc2 | added96 | 96 | OPD_minus_Direct_T1_v | +0.260 | [-0.861, +1.371] | 3 | 1 |
| hc2 | added96 | 96 | OPD_minus_Capped_Direct_T1_v | +0.434 | [-0.137, +1.116] | 1 | 0 |
| vidstg | all128 | 128 | delta_T_OPD_v | +1.416 | [+0.641, +2.282] | 6 | 0 |
| vidstg | all128 | 128 | interaction_OPD_Native_v | +0.273 | [-0.104, +0.619] | 2 | 0 |
| vidstg | all128 | 128 | OPD_minus_Direct_T1_v | -0.166 | [-0.534, +0.232] | 4 | 0 |
| vidstg | all128 | 128 | OPD_minus_Capped_Direct_T1_v | -0.445 | [-0.744, -0.160] | 3 | 0 |
| vidstg | previous_P6_32 | 32 | delta_T_OPD_v | +2.056 | [+0.250, +4.261] | 1 | 0 |
| vidstg | previous_P6_32 | 32 | interaction_OPD_Native_v | +0.422 | [-0.723, +1.475] | 1 | 0 |
| vidstg | previous_P6_32 | 32 | OPD_minus_Direct_T1_v | -0.257 | [-0.871, +0.360] | 1 | 0 |
| vidstg | previous_P6_32 | 32 | OPD_minus_Capped_Direct_T1_v | -0.771 | [-1.680, -0.020] | 2 | 0 |
| vidstg | added96 | 96 | delta_T_OPD_v | +1.202 | [+0.382, +2.129] | 5 | 0 |
| vidstg | added96 | 96 | interaction_OPD_Native_v | +0.224 | [-0.107, +0.531] | 1 | 0 |
| vidstg | added96 | 96 | OPD_minus_Direct_T1_v | -0.136 | [-0.584, +0.331] | 3 | 0 |
| vidstg | added96 | 96 | OPD_minus_Capped_Direct_T1_v | -0.336 | [-0.609, -0.062] | 1 | 0 |

Full Native/OPD/Direct/cap T0/T1 mvIoU, mtIoU, fixed-event msIoU and strict R@.3/.5, each order, complete parent vectors/tails, gained/lost support and event-outside expansion remain in SUMMARY and all anonymous rows. Temporal-only changes leave fixed-GT-event spatial IoU unchanged. Missing decoder support is zero; no extrapolation or GT spatial filling.

The original dataset and source roster is fixed. Both orders are averaged within each parent before 10,000 paired bootstrap resamples with seed20261012; intervals are unadjusted for multiple comparisons. Old32 and added96 are displayed separately. Differences that favor Direct or output-capped Direct are not omitted.

## Locked resource branch to one P7-B

Eligibility requires added96 and all128 OPD T1−T0 means to be positive in both target datasets, after complete independent A root. This was locked before new GT. The CI is reported as uncertainty, rather than fitted into another gate. A positive mean with an interval spanning zero only licenses a bounded mechanism pilot, not a confirmed new method.

Actual post-root branch: **P7_B_eligible=True**. No parent, interval, learning rate, reward, temperature or observation frame is chosen using target GT. B has one fixed full512-cell four-arm experiment; A/B exact old fits and C/D new fits share original Full Before. Production registration and full official-query expansion remain unperformed.
