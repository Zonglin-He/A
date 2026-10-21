# development: fixed physical interval pooling

HC64 repeatedly exposed development panel. Fixed 2x2 readout; MAP is diagnostic. All predictions sealed before labels.

| Method | Full vIoU % | tIoU % | Boundary MAE s | Unobserved sIoU % |
|---|---:|---:|---:|---:|
| Source_Native224 | 26.8938 | 54.8230 | 2.6821 | 47.6990 |
| Hull_MBR | 29.6952 | 55.9613 | 2.6034 | 50.5762 |
| Hull_MAP | 27.4833 | 50.9166 | 2.7920 | 51.1620 |
| Mix_MBR | 26.4743 | 49.2917 | 2.8031 | 50.7342 |
| Mix_MAP | 13.2052 | 23.6288 | 3.9232 | 50.6417 |

| Comparison | vIoU pp [95% CI] | tIoU pp [95% CI] | v improve/harm/same | v decline >5pp |
|---|---:|---:|---:|---:|
| Mix_MBR-Hull_MBR | -3.2209 [-4.5709,-1.8970] | -6.6697 [-8.8123,-4.4825] | 13/48/3 | 18 |
| Mix_MAP-Hull_MAP | -14.2781 [-18.0864,-10.7122] | -27.2878 [-33.2455,-21.6226] | 7/55/2 | 41 |
| Hull_MBR-Hull_MAP | +2.2118 [0.3219,4.1993] | +5.0448 [1.8309,8.4047] | 42/18/4 | 7 |
| Mix_MBR-Mix_MAP | +13.2691 [10.1378,16.6029] | +25.6629 [20.8407,30.7469] | 58/3/3 | 0 |
| Mix_MBR-Source_Native224 | -0.4195 [-2.6418,1.7781] | -5.5313 [-7.5893,-3.4250] | 28/35/1 | 16 |
| Hull_MBR-Source_Native224 | +2.8014 [0.8963,4.7724] | +1.1383 [-0.4703,2.7668] | 41/22/1 | 8 |

## Same frozen spatial trajectory

| Interval source | Fixed-space vIoU % | tIoU % |
|---|---:|---:|
| Source_Native224 | 29.0550 | 54.8230 |
| Hull_MBR | 29.8122 | 55.9613 |
| Hull_MAP | 27.5436 | 50.9166 |
| Mix_MBR | 26.4895 | 49.2917 |
| Mix_MAP | 13.2809 | 23.6288 |

Source_Native224 above is the raw Source overall baseline. Its fixed-space row uses the common BoxFixed boxes and is only a temporal analysis. MAP is diagnostic only; main comparison is Mix_MBR−Hull_MBR. Strong is historical reference, not a gate. All signed/severe changes and costs remain in JSON/TSV.

Stage times may be nested. Posterior construction is compared on shared logits; no end-to-end multiplier inferred. Recheck has one shared224/eight-frame expert per query; cached intervention has no new network. Head/IO/MBR/DP and full capture are separate.
