# hc2: fixed physical interval pooling

Frozen parent/video-isolated DEVELOPMENT RECHECK, not historically unexposed independent confirmation. Both domains have documented full-domain historical exposure; selected parents exclude the current128 panel, not all retired older panels.

| Method | Full vIoU % | tIoU % | Boundary MAE s | Unobserved sIoU % |
|---|---:|---:|---:|---:|
| Source_Native224 | 23.4655 | 50.4639 | 2.8638 | 48.0826 |
| Hull_MBR | 27.8907 | 51.6010 | 2.7202 | 51.3546 |
| Mix_MBR | 27.3192 | 49.2260 | 2.8857 | 50.8784 |

| Comparison | vIoU pp [95% CI] | tIoU pp [95% CI] | v improve/harm/same | v decline >5pp |
|---|---:|---:|---:|---:|
| Mix_MBR-Hull_MBR | -0.5715 [-2.7294,1.7349] | -2.3750 [-5.7250,1.2008] | 11/18/3 | 5 |
| Mix_MBR-Source_Native224 | +3.8537 [-4.6895,12.6308] | -1.2379 [-5.0465,2.9665] | 15/16/1 | 6 |
| Hull_MBR-Source_Native224 | +4.4253 [-3.3213,12.3936] | +1.1371 [-0.6253,3.0265] | 23/8/1 | 3 |

## Same frozen spatial trajectory

| Interval source | Fixed-space vIoU % | tIoU % |
|---|---:|---:|
| Source_Native224 | 27.0355 | 50.4639 |
| Hull_MBR | 27.8907 | 51.6010 |
| Mix_MBR | 27.3377 | 49.2260 |

Source_Native224 above is the raw Source overall baseline. Its fixed-space row uses the common BoxFixed boxes and is only a temporal analysis. MAP is diagnostic only; main comparison is Mix_MBR−Hull_MBR. Strong is historical reference, not a gate. All signed/severe changes and costs remain in JSON/TSV.

Stage times may be nested. Posterior construction is compared on shared logits; no end-to-end multiplier inferred. Recheck has one shared224/eight-frame expert per query; cached intervention has no new network. Head/IO/MBR/DP and full capture are separate.
