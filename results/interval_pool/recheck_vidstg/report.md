# vidstg: fixed physical interval pooling

Frozen parent/video-isolated DEVELOPMENT RECHECK, not historically unexposed independent confirmation. Both domains have documented full-domain historical exposure; selected parents exclude the current128 panel, not all retired older panels.

| Method | Full vIoU % | tIoU % | Boundary MAE s | Unobserved sIoU % |
|---|---:|---:|---:|---:|
| Source_Native224 | 15.6772 | 34.0110 | 5.6558 | 44.0397 |
| Hull_MBR | 23.9576 | 35.6554 | 5.6122 | 60.6281 |
| Mix_MBR | 22.9325 | 33.8749 | 5.4742 | 60.6048 |

| Comparison | vIoU pp [95% CI] | tIoU pp [95% CI] | v improve/harm/same | v decline >5pp |
|---|---:|---:|---:|---:|
| Mix_MBR-Hull_MBR | -1.0251 [-2.8904,0.5591] | -1.7805 [-4.2030,0.3903] | 15/12/5 | 6 |
| Mix_MBR-Source_Native224 | +7.2552 [3.2429,11.8102] | -0.1361 [-1.9684,1.8006] | 23/4/5 | 2 |
| Hull_MBR-Source_Native224 | +8.2804 [3.5570,13.5748] | +1.6443 [-0.0972,3.8050] | 23/5/4 | 3 |

## Same frozen spatial trajectory

| Interval source | Fixed-space vIoU % | tIoU % |
|---|---:|---:|
| Source_Native224 | 22.8442 | 34.0110 |
| Hull_MBR | 23.9576 | 35.6554 |
| Mix_MBR | 22.9344 | 33.8749 |

Source_Native224 above is the raw Source overall baseline. Its fixed-space row uses the common BoxFixed boxes and is only a temporal analysis. MAP is diagnostic only; main comparison is Mix_MBR−Hull_MBR. Strong is historical reference, not a gate. All signed/severe changes and costs remain in JSON/TSV.

Stage times may be nested. Posterior construction is compared on shared logits; no end-to-end multiplier inferred. Recheck has one shared224/eight-frame expert per query; cached intervention has no new network. Head/IO/MBR/DP and full capture are separate.
