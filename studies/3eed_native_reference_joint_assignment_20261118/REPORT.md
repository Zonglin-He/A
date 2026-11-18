# R171 — Native reference candidates and joint spatial assignment

Preserving reference-as-subject candidates directly increased verified reference coverage. The added candidates did not improve verified dual binding or Acc@50 on the new physical frames. [Full report in Chinese](REPORT_ZH.md)

The original subject pool and its unary scores remain fixed. D retains its native proposals, evaluates the inherited camera-left/right factor in the shared scene, and jointly selects the subject and reference. Its controls use the mapped bank, no explicit left/right term, a fixed native reference, and native geometry restricted to previously linked proposals. All 96 R170 records replay the five original endpoints exactly. The additional 72 directional records come from 36 distinct physical frames. Their Drone and Quad recordings have earlier exposure; their physical frames are separate from R170.

| New-frame readout | mIoU | Acc@25 | Acc@50 | Verified dual correct / 32 |
|---|---:|---:|---:|---:|
| R158 with PCA | 15.372% | 27.778% | 15.278% | — |
| C mapped, joint | 16.509% | 30.556% | 18.056% | 2 |
| D mapped, joint | 15.622% | 29.167% | 15.278% | 2 |
| D native, independent | 15.372% | 27.778% | 15.278% | 2 |
| D native, fixed reference | 15.747% | 29.167% | 13.889% | 3 |
| D native, linked only, joint | 16.922% | 31.944% | 16.667% | 3 |
| D native, all references, joint | 15.838% | 29.167% | 13.889% | 2 |

All five R170 mapping losses regained a correct candidate, but one was selected. The one newly correct R170 pair used a previously linked reference. On the new frames, correct reference coverage increased from 13 to 18 out of 32, with one of six old mapping losses selected correctly. Unrestricted native reference competition made two subject predictions worse and none better compared with the linked-only native control. Keep R158 with PCA for deployment. The structural joint-assignment description is supported; a reliable dual-binding improvement has not been validated.

[Frozen protocol](experiment/PROTOCOL.json) · [Evaluation](results/metrics.csv) · [Matched comparisons](results/paired.csv) · [Reference use](results/unmapped_reference_cases.csv) · [Independent audit](SCALAR_AUDIT.json)
