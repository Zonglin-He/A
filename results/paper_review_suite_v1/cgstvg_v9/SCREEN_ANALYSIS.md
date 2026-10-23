# CG-STVG fixed screening results

Native420FP32, official cross-source EMA,64fixed parents/domain. All128predictions sealed before640official/common/dense metric comparisons, maximum discrepancy1.4432899320127035e-15. Four real checkpoint native-head/observer/no-GT/pooling/coupled-replay bridges passed. Full available input segment and official single-grid sampling retained; input last-valid/one-past boundary conversion documented separately. All settings transferred unchanged; no CG target tuning. The original unmodified full executable OOMed; native reference uses the bitwise-audited temporary storage schedule, not an unmodified-executable completion.

| Target | Arm | m_tIoU (%) | m_vIoU (%) | m_sIoU (%) |
|---|---|---:|---:|---:|
| hc2 | SourceNative | 19.716285 | 11.288811 | 56.835074 |
| hc2 | Source | 42.806323 | 22.759664 | 56.835074 |
| hc2 | SharedCorrection | 42.806323 | 19.896443 | 48.063730 |
| hc2 | Space | 42.806323 | 22.289991 | 51.003905 |
| hc2 | Full | 45.752582 | 23.036291 | 50.468761 |
| vidstg | SourceNative | 20.277952 | 10.086774 | 47.151641 |
| vidstg | Source | 24.918847 | 12.768687 | 47.151641 |
| vidstg | SharedCorrection | 24.918847 | 11.949456 | 44.375082 |
| vidstg | Space | 24.918847 | 13.043494 | 45.813463 |
| vidstg | Full | 29.676564 | 15.926685 | 45.957861 |

SourceNative uses original jointMAP. Source uses MBR on the SAME native boxes; all feedback contrasts share that MBR decision rule. NativeMAP→MBR gains must not be attributed to external evidence.

hc2: Full−Source Δt +2.946260pp CI[-0.5175060566830455, 6.704641249777691], Δv +0.276627pp CI[-3.7468462972849017, 4.651515106470197]. Full−Space Δv +0.746300pp CI[-1.010179997515544, 2.642329055670453]. Full−Source unobserved sIoU -6.747711pp CI[-13.499840561219731, -0.17968650089212174]. SevereSource-relative union: SharedCorrection 14, Space 18, Full 29 of64. All harmful cases retained.
vidstg: Full−Source Δt +4.757716pp CI[1.9865207394159992, 7.979454033548317], Δv +3.157998pp CI[0.24202538719760455, 6.252063950622461]. Full−Space Δv +2.883191pp CI[0.8872089933376446, 5.244190070072598]. Full−Source unobserved sIoU -1.422761pp CI[-5.8580849326432824, 2.913641777448819]. SevereSource-relative union: SharedCorrection 9, Space 7, Full 9 of64. All harmful cases retained.

HC Full−Source vCI crosses zero, with negative unobserved spatial quality; Vid v/time CIs are pointwise positive. Screening does not establish broad cross-backbone benefit or a best setting. No accuracy-based qualification: all four bridges and complete both-domain implementation permit the unchanged pre-fixed128parents/domain confirmation. These sets are parent-disjoint for this contrast, but officialfullsets historically exposed. CIs are pointwise without familywise correction.

All signed t/v/s deltas, severe t/v/union-once, timegroups and parent CIs plus original expertcosts and worker-incarnation segments are in each domain folder. Reporting reuses already scored metrics,0additional GTscoring/model/expert. The private native acquisition snapshots/features/media/captions/GT coordinates remain private.

Screen stored Source costs:139 completed forwards =128regular+4native bridge references+7stream compatibility replays. Additional engineering history:4earlier duplicate completed Source references and1completed Source before zeroarea readout failure;4partialOOM Source attempts separately charged with missing full timings unknown. The decode-boundary failure consumed0Source/expert. Stored receipts record125newexpert calls plus3already-paid exact-context acquisitions from earlier preserved failures/references, covering128unique screen contexts; original expertGPUcost retained on cache hits. Migration is not a deployment speed claim. Meanpipeline excludes videoDecode/subjectparse/someinitialization, nestedcosts notadded. See recovery evidence for historical costs and plan-prose clarification.

Confirmation is currently running; no confirmation accuracy is available before its full256-query seal and CPUscore.
