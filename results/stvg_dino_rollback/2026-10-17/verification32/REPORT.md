# Historical DINO P7-A restoration: fixed32 regression validation

User-selected rollback from Qwen to the historical frozen Actionness projection + single DINO + online Gaussian OPD. Qwen official continuation is paused and its partial evidence remains intact. No official full-population or historical controller is restarted.

Original source/fitter/config bytes match the pre-cleanup snapshot. Same128-parent original P0 order1 first16 arrivals per target are fixed before new models or GT. This exposed short-stream regression set is not independent confirmation, not the Qwen development32 cohort, and not a long-stream/full-query result.

| Target | Arm | vIoU % | tIoU % | dense sIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
| hc2 | Frozen_native | 18.831315 | 45.792579 | 37.714391 | 25.000000 | 6.250000 |
| hc2 | Frozen_projection | 24.567791 | 63.582966 | 37.714391 | 25.000000 | 6.250000 |
| hc2 | Before_projection | 25.188800 | 63.582966 | 38.669129 | 31.250000 | 6.250000 |
| hc2 | OPD_native_readout | 18.506924 | 45.792579 | 35.077293 | 18.750000 | 12.500000 |
| hc2 | Full_P7A | 24.189334 | 63.582966 | 35.077293 | 37.500000 | 12.500000 |
| vidstg | Frozen_native | 7.147781 | 31.818184 | 36.753536 | 6.250000 | 0.000000 |
| vidstg | Frozen_projection | 7.276791 | 32.842732 | 36.753536 | 6.250000 | 0.000000 |
| vidstg | Before_projection | 8.380680 | 32.842732 | 40.902953 | 6.250000 | 0.000000 |
| vidstg | OPD_native_readout | 11.077868 | 31.818184 | 46.393736 | 6.250000 | 0.000000 |
| vidstg | Full_P7A | 11.413770 | 32.842732 | 46.393736 | 12.500000 | 0.000000 |

Frozen_projection holds Source boxes and changes only I0 to I1; Full_P7A minus that control measures the complete online Spatial contribution on this fixed temporal readout. Before_projection contains inherited LN but no current-query update. OPD_native_readout holds adapted boxes and restores Native I0 for attribution. All original denominators, negative/neutral/unknown cases and signed paired deltas remain.

The implementation is restored and its checked invariants pass; this does not establish new overall superiority. In this short prefix HC Full falls below its projection-only and inherited-before controls, while Vid improves. The user rollback is not a result-driven retuning decision. Existing development, failed confirmation and negative findings are retained.

Checks cover32 complete fresh Source/DINO/OPD predictions, original Gaussian/IoU/Adam arithmetic, frozen Source and specialist,1792 coordinates, QR/Adam reset, next-arrival LN/writeback, independent direct Actionness-score enumeration and both-Decoder output equality. The16 saved Vid fits and committed histories are exact; HC historical raw fit payloads were retired earlier and are not claimed reloaded. Eight fixed actual Native-anchor image outputs were privately viewed, with zero new independent semantic labels.

All32 complete predictions and GT-free ROOT sealed before offline GT scoring. Independent root readback recomputed 1120 scalar metrics and all summaries/paired deltas exactly. The results omit original Query text, identities, raw frames, reference/GT coordinates, tensors, weights and private paths.

Predictor timing excludes upstream video decoding and extra verification/audit/serialization. It is not full fresh end-to-end deployment latency. No Prompt/parameter/Teacher changes or optional search is performed.
