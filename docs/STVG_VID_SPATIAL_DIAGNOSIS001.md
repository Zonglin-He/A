# VidSTG spatial degradation: sealed confirmation diagnosis

Fixed64-source confirmation,32 per target domain; no new inference, fitting or hyperparameter selection. This offline analysis reads GT only after the already completed joint prediction seal and independent root. It does not use GT to construct or correct references. All64 anonymous rows and positive/negative/neutral cases are retained.

Full and Temporal-only have exactly identical output intervals and tIoU for all64 sources. The VidSTG degradation therefore comes from box geometry under the same interval.

| Vid query group | Sources | Full minus Temporal vIoU (pp) | Dense sIoU delta (pp) | Positive / negative / neutral |
|---|---:|---:|---:|---|
| all | 32 | -0.531916 | -2.296273 | 6 / 10 / 16 |
| question | 22 | -0.543205 | -2.501483 | 5 / 8 / 9 |
| statement | 10 | -0.507081 | -1.844812 | 1 / 2 / 7 |

All10 negative Vid sources have exact GT at one or more actual supervision anchors. On that known-anchor support, the chosen reference has lower mean IoU than the Source in every one of those10 cases. All10 also improve Student agreement with the chosen teacher reference after adaptation. No negative Vid case in this cohort has a better known-anchor mean reference or completely unknown anchor GT.

This supports an observed supervision-quality mismatch: fitting the reference more closely can worsen true localization. It does not prove wrong object identity in every case; a poor box around the same instance can also have low IoU. Known-anchor means cannot establish correctness of unobserved frames or every individual anchor, nor exclude optimization amplification.

HC has7 negative cases:4 have worse known-anchor mean references,3 have no exact GT anchor and remain unknown. Their uncertainty is not filled with zero or interpolated labels. Both query questions and statements are harmed in Vid; this is not solely a question-prompt failure.

The separately prelocked visual-budget P0 changes Temporal observation density/resolution and fixed8-frame selection only. Its16 sources were selected from development parent hashes, with no use of these confirmation scores; it does not by itself repair spatial teacher geometry. All four P0 arms must seal before their own GT evaluation. No new tracking/detection model, GT oracle fit, bootstrap or CURRENT change.

[All anonymous diagnosis rows and exact anchor-IoU metrics](../results/stvg_temporal_evidence_budget/2026-10-16/spatial_diagnosis001/DIAGNOSIS.json)
