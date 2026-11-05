# hc2: matched ReferDINO evidence, common Full interval

Whole128/128 parents independently sealed, official scored and manual-pixel/input/mask/SHA/means/CI audited.

|Arm|tIoU|vIoU|R.3|R.5|Conditional dense unobserved IoU|
|---|---:|---:|---:|---:|---:|
|Full|54.968548|31.791298|50.781250|22.656250|56.502153|
|SAM2MatchedMask|54.968548|33.751466|52.343750|25.000000|60.214570|

Eight original selected ReferDINO masks prompt official SAM2.1 Hiera-L, all native RGB frames, exact same Full event and same128 queries. No Source/expert/optimizer new calls. SAM2 native mask boxes at prompted frames too; no Source, expert or E output overwrite. Current soft E/expert anchor selection unchanged. The additional SAM2 encoder/model/precision/compute differ, so this is matched expert evidence, not equal models or same internal features.

Conditional dense unobserved IoU uses only common predicted-event/GT-event annotated intersection excluding eight exact observed physical times. Empty support null, allqueries retained in primaryv. All positive/negative tails and short/long pointwise parentCIs retained. Checkpoint/provenance IO is inside experimental clock; not deployment latency.

Full-SAM2 vIoU: {'mean': -1.960168111874897, 'CI95': [-3.272077450781841, -0.6769906331061512], 'queries': 128, 'parents': 128}.
