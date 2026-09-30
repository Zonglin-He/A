# Paper48 P4: measured uncached cold-start efficiency

P4 completed100 fixed hash-selected VidSTG queries under both Frozen and Ours,200 native process runs total. The same frozen Vid-trained TA-STVG checkpoint, J0.1 update and input interface used in P1–P3 apply. Ours maintains1792 spatial parameters across the100-query stream and schedules both specialists every fourth arrival:25 new spatial and25 new temporal calls, with24 nonempty spatial updates. There is no GT scoring in P4.

## End-to-end measurements

| Arm / arrival type | Queries | Mean seconds | Median seconds | p95 seconds | Peak allocated GPU GiB |
|---|---:|---:|---:|---:|---:|
| Frozen |100|8.3874|8.3564|8.8556|7.8306|
| Ours, all |100|18.4230|9.3278|45.5977|9.3943|
| Ours, expert scheduled |25|46.2577|44.4986|55.5899|9.3943|
| Ours, nonexpert |75|9.1448|9.1003|9.7371|7.8672|

These are real serial cold-start process times on one RTX5090 with32GiB. Every native arrival starts a new Python process and loads its model; every scheduled specialist also starts and loads afresh. End-to-end times include process launch, contract/checkpoint verification, loading, decoding, inference, native probes, update, serialization and model switching. Persisted student state is restored between Ours arrivals, but input features and specialist calls are not reused. The benchmark is not a warm resident service latency or a GPU-kernel-only benchmark. GPU memory is PyTorch peak allocated memory, not total device/process reserved memory.

The all-arrival mean is18.4230s for Ours versus8.3874s Frozen under this loading model. Ours nonexpert arrivals average9.1448s, while its scheduled expert arrivals average46.2577s. Within an expert arrival, spatial and temporal specialist processes average18.2585s and18.2972s respectively, including their independent loading. Native decode/forward and adaptation components are separately available in SUMMARY.json and all200 timing rows in TIMINGS.csv. Adaptation averages0.5650s on expert arrivals and0.0430s on nonexpert arrivals; these components alone do not represent end-to-end deployment cost.

## Verification and engineering history

Root readback verified all200 prediction hashes and timing receipts,100 persistent state links and all50 newly executed specialist calls. All100 Frozen outputs are bitwise equal in boxes and identical in decoded temporal indices to the saved same-input P1 source-native predictions. The first Ours current boxes equal Frozen before its spatial update. No GT was read during these checks and no new inference was performed by the auditor.

An independent public auditor reconstructs all mean/median/p95 timings, schedules, call counts and peak-memory aggregates from anonymous timing scalars:1160 checks pass. Source media, captions, original predictions, model states and weights are not exported.

Two preliminary attempts remain archived and excluded from the final timing table. First, an early direct function import bypassed the clean-loader wrapper; the no-GT guard blocked constructor annotation access before any valid timing row. Revision006 fixed the binding order. Second, a raw AMP Frozen path omitted the already established FP32 post-encoder interface;12 rows totaling101.284181s were preserved, then regenerated after revision007 restored the original precision hooks. The final table contains only the200 correctly matched rows. Neither recovery changed source selection, method, checkpoint, expert availability or output policy.

## Required P5 continuation

P5 reached its spatial stage but stopped at18 receipts when the shared raw-frame decoder could not find a requested endpoint. The first18 receipts and3 new expert executions are preserved as an excluded engineering attempt. For the failing file, raw vsync=0 decoding yields499 frames, whereas the official HC loader's default FFmpeg output timing yields500; the locked final index499 exists on the latter grid. Revision008 binds only P5 to the official HC decoding semantics, then selects the original frozen indices. No manual padding, clipping, substitution or roster change is used. All128 sources pass the new CPU decode preflight. Expert observations are regenerated consistently, and the required P5 queue continues; P4 completion does not mean all phases are complete.
