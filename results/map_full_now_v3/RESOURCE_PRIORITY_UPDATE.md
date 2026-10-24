# Runtime resource protection update

The four CPU MAP jobs were actually started with the original CG-Vid untouched. After a rolling Vid pipeline rise, root paused ONLY its own MAP groups for60seconds; the automatic guard then took over and resumed them. Neither the original CG controller/predictor/teacher nor unrelated services received signals or priority/affinity changes.

During the pause,11new ordinary Vid queries averaged5.380399s pipeline,1.950742s Source and0.577596s feedback. Reconstructed pre-CPU baseline:5.050397s pipeline,1.944960s Source and0.592333s feedback. The CPU-on window before the precaution was7.428966s pipeline,2.445974s Source and1.099000s feedback. These are changing-video observational windows, not a randomized causal measurement or a zero-interference guarantee.

Root further narrowed ONLY its own MAP job groups and descendants to CPU14–15 (two physical cores), nice19 (lowest CPU priority), retaining idle-class3 disk I/O and one compute thread. New stage children inherit those restrictions. This narrows the initial four-core/nice15 upper resource allowance; the MAP posterior, DP, precision, seed, profiles, cohort and metric code are unchanged. Four tasks remain concurrent; only their CPU share is reduced to favor original Vid.

The original Vid remains nice0/CPU0–19 and keeps the shared GPU lock. MAP processes do not appear among GPU compute applications. The existing lowRAM/rate precaution continues to pause/resume ONLY owned MAP groups. Timings include shared-host resource controls; do not call them deployment latency. Exact process identities, pause/resume times and observations are private, and the [scalar protection receipt](resource_priority_override.json) is public.
