# Four-way CPU scheduling, same frozen MAP study

The user's follow-up explicitly requests CPU parallelism. The original MAP scientific plan, cohorts, profiles, q1 decoder, new-interval DP, metrics and seed are unchanged. Independent `studies/map_full_v1/parallel_v2.py` changes only execution scheduling and records its own fingerprint. The original `run.py` and `map_core.py` remain byte-for-byte frozen.

The host reports 20 logical CPUs and 14 physical cores. At preparation, available RAM was about16.6GiB with the CG-Vid model and expert still resident. The planned concurrency is **four independent backbone/domain jobs**, not20 decoder threads or extra models:

| Concurrent domain job | Sequence inside the job | Compute threads per subprocess |
|---|---|---:|
| TA–HC1902 | original predict → original score → original report | 1 |
| TA–Vid10303 | original predict → original score → original report | 1 |
| CG–HC1902 | original predict → original score → original report | 1 |
| CG–Vid10303 | original predict → original score → original report | 1 |

Each stage is launched using the **unchanged frozen `run.py` commands** in a separate subprocess. In particular, prediction's no-GT audit guard cannot leak into the scoring process. A domain must finish its entire prediction seal before that domain accesses official labels; other independent domains may still be predicting. All jobs first wait for the existing both-CG-full seal/score/audit/named-report/owned-exit gate and successful original eight-query preflight.

The controller requires at least8GiB available RAM before starting another domain job. CUDA is hidden; OMP/OpenBLAS/MKL and Torch compute threads are limited to one per active computational subprocess. The launch cap remains four domain jobs. This leaves system resources available and limits simultaneous cache I/O. No speedup factor or completion time has been measured yet.

Only the prior **waiting**, zero-prediction, childless, no-GPU MAP controller was stopped. Its controller/state/launch/log and scheduling authorization/exit evidence were preserved privately. Current CG-Vid continues untouched. The strict-deletion study remains deferred. After all four MAP jobs have successful complete official scores and replay reports, the new controller launches the same original strict-deletion entrypoint; it cannot bypass an unsuccessful MAP domain.

Each job has its own state, stage child PID/start, exit receipt and logs. All CPU query costs retain the real scientific worker PID/start rather than being combined across workers. A failure is preserved without retry, fallback or sample deletion; other independent jobs finish their evidence, and any failed job prevents the strict-deletion chain from starting. Root inspects completion and failure states and publishes every signed positive/negative result.

The [parallel execution protocol](parallel_v2_protocol.json) augments the [original scientific protocol](protocol.json). This is a queued engineering change, not a completed MAP experiment or a measured deployment acceleration.
