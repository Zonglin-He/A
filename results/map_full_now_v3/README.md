# MAP + recomputed spatial DP — CPU parallel execution now

The latest direct user request explicitly changes the previous wait: **start CPU-parallel MAP replay now, alongside the original CG-Vid full, without interrupting it**. This independent namespace replaces the zero-work waiting scheduler. Original MAP-v1 scientific source/plan, parallel-v2 source/plan, every prior output and current CG inference remain immutable.

The six arms and mathematical definitions are unchanged from the [original MAP protocol](../map_full_v1/README.md): original Source, Source+MBR, q0 joint-MAP Source, FullMBR, fixed-box MAP and **FullMAP with spatial DP genuinely recomputed at the new q1 MAP interval**. The same eight observations, upstream trajectory/features/q0/q1, domain profiles, seed and metrics remain fixed. `frozen.calculate` and the original `map_core.py` are reused without modification. This is not an additional parameter search or an assurance about reviewer acceptance.

## Actual work and readiness

Four low-priority, single-thread CPU jobs started:

| Job | Parent input state | When it may access GT |
|---|---|---|
| TA–HC1902 | Original full is already complete/sealed | After all1,902 MAP predictions seal. |
| TA–Vid10303 | Original full is already complete/sealed | After all10,303 MAP predictions seal. |
| CG–HC1902 | Original full is complete/sealed/scored | After all1,902 MAP predictions seal. |
| CG–Vid10303 | Follow only committed immutable per-query receipts/compact/predictions | After all10,303 MAP predictions seal **and** the original CG-Vid full seals/scores. No partial-domain scoring. |

CG-Vid's atomic receipt is written only after its own prediction/compact cold replay succeeds. The MAP consumer verifies that receipt's original input binding, frozen parent fingerprint and all three file SHAs; it never reads active raw Source scratch or incomplete files. Once the full parent seal exists, every streamed parent SHA is checked against it before the MAP domain can seal. The consumer may catch up with the producer and wait for the next committed query; it does not acquire new Source/expert evidence or skip pending queries.

The original scoring and reporting **code objects are reused unchanged** in separate subprocesses, rebinding only this new output directory, frozen-plan validation and per-domain readiness. The prediction no-GT audit guard stays in its own process. This new authorized gate does not relax any source/input hash or permit early CG-Vid labels.

## Resource priority

The new controller and its descendants have CUDA hidden, nice15, idle-class3 disk I/O and CPU affinity12–15 (four physical cores on this host); each active computational process uses one Torch/BLAS/OMP thread. The original CG controller/predictor/teacher retain their original priorities, CPU affinities, PIDs, parameters and GPU lock. No unrelated service is signalled.

Available-RAM launch floor is8GiB. If available RAM falls below8GiB, or the latest50 ordinary CG-Vid query pipelines exceed the recorded pre-launch baseline by1.35x after25new queries, the controller may SIGSTOP **only its own MAP process groups**. Resume requires available RAM>10GiB and at least60seconds pause. The rate threshold is a precaution, not a claim that a slowdown was causally caused by MAP. Shared-host CPU and disk use cannot guarantee mathematically zero interference; resource priority and observations are recorded explicitly.

Eight actual committed-cache CPU bridges passed before the jobs started: saved FullMBR arrays/intervals are reproduced bitwise, MAP outputs including new-interval DP replay exactly, and TA MAP intervals match the previous P1MAP controls. **Zero new Source, prefix, expert, RGB, optimizer or GPU calls.** These bridge checks do not constitute full MAP performance results. See [bridge receipt](preflight_receipt.json), [launch snapshot](launch_running_receipt.json) and [frozen current protocol](protocol.json).

All whole-domain outputs will be independently scored and audited, with every signed positive/negative, temporal/spatial severe union counted once, rescues, short/long parent-cluster pointwise CIs, Source/readout distinctions and per-worker conditional CPU wall cost. Paused or shared-host timings are not uncached deployment latency. All full sets remain historically exposed; no hyperparameters or samples are selected from these results.

The original four-row strict-deletion study remains deferred. Even if some MAP domains finish early, its GPU reconstruction cannot start until **all four MAP domains** have complete score/replay reports **and** both original CG fulls have their audit/named-source reports and actual owned-process exit. Root reports each valid complete result when available and preserves every failure without fallback or retry.


## First completed domain

[TA-HC1902 complete MAP+DP result](TA/hc2/ANALYSIS.md): t48.586628/v26.731488%; versus FullMBR−.990214t/−.486611v pp. Official11412checks and9510SHA/all-array replay complete. Other domains remain in progress; see [resource protection update](RESOURCE_PRIORITY_UPDATE.md) for the current narrower CPU14–15/nice19 setting rather than the initial launch setting.
