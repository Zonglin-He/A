# Complete MAP decoding with spatial DP recomputed — queued

The user requests the complete MAP version of both **TA-STVG and CG-STVG**, on **HC-STVG-v2 1,902 queries and VidSTG 10,303 queries**. These are four backbone/domain evaluations, not parameter searches. This protocol is queued; no real-cache prediction or new metric evaluation has run yet.

For each query, retain the frozen upstream eight observations, Source boxes, geometric candidates, full-video auxiliary trajectory, original pooled features and visual feedback posterior **q1**. Select the highest-mass legal physical interval from q1, then run the **unchanged spatial DP at that new MAP interval**. The upstream auxiliary path is interval-independent and remains unchanged, so no raw-grid repooling or new model inference is required for this specific final-readout change.

MAP is `numpy.argmax(q1)` in C order, with the earliest flattened index breaking exact ties; inclusive native endpoints become `[frame_ids[start], frame_ids[end]+1)`. This is joint physical-interval MAP. TA's untouched Source instead takes per-offset MAP followed by a physical hull; these are explicitly different readout constructions despite both using the name MAP.

## Reported arms

| Arm | Boxes | Time interval | Purpose |
|---|---|---|---|
| Source | Original source boxes | Original native readout | Untouched readout reference under the local frozen input/precision protocol. |
| SourcePlusMBR | Same source boxes | q0 MBR | Preserve the existing readout-only control. |
| SourceJointMAP | Same source boxes | q0 joint MAP | Compare against the same joint-MAP decoder without visual posterior refinement. |
| FullMBR | Original final DP | q1 MBR | Must reproduce all completed Full boxes/intervals bitwise. |
| FullMAPFixedBoxes | Existing FullMBR boxes | q1 MAP | Isolate the readout effect on fixed boxes; not presented as the complete MAP pipeline. |
| **FullMAP** | **DP recomputed at q1 MAP** | **q1 MAP** | **Requested complete MAP output.** |

TA keeps its original kappa64/beta1/context1.5 configuration. CG keeps the already frozen HC gamma0/u0/link0/beta0.5/kappa64/context1.5 and Vid gamma0.5/u0.5/link0.25/beta1/kappa64/context1. No accuracy-based combination, backbone change or configuration reselection is performed. In particular, CG-HC retains Source boxes on unobserved frames; evaluating its MAP output does not turn that conservative setting into full spatial propagation.

## Dependencies, cache and validation

Current CG Vid full inference continues unchanged. This new controller waits without acquiring a GPU lock until both CG full domains have complete seals, successful official score receipts, exact-replay audit reports, named-source semantics reports and actual owned-process exit. No real feature cache, video, annotation or model is opened for this study before that gate. Original sufficient caches hold all needed geometry/q0/pooled features/native intervals; the budget is **zero Source forwards, zero appearance-prefix forwards, zero expert calls, zero optimizer steps and zero GPU work**.

The previously waiting strict-deletion controller was stopped only while it had zero predictions, no child, no GPU lock and no experimental work. Its identity/state/log were preserved; its frozen source/plan/scientific definitions were not changed. Order is now:

1. Finish and report the current CG HC/Vid fulls.
2. Run the four complete cache-only MAP/DP evaluations (TA-HC, TA-Vid, CG-HC, CG-Vid).
3. After all MAP prediction seals, official scores and postseal reports succeed, launch the **unchanged** four-row strict-deletion entrypoint. Its separate frozen MBR protocol remains a historical matched study; this MAP comparison does not silently rename or overwrite it.

Four synthetic CPU checks cover MAP endpoint conversion and exact-tie behavior, without real evidence or GT. After the dependency gate, an eight-query bridge (two per backbone/domain) must reproduce every FullMBR result exactly and replay MAP+DP outputs deterministically; TA MAP intervals must match the already sealed P1MAP conditional controls. The new output must actually solve the spatial DP at MAP rather than reuse FullMBR boxes.

Each complete backbone/domain prediction set seals before independent CPU official scoring. Reports preserve all signed t/v/s values; FullMAP versus FullMBR, Source, SourceJointMAP and fixed-box MAP; severe t/v loss >5pp and union counted once; rescues; short/long event groups; parent-cluster pointwise CIs; parent Source/FullMBR metric equality; every prediction/receipt/input SHA; and all-array replay. CPU costs are segmented by actual PID/start. They are conditional cache-replay costs, not deployment latency; original Source/expert acquisition provenance remains bound separately.

All official full sets are historically exposed. CG-HC all227 parents and CG-Vid64 development queries informed their fixed profiles. This is a fixed decoder comparison, not fresh confirmation or justification for another search. Both positive and negative MAP results will be retained. MAP simplifies the stated final decision rule; it does not establish immunity to reviewer criticism or erase the distinction from the original native Source.

See [frozen protocol](protocol.json) and [unfinished dependency refusal](deferred_gate_test.json). Queue preparation and synthetic checks are not empirical full MAP results.

Latest user requests CPU parallelism. The four backbone/domain jobs now run concurrently after the same gate and preflight, with one compute thread per subprocess and an8GiB available-RAM launch floor. The original scientific source/plan is unchanged; the prior sequential waiting controller was preserved and stopped at zero work. See [parallel scheduling](PARALLEL_SCHEDULING.md). Original strict deletions still follow all four MAP score/report completions.
