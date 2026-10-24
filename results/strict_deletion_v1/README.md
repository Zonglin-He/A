# Strict deletion ablations — queued after both CG full evaluations

The user authorizes four rows for the fixed **TA-STVG main method**, evaluated on all HC-STVG-v2 1,902 queries and VidSTG 10,303 queries. This is a new independent study, not a rerun of the selected CG-specific profiles. No predictions or experimental measurements have been produced for these deletions yet.

| Row | Shared correction | Spatial trajectory inference | Temporal inference | Exact bypass |
|---|---|---|---|---|
| Full | retained | retained | retained | Frozen original main configuration; every Full output must reproduce the completed TA full bitwise. |
| w/o shared correction | deleted | retained | retained | Set the existing corrected initialization E to the original Source boxes, rebuild all dependent geometry/paths, repool features and recompute the posterior and final path. |
| w/o spatial trajectory inference | retained | deleted | retained | Use the existing upstream E directly for boxes and ROI pooling. Delete propagation candidates, path scoring and DP; no interpolation, greedy path or direct expert overwrite is introduced. |
| w/o temporal inference | retained | retained | deleted | Delete visual posterior refinement and the added MBR readout together. Use the original per-offset native MAP physical hull, then compute the retained spatial path at that interval. |

The original Source and Source+MBR are separately named reference rows in the supporting report. MBR is an implementation component of temporal inference, not an untouched Source behavior or an additional claimed innovation. The prior NoLocal/Independent controls and OursS with MBR are not reused as deletion outcomes.

## Order and cache use

The controller waits without taking the shared GPU lock. It may open feature caches or produce experimental predictions only after **both current CG full domains** have their complete prediction seals, official score receipts, postseal audit reports, named-source comparison receipts and actual owned-process exit. Both domain score/seal/report SHA bindings are checked. A current CG failure stops the deferred queue for inspection; it does not trigger an automatic model retry.

There are 384 known, previously validated raw-grid caches (192 per direction). Other SHA/input-bound legacy caches are inspected only after this gate. Compatible full raw grids require no GPU work. If only the original post-fusion spatial grid exists, reconstruct the original pre-text appearance prefix; if the spatial grid is missing, reconstruct the original Source224 once. Rebuild geometry and actually repool after a deletion changes its ROI path. Saved Full pooled features do not substitute for new-path pooling.

The maximum number of new Source-or-prefix reconstructions is 11,821 queries combined, reduced by any additional compatible raw caches found. There are **zero new expert observations or expert calls**, zero optimizer steps and no parameter search. Original evidence at the same eight positions is reused with SHA bindings. The production profile, current CG fulls and archived roots are unchanged.

After the gate, a four-query CPU bridge checks raw-grid-to-Full pool equality, exact reproduction of saved Full outputs, the three deletion invariants and all-array compact replay. Each whole domain seals all predictions before independent CPU official scoring. Postseal verification checks every prediction/capture/receipt SHA and exact compact replay, and exports all signed outcomes, severe temporal/spatial union counted once, short/long groups, parent-cluster CIs and real Source/prefix costs by worker PID/start. Original acquisition provenance and conditional cache replay times are distinct from deployment cost.

All full sets are historically exposed; no fresh or independent-test selection claim is made. Results, including negative outcomes, will be published without changing the fixed configurations. The frozen protocol is [protocol.json](protocol.json); the unfinished-dependency refusal test is [deferred_gate_test.json](deferred_gate_test.json). Queue preparation is not completion of the experiments.
