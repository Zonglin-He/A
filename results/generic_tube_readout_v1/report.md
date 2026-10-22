# 通用时空重叠读出

Status: Requested generic interface, failed measured adoption criteria. All outputs completed and independently scored; no method adopted.

All128 queries have repeated development exposure. HC32/Vid32 are parent-isolated historical development checks, not untouched confirmation. Every complete panel sealed before this panel's new GT scoring. GT-derived error/rescue subsets are offline diagnosis only.

## Signed effects relative to Global6 + original BoxFixed

| Cohort | Method | ΔtIoU pp | Δfull vIoU pp | Δfixed-track vIoU pp | tIoU .5 rescue/loss | Changed intervals |
|---|---|---:|---:|---:|---:|---:|
| HC64 | ExpectedTube | -0.3722 | -0.1076 | -0.1354 | 0/0 | 10 |
| HC64 | FixedTrackTube | -0.3016 | -0.0883 | -0.0937 | 0/0 | 9 |
| HC64 | MeanQuality | +0.0586 | +0.0294 | +0.0304 | 0/0 | 2 |
| HC64 | ShiftedQuality | +0.1689 | +0.0604 | +0.0209 | 0/0 | 11 |
| HC64 | PriorMatched | -0.0447 | +0.0316 | +0.0093 | 0/0 | 10 |
| HC64 | RandomMatched | -2.5165 | -1.2622 | -1.2622 | 0/2 | 10 |
| HC64 | OldJoint10 | -0.1175 | -0.0263 | -0.0175 | 0/0 | 5 |
| HC32 | ExpectedTube | -0.2739 | -0.0844 | -0.0844 | 0/1 | 6 |
| HC32 | FixedTrackTube | -0.2739 | -0.0844 | -0.0844 | 0/1 | 6 |
| HC32 | MeanQuality | +0.0000 | +0.0000 | +0.0000 | 0/0 | 0 |
| HC32 | ShiftedQuality | +0.4028 | +0.2708 | +0.2222 | 0/0 | 7 |
| HC32 | PriorMatched | -0.5662 | -0.2233 | -0.2233 | 0/1 | 6 |
| HC32 | RandomMatched | -0.1559 | +0.0420 | +0.0780 | 0/1 | 6 |
| HC32 | OldJoint10 | +0.4082 | +0.2664 | +0.2732 | 0/0 | 2 |
| Vid32 | ExpectedTube | +0.0138 | +0.0155 | +0.0155 | 0/0 | 1 |
| Vid32 | FixedTrackTube | +0.0138 | +0.0155 | +0.0155 | 0/0 | 1 |
| Vid32 | MeanQuality | +0.0000 | +0.0000 | +0.0000 | 0/0 | 0 |
| Vid32 | ShiftedQuality | +0.0718 | +0.0127 | +0.0310 | 0/0 | 5 |
| Vid32 | PriorMatched | +0.0138 | +0.0155 | +0.0155 | 0/0 | 1 |
| Vid32 | RandomMatched | -1.3117 | -1.1070 | -1.1070 | 0/1 | 1 |
| Vid32 | OldJoint10 | +0.0732 | +0.0028 | +0.0028 | 0/0 | 1 |

All per-query signs, boundary errors, severe losses, observed/unobserved spatial scores, confidence intervals and Source-relative outcomes remain in report.json and signed_queries.tsv. Candidate oracle is diagnostic, never an inference input.

## Interpretation

The NumPy core reads only a temporal interval posterior, candidate complete-box tracks, and available spatial box hypotheses. No actionness, hidden, attention, backbone auxiliary head, learned parameters or labels enter the algorithm. Cached-framework-specific adapters only recover those neutral inputs and execute the unchanged BoxFixed DP. The old hidden-based Joint10 result is an explicitly historical comparison, not a method input.

For each candidate interval, obtain its actual spatial-DP trajectory. Estimate per-node spatial quality by mean IoU with the available ERef/forward/backward complete-box hypotheses. Integrate that quality over the intersection with each native posterior interval, normalize by temporal union, and average over the posterior. This is a **surrogate**, not a calibrated expected true vIoU. Correlated hypotheses can agree on a wrong target; their agreement does not encode event semantics.

Controls: quality1 restores original temporal MBR utility; FixedTrackTube fixes the baseline spatial track; MeanQuality removes temporal alignment; ShiftedQuality shifts the quality sequence by half a video; PriorMatched and RandomMatched intervene on exactly the same queries. No strength, temperature, threshold or weight sweep.

The constant-quality positive control max error was 1.78e-15. All14 baseline-wrong candidates capable of crossing tIoU .5 remained unrecovered. Correct candidates in those14 cases still scored below the incumbent, so the spatial consensus proxy did not supply the required event correction. ShiftedQuality is better in HC point estimates; this defeats a placement-specific contribution claim. Vid's one-query small change also occurs under PriorMatched, so it does not establish spatial value.

The fixed gates failed. Do not deploy, scan weights, or rewrite the active full queue. Generality of the interface is achieved; reliable incremental event selection is not.

Cached CPU work excluding IO: HC64 17.383 ms/query; HC32 16.964; Vid32 34.346. This includes all candidate DP and all score controls; it is not end-to-end deployment latency.

## Evidence and scope

Official/dense comparisons: 2304; max absolute discrepancy 1.44e-15. Training/GPU/new experts0. Original inputs, predictions, receipts and seals are preserved privately; no raw boxes/features/weights/query or GT coordinates are published.

The running full evaluation is a separate immutable protocol. All434 frozen sources and its plan remain unchanged. Current full-domain results remain local until both domains complete.
