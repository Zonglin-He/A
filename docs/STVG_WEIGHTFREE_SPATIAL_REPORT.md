# Weight-free Spatial proposals: bounded development experiment

Date: 2026-10-15. **Measured and complete for eight historical development cases; not adopted as a replacement for the frozen external-expert method.**

## Question and actual configuration

Can classical whole-image region generation replace YOLO/SAM2 while preserving full Student tube quality? Eight previously exposed sources (HC01–HC04, V01–V04), one query each, four QA-centered five-frame windows each. Student remains the original cross-domain TA-STVG source checkpoint, episodic Query/LN reset, 1792 spatial parameters, original five-step Gaussian IoU-OPD and training-reference acceptance. Temporal remains the saved four-candidate Caption–Query/512-parameter Adam result. Source/Temporal/original external-expert controls are exact saved results, not newly inferred controls.

Selective Search Fast proposes whole-image rectangles at maximum edge 320; first 256, appearance/size diversification, up to 24 GrabCut refinements, NMS 0.7, at most eight new regions. DIS adjacent forward/backward consistency propagates only the local five-frame mask, with two-pixel cycle filtering. Source is retained. Fixed budgets expose four or eight new candidates to frozen STVG-R1-7B. Candidate IDs are local to each window; there is no established long-range object identity. The 448-pixel crop-contact-video prompt is a changed R1 input protocol, not equivalent to native whole-video marked-instance input. OpenCV-contrib 4.11.0.86 is isolated in this artifact directory; no detector/segmenter weights were downloaded. R1 and TA-STVG remain learned models: only the proposal generator is weight-free.

No GT was used to generate regions, select references, update Student, or accept updates. Each phase sealed all predictions before official scoring. The second phase was designed after inspecting first-phase development results; it is exploratory and cannot count as independent confirmation. No GT Oracle fits, full Table1 execution, CURRENT promotion, or restart of stopped queues occurred.

## Full-tube results

Official query-macro vIoU in percent, four queries per domain. With one query per source here, these means also equal source-macro; eight exposed cases do not support a population claim. Every positive, negative, neutral and undefined result is retained in the accompanying JSON.

| Method | HC-STVG-v2 | VidSTG |
|---|---:|---:|
| Frozen Source | 22.210 | 14.490 |
| Temporal QA | 31.286 | 22.442 |
| Temporal + original YOLO/SAM2/R1 OPD | 33.601 | 31.899 |
| Temporal + weight-free 4 | 29.854 | 25.347 |
| Temporal + weight-free 8 | 33.660 | 19.203 |
| Temporal + 8 + three-order majority | 29.775 | 21.253 |
| Temporal + 8 + nested-region rejection | 33.660 | 19.203 |
| Temporal + 8 + majority + rejection | 29.775 | 21.739 |

The four-proposal variant loses 3.747 pp HC and 6.552 pp Vid relative to the original full dual branch. Eight proposals approximately preserve HC (+0.059 pp) but lose 12.696 pp Vid. No one new recipe preserves both domains. Choosing four for Vid and eight for HC after these scores would be dataset-specific development selection and was not adopted.

## Controlled debugging and negative evidence

The second phase fixes all eight-proposal regions and tests two factors: (1) original, rotate-by-three and reversed tile/ID presentations, choosing a canonical proposal only with at least two votes; (2) reject a selected region if at least 90% of its area is inside Source while its area is below two-thirds of Source. The latter is purely predicted geometry and can also reject a useful correction to an oversized Source box.

Three-order voting yields 12 non-Source anchors, 11 Source choices and nine abstentions/disagreements. It improves the eight-proposal Vid mean but damages HC; stable majority is not evidence of semantic correctness. The nesting rule rejects **zero original eight-proposal anchors**, so it does not explain or cure the original negative updates. Combined with majority it rejects one V01 anchor and only partly reduces its harm. Both results, including the zero-effect control, are retained.

Private visual review of the HC01 first window and all eight cases at center-slot 2 shows body parts, texture and background proposals. This is evidence that the generator lacks an object-completeness constraint, not proof that every selected bad reference is a nested body-part box. We did not claim all 32 windows or all video frames were visually reviewed. The existing native teacher-reference agreement diagnostic is an imperfect reference-system comparison, not GT coverage or oracle truth.

Notable first-phase individual changes versus Temporal alone: HC03 -5.728 pp with four proposals versus +8.326 pp with eight; V01 -15.649 pp with eight; V04 +11.620/+11.986 pp with four/eight. Positive recovery and severe negative transfer coexist. More proposals and more voting do not monotonically help.

## Actual work and cost

32 windows yielded 256 new region hypotheses. Initial phase: 64 R1 calls and 12 new complete Student fits/redecodes. Diagnostic phase: 64 additional R1 calls and five new complete fits/redecodes. **Total 128 R1 calls and 17 new fits**, not 34 fits: the diagnostic reader's 22 logical fit records include exact original/same-reference reuse. Copied evidence and repeated controls are not independent inference.

Cached-window candidate generation took 65.162 CPU wall seconds. Initial R1 generation took 28.183 seconds (request total 30.131); the additional 64 diagnostic calls took 28.203 generation seconds. Student worker totals were 73.809 and 58.973 seconds. These exclude the already-existing Temporal results, original video preprocessing, cache preparation, and other separately scoped work. R1 times exclude checkpoint loading/hash verification. The short 96-token output protocol is different from the native teacher protocol. These component timings cannot establish a full-video speedup ratio or a reliable Table1 ETA. Units exited normally; no OOM was observed in these runs.

## Verification and decision

Eight synthetic CPU contracts passed. Source frames/pixels/frozen outputs are matched on recapture; actual updates are followed by complete model redecoding and cached/full decoder equality. Independent CPU readback checks reference binding, float64 IoU, acceptance, saved Adam/parameter rollback and final output reconstruction. It does **not** independently rerun backward gradients. Official versus independent scoring maximum absolute discrepancy is 1.22e-15; all summary fields are independently reaggregated. Existing original-expert baseline has no new-reference supervision partition in these exports; compare overall metrics, not its empty supervised-frame split against new arms.

The tested pure-algorithm replacement is fast on cached windows but has not met the user's low-degradation objective. Preserve the original frozen A method and all positive/negative evidence; do not expand this replacement to Table1. These failures apply to these finite configurations, not all weight-free segmentation. A future speed experiment should isolate the native tracker's repeated suffix propagation/cache scheduling from its object proposal quality, with actual profiling and matched outputs before claiming equivalence. That change has not been implemented or benchmarked here.
