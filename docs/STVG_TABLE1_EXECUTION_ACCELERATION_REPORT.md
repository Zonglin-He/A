# Table 1: frozen method and bounded execution acceleration

Date: 2026-10-15. Status: engineering preflight running; full 13,785-query evaluation is not complete and has not been scored.

The final controlled comparison selected one global recipe before reading the new development labels: four-candidate Caption–Query Alignment with a 512-parameter Adam update, followed by WHO/WHEN-decoupled native YOLOv12-X/SAM2.1/R1 instance evidence and 1,792-parameter Gaussian IoU OPD with training-reference validation (A). Cross-frame validation B did not pass the prelocked label-free condition in both directions. Its later positive development vIoU differences remain reported in the cross-frame report; they were not used to overturn the selection.

A single two-candidate Temporal speed ablation reduced full-method development vIoU by 6.357462 pp on HC and 5.611563 pp on VidSTG. It was rejected. No additional model, candidate mechanism, prompt sweep or learning-rate search is being introduced.

## R1 execution measurement

The original cold R1 generation configuration disabled KV caching. The first hot, full-GPU execution test crossed the task's 4 GiB free-GPU floor and was stopped by its own resource guard before a complete generation; this was not an observed OOM. That result and the original source remain preserved.

An explicit `use_cache=true` variant kept the original BF16, SDPA, unquantized weights, input prompts, video inputs, 2,048-token generation cap and sampling configuration. The eight historical sources were each run twice, with one model load:

| Measurement | Actual value |
|---|---:|
| Complete generations | 16 |
| Model loads | 1 |
| Model load | 4.801 s |
| Mean generation | 2.306 s |
| Generation range | 2.094–2.726 s |
| Total worker wall time | 58.413 s |
| Peak allocated GPU memory | 17,445,963,776 bytes |
| Exact old input bindings | 16/16 |
| Exact repeated text within this run | 8/8 pairs |
| Old selected Object ID unchanged | 14/16 calls |

The text and selected IDs are **not generally bitwise equivalent** to the old execution. Both V03 repetitions changed the selected ID from 19 to 10. Independent downstream reconstruction found identical effective supervision in this bounded set: 22 sparse anchor geometries and actual A fit weights, 708 TeacherGeometry positions, seven eligible cases and one exact no-update case. V03's new ID has masks elsewhere but none in its fixed QA event; its eligibility and final geometry fallback therefore remain unchanged. This is bounded supervision equivalence, not proof of identical semantics or performance on unseen queries. No new Student fit was used to claim this execution qualification.

The new explicit execution variant is locked globally for engineering preflight. R1 is kept hot across queries; its output cap is not shortened. Timing above excludes full tracking, Temporal QA and Student processing and must not be extrapolated as whole-pipeline throughput.

## Independent-source preflight

The fixed roster contains one representative query from each of 32 noncurrent-development parent sources per direction, 64 queries total. All 64 new DINO-free Source captures are sealed, with 128 offset forwards and two model loads. Root independently matched all Frozen boxes, source grids and intervals to existing sealed Source outputs. No new GT has been read.

The first new preflight case completed the full native tracker, nine real teacher calls (four captions, four query alignments, one independent WHO selection), four supported anchors, five accepted parameter updates and one complete Student re-decode. Teacher calls took 9.250 s for this case, plus a separate 7.503 s model load. Root checked custody, temporal replay, arm composition, independently recomputed reference IoUs/acceptance and exact full-versus-cached decoder output. This is one actual engineering qualification, not the complete 64-case preflight and not a performance result.

Two import/metadata dispatch errors occurred before Student model execution: a missing project import path and a legacy `parent` field absent from the opaque roster. Original logs/sources were preserved; separate dispatch revisions use the existing project path and the exact parent from the pinned official PLAN. No scientific parameter or output was changed. A prior tracking import-path error was similarly saved and corrected before actual tracking. Four predeclared center slots must not be confused with four visible anchors: the frozen A rule allows one or more supported anchors; the incorrect prelaunch adapter check was corrected before GPU use.

## Full evaluation and storage

The official population remains HC-STVG-v2 validation (3,482 queries, 237 parents) and VidSTG test (10,303 queries, 732 videos). All 13,785 domain-namespaced query-key hashes match their pinned official PLAN metadata. Source, Temporal-only, Spatial-only, Temporal+Spatial and Temporal+TeacherGeometry outputs will be sealed before new evaluation. Spatial-only retains QA-conditioned evidence selection while using Native output time; it does not remove QA computation.

Existing baseline custody has been checked at actual file level: 120,726 physical prediction files, 13,751,847,014 bytes, all matching sealed SHA-256. The complete matching Source/DINO/TENT/SAR rows can be reused with their original settings clearly labeled; missing old Spatial payloads are not replaced by development scores. EATA is not presented as complete in both directions.

Cache reuse is keyed by actual video bytes and frame clock/bounds, not merely parent ID. HC clips sharing a parent may still require separate tracking. Full processing is being organized into bounded video batches to respect the 30 GiB disk floor. Only newly generated, independently verified reproducible intermediates may be retired; historical measurements, negative results and source assets are preserved. Own task cgroups and RAM/GPU guards remain active. No stopped historical queue or CURRENT method is restored/promoted.

The teacher's possible target-data training overlap is still unresolved. The method is frozen external-expert-guided TTA, not a claim of a teacher without target-domain prior knowledge. Final official scores, 10,000-pair parent bootstrap, negative tails and cost estimates remain pending actual full execution.
