# Three-strategy spatial comparison — completed measurements

The latest direct user narrows all new work to this one fixed-panel comparison and explicitly cancels manuscript work. All previous accuracy predictions and official scoring are reused. This adds only actual CPU spatial-function timing for PureExpert, Source-feature matching propagation and unchanged Current.

- Same existing 128 queries and 128 distinct parents per dataset; same eight expert observations/physical positions, Source feature grid and Current event interval.
- 2,304 actual formal calls (256 queries × 3 strategies × 3 repeats), all reference output arrays bitwise exact; six separate warmups excluded.
- CPU input-ready spatial processing on Intel i5-14600KF, one Torch/BLAS/OpenCV thread, CPU14–15, nice19/idleIO. No GPU, new Source, expert, optimizer, GT scoring, or accuracy prediction generation.
- Source/expert/decoding/feature extraction/temporal posterior and experiment IO are outside this spatial timing boundary. Current really rebuilds E initialization, candidates, scores, links and final interval DP. No precomputed geometry/path or historical-clock subtraction.
- The Source-feature VideoWalk module adapter is functional and source-validated, with source-selected temperature.05/topK10/context20. It remains weaker than pure interpolation on both source validation and target panels; this is not a reproduction of its learned encoder or proof against a strong propagation baseline.
- Current is reused as implemented: expert boxes are soft candidates at observed frames; Pure/Matching hard-retain the original eight boxes. This preserves, rather than fixes, the previously diagnosed observation-contract mismatch. No clamp, main-method or paper edit was made.

[Accuracy–cost analysis](ANALYSIS.md), [root interpretation](ROOT_FINDINGS.md), [per-call measurements](all_measured_calls.tsv), [query means](per_query_three_repeat_means.tsv), [independent audit](audit_receipt.json), [root scalar/worker/provenance check](root_validation_v1.json).

Original source validation and signed accuracy/recall/severe-tail/short–long/distance reports are reused from [the completed matching pilot](../spatial_temporal_six_v1/matching_pilot/hc2/ANALYSIS.md) and [VidSTG pilot](../spatial_temporal_six_v1/matching_pilot/vidstg/ANALYSIS.md). All older full pure-interpolation and temporal results remain intact; these 128-query values do not replace full-dataset values.

All other new tasks are paused: full-target matching acquisition, five-strategy/end-to-end timing, density sensitivity, optional perturbations, CG/fullgamma/fullcorruption/generic tasks. No old dispatcher may be resumed. The earlier six-item scope is superseded, not falsely marked experimentally completed.

An initial generic Python module-name collision stopped before controller registration, child/preflight/model/Source/expert/GT or measurement; original source/plan/log/diagnosis are preserved. Independent execution v2 only changes import wiring, with exact reverse-edit checks and unchanged frozen numerical sources/plan.
