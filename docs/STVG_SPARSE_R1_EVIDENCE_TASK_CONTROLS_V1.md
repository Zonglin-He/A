# Sparse STVG-R1 evidence and task controls

**Scope:** finite P0-1 / P0-2 development diagnostic, completed 2026-10-14. This package preserves anonymous results and response metadata; it does not publish raw model responses, RGB/video, query text, GT coordinates, weights, or private paths.

## Protocol and data boundary

The [protocol record](../results/stvg_sparse_r1_evidence_task_controls/2026-10-14/protocol/PROTOCOL_RECORD.md) captures the actually run temporal/spatial representation controls and query-conditioned task controls. The [adaptation interface record](../results/stvg_sparse_r1_evidence_task_controls/2026-10-14/protocol/ADAPTATION_INTERFACE.md) states the allowed target/update semantics and the observed zero-fit boundary.

Eight historically exposed development queries were used. No fresh confirmation set was used, and teacher-training overlap is unknown. The root receipt identifies a post-seal scoring read of ground truth; this is not a no-GT result.

## Readout

There are 64 candidate metric rows, 56 paired/control rows, 16 selected candidate records, and 232 response metadata rows. The official-versus-independent metric maximum absolute difference is 1.2212453270876722e-15. Output validity was 136/136 for native-style responses, 46/48 for query-free facts, and 48/48 for alignment responses; none were truncated. The two invalid fact outputs remain represented with zero valid fact fields.

The four high-view alignment cells had no decisive stable preference direction. That is a small-sample task-control observation, not a claim that alignment is universally ineffective. For wrong-query controls, all 16 calls retained the base ID preference, while the replacement query text was byte-distinct; shared visual semantics remain possible, so this does not prove query blindness. Eighty-five of 136 native-style answers copied the prompt example’s 0.00-to-1.00 time range.

Actual frozen teacher inference cost was 328.10738354129717 seconds, 786,935 input tokens, 6,601 output tokens, and 18,715,143,168 peak allocated GPU bytes (about 17.43 GiB). This excludes source capture, view decoding/loading, and student adaptation.

## Adaptation and limits

No conditional-fit case was eligible and no student fit ran. Zero fits do not establish student failure. Temporal fitting to Native is a no-op; the interface requires a non-no-op Projection target. The finite comparison does not establish causal resolution effects because evidence and pixel budgets differ, does not measure online carry, and does not settle all MLLM feedback or OPD approaches.

## Files

- `results/stvg_sparse_r1_evidence_task_controls/2026-10-14/ANONYMOUS_RESULTS.json` contains the complete anonymous scalar export, including positive, negative, tie, uncertain, and unknown outcomes.
- `ANONYMOUS_RESPONSE_ROWS.json` contains 232 anonymous metadata rows and response/RGB digests; it contains no raw response text or RGB.
- `EXTRA_FORMAT_AND_QUERY_DIAGNOSTICS.json` and `ROOT_READBACK.json` preserve diagnostic and post-seal audit metadata.
- `source/` contains the seven scope source files; `report_assets/` contains the PDF and three Poppler page renders. The PNGs are PDF rasterizations only; no visual inspection of study RGB or video is claimed.
