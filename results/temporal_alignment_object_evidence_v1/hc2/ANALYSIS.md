# Temporal alignment of object evidence: hc2

128 queries / 128 distinct parents; complete prediction seal before independent official scoring (256 checks, max error 8.8817841970012523e-16), independent exact cold-array/input/parent/SHA and mean audit. Root analysis only reads audited scalar metrics and saved prediction arrays; no new GT scoring or model/acquisition.

|Method|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|CurrentAligned|54.968548|31.791298|50.781250|22.656250|
|FixedTemporalPermutation|50.652302|29.643808|44.531250|23.437500|

|Subset|Queries|Aligned minus permuted t pp [95% CI]|Aligned minus permuted v pp [95% CI]|
|---|---:|---:|---:|
|whole|128|+4.316246 [2.013873, 6.789220]|+2.147490 [0.677082, 3.719394]|
|short_le5s|42|+1.000981 [-2.650855, 4.697703]|+0.691035 [-1.625807, 3.222180]|
|long_gt5s|86|+5.935329 [2.964547, 9.041219]|+2.858783 [0.954406, 4.859802]|

Positive deltas favor the original temporal correspondence. All full E-ROI vectors are reassigned to fixed physical temporal nodes by a single predeclared query-specific SHA256 permutation; the vector multiset is exactly preserved, while temporally weighted prototypes can change by design. Source q0/boxes, eight anchor positions, spatial geometry/candidates/unary/link features and all parameters remain fixed. Each arm recomputes prototype/evidence/posterior/legal joint MAP interval and final trajectory DP in its own interval. CurrentAligned reproduces the audited current method bitwise. This intervention tests temporal correspondence of local object evidence conditional on this pipeline; it does not isolate action semantics, prove self-reinforcement causality or compare learned action recognition.

Short means annotated event duration <=5s; long >5s. Bootstrap uses10000 paired parent resamples, seed20260920, pointwise95% intervals, not familywise. The panel was previously exposed; this is not unknown independent confirmation. No seed, parameters or sample counts were chosen from these scores. Null/negative findings and all signed metrics are retained. Fixed-support spatial metrics, boundary errors, recall, severe >5pp t/v union-once and rescues are fully reported in JSON/TSVs; a positive mean does not imply every-query improvement.

## Permutation diagnostics

{
  "total_nodes": 8320,
  "fixed_points": 106,
  "nonintervened_queries": 0,
  "query_mean_index_displacement": 21.834134615384613,
  "query_mean_seconds_displacement": 6.820784624380204,
  "maximum_seconds_displacement": 19.92,
  "changed_intervals": 111,
  "changed_final_box_arrays": 92,
  "changed_posteriors": 128,
  "all_vector_multisets_inverse_exact": true,
  "all_spatial_inputs_unchanged": true,
  "aligned_arrays_equal_current": true
}

## Tails: permutation relative to original alignment

{
  "whole": {
    "severe_t": 48,
    "severe_v": 36,
    "severe_union_once": 49,
    "rescue_union_once": 26,
    "v_improve": 40,
    "v_harm": 60,
    "v_tie": 28
  },
  "short_le5s": {
    "severe_t": 12,
    "severe_v": 10,
    "severe_union_once": 12,
    "rescue_union_once": 12,
    "v_improve": 13,
    "v_harm": 16,
    "v_tie": 13
  },
  "long_gt5s": {
    "severe_t": 36,
    "severe_v": 26,
    "severe_union_once": 37,
    "rescue_union_once": 14,
    "v_improve": 27,
    "v_harm": 44,
    "v_tie": 15
  }
}

## Actual costs

{
  "new_Source": 0,
  "new_expert": 0,
  "new_model": 0,
  "new_optimizer": 0,
  "new_acquisition": 0,
  "inherited_Source_expert_acquisition_nonzero": true,
  "prediction_worker": {
    "pid": 3662097,
    "startticks": 265185287
  },
  "prediction_cached_elapsed_wall_seconds": 6.429496744181961,
  "conditional_query_elapsed_sum": 6.140178780071437,
  "score": {
    "status": "scored_whole128_after_prediction_seal",
    "queries": 128,
    "parents": 128,
    "official_checks": 256,
    "max_error": 8.881784197001252e-16,
    "worker": {
      "pid": 3662789,
      "startticks": 265186126
    },
    "elapsed_wall_seconds": 2.117183072026819,
    "seal_sha256": "30569e0c9a72677de9531cbdfbe2762b2dbd07d1c73dcdfb025ef9418946eee3",
    "metrics_sha256": "a263ab980c23bc0086d32520b3c71356f10ac348e8ec2c16a4265e23c3187bc2"
  },
  "audit_worker": {
    "pid": 3663187,
    "startticks": 265186602
  },
  "audit_elapsed_wall_seconds": 6.334870230872184,
  "actual_stage_exits": {
    "predict": {
      "worker": {
        "pid": 3662097,
        "start_ticks": 265185287
      },
      "exit_code": 0,
      "actual_subprocess_wait": true,
      "elapsed_wall_seconds": 8.39132705423981,
      "log": "outputs/temporal_alignment_object_evidence_v1/hc2/predict_execution_v1.log"
    },
    "score": {
      "worker": {
        "pid": 3662789,
        "start_ticks": 265186126
      },
      "exit_code": 0,
      "actual_subprocess_wait": true,
      "elapsed_wall_seconds": 4.761048079002649,
      "log": "outputs/temporal_alignment_object_evidence_v1/hc2/score_execution_v1.log"
    },
    "audit": {
      "worker": {
        "pid": 3663187,
        "start_ticks": 265186602
      },
      "exit_code": 0,
      "actual_subprocess_wait": true,
      "elapsed_wall_seconds": 8.209138897247612,
      "log": "outputs/temporal_alignment_object_evidence_v1/hc2/audit_execution_v1.log"
    }
  },
  "scope": "Cached replay/input checks and official CPU evaluation; not process CPU time or deployment latency. Stage clocks include nested work and must not be added to inherited acquisition as a speed benchmark."
}
