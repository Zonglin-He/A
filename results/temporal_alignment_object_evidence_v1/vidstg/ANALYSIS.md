# Temporal alignment of object evidence: vidstg

128 queries / 128 distinct parents; complete prediction seal before independent official scoring (256 checks, max error 1.2212453270876722e-15), independent exact cold-array/input/parent/SHA and mean audit. Root analysis only reads audited scalar metrics and saved prediction arrays; no new GT scoring or model/acquisition.

|Method|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|CurrentAligned|35.202435|15.339259|19.531250|10.156250|
|FixedTemporalPermutation|33.318565|14.432890|17.968750|7.812500|

|Subset|Queries|Aligned minus permuted t pp [95% CI]|Aligned minus permuted v pp [95% CI]|
|---|---:|---:|---:|
|whole|128|+1.883870 [-0.172891, 4.210838]|+0.906369 [0.020873, 1.897250]|
|short_le5s|78|-0.148344 [-1.525738, 1.281639]|-0.193851 [-0.771733, 0.371329]|
|long_gt5s|50|+5.054124 [0.271509, 10.434531]|+2.622713 [0.573931, 4.869615]|

Positive deltas favor the original temporal correspondence. All full E-ROI vectors are reassigned to fixed physical temporal nodes by a single predeclared query-specific SHA256 permutation; the vector multiset is exactly preserved, while temporally weighted prototypes can change by design. Source q0/boxes, eight anchor positions, spatial geometry/candidates/unary/link features and all parameters remain fixed. Each arm recomputes prototype/evidence/posterior/legal joint MAP interval and final trajectory DP in its own interval. CurrentAligned reproduces the audited current method bitwise. This intervention tests temporal correspondence of local object evidence conditional on this pipeline; it does not isolate action semantics, prove self-reinforcement causality or compare learned action recognition.

Short means annotated event duration <=5s; long >5s. Bootstrap uses10000 paired parent resamples, seed20260920, pointwise95% intervals, not familywise. The panel was previously exposed; this is not unknown independent confirmation. No seed, parameters or sample counts were chosen from these scores. Null/negative findings and all signed metrics are retained. Fixed-support spatial metrics, boundary errors, recall, severe >5pp t/v union-once and rescues are fully reported in JSON/TSVs; a positive mean does not imply every-query improvement.

## Permutation diagnostics

{
  "total_nodes": 13502,
  "fixed_points": 131,
  "nonintervened_queries": 0,
  "query_mean_index_displacement": 35.1671641032805,
  "query_mean_seconds_displacement": 8.798515879819274,
  "maximum_seconds_displacement": 85.41866666666667,
  "changed_intervals": 114,
  "changed_final_box_arrays": 90,
  "changed_posteriors": 128,
  "all_vector_multisets_inverse_exact": true,
  "all_spatial_inputs_unchanged": true,
  "aligned_arrays_equal_current": true
}

## Tails: permutation relative to original alignment

{
  "whole": {
    "severe_t": 28,
    "severe_v": 13,
    "severe_union_once": 28,
    "rescue_union_once": 30,
    "v_improve": 34,
    "v_harm": 47,
    "v_tie": 47
  },
  "short_le5s": {
    "severe_t": 12,
    "severe_v": 2,
    "severe_union_once": 12,
    "rescue_union_once": 16,
    "v_improve": 20,
    "v_harm": 22,
    "v_tie": 36
  },
  "long_gt5s": {
    "severe_t": 16,
    "severe_v": 11,
    "severe_union_once": 16,
    "rescue_union_once": 14,
    "v_improve": 14,
    "v_harm": 25,
    "v_tie": 11
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
    "pid": 3663953,
    "startticks": 265187423
  },
  "prediction_cached_elapsed_wall_seconds": 20.88328970875591,
  "conditional_query_elapsed_sum": 20.237101055681705,
  "score": {
    "status": "scored_whole128_after_prediction_seal",
    "queries": 128,
    "parents": 128,
    "official_checks": 256,
    "max_error": 1.2212453270876722e-15,
    "worker": {
      "pid": 3665882,
      "startticks": 265189707
    },
    "elapsed_wall_seconds": 2.8862265637144446,
    "seal_sha256": "5f68db1140da271fb313732595f012d6cb5821a38fa1487a779b5198f4cc6ca4",
    "metrics_sha256": "8ed002bb6220530fc5f85152961db1815f4d1c6c00dd8d175ff4e674b0ab91ff"
  },
  "audit_worker": {
    "pid": 3666694,
    "startticks": 265190715
  },
  "audit_elapsed_wall_seconds": 20.613354227039963,
  "actual_stage_exits": {
    "predict": {
      "worker": {
        "pid": 3663953,
        "start_ticks": 265187423
      },
      "exit_code": 0,
      "actual_subprocess_wait": true,
      "elapsed_wall_seconds": 22.843574600294232,
      "log": "outputs/temporal_alignment_object_evidence_v1/vidstg/predict_execution_v1.log"
    },
    "score": {
      "worker": {
        "pid": 3665882,
        "start_ticks": 265189707
      },
      "exit_code": 0,
      "actual_subprocess_wait": true,
      "elapsed_wall_seconds": 10.078829466830939,
      "log": "outputs/temporal_alignment_object_evidence_v1/vidstg/score_execution_v1.log"
    },
    "audit": {
      "worker": {
        "pid": 3666694,
        "start_ticks": 265190715
      },
      "exit_code": 0,
      "actual_subprocess_wait": true,
      "elapsed_wall_seconds": 22.537487814202905,
      "log": "outputs/temporal_alignment_object_evidence_v1/vidstg/audit_execution_v1.log"
    }
  },
  "scope": "Cached replay/input checks and official CPU evaluation; not process CPU time or deployment latency. Stage clocks include nested work and must not be added to inherited acquisition as a speed benchmark."
}
