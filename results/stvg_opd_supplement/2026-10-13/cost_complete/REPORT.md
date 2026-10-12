# Original audited Full cost: all32 actual paths

Original locked historical16 queries per target, cohort seed2026101104. All32 measured Full trajectories and all fresh detector/input/native/source outputs exactly match their saved original scientific fits. Original parameters, frames, optimizer/readout and five-stage timer are unchanged. The original cancelled-baseline barrier dependency was replaced by genuine C/N FINAL; no baseline barrier was fabricated and no paused baseline was resumed.

| Target | Paths | Total median (s) | Total p95 (s) | Actual DINO calls | Zero DINO | Zero admitted | Zero backward |
|---|---:|---:|---:|---:|---:|---:|---:|
| hc2 | 16 | 5.1417 | 6.8284 | 64 | 0 | 0 | 2 |
| vidstg | 16 | 3.6825 | 4.2603 | 48 | 4 | 4 | 4 |

| Target | Stage | Median (s) | p95 (s) |
|---|---|---:|---:|
| hc2 | decode_preprocess_seconds | 0.52082 | 1.93571 |
| hc2 | native_student_prefix_seconds | 2.58599 | 2.76522 |
| hc2 | fresh_expert_seconds | 0.46980 | 0.57184 |
| hc2 | adaptation_seconds | 1.51158 | 1.56024 |
| hc2 | final_readout_seconds | 0.01144 | 0.01189 |
| vidstg | decode_preprocess_seconds | 0.33647 | 0.65574 |
| vidstg | native_student_prefix_seconds | 2.52822 | 2.97727 |
| vidstg | fresh_expert_seconds | 0.46932 | 0.54199 |
| vidstg | adaptation_seconds | 0.37735 | 0.43125 |
| vidstg | final_readout_seconds | 0.01160 | 0.01341 |

The five stages are summed separately for every path. The median or p95 of a sum is not the sum of stage medians or p95s. All32 raw anonymous timings, actual detector/admission/backward counts, startup and this-process peak allocated/reserved memory are exported; no slow or no-information path is dropped.

Weights are resident, the application frame cache is empty for every query, OS pages are not forcibly evicted, and query parsing is saved. All replay calls are CUDA synchronized; the qualified engineering recorder is included. The exact O012 frozen capture, inactive expert storage/restoration and S-identity check are included in the native-prefix timer. CPU hash/math/serialization follows the timed interval. The cancelled S optimizer/model CUDA context remains resident in its original stopped processes. No measured overhead is subtracted to claim production latency.

These historical audited implementation measurements do not estimate fresh-population production throughput or whole-GPU memory use. They use a different fixed cohort and instrumentation from C's128 Full/Frozen paired deployment paths; their absolute seconds cannot be pooled as one speed benchmark. This study reads no GT and changes no method, gate, teacher or manuscript. Actual PDF pixel view, all anonymous public scalars, remote bytes/SHA/Git blob/mode/tree and archive precede genuine FINAL.
