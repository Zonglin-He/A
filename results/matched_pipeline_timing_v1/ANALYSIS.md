# Matched complete-pipeline timing and peak memory

Both pipelines include actual Source inference, decode, parse, transfers/offload and their own complete inference to CPU boxes/interval. All 1536 measured calls and all outputs were independently checked exactly; no GT scoring or model rerun was needed for this analysis.

Fixed128 distinct parents/domain, three actual repeats per query/arm. Average the three latency repeats within each query before the 128-parent paired bootstrap (10000, seed20260920). The panel has prior development exposure. Confidence intervals are pointwise, not familywise.

| Domain | Current mean / p50 / p90 seconds | Source-box EdgeTAM mean / p50 / p90 seconds | Edge minus Current seconds (95% CI) | Edge / Current ratio (95% CI) |
|---|---|---|---|---|
| hc2 | 7.269798 / 6.633048 / 8.750036 | 10.260872 / 9.605960 / 11.856661 | 2.991074 [2.945780, 3.041778] | 1.411438 [1.397242, 1.424453] |
| vidstg | 7.380069 / 7.136415 / 9.312730 | 12.719627 / 11.965546 / 20.058002 | 5.339557 [4.719511, 5.971220] | 1.723510 [1.655023, 1.789438] |

Current has lower mean latency for every one of the128 repeat-averaged queries in each domain. This is a comparison of the complete deployment choices under their original component backends, not a claim of isolated propagation superiority, equal compute/knowledge, full-dataset throughput or uniformly matched arithmetic kernels.

| Domain / pipeline | Max PyTorch allocated GiB | Max PyTorch reserved GiB | Max sampled active-pipeline NVML GiB |
|---|---|---|---|
| hc2 / Current | 2.636429 | 4.046875 | 5.224609 |
| hc2 / EdgeTAM | 2.298334 | 2.751953 | 3.341797 |
| vidstg / Current | 5.683005 | 6.775391 | 7.974609 |
| vidstg / EdgeTAM | 5.683005 | 6.775391 | 7.367188 |

The memory tradeoff is not an unconditional Current advantage. Current has a larger active-process NVML peak in both domains; Source dominates the maximum PyTorch allocated and reserved peaks for both Vid pipelines, making those Vid maxima equal. Mean/median/p90 memory over the per-query maximum of three repeats are reported separately.

PyTorch allocated/reserved are tensor/allocator counters. For Current, the exact pipeline peak is max(parent whole-call peak, parent live-at-synchronous-teacher-request plus teacher-child peak). Source is offloaded before teacher work and the parent waits; peaks from non-overlapping phases are not added. NVML samples only active pipeline PIDs every10ms and includes their CUDA contexts; unrelated processes and the inactive teacher PID for Edge are excluded. Sampling can miss sub10ms transients and is not an exact device-wide peak.

RTX5090 / i5-14600KF, shared host, CPU14-15 with original Source/pool Torch4. Source and ReferDINO preserve cuDNN TF32=True/matmul=False; Edge preserves its original cuDNN=False/matmul=False and official BF16-memory/lossless FP32 V-input compatibility. Both arms use the same original Source backend. These are original-component backend results, not a uniform-precision operator experiment.

All models remain CPU-resident between sequential GPU stages; normal transfers and offload are timed. Decode, parsing, Source, the respective complete expert/tracker path and output readout are included. There is no cross-query RGB/feature/teacher/prediction cache reuse. Experimental attempt/reference validation/storage/auditing occurs outside the timer. Startup and four real warmup calls are excluded and exported separately. Official native tracking progress output remains part of its normal implementation.

Stage timings in all_measured_call_scalars_v1.json and root_full_analysis_v1.json are nested and differently scoped: expert request contains transfers/serialization/forward; finish contains pooling/prototype/MAP/finalDP, and the two arms have distinct Source-wrapper boundaries. Do not add these stages or interpret them as isolated causal contributions. Model-load scopes are per actual PID/start; Edge wrapper load includes its nested strict-load duration, not two loads.

Repeat-level means are also retained. HC Current7.754915/7.017038/7.037440s and Edge10.845183/9.965312/9.972120s; Vid Current6.910008/6.813858/8.416342s and Edge12.039691/11.820441/14.298748s. Both Vid arms are slower in repeat3. Alternating arm order and paired query averaging limit fixed-order confounding but do not eliminate shared-host temporal variability. Do not attribute this drift to a specific service or claim an isolated hardware benchmark.

Accuracy is reused only from the old complete128-parent per-domain independent scoring. All1536 actual timed predictions exactly match those references: HC current31.791298 vs Edge26.227517v; Vid15.339259 vs15.257178v. The prior Vid accuracy difference CI crosses zero, so latency advantage does not establish a Vid accuracy advantage or equivalence. No new GT scoring, target tuning or accuracy-based sample choice.

Failure lineage is fully retained: two pre-formal warmup Source/ReferDINO failures, one additional actual original-backend expert diagnostic, and a zero-model namespace-launch failure. The first RAM-auth failure did not retain every Source intermediate; missing clocks are null, not zero. The independent CPU report then completed its checking/statistics loops but failed at hardware-path assembly. A separate report-v5 fixes only that path, retains the original code/logs and all1536 prediction hashes, and successfully rechecks every record without new inference. Successful GPU and teacher actual waits are0; original report wait1; repaired report actual wait0. No old controller was resumed.

This completes the numerical/root analysis of timing, pending verified publication. The separately authorized cached temporal-correspondence study remains gated on timing publication. All older CG/gamma/full-corruption/generic pauses and cancelled CG-HC control remain unchanged.
