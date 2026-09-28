# Current: A0 completed in31.8min; direction gate failed

|Direction cosine|Train128 /95parents|Dev64 /16parents|
|---|---:|---:|
|Mean|0.029865917|0.024222450|
|Median|0.007693602|0.002401304|

192/192 directions defined. Fixed existing StateAwareDirectionMixer, one seed,200 actual AdamW steps/800 query occurrences, only coefficient cosine loss. **Dev median<0.1 and Train median<0.3: capacity/optimization screening branch.** Native inference was correctly skipped; there is no A0 native t/s/v result, full618/447 expansion or fresh388 use.

The cache completed128 new source native captures/253 backwards; Dev64 reused all native/oracle evidence and only extracted missing frozen features. Complete NumPy projection/state/cosine audits passed. All8 tensors changed, clip0, frozen union unchanged, Adam integer/live counters200. Weak fit is not a proven capacity root cause. Source GT privilege and exposed development are explicit.

[Full anonymous results](../../results/desta3d_v3/2026-09-29/A0_FAST_SCREEN.md) and [machine-readable report](../../results/desta3d_v3/2026-09-29/A0_FAST_SCREEN.json). GPU exited and this A0 monitor was deleted. Only proposed next: internal width128→256, preserving feature128/state33/cache/loss/radius/steps. It has not run. Prior negative results and the CPU controller compatibility repair are retained.

## Historical registration and earlier results

# A0 execution and boundaries

The user-authorized run is Train128 / exposed Dev64, not full618. See the [locked protocol](../../protocols/desta3d_v3_a0_fast_screen_v1.md).

1. `desta3d_v3_a0_fast_screen.py prepare` locks hash-selected manifests and CPU contracts before GPU.
2. `run_desta3d_v3_a0_stage.py cache --name a0_cache001` performs Train native-gradient caching and Dev frozen-feature extraction. Dev oracle directions and native state reuse the sealed447 Gap audit.
3. `supervise_desta3d_v3_a0_screen_v2.py` attaches to that existing cache wrapper, waits for its actual exit, audits all caches, runs exactly200 cached training steps, and independently audits terminal cosines.
4. Only a passing Dev cosine gate calls the separate `desta3d_v3_a0_native.py prepare/run` and seal-first `score_desta3d_v3_a0_native.py`. The generic fast-screen script intentionally refuses unregistered native execution.
5. `summarize_desta3d_v3_a0_screen.py` retains every executed branch, missing direction, failure limitation and negative result. No controller command expands to full data or reads fresh confirmation.

The initial orchestration version failed before waiting because Python3.8 lacks `os.pidfd_open`. The cache GPU process was uninterrupted. Isolated v2 obtains the same kernel pidfd through Linux x86_64 syscall434 and `select`; this is a process-wait compatibility repair only. The original failed code is retained. No GPU replay, parameter or scientific-support changes resulted.

Current stage: cache running; CPU contracts and first real train-cache NumPy projection/state check passed. No final train/dev cosine or A0 native result is available at this registration. The full post-cache chain is implemented and locked; any failure stops for root review, with no retry or sample replacement.
