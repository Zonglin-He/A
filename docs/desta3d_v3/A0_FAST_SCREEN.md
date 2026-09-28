# A0 execution and boundaries

The user-authorized run is Train128 / exposed Dev64, not full618. See the [locked protocol](../../protocols/desta3d_v3_a0_fast_screen_v1.md).

1. `desta3d_v3_a0_fast_screen.py prepare` locks hash-selected manifests and CPU contracts before GPU.
2. `run_desta3d_v3_a0_stage.py cache --name a0_cache001` performs Train native-gradient caching and Dev frozen-feature extraction. Dev oracle directions and native state reuse the sealed447 Gap audit.
3. `supervise_desta3d_v3_a0_screen_v2.py` attaches to that existing cache wrapper, waits for its actual exit, audits all caches, runs exactly200 cached training steps, and independently audits terminal cosines.
4. Only a passing Dev cosine gate calls the separate `desta3d_v3_a0_native.py prepare/run` and seal-first `score_desta3d_v3_a0_native.py`. The generic fast-screen script intentionally refuses unregistered native execution.
5. `summarize_desta3d_v3_a0_screen.py` retains every executed branch, missing direction, failure limitation and negative result. No controller command expands to full data or reads fresh confirmation.

The initial orchestration version failed before waiting because Python3.8 lacks `os.pidfd_open`. The cache GPU process was uninterrupted. Isolated v2 obtains the same kernel pidfd through Linux x86_64 syscall434 and `select`; this is a process-wait compatibility repair only. The original failed code is retained. No GPU replay, parameter or scientific-support changes resulted.

Current stage: cache running; CPU contracts and first real train-cache NumPy projection/state check passed. No final train/dev cosine or A0 native result is available at this registration. The full post-cache chain is implemented and locked; any failure stops for root review, with no retry or sample replacement.
