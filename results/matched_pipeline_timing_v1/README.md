# Verified complete-pipeline timing and memory

1536 actual matched Source calls;128 distinct parents per domain, three repeats, Current and Source-box-prompt EdgeTAM. Both include Source/decode/parse/transfers and full own-model readout. No cross-query prediction/feature/RGB reuse. Read [root analysis](ANALYSIS.md), [all-call scalars](all_measured_call_scalars_v1.json), [paired report](paired_timing_memory.json), [startup/warmups](startup_warmup_v1.json) and [all failure/actual costs](actual_exit_cost_lineage_v5.json).

Active science2 preserves original component backends; earlier protocol/runtime files are historical and explicitly superseded by active_protocol_v2.json. Original GPU/teacher actual waits0, original CPU report wait1 due hardware path at final assembly, separate report-v5 only fixes the path and succeeds with full revalidation/wait0. Original files and all paid outputs retained; no model rerun.

Current is faster on this panel but has higher sampled active-process NVML memory. Vid accuracy advantage over Edge is not established; old score references are reused only after every actual timed prediction matches exactly. This is a shared-host complete-pipeline comparison, not isolated propagation, equal knowledge/compute, uniform precision kernels or full-target throughput. The panel was exposed in earlier development.

The separately authorized cached temporal-alignment comparison remains queued until this timing publication is fully verified. Paused older scopes stay paused.
