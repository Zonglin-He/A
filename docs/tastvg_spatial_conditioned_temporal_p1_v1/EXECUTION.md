# Finite P1 execution

Use the isolated `scripts/run_tastvg_spatial_conditioned_temporal_p1_v1.py`.
The scientific policy is `protocols/tastvg_spatial_conditioned_temporal_p1_v1.md`.
Model runtime is `.venv-exost/bin/python`; CPU geometry/audits use
`.conda/tubedetr/bin/python`. No installs/downloads.

1. Run CPU tests, prepare immutable runtime/config/input locks.
2. Run `smoke`; root checks SMOKE and writes SMOKE_ROOT_ACCEPTANCE once.
3. Run `Soft` once under the GPU lease. Observe bounded progress; don't spawn
   another worker. Completion is GLOBAL_PREDICTION_BARRIER, not a plan or code.
4. Only then run CPU score; independently root/public audit, plot/report.
5. Explicit public allowlist, remote byte/hash validation, FINAL_COMPLETION and
   local archive check/snapshot/check. No method promotion or further GPU job.

If an engineering failure occurs, retain status, log, allocations and all
partial receipts in recovery. Fix only engineering, pin revision, and resume
the same inputs. Never change alpha, sources or scoring based on results.
All old P0, full-query and paused queues remain stopped.
