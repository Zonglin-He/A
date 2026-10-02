# Same-state U/R decomposition execution

New isolated namespace: artifacts/tastvg_ur_write_decomposition_v1. Historical
accumulation/transfer/online predictions, GT exposure and production registration
are read-only. Scientific specification: protocols/tastvg_ur_write_decomposition_v1.md.

1. CPU prepare_tastvg_ur_decomposition_v1.py locks 96 donors, common R pre-states,
   cached Uniform/Routed evidence and GT-free self/future targets.
2. test_tastvg_ur_decomposition_v1.py verifies six analytic controls.
3. Exclusive GPU run_tastvg_ur_decomposition_v1.py smoke (via with_local_cuda.sh):
   two clean donors, common probes and original R first-step bitwise positive control.
4. CPU audit_tastvg_ur_write_v1.py writes SMOKE_ROOT_ACCEPTANCE before full replay.
5. Exclusive GPU run_tastvg_ur_decomposition_v1.py full: reuse accepted smoke writes,
   complete 96 writes/1392 target predictions and GLOBAL_PREDICTION_BARRIER.
6. CPU score_tastvg_ur_decomposition_v1.py: independently reconstruct input/target
   bindings, reward/rank/KL/SGD/gradient residual and official dense metrics, then
   anonymous rows/paired donor-source summaries/target clustering/controls/cases.
7. audit_tastvg_ur_decomposition_public_v1.py independently rebuilds anonymous
   scalar statistics and 10000 bootstrap, without private inputs.
8. draw_tastvg_ur_decomposition_v1.py generates PNG/PDF/SVG figures; inspect actual
   rendered files. Report and retain negative/uncertain locality findings.
9. Publish explicit allowlist of code/protocol/report/anonymous results only to
   Zonglin-He/A, compare every fetched remote byte/hash and rerun public audit.
   Update RESEARCH_HISTORY via check/snapshot/check, FINAL_COMPLETION and pause
   the existing ta-stvg-b1-luna monitor through automation_update.

The hourly Luna max performs one bounded read-only check, never reads GT/raw
predictions/weights/parameter tensors or result scores, and cannot alter/restart
the job. Root handles recoverable engineering failures with preserved originals
and revision pin; new scientific changes require a concrete decision. No full
Slow–Fast method, token cosine, latest reset or other historical queue starts here.
