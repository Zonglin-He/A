# Execution

Isolated finite run, no deadline extension or old queue resumption. Entry:
`.conda/tubedetr/bin/python -B scripts/run_tastvg_temporal_latent_quality_v1.py prepare`.
Then `capture_source vidstg`, `capture_source hc2`, `fit`, `capture_target vidstg`,
`capture_target hc2`, `seal`, `diagnose`, followed by independent root audit,
public audit, report/plots, archive check/snapshot/check, GitHub publication and
per-file remote readback. GPU work is serial under the established GPU lock.

Preparation pins all inputs and implementation. Failure preserves originals;
engineering-only repairs use a revision pin and retain failure records. A
controller's completion is pending root audit/publication, never publication by
itself. No monitor automation is started for this run. Progress is inspectable
in artifacts/tastvg_temporal_latent_quality_v1/STATUS.json and worker logs.
