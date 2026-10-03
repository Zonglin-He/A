# Bounded cached temporal evidence audit

Use the project's existing CPU environment. Run sequentially:

```
.conda/tubedetr/bin/python -B scripts/test_tastvg_large_evidence_v1.py
.conda/tubedetr/bin/python -B scripts/run_tastvg_large_evidence_v1.py prepare
.conda/tubedetr/bin/python -B scripts/run_tastvg_large_evidence_v1.py signals
.conda/tubedetr/bin/python -B scripts/run_tastvg_large_evidence_v1.py diagnose
.conda/tubedetr/bin/python -B scripts/audit_tastvg_large_evidence_public_v1.py results/tastvg_large_correction_evidence/2026-10-03
.conda/tubedetr/bin/python -B scripts/report_tastvg_large_evidence_v1.py
```

`signals` refuses GT/results reads and GPU use, verifies original receipt hashes,
and writes a global signal/decision seal. `diagnose` must verify that seal before
joining the previously published anonymous candidate metrics. No media, weights,
model forward, decoder, new expert or online stream execution is required.
Preserve failed attempts and revision pins for any engineering repairs. Do not
change scientific constants after seeing labels. A-current native logits are
unavailable; source-checkpoint logits are explicitly a limited proxy.

After valid measurement, record a decision and limits, review the figures, update
RESEARCH_HISTORY and run check/snapshot/check. Publish only the explicit export
manifest through the existing GitHub public workflow, then fetch/read back every
file and run the standalone public auditor. Save FINAL_COMPLETION and perform
the final archive check/snapshot/check. Do not restore paused jobs or monitors.
