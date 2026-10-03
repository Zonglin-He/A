# Temporal Latent Information Atlas v1 execution

P0 characterization completed entirely on CPU using immutable predecessor caches.
Do not repeat predecessor inference, create candidates, call experts, update A,
or resume historical queues. Query swap/layerwise/app-motion were not started.

Private root: `artifacts/tastvg_temporal_information_atlas_v1`.
Public export: `results/tastvg_temporal_information_atlas/2026-10-03`.
Scientific lock: `protocols/tastvg_temporal_information_atlas_v1.md`.

The sequential stages were `prepare`, `fit`, `readout`, `diagnose` in
`scripts/run_tastvg_information_atlas_v1.py`, using the existing CPU environment
`.conda/tubedetr/bin/python -B`. All 136 probes were source-frozen before target
readout, and all readouts were sealed before the diagnostic label join. Four
cached target-span schema inspections before locking are explicitly disclosed.
Do not call `prepare` against the existing immutable runtime lock.

Root audit: `scripts/audit_tastvg_information_atlas_root_v1.py`; requires local
SHA-bound private predecessor assets. It independently checks physical labels,
features, all selected ridge models/all ridge validation paths, selected
logistic KKT and all frozen predictions. It does not re-run a backbone.

Anonymous/public-only checks:

```bash
python -B -m scripts.test_tastvg_information_atlas_v1
python -B scripts/audit_tastvg_information_atlas_public_v1.py results/tastvg_temporal_information_atlas/2026-10-03
```

Figures: `scripts/draw_tastvg_information_atlas_v1.py <results_dir>`.
Order supplementary statistics: `scripts/supplement_tastvg_information_atlas_v1.py <results_dir>`.
These use saved anonymous statistics only; no new fit, predictions, labels or GPU.

Public audit reconstructs source/condition/order moments and 10000-source-bootstrap
intervals, not raw private labels or model fitting. JSON compaction for export
preserves parsed values exactly. Full uncertainty/undefined statistics and source
paths remain published. The AP rounding tolerance repair changed only an audit
range assertion; raw failure evidence remains in the private recovery directory.

Completion requires both audits, report and visually checked PNG/PDF figures,
archive check/snapshot/check, explicit public-export allowlist, non-force GitHub
push and every-file remote byte/SHA verification. The main RESEARCH_HISTORY is
local continuity evidence, not exported wholesale. No production promotion.
