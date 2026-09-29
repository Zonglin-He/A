# B1 temporal environment recovery — 2026-09-30

B1 remains running; no full online predictions or benchmark metrics are complete.

The spatial stage completed all 86,352 receipts (16,326 unique specialist inferences). The subsequent temporal launch failed before producing a receipt because the generic TA-STVG Python environment did not contain `mamba_ssm`. The existing `.venv-exost` environment contains the UniversalVTG dependencies and passes the actual model-loading and one-query inference check. No dependency installation, specialist code modification, checkpoint change, or method adjustment was needed.

The pipeline now launches only the temporal specialist with `.venv-exost/bin/python`; other stages retain `.conda/tubedetr/bin/python`. The initial failed status and logs were preserved. The single successful bounded inference and failed attempt remain charged to the original runtime budget. Spatial barrier SHA and all expert-worker/scientific pins remain unchanged. Original locks remain immutable; `POSTPROCESS_REVISION_001.json` records the supervisor-only override.

The full temporal stage resumed and produced additional receipts. The paper queue resumed waiting for B1. Its scheduling correction (`READY_REVISION_001.json`) now requires the root to verify and publish B1 before A2 can start: `artifacts/tastvg_full_b1_v1/GITHUB_PUBLICATION.json` must have `status=verified`, a verified remote `commit`, and the matching `completion_sha256`. The root must create this receipt only after actual B1 closure. A2, baseline GPU smoke and paper readouts have not started.

Frozen method, cohort (9,411 queries / 670 sources), three orders, 16 conditions, 451,728 target online arrivals, no-label inference and global prediction-before-scoring boundary remain unchanged. This is an engineering recovery, not a new efficacy result. Completed historical negative findings and all original failed attempts remain preserved locally.
