# Finite matched proximal ablation and native-head follow-through

This is the user's 2026-10-02 newly authorized task. Never resume the paused
full-query, B1, old paper matrix, PTD or TubeDETR controllers.

Private workspace: `/home/wwww/visual grounding`.
Artifacts: `artifacts/tastvg_proximal_ablation_v1`.
Input pool remains read-only: `artifacts/tastvg_extended_sensitivity_v3`.

Two finite processes were launched: `continue_tastvg_proximal_v1.py` generates
A on both datasets, no-GT reward calibration, then B/C/D; the separate
`continue_tastvg_proximal_finish_v1.py` waits for its global prediction barrier,
scores all eight jobs on CPU, runs the prescribed fixed-mixture control only
where B has positive future mean, seals spatial selection, checks the native
head interface, generates T on both datasets, seals T predictions and scores
and audits the head. Do not create a duplicate process. Inspect current
`STATUS.json`, `LAUNCH.json`, `FOLLOWTHROUGH_STATUS.json` and actual commands.

All spatial batch configuration and input plan files are pinned in
`RUNTIME_LOCK.json`; additional follow-through code in
`FOLLOWTHROUGH_RUNTIME_LOCK.json`; native head code in
`TEMPORAL_RUNTIME_LOCK.json`. Never alter a running pinned implementation.
For a recoverable engineering error preserve failed logs/status/code, document
a revision and its hashes, and resume only this queue from matching receipts.
Keep scientific configuration, cohort and failure records intact.

Baseline A must match all 384 pre/post states, predictions and gradients of the
selected v3 development stream. It logs missing raw rewards. First-step spread
calibration uses no GT and seals in `CALIBRATION_BARRIER.json`.
Four-arm global seal is 3072 arrivals. T has a separate 768-arrival seal; any
conditional fixed-control receipts are also retained. Existing H and expert
inference are reused. All current outputs are generated before the arrival's
updates, and incoming states reset per condition/order.

Completed score/audit logs are not the end of the task. Root independently
reviews state/projection/target checks, per-arm coverage, probability semantics,
selection rules, all negative results and paired source bootstrap. Public
results must use anonymous source/parent ordinals, excluding captions/media,
annotations, weights and raw H/state/gradient tensors. Publish code/protocol,
all scalar outcomes and conditional branches, report and meaningful costs to
`Zonglin-He/A`, verify every remote file hash/byte count, update
`docs/RESEARCH_HISTORY.md`, and run archive check/snapshot/check. Only after
`FINAL_COMPLETION.json` and verified publication should hourly Luna monitoring
be paused. Model promotion is not authorized by this development run.

No GT threshold, reference resampling, event-support module, HC anchor, external
baseline, new teacher or additional hyperparameter sweep is included.
