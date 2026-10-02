# Finite two-round execution and root handoffs

Namespace: artifacts/tastvg_current_correction_views_v1. Read the locked
protocol and actual STATUS; prepared, sealed, scored and published differ.
Historical fullquery and other paused queues remain paused.

Controller .conda/tubedetr/bin/python -B
scripts/continue_tastvg_correction_views_v1.py STAGE SPLIT, under
bash scripts/with_local_cuda.sh, owns PROCESS.lock. Stage is round1/round2;
split search/confirm. The controller chooses original .venv-exost/bin/python
for UniversalVTG and .conda/tubedetr/bin/python plus .runtime/sa2va_deps
PYTHONPATH for Sa2VA. All child GPU processes run serially. Do not duplicate
a live controller or start historical workers.

1. round1 search: input-hash matched Routed observations, Vid/HC same-A-state
   current outputs. Return sealed_pending_root_audit with 768 cells and both
   model barriers. CPU score_tastvg_correction_views_v1.py round1 search
   checks masks, gradient targets, SGD, persistent A hashes and official dense
   metrics; then seals ROUND1_SELECTION.json. Never use confirmation to select.
2. round2 search: fresh real shifted temporal image features, cache old phase0
   exactly; min-view output and chosen-interval five-frame acquisition; separate
   Uniform5 midpoint current-readout control. Same-state suffix correction,
   both datasets seal, CPU root scoring seals FINAL_SELECTION.json.
3. round1 confirm then round2 confirm, on the frozen v3 16-source cohorts.
   No confirmation GT is opened until BOTH stages and BOTH datasets have
   sealed. Score both stages only afterwards; no reselection or implicit
   per-dataset mechanism. Confirmed effects are descriptive evidence.
4. report_tastvg_correction_views_v1.py draws development/confirmation paired
   source CI figures and saves the complete review. Public auditor
   audit_tastvg_correction_views_public_v1.py RESULTS_DIRECTORY independently
   checks anonymous rows, bootstrap, tails, critic ties and decisions.
5. Root examines actual positive/negative cases, controls and resource receipts,
   including source influence. Publish explicit code/protocol/results allowlist
   to Zonglin-He/A, verify remote bytes/tree/commit and run remote anonymous
   auditor. Update RESEARCH_HISTORY with check/snapshot/check, write remote
   receipt and FINAL_COMPLETION, then pause the sole existing monitor.

Audit invariants: CURRENT_METHOD unchanged; source checkpoint restored; current
readout physical intervals correspond to fixed output indices; same nine probes
come from A pre-state; no temporary offset leaks into post-state. Nonempty flat
reward retains original RKL; Specific requires both evidences; top ties keep
center. Empty Routed never produces an anti-Uniform correction. Temporal native
is fixed across both views and chosen only on a unique strictly positive
min-view difference; ties/native use original native interval.

Live A prediction/gradient/state controls are bitwise. UniversalVTG cached
phase0 outputs remain exact immutable inputs; its fresh FP32 head control is
bounded (<.1 physical frame, <1e-4 score/confidence, same student selected
candidate), with errors disclosed, since cached-feature live replay exhibited
sub-frame numerical variance. Sa2VA original Uniform5 mask/box/text live
control is bitwise before first new spatial inference. Runtime revisions are
applied in explicit numeric order, not lexical name order. Old rejected freeze
decoder/environment/numeric-control/revision-order attempts are preserved in
recovery, never erase or mistake them for current failures.

Luna only inspects metadata, progress, process/GPU/disk and updates the single
monitor baseline; no GT, caption, predictions, parameters, gradients, weights,
scores or restarts. pending_root is an explicit handoff. No overall deadline,
no new search, training, new models or fast memory. Full stage completion and
publication, not code/plan/smoke alone, closes the task.
