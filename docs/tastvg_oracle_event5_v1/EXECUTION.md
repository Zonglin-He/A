# Finite fixed-A oracle and event5 diagnosis

Read protocols/tastvg_oracle_event5_v1.md and actual STATUS. This is separate
from all paused/completed old tasks. Production remains CURRENT_METHOD.

1. CPU `.conda/tubedetr/bin/python -B -m scripts.test_tastvg_oracle_event5_v1`
   validates analytical endpoint/support/upper-bound controls. Prepare once with
   `-m scripts.prepare_tastvg_oracle_event5_v1`: pins A trajectories, original
   nine probes, U/U2/R evidence and label-derived event-only positions. Labels
   are explicitly authorized for this oracle route and have historical exposure.
2. CPU `-m scripts.score_tastvg_residual_oracles_v1` scores all 1152 A/GT time/
   dense GT space/Joint GT readouts and only the 288 cached candidate grids.
   Return EXPERIMENT1_ROOT_AUDIT plus source bootstrap. Root reviews this before
   starting the next experiment. No GPU or new expert calls in this step.
3. Under `bash scripts/with_local_cuda.sh`, run
   `.conda/tubedetr/bin/python -B scripts/continue_tastvg_gt_event5_v1.py` once.
   It owns PROCESS.lock, serially uses Sa2VA with .runtime/sa2va_deps on PYTHONPATH,
   then TA-STVG Vid and HC suffix-only temporary SGD workers. The logical request
   upper bound is 288. Pre-inference EXPERT_ATTEMPTS ledger counts actual new
   unique input calls. No requests for insufficient event sample support. No
   raw GT/metric files may be opened by the model worker; only the CPU event
   position plan is allowed. A's pre/post state is never overwritten.
4. GLOBAL_INTERVENTION_BARRIER seals both datasets before CPU
   `-m scripts.score_tastvg_gt_event5_v1`. All four observations use the same
   eligible donors; compare GT and old A intervals without decoder intervention.
   Preserve unsupported full-scheduled no-op sensitivity and negative cases.
   The old confirmation caches omit 96 U first-step post-tubes. Complete these
   once with `scripts/replay_tastvg_cached_uniform_readout_v1.py vidstg` and then
   `hc2` under with_local_cuda, using the saved first-step states and H caches.
   Each dataset adds 48 post suffix calls and two pre-state bitwise controls;
   no SGD, experts or backbone. Preserve the valid development partial and
   its hashes before retrying CPU scoring. Engineering revision 005 pins the
   cache-contract repair. These fixed states are never selected using GT.
5. Root checks coverage, official metrics, scalar reward/KL/SGD, same-state U
   gradient controls, code/input/checkpoint hashes, dedup budget and all old A
   state chains. Save DECISION.json recommending only ONE next research variable;
   this does not authorize its execution or a production change.
   Run `-m scripts.audit_tastvg_oracle_event5_root_v1` after completed scoring to
   rehash all 6104 old inputs, verify both U reconstruction barriers, and check
   exact byte reproduction of the retained valid development partial.
6. `-m scripts.report_tastvg_oracle_event5_v1` draws paired uncertainty figures,
   writes docs/TA_ORACLE_EVENT5_REVIEW.md. Independently run anonymous auditor
   `scripts/audit_tastvg_oracle_event5_public_v1.py RESULTS_DIRECTORY`, visually
   inspect figures, export explicit manifest using export_tastvg_oracle_event5_v1.
   Publish only listed code/protocol/anonymous outcomes to Zonglin-He/A, verify
   every remote file and rerun the anonymous audit from the remote checkout.
   Update RESEARCH_HISTORY check/snapshot/check and save FINAL_COMPLETION.

Failed engineering attempts and original code/logs belong in recovery before a
pin revision; never rewrite scientific configuration or old predictions to
resolve an engineering issue. pending_root is a handoff, not completion. No
Specific_temp, new temporal views/loss search/baselines/backbones/full online
stream or fresh source choice. No extra recurring automation was created.
