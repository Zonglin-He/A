# Main-method spatial OPD sensitivity search

2026-10-06 user selected spatial OPD as the main method, authorized obsolete
cache cleanup and paused other method experiments. The later instruction
fixes execution order: finish tuning first, then resume saved paper baselines.

Fixed: Native WHEN, original Uniform4, single frozen Grounding DINO, original
admission/Top1, joint1792, Adam with original beta/epsilon, 32 antithetic actions,
detached IoU feedback, Gaussian likelihood/control variate, final-round output,
per-query residual/Adam reset, separate persistent LN chain per trial/order.

Only five parameters vary: lr {0.001,0.003,0.01,0.03,0.1}; sigma {0.1,0.25,0.5};
tau {0.1,0.25,0.5,1}; steps {3,5,10,20}; writeback {0,1/32,1/16,1/8}.
The original (.03,.25,.25,10,1/16) is retained. This yields 16 distinct
one-factor configurations per dataset. Screen uses 16 deterministically chosen
development parent sources and two complete orders, clean cross-domain only.

After all screen predictions are sealed, offline development GT scores rank
parameter sensitivity by max-minus-min source-macro delta-vIoU, with parameter
name as deterministic tie break. The two largest effects enter 12 finite
Optuna TPE coordinate trials on all original 32 development sources, two orders.
The first refinement trial is the best screen configuration. The next 11
change one of the two sensitive coordinates at a time, using categorical
values from the same finite ranges. Coordinate incumbent updates use only
completed sealed development trials. Trial duplicates are reused, not rerun.

Each target dataset receives one unified configuration. Selection maximizes
source-macro After-Frozen vIoU; exact ties use fewer >20pp harmed sources and
lower measured GPU fit cost. No query/corruption/source-specific selection.
The old 128-source confirmation scores do not participate in parameter search.
All development data have prior exposure; selected scores are development
scores, not independent efficacy claims. No new heldout/main-table/corruption
experiment of Ours is authorized by this task.

GPU prediction workers reject GT/results access. A trial's complete prediction
barrier precedes its CPU scorer. Offline GT may guide the next development
trial, never the online objective, admission, output choice or reset decision.
Numerical-invalid trials retain their failure/prefix and cannot win; no query
is silently skipped or assigned a fallback score. Independent Gaussian/IoU/
Adam arithmetic, exact default equivalence and final-step/state-chain checks
qualify the configurable implementation before the finite search.

After both dataset selections and their actual configuration files are sealed,
the original TENT optimizer/LN/prediction chain resumes at the saved cursor,
then the remaining SAR and EATA baselines retain their original parameters,
sampling, source checkpoints and input rosters. The registry change is recorded
in a process-local verification bridge; original paper code/runtime locks and
existing predictions stay immutable. Missing HC source-training media remains
an explicit EATA dependency, with no substitution of target validation data.

Results, actual configurations, numerical failures, search cost and cleanup
receipt must be recorded in the research archive and exported anonymously to
Zonglin-He/A. Historical reports/configs/receipts and baseline states survive
cleanup; obsolete raw caches are unavailable for later replay by design.
