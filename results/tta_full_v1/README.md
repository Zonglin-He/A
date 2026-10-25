# Full-target Tent / SAR baselines — launched, measurements pending

The user explicitly requested full-target TTA baselines. Coverage is **HC-STVG-v2:1902 queries/227 parents** and **VidSTG:10303 queries/732 parents**. Tent and SAR each cover all12205 queries: **24410 method/query results**, including512 already-complete results reused after exact file/hash/trial/reset/native-and-joint-readout verification. Existing128-parent confirmation panels remain historical results, not relabeled full-target measurements.

Scientific plan `4b147ab88d63b14bd394d8d4f94a48a4f1e30bcdb84d3da54b7abc6d09f3d2ab`; active independent execution `e9bfd4dfc63c8318fafd8bf3e4118e72ec1ec83894f6ab6241bfca0fe317e7ac`. Current entry: `studies/tta_full_v1/controller_runtime_v2.py queue`, with live identities/state/children under `outputs/tta_full_v1`. Controller1924421/start245882503 was actually launched; current state waits for the already-running HC grid collector to finish. Protocol/preflight/startup are not completed full-target measurements.

## Fixed algorithms and comparison

| Method | Fixed configuration | Update scope | Primary readout |
|---|---|---|---|
| Tent | SGD lr0.0001, momentum0.9,5 iterations/query | Existing temporal-decoder LayerNorm affine only | Original legal per-offset MAP then physical hull |
| SAR | SAM rho0.05, SGD lr0.0001/momentum0.9,5 iterations, reliable-entropy margin0.8, reset constant0.2 | Same decoder LayerNorm affine | Same native readout |

Pinned original algorithm files come from `mr-eggplant/SAR` commit20f6e24b17525f34503510afccedc0629b67b7c4 with exact SHA bindings. Both use episodic per-query/method parameter/gradient/optimizer/SAR-state resets, preserved eval/dropout behavior, frozen Source boxes and exact original seven TimeDecoder inputs. Native logits before updates and after reset must be bitwise original. Adapted affine weights, logits, entropies, optimizer iterations, decoder calls, parameter changes and exact resets are saved. Empty official reliable filters are recorded; numerical/native/reset failures stop without Source fallback or skipped queries.

This remains the previously defined temporal-decoder LayerNorm STVG adapter. Full target coverage does not turn it into canonical BatchNorm Tent, full-network adaptation, continual/online adaptation or a universal TTA ranking. Separate jointMAP endpoints isolate readout effects. Ours is the currently selected **kappa0 FullMAP** from already audited full-domain predictions. Ours uses eight expert observations and no optimizer; these baselines use no expert and five adaptation iterations. Different budgets are explicit. All fullsets were historically exposed; no new hyperparameter selection or independent-unseen confirmation claim.

## Input coverage and finite acquisition

The full main-method captures intentionally did not retain large decoder memories. Actual original input caches currently cover **294 HC +315 Vid =609 queries**. Existing complete Tent/SAR predictions cover128/domain (256 queries,512 configurations). The other known caches are used only as original input evidence, not as already-run official Tent/SAR results.

Four real cached input bridges verified all seven kwargs, original strides/layouts, physical frames and native logits against the input-bound full Source parents. All256 reused complete queries were checked for prediction/logit/receipt SHA, selected trials, reset proof and both native/joint readout arrays. This preflight used zero Source/expert/GPU/optimizer/GT scoring. Actual full GPU updates are still pending; every new GPU query must pass native replay and post-update reset before proceeding.

Missing temporal inputs require original224 Source reconstruction; spatial boxes/features cannot impersonate those inputs. The finite worst-case cap without future sharing is **1608 HC +9988 Vid =11596 additional Source queries**. Source and optimizer acquisition are checkpointed per query and charged explicitly. **Zero new expert calls.** No new backbone, teacher, reduced frame count/resolution/precision or optimizer sweep.

Scheduling preserves already-paid work:

1. Leave active HC structural/grid collector1518134/start245498341 untouched until its successful full seal/exit.
2. Execute HC full Tent/SAR, reusing all compatible inputs/completed results first.
3. Measure actual current-kappa0 timing using the previously fixed16/domain×3 repeats. Pause only owned CPU groups during measurement; resume them even on failure. A pre-launch execution-v2 adapter prevents new CPU jobs being launched during this pause.
4. Run original Vid grid collection and full Vid TTA in one versioned worker: the **same original Source forward** yields spatial grids and actual TimeDecoder kwargs. Save temporal input+SHA before adaptation; preserve original raw-grid checkpoint; reset all model state and restore Torch4 before original geometry/ROI-pool/legacy-control computation. Every original Full-control array must remain bitwise exact. Shared Source work is charged once to its producing process. Where existing inputs suffice, use them; where an input is still missing, acquire one original Source query.

The old kappa0 dispatcher1667978/start245631555 was stopped only after exact identity verification and complete code/plan/state/authorization/exit preservation, to prevent a duplicate independent Vid launch. Active HC Source work and existing Vid component/CG CPU domains were not stopped or restarted. The new owner adopts these domains and preserves their frozen science. Old dispatcher/queue/source files are not edited or resumed.

## Evidence and completion requirements

Only a complete1902/10303-query TTA domain seal permits independent official CPU scoring. Each result retains Source/native controls and current kappa0 Ours comparisons, all means/R.3/R.5, signed positives and negatives, severe t/v/union counted once (>5pp loss), rescues, short/long parent-cluster CIs, actual reliable-filter intersections and per-PID/start Source/update/load costs. Reused acquisition is not zero deployment cost; cached decoder adaptation time is not complete deployment latency; nested/shared durations must not be added twice.

Temporary native input from a new active query is retained through prediction/logit/weight/receipt commit and cold readback of all frozen readouts. Only then may that new raw input be removed with its original SHA and disposal proof. This sufficient saved result supports frozen readout replay, not arbitrary future optimizer/context replay. Unresolved paid attempts, uncommitted inputs and all historical caches are preserved.

Limits: one owned GPU worker, at most2 shared CPU compute domains on CPU14-15/nice19/idleIO, one compute thread (original Source/spatial numerical context and current budget derive retain their separately specified Torch4 contexts), minimum8GiB availableRAM,80GiB disk free,160GiB new private storage,72h/domain. All code/plan/asset mismatches hard-fail. No automatic retries, query skipping, fallback or frozen-hash relaxation. Source/teacher media, captions, GT coordinates, raw features/masks/weights and credentials remain private. Root exports complete results and verifies both public refs/all remote file modes/bytes/hashes/tree/ancestors before completion. Same heartbeat remains active through full TTA and all ongoing kappa0 tasks. TubeDETR is excluded.
