# Weight-free eight-candidate sequential tuning and expanded confirmation

Date: 2026-10-15. User explicitly requests diagnosing the new method, tuning one parameter to the best measured value before the next, tuning both datasets, and expanding the cohort.

## Scope and data roles

TA-STVG source checkpoints, frozen STVG-R1, existing pure-algorithm eight-region proposals plus Source, Gaussian IoU OPD and episodic reset remain. No new pretrained models, YOLO/SAM, GT-reference fitting, CURRENT promotion or Table 1 launch.

64 distinct source parents: 32 per direction, one query each, using the immutable former 64-source engineering roster. The first 16 per domain were scored in the preceding experiment. They are explicitly reclassified as **development**, not untouched confirmation. The other 16 per domain are excluded from every selection stage, then measured only after all parameters are locked. Historical project exposure exists; neither cohort is claimed globally unseen. Previous old-eight source aliases are unrelated to these 64 aliases.

## Spatial coordinate search

Start with the preceding development-selected configuration: HC lr=.01, sigma=.025, tau=.05, steps=3; Vid lr=.0075, sigma=.05, tau=.25, steps=5. Each coordinate inherits every preceding winner, separately for each dataset.

1. Learning rate: original-domain lr times {1/16,1/8,1/4,1/2,1,2}, plus incumbent.
2. Gaussian logit-space sigma: original-domain sigma times {.25,.5,1,2}, plus incumbent.
3. Reward temperature: original-domain tau times {.25,.5,1,2,4}, plus incumbent.
4. Maximum update steps: {1,2,3,5,8}, plus incumbent. Actual informative and accepted steps reported separately.
5. Learning-rate neighborhood check after the other coordinates: incumbent times {1/sqrt(2),1,sqrt(2)}. This is a finite interaction check, not an unbounded search.

Each stage completes all 32 predictions and independent geometry/rollback/full-readout checks before GT scoring. Select maximum development mean T+S official vIoU per direction; ties within 1e-12 retain incumbent, otherwise choose smaller searched value. Boundary winners are disclosed; no global-optimum claim. Cap 600 new spatial fits. Exact same config/source/reference/seed results may be reused with explicit custody and zero new-fit accounting. Every changed configuration starts from source and actually optimizes and fully redecodes the Student. Fixed seed is not selected.

Teacher choices, observation positions, reliability weights, Temporal outputs, source weights, 32 antithetic samples, Adam betas/epsilon and acceptance rule are fixed in the spatial search. The writeback parameter does not govern this episodic experiment and is not presented as a useful tuning knob.

## Temporal diagnosis and subsequent isolated coordinates

First reproduce the saved 512-parameter temporal readout from frozen features and sealed teacher feedback. Separate Projection zero-state, teacher candidate ranking, and Adam trajectory effects, preserving failures and successful cases. Any later temporal parameter grid is separately locked before running; it does not move the spatial observation positions already fixed for this controlled experiment. If temporal output parameters change, the final recipe must explicitly retain the original QA observation clock while using the selected output interval; do not call old references evidence generated at a new interval.

## Execution and reporting

One own GPU unit, MemoryHigh16GiB/Max20GiB/SwapMax1GiB; preserve RAM/GPU/disk margins. Existing protected source/runtime and CURRENT remain unchanged. Independent metric implementation, complete per-source positive/negative/neutral rows, actual fits versus reuse, parameter trajectories, exact cohort exposure and bounded selection bias must be reported. Following selection, the additional 32 sources receive the same frozen recipe; their scores cannot trigger a new search in this round. Full results/code published through the existing anonymous GitHub export with actual remote verification and archive check/snapshot/check.

## Locked temporal coordinate addendum (before temporal trials)

After all spatial coordinates finish, separately optimize the shared temporal output head with the same32 development sources, per direction: lr {0,.0003,.001,.003,.01,.03}; tau_T {.25,.5,1,2,4}; steps {1,3,5,10,20}; l2 {0,.0001,.001,.01,.1}. Each inherits the preceding winner. Before any temporal trial, addendum002 fixes selection to T+S mean official vIoU with the final spatial winner held fixed, using the same incumbent-first tie rule. Temporal-only is retained as an attribution control. The earlier unexecuted T-only selection proposal is retained in TEMPORAL_DESIGN_LOCK.json; TEMPORAL_DESIGN_LOCK002.json is authoritative. The zero-lr arm is explicitly Projection-only, not evidence of successful learned adaptation. Four originally unavailable temporal inputs retain Native fallback. Cached original source features and sealed four-candidate teacher responses are reused exactly; all shared-head parameter paths and complete grid interval readouts are saved. This is real512-parameter head optimization, not fresh backbone/MLLM inference.

The spatial observation clock is deliberately fixed to the original five-step QA result in both development and confirmation. The tuned temporal output is decoded separately from the same cached feedback. This isolates output adaptation from changing candidate/reference construction. Confirmation exposes Frozen, original Temporal, tuned Temporal, prior-selected Spatial and its original Temporal combination, new Spatial, original Temporal plus new Spatial, and tuned Temporal plus new Spatial. No source-specific or GT-based online choice between these arms is permitted.
