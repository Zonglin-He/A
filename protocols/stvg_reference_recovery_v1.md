# Fixed-reference and LN-state recovery experiment

Date: 2026-10-15. TA-STVG only; CURRENT is protected. This is a bounded, previously exposed development-cohort mechanism test, not independent confirmation or Table 1. No bootstrap and no GT-assisted reference fitting.

## Fixed cohort and supervision

Use the exact 32 locked historical development sources, 16 in each direction, and the frozen `round2_prompt009/new_crop` Weight-free STVG-R1 Top-1 selections. There are 49 accepted non-Source anchors across 23 sources. Source checkpoint, input pixels, frame clocks, reference geometry/weights, target teacher and Temporal Offset-Trust readouts remain unchanged. No new teacher inference or candidate construction.

## E1: cache-only reference swap feasibility

Match query identity, source video and pixel hashes, source checkpoint/model state and exact physical observation frames. Alias equality does not qualify. An old expert reference at another frame cannot be interpolated into a matched reference without a new, separately declared measurement. Inventory all failures as well as matches; a missing common support prevents this comparison and is not a negative teacher-quality result.

## E2: Source-reset configuration-package replay

Hold references and the existing validated Gaussian-IoU loss fixed. Compare exact saved current-five-step predictions against new fits using the original frozen domain configuration: HC lr=.01/sigma=.025/tau=.05/40 steps, Vid lr=.03/sigma=.1/tau=.25/10 steps; 32 antithetic samples. Query256+LN1536 update; Source reset and fresh Adam per query. Both arms retain the current reference-agreement acceptance/rollback rule. This isolates an entire configuration package, not individual hyperparameters, and does not reproduce the complete old DINO/online method. Independent full-model readout must match cached decoding and restore Source. Empty-reference cases reuse the exact unchanged arm.

## E3: Episodic vs original LN writeback

Keep the current five-step configuration and references fixed. Use the exact current cohort's query-ordinal subsequences in each domain's original P1 `order1/order2/order3`, yielding six 16-query streams and 96 arrivals from 32 independent sources. Every stream starts at Source. Every arrival resets Query residual256 and Adam, while LN1536 starts from the prior commit. The commit is `incoming_LN + alpha*(accepted_fit_LN-incoming_LN)` with original alpha HC=.0625/Vid=.125; committed Query is zero. Empty-reference arrivals must retain and actually decode incoming LN, not substitute Frozen.

Record every arrival state, proposed/accepted fit states, rejected-step rollback, next committed state and predecessor hash. Actually run the full Student Before and After decoders on every arrival; verify Source restoration after each. Retain exact parent F, T, Episodic-Native and Episodic-Temporal predictions as controls. Native and fixed Temporal intervals are two readouts of the same spatial boxes; they are not two independent fits. Three streams are repeated orders of the same sources, not 96 independent videos or three independent Episodic runs.

## Scoring and decision

Seal every stage's predictions and perform independent Source/state/arithmetic/readout checks before opening GT. Report complete per-source vIoU, tIoU, dense sIoU and supervised/unsupervised support metrics, including negative, neutral and unknown outcomes. E3 primary contrasts are Before-Frozen (incoming memory), After-Before (current update), and OnlineAfter-Episodic (carry effect), separately for Native and Temporal support and for every stream order. These are descriptive matched contrasts; do not describe an additive semantic causal proof or infer long-stream stability from 16-source streams. No GT selection of best step, source, order, parameter or reference.

## Execution

One GPU process chain at a time under its own user unit; host MemoryHigh16GiB/Max20GiB/SwapMax1GiB, resource admission and own-unit guard. Old queues remain stopped. Preserve partial outputs and failure records; engineering fixes cannot silently alter the scientific recipe. Completed bounded measurements require anonymous positive/negative/null output, source/configuration bindings, root readback, actual figure review, verified GitHub byte/blob/tree/ref publication and archive check-snapshot-check.
