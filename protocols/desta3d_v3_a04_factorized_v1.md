# A0.4 Fixed Shared-R16 Factorized Direction Screen

Registered after A0.3, before any A0.4 GPU work. User attachment 30d6fe14 authorizes this bounded screen. A0/A0.1/A0.2 failed full256 multi-query fitting; A0.3's train-only R16 retained 80.6687% mean Dev energy and 83.3888% oracle vIoU gain. These are source GT oracle results, not learned success. Test whether a fixed low-dimensional output makes the original local predictor learnable.

## Locked intervention and data
Reuse all sealed A0 Train128 (95 training parents) and Dev64 (16 repeatedly exposed parents, four queries each), unchanged order, feature128, qT/qS128, state33, evidence8, union256 and radius .13545580427763146. B16 is precisely the first16 columns of A0.3 TRAIN_BASIS.npz, fitted using Train128 only, unchanged sign/order. Full matrices and original caches remain frozen and hash-pinned. No new labels, PTD cache extraction, fresh31/388, target, full618, expert or OPD.

Original local H128 input425 ->128, depthwise3x3x3, mix128 and SiLU stay unchanged. Only output128->256 becomes128->16. All hidden parameters use identical original seed initialization; the new output uses the same construction-order Gaussian std.001/zero-bias rule, not a sliced old head. 76688 trainable parameters; frozen Q_U, B16 and Q_C=Q_U B16 are buffers. No GMean, normalization, attention, extra loss or magnitude learning.

Derived target16 is full query oracle256 @ B16, CPU FP64 matmul then FP32 for cached training; preserve full original targets for diagnostics. It is computed once and sealed. BLAS thread policy fixed at4 for CPU construction/audit and Torch4, CUDA deterministic/CUBLAS :4096:8/TF32 disabled for fit. Independent NumPy FP64 verifies projection, geometry and cosines; target rounding must stay within 2e-6 relative L2. Fit gate uses FP64 dot/norm of saved FP32 predictions and exact FP64 projected original target. Training uses the rounded target only, cosine-only over the entire query, as in A0. R16-vs-full cosine identity is checked query by query, not products of averaged statistics.

## Training and predeclared routing
Seed20260928, AdamW lr.001/wd0, clip1, batch4 with per-query loss averaged, unchanged deterministic original shuffled cycles. First200 actual steps, fixed terminal; no best-step. Checkpoint stores integer Adam keys, live-bound counters, Python selection RNG/order, Torch CPU/CUDA RNG, initial and terminal model. Independent raw CPU gate audit precedes subsequent work.

- Train median >=.3 AND Dev >=.1: stop fitting and run Dev64 native.
- Train median <.3 at200: only continuation to total2000, same model/Adam/RNG/order trajectory. No restart or new seed.
- Train >=.3 but Dev <.1: stop, conditioning/generalization candidate.
- Still Train <.3 at2000: stop offline predictor tuning; R16 coefficient-field test-time optimization is the next feasibility proposal, not automatically launched here.

Report both split mean/median/count>.1/count>.3, full256 reconstructed cosine, terminal batch and dataset loss, gradient norm/clipping and actual counters. Save all192 prediction coefficient fields at each reached endpoint. Undefined/zero targets or predictions are explicit failures, never filtered. CPU scope, local-path, projection, budget, exception/restore and route controls are synthetic engineering checks, not measured native effect.

## Conditional native contract
Only after direction gate independently passes, register native stage with immutable selected checkpoint. Frozen official PTD4B/B1, exact cached physical input and context replay; source GT is privileged evidence, not forced prefix/selection. Generate Learned-R16 only for all Dev64. Ahat=pred16 B16.T, delta=Ahat Q_U.T, normalize the realized complete field to r||F||; same delta in both native passes, 0 optimizer. Reuse sealed B1/Oracle-R16 (Full Oracle as reference) only after input/hash equality. Preserve format failures/invalid geometry, seal all predictions and independently audit scope/field then score with scalar and tensor metrics.

Parent macro t/s/v, paired parent bootstrap CI, query harm>5pp, B1-good v/t>.5 retention; learned gain retention uses aggregate (Learned-B1)/(OracleR16-B1). Collapse only if branch mean <=-1pp AND >=12/16 negative parents. Positive v and no collapse permits a separately registered same-model exposed447 forward-only screen, not full618 caches now. Direction pass with native v<=0 routes to trust/no-op proposal. No outcome-based steps/rank/radius selection.

## Resources and evidence
GPU serial lease, 8GiB disk floor, 900s bounded fit allocation per endpoint, at most2000 total steps, output<=2GiB. Cumulative GPU cap=null. Worker/loading/failure/replay plus nonoverlap wrapper actual time recorded; CPU audits separate. Preserve failures and partials, fix isolated versions; never silently replay. Before registration disk/GPU checked. Source exposure, old optimizer integer-key correction, cancelled full-source/scalar studies and CURRENT are unchanged. Archive check -> snapshot complete -> check; publish code/protocol/anonymous aggregates only, no cache/weights/raw/IDs.
