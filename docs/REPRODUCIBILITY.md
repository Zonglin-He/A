# Scope and experimental boundaries

This repository contains reusable code plus selected research execution scripts. It does not contain a turnkey data preparation pipeline for every historical experiment.

## Current v2 source comparison

- Frozen official PTD-4B, hidden width 128, P1 motion/pyramid features disabled.
- VidSTG source-only: 618 training queries / 95 parent videos; 198 development-validation queries / 31 parents.
- Three repaired arms: early-factorized, shared-3D and dual-3D. One auxiliary evidence epoch, then at most five integration epochs. Reader LR 3e-5, head/out/gate LR 1e-4, AdamW, no weight decay, accumulation 4, 5% warmup and cosine schedule, gradient clip 1.
- Current-recipe control: same dual-v2 initial state and split objective, six integration epochs, AdamW 1e-4, accumulation 1 and constant learning rate.
- The two dual arms start at exactly the same state. Shared modules across architecture controls are copied where names/shapes match. Parameter counts differ slightly; this is not exact capacity matching.
- Source evidence uses referent occupancy on valid annotated support and independent frame event labels. Missing boxes do not erase event supervision. Six queries lacking valid PTD responses still receive auxiliary supervision.
- All arms' current-epoch validation predictions are sealed before source validation labels are read. The executed common eligible window is epochs 1–5 for all arms; `dual_current` epoch 0 is integration but excluded. The old shorthand "integration epochs only" was imprecise. Selection uses parent-macro full-tube vIoU, with ties retaining the earlier state. This is development selection, not an untouched target estimate; E0 has not been retrospectively re-ranked.
- Target labels never select updates, checkpoints or sources. HC mixed-source media disjointness was not resolved for this run; later HC evaluation would be cross-dataset transfer.
- Full v2 TTA is pending the source gate. The one-source-query ephemeral calibration check updates 66,816 FiLM/LN parameters and freezes two residual gates because its objective is pre-gate. It is a real gradient/output-connectivity check, not a completed target evaluation. Alignment and joint weights are zero; future alignment requires new v2 query-conditioned statistics.

## What can run from this repository alone

The synthetic CPU example and the three v2 unit-test modules exercise tensor shape, query conditioning, branch separation, channel-only normalization, physical time differences, event targets, optimizer groups and the TTA objective. They require no data or weights.

`scripts/check_desta3d_v2_shared_reference.py` and `scripts/check_desta3d_v2_shared_reference_cached.py` additionally exercise the official PTD parser/generation contract using mocks. Install the pinned official PTD source and its dependencies from `docs/DEPENDENCIES.md` under `external/ParallelTubeDecoding` before running them with CUDA disabled. These checks open no checkpoint, media or labels. The first script also preserves tests for the historical whole-prefix attempt; real zero-gate acceptance applies to the cached revision only.

The real-PTD P0 and source-fit scripts still require local `artifacts/` metadata, fixed source records, original model dependencies, tokenizer files, model weights and legally acquired video data. They deliberately retain registration and sealing checks. `configs/desta3d_v2_source.json` is a readable copy of the actual configuration, not a substitute for those inputs.

Historical support scripts may reference old artifacts, oracle labels, or superseded methods. They are included because of imported helper dependencies and to make the implementation inspectable, not as recommended jobs to run. No schedule or training job is started on import by this publication procedure.

## Snapshot integrity

The live research workspace and all running experiment pins remain unchanged. Public copies replace machine-specific paths with relative placeholders, including private authorization-note locations. These historical authorization files are not published, and those registration paths need explicit local setup before reuse. The initial and incremental publication manifests record original and exported hashes.

No trained model, video, annotation, per-query caption/media manifest, raw prediction, personal conversation, credential, or full private research ledger is included. No new open-source license has been selected in this snapshot; third-party rights remain with their owners.

## Published aggregate results

`results/desta3d_v2/2026-09-27/` exports audited E0 and shared-reference source-validation summaries, calibration connectivity and the small gradient-scale panel. Macro IoUs are stored as fractions; delta/CI fields explicitly use percentage points. The summaries retain failed/neutral/negative findings and denominators. They omit per-query/parent identities, raw annotations and output tensors. Local evidence SHA-256 values identify the original reports but are not a substitute for publishing the underlying inputs: the public summaries alone cannot reproduce every metric from scratch.

The shared-reference checkpoint is fixed at 155 evidence steps plus six integration steps, not a performance-selected final checkpoint. Two decode paths gave identical boxes/tubes for all 198 queries. Only 197 nonempty valid reference pairs are comparable; one format failure remains in the 198-query task denominator. The source E0 controls have different stages (A for repaired arms, B for current recipe), so they do not isolate a single optimization trick.
