# DESTA-3D v2: shared reference/time and calibration connectivity audit

User authorization: private authorization (not distributed), 2026-09-27. Incremental audit; original four-arm training, locks, selection, and predictions remain unchanged.

## Fixed state and paired decode

Use the first safe post-review checkpoint, dual_repaired E1 cursor24, after 155 evidence and 6 integration optimizer steps. Its exact file and adapter SHA are in `artifacts/desta3d_v2/shared_reference_v1/CHECKPOINT.json`. This checkpoint is not selected by readout performance. E0 weights were already overwritten before the review; E0 predictions remain available and must not be represented as reconstructed weights.

Compare the existing independent spatial semantic reference (event time forced) against event-generated reference **and** time tokens, freshly prefetched under the spatial visual branch. No event KV reuse, no new head or training. The frozen PTD weights, pixels, source metadata, adapter, event reference/time decisions and evaluation support match. A format failure remains a failure, with no fallback to independent reference.

First run four source-validation queries, selected by sorted metadata keys and distinct parents. Check real zero-gate old/new interval, reference and box equality, actual nonzero residual injection, grammar and fresh-cache behavior. No IoU gate is used for interface acceptance. If zero-gate geometry differs, preserve it and inspect numerical/position behavior before expansion. After engineering acceptance, register the entire existing 198-query/31-parent source validation cohort; do not select a subset by effect. Source labels are opened only after all paired predictions are sealed.

Report source parent-macro vIoU/sIoU/tIoU, 10,000 paired parent bootstrap intervals (seed20260927), original-good retention and source damage >5pp, invalid formats, independent/event reference differences, direct event readout versus endpoint quality, actual cast residual ratios and changed elements. No claim about target TTA follows from this source-only comparison.

## Actual calibration gradient check

On the lexical first source-training query only, use clean frozen features as detached teacher and the same physical frames with gamma0.9 as the student view. No source or target label is read. Take one ephemeral AdamW step (LR1e-5, wd0, clip1) then reset exactly. The concrete loss is normalized reader-feature MSE plus referent and direct-event Bernoulli KL, plus mean squared parameter anchor. Alignment=0 and joint=0. The source moments for later alignment must be newly constructed from v2 query-conditioned features; v1 moments are prohibited.

Only branch FiLM and channel-LN affine parameters update: **66,816 parameters**. Gates are frozen because all these losses precede the residual gates. Do not manufacture a gate gradient with a fake correction loss. Record each group’s None/zero/nonzero tensor counts, gradient norm, exact changed parameters and frozen-backbone gradients. After the step, force the same pre-step event reference/time and measure **actual PTD coordinate logits**, KL, box changes and BF16 residual changes. A finite/nonzero gradient is engineering connectivity, not localization benefit; a zero output change is reported rather than tuned away.

## Selection and continuity clarification

Preserve the executed common eligible source-validation window, epochs1–5, for **all** four training arms. `dual_current` epoch0 is integration but excluded from this common window; the original shorthand “B epochs only” was imprecise. Do not retroactively rerank E0. An isolated continuation wrapper saves immutable all-arm adapter snapshots from E1 onward at cursor618 before validation. It does not change losses, optimizer state, RNG, query order, patience, or the original runner.

Keep joint=0; occupancy top-k is not a calibrated referent-existence probability. No offset, zero-init search, new teacher, P1 temporal pyramid, or target64 sweep is added. The source resource gate remains before target8→64. Subsequent TTA reports Frozen, source no-TTA and TTA separately, and view-only versus +alignment in matched controls.

## Resources and integrity

GPU is serial under the existing lease. Each audit invocation has a 1,800s engineering review window and 8GiB disk reserve; cumulative cap is null. Record loading, failures, replay and wrapper overhead in additive receipts. Preserve failed invocations and their code hashes before any repair. Each prediction is atomic and write-once, all original training pins and production CURRENT are retained. One bounded Luna max watcher monitors the current GPU stage.

## Executed engineering revision and small scale panel

Pilot001 failed zero-gate box equality in four of four queries despite identical reference/time. Preserve the entire whole-prefix prefill implementation and predictions; the failure alone does not establish whether BF16 arithmetic or a cache-state difference caused it. Pilot002 failed during import before model loading, also retained. Pilot003 uses the official incremental semantic/time/box cache schedule under a fresh spatial prefill. Semantic/time parser outputs are fixed to event-generated tokens; their probe computations remain only to match the original execution path. This version passed four of four exact zero-gate box/reference/time checks. Coordinate logits represent xyxy numeric-token positions; boxes are separately converted to cxcywh.

The one-query gradient check found a large weighted-auxiliary/task gradient norm ratio. A separately registered eight-parent source development panel (`gradient_scale_panel_v1`) uses the lexical first training query of each of the first eight distinct parents; it reuses the already sealed first-query result and evaluates seven more without optimizer steps. This is not independent confirmation: the first result motivated the panel, and source-training data are used. It measures identical shared_stem parameter gradients for temporal CE, spatial CE and the two auxiliaries, including .1 weighting and undefined angles for zero norms. No PCGrad, rescaling, extra loss, source sampling or active optimizer change is made on the basis of this diagnostic.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
