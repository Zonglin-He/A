# Candidate-conditioned temporal latent quality audit

User authorization: attachment 77c82973, 2026-10-03. The measured scalar failures
motivate an accessibility test; they do not establish information loss as their
cause. This run tests a finite **source-supervised linear probe**, not latent TTA.

## Frozen target contract

Reuse the exact 1,152 A arrivals (32 development and 16 historically exposed
confirmation sources per dataset, two orders, clean and five 5% corruptions),
including the 288 expert arrivals with sealed Old8/Expanded32 support. All
candidate intervals, A spatial boxes, pre/post persistent states, expert schedule,
pixels, sampling, and checkpoints remain immutable. No new specialists, views,
candidates, spatial updates, or backbone forwards on target data. Extract features
with cached encoder H and exact native suffix replay at each saved A pre-state.
Source and target checkpoint are the same official checkpoint for each dataset.
No deployed-method promotion or historical-queue continuation is authorized.

## Source supervision and budget

Vid: reuse 126 already available official train sources, one pre-existing
metadata-chosen query each, and the existing 95 train/31 validation source split.
HC2: reuse 64 available official train sources from the existing ViTTA source
manifest; SHA256 `temporal-latent-v1|source` chooses 16 validation and 48 fitting
sources. Verify source IDs and media hashes have no overlap with target sources.
No label-based source filtering, event-dependent sampling, new downloads, or
target training labels. These sources have previous project exposure and are
not a new untouched benchmark. Training-derived validation selects probe ridge
strength and is explicitly not an independent final evaluation.

Vid uses the existing physical 5fps/max200 metadata-only input grid; HC2 uses
the same 64-frame official timing/index recipe as the fixed target panel. RGB
normalization, two offsets, query-subject parser, and FP16 encoder/FP32 native
suffix match the established interface. Source capture is once per query; no
backbone training or per-optimization-step backbone inference.

Generate source Old8 using the same six decoder-layer and final span-pair
interface, then append the existing 24 deterministic maximin intervals. Source
GT is used only to label these fixed intervals with physical-time tIoU. It is
never used to generate intervals. No source expert is needed.

## Representation and probe

Hook `model.temp_embed` input, final (sixth) temporal decoder layer, after native
second-pass rerouting. Verify each saved hidden tensor reproduces all head logits
bitwise. Merge the two offsets by their exact physical frame IDs. Spatial decoder
parameters are held at A pre-state during target replay; no gradients or writes.

For interval `[frame_i,frame_j+1)`, concatenate 256D `h_i`, `h_j`, mean of observed
states i through j inclusive, left and right observed-state means in **1-second**
local windows, `h_i-mean_left`, and `h_j-mean_right`. Left frames are in
`[start-fps,start)`, right frames in `[end,end+fps)`. Missing exterior context is
zero and is counted in diagnostics. Total 1,792D. No PE cosine or spatial cue.

Standardize each dimension using training rows only; std below 1e-8 becomes 1.
Fit intercept plus ridge least squares in FP64, objective mean squared tIoU
error plus alpha squared coefficient norm, alpha fixed grid
`[.001,.01,.1,1,10,100,1000]`. Choose maximum source-validation top1 tIoU, then
minimum validation MSE, then smallest alpha. Do not refit on validation sources.
Freeze the chosen coefficients and normalization before target scoring.

Matched geometry-only linear control has normalized start, end, and length,
same training labels/splits/ridge rules. It tests positional/duration shortcuts;
it is not a new expert or additional deployment mechanism. Raw regression
scores are used for ordering, without clipping or target calibration.

Evaluate both probes on Old8 and Expanded32. A8 is retained unless a unique
score maximum improves strictly over its score (tie epsilon 1e-12), identical
to the previous N/U/S audit. Unscheduled arrivals exactly retain A. Scores and
selections for **both datasets** must be sealed before target GT-derived labels
are opened. Existing target GT exposure is disclosed, not claimed absent.

## Outcomes and scope

Report top1 tIoU/vIoU, same-support oracle regret, paired gain/loss, >5pp harms,
old-good destruction, and large-correction benefit/harm AUC, precision, recall,
harm acceptance using the predecessor's radius .5 and operation decomposition.
Compare L/G with A8 and N/U/S on the exact same candidate metrics. Aggregate
within condition/order then source; 10,000 paired source bootstrap; clean and
expert/nonexpert subsets separate. Preserve positive/negative cases and source
concentration. The supervised target is tIoU, whereas vIoU also depends on fixed
spatial boxes; neither a negative vIoU result nor failure of this finite linear
probe proves all native latent information absent.

Do not automatically add MLP, latent adaptation, new expert, or parameter search.
Those are later research decisions. Source fitting costs, target suffix costs,
stored feature bytes and actual calls are reported. Keep learned weights/raw
latents/source annotations private under the existing export exclusion; release
code, configuration, anonymous scores/labels/metrics, fit summaries and hashes.
