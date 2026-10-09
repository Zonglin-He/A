# Figure 1 Panel B: native cross-domain error composition

Human scope: on 2026-10-09 the user specified that Panel B must be tested on
cross-domain data. This supersedes the same-domain Panel B proposed in v5.
The old v5 files are retained as a historical diagnostic, not the new result.

## Source and target binding

Evaluate frozen TA-STVG and TubeDETR in both directions:

| Direction | STVG training checkpoint | Target split | Queries / parent sources |
|---|---|---|---|
| VidSTG → HC2 | released VidSTG checkpoint | official HC-STVG v2 validation | 128 / 128 |
| HC2 → VidSTG | released HC-STVG v2 checkpoint | official VidSTG test | 128 / 128 |

Checkpoint paths, byte sizes and SHA256 digests are registered before model
execution. TubeDETR's local upstream README identifies separate VidSTG and
HC-STVG v2 training checkpoints; the historical intake hashes are retained.
TA-STVG uses the project's previously qualified source loaders and checkpoints.
Shared upstream pretraining is not claimed to exclude every target-domain
image. The comparison is across released source-trained STVG systems, not an
architecture-only controlled training comparison.

The local ParallelTubeDecoding-Qwen3-VL-4B model card lists VidSTG and HC-STVG
v1/v2 jointly in supervised fine-tuning. It therefore cannot be labeled a
source-only cross-domain checkpoint and is excluded here. No new backbone,
weight download, training, expert, adaptation, oracle readout or tuning is
authorized by this diagnostic.

## Target cohorts and model inputs

HC2 reuses the original locked P0 confirmation list: 128 distinct parent
movies, one official validation clip/query per parent, without selecting by
score. VidSTG reuses the original native-support 128-parent roster and its
one query per video. These panels are historically exposed; they are not a
fresh held-out confirmation, nor the full official benchmark.

For each target query, choose at most 64 evenly spaced positions from its
original registered frame grid, preserving the first and last frame. Both
backbones receive byte-identical RGB observations and exactly the same frame
IDs and query. VidSTG uses the verified zero-based original-frame decoder;
HC2 uses the verified official FFmpeg output timing. Model-specific released
preprocessing remains native: TA-STVG 224, two temporal offsets, encoder
FP16/native suffix FP32; TubeDETR 224, stride 2, fast branch, BF16. Only the
native final decoder output is scored. There is no best-layer/interval choice.

## Execution order and leakage barrier

Current OPD P1 retains priority. The finite CPU controller waits for the
current P1 global prediction seal and P1 GPU completion receipt. It neither
reads P1 prediction payloads nor interrupts its controller/worker. It acquires
the project's existing exclusive GPU lease for each native worker; a busy
lease means waiting, never parallel GPU execution. It stops for root diagnosis
if the current P1 ends in an unexplained failure or a new native worker fails.

Each of the four model/direction cells has two predetermined qualification
queries (the first two roster entries). Qualification checks native parity,
repeated deterministic outputs, identical observation hashes, no parameter or
buffer changes, and denied annotation access. These eight qualification cells
are retained separately and are not accepted as formal predictions. Synthetic
CPU contracts do not count as real GPU qualification.

Only after all four qualifications pass, execute 128 native queries per cell,
512 formal outputs in total, serially. Save immutable per-output receipts and
four prediction seals, then verify every prediction byte/receipt/pixel binding
and write the new global barrier. Only this barrier authorizes the new
diagnostic's CPU GT scoring. The separate OPD P1 global barrier is unchanged.
An operational inference exception is preserved and stops execution; there is
no skipped query, rerun with another model, or silent denominator reduction.
Any native format/spatial/temporal failure that is a returned prediction stays
in the denominator and scores zero for the missing component.

## Predeclared four quadrants

Main thresholds are fixed before inference and scoring:

- temporal correct: native tIoU **strictly greater than 0.5**;
- spatial correct: native **GT-frame mean IoU strictly greater than 0.5**.

Spatial accuracy uses the full fixed GT-frame denominator, independent of
the predicted temporal interval. Use the original official linear interpolation
only inside the predicted frame-anchor hull, with zero IoU for uncovered frames
and invalid geometry. HC2's existing official lower-bound box clamp and literal
GT endpoint convention are retained; no per-model clipping adjustment is made.
An independent NumPy geometry/interval implementation must match the relevant
official TA-STVG evaluator for every cell. vIoU is not substituted for sIoU:
it already couples temporal and spatial error and would blur the two axes.

Count all four categories in each model/direction:
T+/S+, T+/S−, T−/S+, T−/S−. Report exact N and category counts/percentages.
Keep the two directions separate; do not pool unequal domains to obtain a
preferred result. Register 10,000 paired parent-source bootstrap draws with
seed 20261009 for category intervals, the T+/S− minus T−/S+ contrast, and
between-backbone contrasts on identical target parents. No threshold, cohort,
model or source checkpoint is selected from these results. Both positive and
negative contrasts are preserved. The motivating inequality is a hypothesis
until these cross-domain counts support it.

## Figure and closing

Panel B uses a compact four-category comparison with a separate block per
direction. The complete figure keeps a white background, short labels,
consistent colors and no outer/panel boxes or divider lines. No overall
“Figure 1” title or method pipeline is added. Panel A may use a matched real
cross-domain case after sealing; any selected illustration is explicitly a
case, not a prevalence estimate. Actual pixels, native intervals/boxes and GT
are never fabricated. Root must inspect real PNG/PDF exports before closing.

The controller's CPU handoff is not figure/paper completion. Root must finish
independent scalar/coverage/source audits, case readback, actual image review,
anonymous safe code/config/scalar-result export (no private media/annotations,
query text, weights or raw predictions), exact remote file verification and
RESEARCH_HISTORY check/snapshot/check. EATA, tuning and historical queues remain
paused. This diagnostic does not alter CURRENT or any OPD method/science lock.
