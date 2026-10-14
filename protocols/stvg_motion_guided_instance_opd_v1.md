# Temporal QA + Motion-Guided Instance OPD: fixed small experiment

2026-10-14. Authorized by the attached two-round review. This is a bounded
research candidate, not a CURRENT promotion. Negative findings remain valid.

## Question and cohorts

Does candidate-local visual correspondence increase usable instance support,
does R1 selection survive presentation changes, and does its selected reference
improve a freshly decoded student? Supervision quality, candidate support and
optimizer behavior remain competing explanations.

Use hc2_000–007 and vidstg_000–007 in the existing source-only roster. 000–003
are the old eight-case mechanism panel; 004–007 are eight additional cases chosen
by stable ID before this experiment. All belong to historically exposed sources;
the additional panel is not an untouched confirmation set. Report cohorts apart.
TA-STVG starts at its original cross-domain source checkpoint. Existing Temporal
QA outputs are reused exactly for the old eight only. New eight report spatial
effects with Native and Projection baselines; they have no new Temporal QA fit.

## P0: local correspondence, not an identity guarantee

Old eight: reuse all 43 original GT-free Source plus marked/masked Top3 union
seed windows, including previously rejected whole-trajectory proposals. Track
each non-Source seed from the second original spatial observation using pyramidal
LK. Source itself remains its original tube. Each candidate center follows its
own local point displacement; width/height remain the Source per-frame widths
and heights. This is neither a new detector nor learned tracker.

Additional eight: construct the same box-scale full-image coarse grid at that
anchor; deterministically choose at most four grid windows by farthest-center
coverage, first farthest from Source then farthest from selected centers, ties
by window ID. No R1 retrieval score or GT selects these new seeds. This different
seed proposal rule is disclosed and cannot be pooled as a matched tracking effect.

Original four observation frames are fixed. Decode every fourth physical frame
between first and last, plus exact observations, with chronological mapping.
LK uses gray long edge at most 640, 80 corners, quality .01, minimum spacing 5,
window 21, 3 pyramid levels, 30 iterations / .01 tolerance. Forward/backward
threshold 1.5 pixels is in working-gray coordinates, >=4 retained points and
retained ratio >=.5. Median displacement; backward and forward directions are
independent. Lost correspondence or out-of-bounds gives explicit Missing, then
that direction remains Missing. No geometry clipping, GT recovery or replacement.
The anchor alone is not evidence of a successfully tracked object.

P0 saves all proposals and full transition evidence. P1 displays Source plus
the first four non-Source seeds in the locked proposal ordering, regardless of
tracking success. Missing frames are not drawn and are declared in the prompt.
Full-frame support count, >=2/4 and 4/4 support, Source-only frames and unknown
identity are reported separately; preserving an anchor must not inflate success.

## P1: fixed visual protocol and six controls

Four full-scene 448px frames, actual timestamps, at most five stable IDs. Base ID,
palette and draw order are independently permuted by a fixed case-ID hash. Run
base, ID-only cycle, color-only cycle, draw-order reversal, joint permutation,
and wrong-query cyclic donor within the same domain/cohort. Total 96 requests.
No prompt search. Scene pixels/geometry remain identical; annotation pixels
necessarily change in presentation controls. Missing means unavailable evidence,
not negative semantic evidence. Return strict JSON ID, 0 for none/uncertain.
Wrong-query may refer to the same object; unchanged choice is diagnostic, not
proof of ignoring language. Palette/order/ID consistency is not GT identity proof.

Frozen official STVG-R1-7B BF16/SDPA, four frames, <=4096 input tokens, <=96
output tokens. No DINO/YOLO/SAM, new weights, or alternative teacher. Retain actual
token counts and complete costs including input construction/model load separately.

## P2 and fixed V03 reference control

Before labels: Source ID, none, invalid output -> exact Spatial No-update.
For a non-Source base choice use only its non-Missing original observation boxes.
Run Direct / IoU-OPD / DIoU-OPD, five steps each, original per-domain hyperparameters,
same source state, random key and 1792 Query/LN parameters; no GT step selection.
Controls do not change reference after looking at results. Verify final full-model
re-decode, Source restoration and all negative effects. Empty supervision -> no-op.
Report partial support and original normalization semantics rather than silently
rescaling loss for Missing. No assumption that OPD must beat Direct.

Separately, vidstg_002 (old V03): freeze old legal Source, old R1 ID2/window4,
and old alternate ID3/window5 geometry before any new GT reading. Three actual
IoU-OPD fits with identical initial state/config/random key, including Source
self-distillation as a diagnostic exception. Do not mix these with main No-update.

All predictions and selections must seal before new GT scoring. Root independently
checks rewards/weights/updates, complete readouts and metrics. Report candidate
oracle only post-seal, known/unknown original anchors, supervised/unobserved frames,
paired results and negative tails. Small historical panels do not establish
generalization or identity correctness. No automatic 32-source expansion.

QA-interval anchor placement is a separate optional future ablation, not part of
this motion-versus-template test; no temporal change or new temporal training here.

## Execution, custody and closure

One current-scope GPU unit, RAM high16GiB/max20GiB/swap1GiB, existing per-process
resource guards, disk floor30GiB. Do not resume stopped old queues. No whole-run
deadline or hyperparameter search. Preserve failures/partial results separately.
Public export only anonymous complete positive/negative/unknown results and
allowlisted implementation/protocol; exclude private media/query/GT coordinates,
weights, optimizer states and conversations. Actual root/pixel/source/scalar
readback, remote bytes/SHA/blob/mode/tree/two refs and archive check-snapshot-check
are required for finite FINAL. Prepared/running are not completion.
