# Three native model rows following the human sketch

Each row contains a real filmstrip with GT and native predicted boxes,
followed by separate Prediction/GT event ranges and two native tIoU/sIoU
columns on one unchanged zero-to-one scale. The human's subsequent instruction
allows the six native score values at three decimal places above these columns.
No tick numbers, percentages or timestamps appear elsewhere. Green dashed rectangles are
GT; coral solid rectangles are predictions. Blue and coral score columns
represent tIoU and fixed-GT-frame sIoU respectively. Gray backgrounds show
the common full scale. Extremely low spatial scores are not enlarged.

TA-STVG and TubeDETR use their actually completed HC2-source-only → VidSTG
cross-domain native outputs. The human explicitly authorized the third MLLM
row as a labeled joint-trained reference. It reuses native candidate zero
from PTD/Qwen3-VL, which has joint VidSTG/HC-STVG supervision; it is not a
source-only cross-domain model and is not inserted into the cross-domain
aggregate statistics. All three inputs, frame IDs, canonical RGB and the
single ground truth instance are exactly matched. No model is rerun.

The previous stage-performance case has a spatially successful MLLM output
and is retained as an earlier artifact. The new qualitative example is
chosen post hoc from five existing matched cases where all three native
outputs satisfy tIoU>.5 and sIoU<=.5. Deterministic ranking is descending
mean(tIoU−sIoU), then ascending locked parent ordinal. The selected ordinal
is 61. This selection only illustrates a possible failure and cannot
estimate its prevalence or establish cross-backbone spatial-error dominance.
The previous negative registered contrasts and all source rows are preserved.

## Paper caption

Event coverage does not ensure instance grounding. On the same video and
query, native predictions from TA-STVG, TubeDETR and an MLLM reference overlap
the event but disagree with the referred instance. Green dashed boxes are
ground truth and coral solid boxes are native predictions. The event ranges
and native temporal/spatial score columns separate timing from instance
localization. TA-STVG and TubeDETR use source-only cross-domain checkpoints;
the MLLM is a separately marked joint-trained reference. This case is chosen
post hoc for illustration and is not an aggregate error-frequency estimate.

The unchanged full cross-domain quantitative companion remains in
`results/stvg_motivation_cross_domain/2026-10-09`. It does not support the
original spatial-only-error dominance hypothesis. The figure motivates
spatial correction as a task need; OPD efficacy, expert reliability and
on-policy mechanism remain questions for their authorized experiments.

## Evidence and export

Local source bindings contain the exact native prediction/receipt hashes,
canonical RGB hash, GT provenance and candidate-zero scalar provenance.
Real frames are uniformly selected inside the shared annotated/predicted
event support, without cropping or geometry modification. PNG, independent
PDF raster, and editable SVG must be reviewed at root. Dataset pixels, query,
GT boxes and raw outputs remain local. Public files contain authoring code,
configuration, anonymous scalar-height integrity and the factual caption.
No active P1 output is used for this visualization.
