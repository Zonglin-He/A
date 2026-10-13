# Fig. 1: spatial frames and three temporal lanes

This revision applies the human-provided spatial/temporal layout to the
latest reviewed v11 figure. It retains one shared real case, five frames and
three rows: TA-STVG, TubeDETR, and PTD / Qwen3-VL. Each row shows the original
spatial GT and Native boxes above GT, Native, and Projected temporal lanes.
The former native-score columns are removed. GT uses dashed green outlines,
Native uses orange, and blue is reserved for DINO and Projected evidence.
PDF, PNG and editable SVG are local deliverables.

## What is measured

All fifteen image panels use the exact original RGB pixels and saved GT and
Native geometry. The three Native intervals and common GT interval retain
their physical frame coordinates. No interval is expanded for visual effect.
The original case was selected after the historical statistics; it is a
qualitative illustration, not an estimate of failure prevalence.

TA-STVG and TubeDETR use the HC-STVG2 source-only checkpoints evaluated on
VidSTG. PTD uses the official Qwen3-VL-4B jointly trained reference. The
third row is not another source-only cross-domain result.

## Evidence needed for the requested additions

The exact selected case has no saved Projected interval for any of the three
models. Native prediction JSONs do not contain the necessary raw temporal
heads. In particular, PTD's structured temporal candidates cannot be relabeled
Projected, and TA's Actionness Projection cannot be assigned to other models
without a matched, explicitly defined readout.

The matching old TA input has four proposed expert positions but zero actual
DINO observations and zero anchors. Proposed positions are not detections.
None coincide with the five original displayed frames. Another historical
input with detections has a different query, RGB, frame grid and predictions;
those boxes were rejected as incompatible.

The delivered layout therefore draws no DINO boxes or observation dots and
no Projected interval. All three Projected rows explicitly say that they are
not yet measured. The figure is marked LAYOUT PREVIEW. It cannot support a
claim of expert correction, recovered event coverage or new method efficacy.

The human's outstanding preference is whether the completed follow-up should
be a matched actual-result figure or a clearly labeled illustrative method
diagram. Those are different evidence contracts; the supplied screenshot is
a layout reference, not experimental evidence or authority to fabricate data.

## Verification and privacy

The renderer locks the original case and prior source bindings, checks all
five PNG byte hashes and decoded pixel hashes, and reads original geometry
directly. The latest PDF must be independently rasterized and visually
reviewed before its finite layout completion receipt is written. Earlier
spacing previews remain preserved.

Dataset RGB, exact query, GT coordinates, Native arrays, media-bearing PDF,
PNG and SVG stay local. Public export is limited to rendering code, this
description, source hashes, availability counts and actual visual-review
metadata. Missing measurements are not represented as failed scientific
results, and layout completion is not scientific completion.

This operation runs no model, GPU inference, DINO, training, optimizer or new
GT scoring. It does not read the current unsealed P7-A suite or interrupt its
worker. The production method, research pins, cancelled baseline standby and
the requested P7-A monitor remain unchanged.
