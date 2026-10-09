# Final named backbones, larger text and explicit spatial GT overlays

The latest human edits name the third row PTD / Qwen3-VL, remove the training
subtitle from the graphic itself, enlarge the query and content labels,
and mark every correct spatial GT box clearly. Each of fifteen unchanged
official GT rectangles has a heavier green dashed line with white contrast
stroke and a green GT badge. The coral native prediction rectangles retain
geometry. Prediction/GT event ranges and all six native score labels are
unchanged, on the original common linear scale.

PTD uses the Qwen3-VL-4B backbone and joint VidSTG/HC-STVG supervision. This
training scope remains disclosed in the paper caption and this record.
It is a separately authorized reference, excluded from the strict source-only
cross-domain cohort. The other two rows are TA-STVG and TubeDETR, using
HC2-source-only to VidSTG native outputs. No model is rerun.

The actual private PNG, independent PDF raster and editable SVG have been
reviewed. Dataset RGB, exact query, GT coordinates, raw temporal ranges and
prediction arrays remain local. This public folder contains only code,
factual caption, design/runtime bindings and integrity evidence.

The original post-hoc illustrative case and all negative aggregate findings
are unchanged. Case score readback remains public in
results/stvg_motivation_sketch/2026-10-09. All 512 native cross-domain scalar
rows, registered negative contrasts and bootstrap results remain in
results/stvg_motivation_cross_domain/2026-10-09. The conditional companion in
results/stvg_motivation_filmstrip/2026-10-09 remains labeled post hoc.
This figure illustrates that event coverage does not ensure instance
grounding; it does not establish prevalence, OPD efficacy, expert reliability
or an on-policy causal mechanism.

The independent public primary and conditional scalar check remains:

```bash
python -B scripts/audit_stvg_motivation_filmstrip_public_v7.py
```

Only this finite visual revision closes here. Original P1-P6 actual root,
math/state/dense/view/publication and archive obligations remain active.
