# Temporal and spatial grounding shown as actual filmstrips

The latest human presentation instruction supersedes the v8 column chart:
remove every numeric annotation, and show temporal and spatial grounding as
two image strips. This is a layout revision of the completed v6 source-only
cross-domain native study; it adds no model, expert, optimizer, GT scoring,
threshold, cohort, or P1 payload access.

The left comparison retains the actual GT, TA-STVG and TubeDETR rows. The
right panel uses the same previously selected HC2-source → VidSTG example.
Its upper filmstrip shows uniformly sampled context frames and the two actual
GT/TA-STVG event intervals on a common physical-time extent. No interval is
extended or clipped to make the temporal result look more accurate.
Its lower filmstrip overlays the unchanged GT and TA-STVG rectangles on the
previous uniformly sampled common-event frames. Dashed green is GT; solid
coral is prediction. Full real frames are preserved without cropping or
warping. The exact caption and dataset pixels remain local.

## Suggested paper caption

Cross-domain instance ambiguity. Native predictions from distinct STVG
backbones overlap the annotated event while disagreeing with the referred
instance. The right panel separates event coverage from frame-level spatial
grounding on the same video and query: temporal overlap does not ensure the
correct person is localized. Ground truth is green, and the native prediction
is coral. This is an illustrative case, not an error-frequency estimate.

The quantitative companion is unchanged: all 512 sealed native outputs,
the four registered categories, strict tIoU and fixed-GT-frame spatial
thresholds of .5, all 10000 paired-parent bootstrap results, and the negative
primary contrast are retained in
`results/stvg_motivation_cross_domain/2026-10-09`. The later conditional
spatial-failure description remains explicitly post hoc in
`results/stvg_motivation_filmstrip/2026-10-09`.
Neither the illustration nor its new style establishes spatial-error
dominance across backbones, overall temporal reliability, OPD efficacy,
DINO trustworthiness, or on-policy optimization necessity. Mechanism and
efficacy claims require their authorized original paper experiments.

## Reproduction and publication

`scripts/render_stvg_motivation_visual_grounding_v9.py` prepares only the
same sealed case's context RGB, verifies byte/pixel bindings, and emits PNG,
PDF and editable SVG. The private full and panel-only figures stay local.
The public companion is clearly labeled a layout schematic, contains no
dataset pixels or experimental values, and is not scientific evidence.
Earlier v6/v7/v8 artifacts are preserved. The rendering manifest checks that
no plotted text contains a digit. Root review must inspect both the PNG and
an independent PDF raster before claiming completion.
