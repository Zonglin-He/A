# Four actual failure visualizations

Four deterministic post-hoc negative extremes from previously whole-sealed, independently officially scored and all-array audited predictions. These examples explain real failure modes; they do not estimate failure prevalence. All original prediction arrays, expert observations, current method and numerical parameters are unchanged.

| Case | Domain | Comparison (Ours minus baseline) | tIoU delta (pp) | vIoU delta (pp) |
|---|---|---|---:|---:|
| 1 Temporal event | HC | Native Source | -69.594595 | -28.077164 |
| 2 Unobserved spatial trajectory | Vid | Native Source, exact same interval | 0 | -6.212772 |
| 3 Final path decoding | Vid | Same-candidate independent, same interval | 0 | -10.321734 |
| 4 Temporal appearance ROI | Vid | Source-box temporal appearance, full main spatial geometry | -67.023814 | -6.651527 |

Case 1 selects the wrong early event prefix. Its native Source comparison includes a temporal readout change and is not isolated posterior attribution. Case 2 has a -77.018791pp fixed-support unobserved sIoU change; two displayed predictions are outside the image and are labeled explicitly without clipping or changing coordinates. Case 3 illustrates the final path choosing another person despite a better independent candidate on the selected frames. Case 4 expands the temporal interval too early; the same spatial candidate construction and DP parameters are retained, each pooling arm using its own prototype, posterior, legal MAP interval and final DP. Neither spatial trajectory in that example is claimed fully correct.

Each figure contains five actual native sampled RGB frames, including two observed and three GT-supported unobserved positions, four saved trajectories, actual GT overlays only where annotations exist, event timelines and all eight expert-position markers. Frame selection uses offline GT to expose the specified failures, never for online inference. Gray dashed boxes are inactive outside that method's event, rather than missing predictions. Green is annotated GT and red is Ours. Original same-eight center/logsize interpolation is reproduced exactly. Full and independent arrays are cold-replayed exactly; case 4 additionally cold-replays Source-ROI prototypes/posterior/MAP/final-DP arrays.

The original renderer produced cramped layouts. Independent display_v2 changes layout only and display_v3 adds the outside-image notice. Original code, plans, images and receipts remain preserved. All three versions have identical selected cases/frames/RGB hashes/prediction hashes/signed metrics. Root actually viewed all four final 3360x2100 PNGs and checked labels, timelines, missing GT, inactive predictions and off-canvas outputs. The authoritative scalar/hash inspection is display_v3/visual_inspection.json; earlier raw render receipts intentionally retain their pre-inspection status.

All four PNG/PDF pairs remain private local artifacts. Public evidence consists only of code, protocols, anonymous scalar metrics, hashes and inspection receipts. The earlier authorization for three existing positive mechanism figures is not silently extended to these new RGB/caption/GT-overlay failure figures. No new model, Source forward, expert, optimizer or official GT scoring was performed. EdgeTAM has no actual propagated predictions yet and is not fabricated as a row. Render elapsed times are display work, not model deployment measurements. Corruption remains user-paused; other ongoing experiments are unaffected.
