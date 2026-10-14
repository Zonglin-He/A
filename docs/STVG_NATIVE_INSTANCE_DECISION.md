# Native instance spatial adaptation: finite development decision

The eight-source experiment completed its measurements. This is an exposed development panel, not independent confirmation. Temporal Caption–Query remains fixed; the spatial references came from all-category YOLOv12-X/SAM2 tracks selected by frozen STVG-R1. No GT Oracle fit was performed. Annotation use began after prediction sealing, for evaluation and candidate-pool coverage diagnostics only.

T + IoU OPD improved mean vIoU over T by 1.150 percentage points in HC and 3.236 in VidSTG. HC contained one positive, two negative and one neutral source. VidSTG's entire improvement came from V04; the other three sources did not update. HC dense sIoU on unsupervised frames decreased by 3.527 points. The fixed mask-continuity reliability weighting did not establish additional benefit. Neither a robust spatial method nor independent-source expansion is established by these means.

## Evidence bottlenecks

The filtered pool contains boxes overlapping the annotated target at IoU >= 0.5 on 587/729 HC frames and 1860/1860 VidSTG frames. This is post-seal geometric pool coverage, not raw detector recall, continuous correct identity, R1 selection accuracy, or a GT-selected fitting reference. It cannot be numerically compared with the earlier short-window oracle without matching support.

Two VidSTG responses returned an object ID but were rejected by the preregistered parser because their temporal interval exceeded the clip clock. The current implementation couples WHO acceptance to native time validity, even though Student temporal support comes from the fixed QA branch. These are invalid complete responses, not proof that the returned IDs identify the wrong object. They remain unknown in the direct-expert metric and exact no-update in Student arms. The third no-update case had a valid response but no selected-object mask at any of the four fixed QA centers. No response was repaired, clipped, reprompted or admitted after looking at GT.

A possible subsequent protocol would assess identity validity independently from native temporal output while preserving QA support. That would be a new, prospectively locked experiment; it was not executed here and no gain from it is claimed. Likewise, the current continuity weight can penalize genuine motion and must not be described as calibrated reliability.

## Decision

Retain both research branches and the existing Temporal configuration. Do not expand this fixed spatial recipe to 32 independent sources per domain or full-data experiments, and do not change CURRENT. Preserve all positive, negative, neutral and unknown outcomes. This closes the authorized eight-source measurement and evidence package; it does not establish a final deployable method or close future research.

The complete [reviewed report](STVG_NATIVE_INSTANCE_V1_REVIEWED.md) retains all 72 source-arm rows, 90 summary cells, 80 paired contrasts and measured cost boundaries. The [pool appendix](STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md) retains every source's coverage diagnostic.
