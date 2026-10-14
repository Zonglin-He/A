# Filtered instance-pool geometric coverage

This post-seal diagnostic measures how well the available filtered YOLOv12-X + SAM2.1 mask pool covers official ground-truth boxes geometrically. For each officially annotated physical frame, it takes the maximum axis-aligned box IoU between the GT box and any nonempty filtered instance mask box. A GT frame outside the canonical tracker clock or with no nonempty candidate mask contributes zero.

The denominator is every physical frame returned by the official annotation reader for each locked case. The report includes both the count and rate at IoU ≥ 0.25 and IoU ≥ 0.50, plus the frame-weighted mean of per-frame maximum IoU. Inclusive mask pixels are converted to continuous half-open boxes with the required +1 right/bottom edge before pixel geometry is compared. No clipping or GT-based candidate selection is applied.

All-category tracking used an empty metadata mapping, so category labels did not restrict the candidate pool. This is geometric availability of the filtered tracker output. It is **not raw YOLO detector recall**, does not assess semantic identity correctness, does not fit an oracle, and does not change any locked reference or prediction.

## Per-case results

| Case | Domain | GT frames | In clock | Outside clock | Empty pool | Mean max IoU | ≥0.25 | ≥0.50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HC01 | hc2 | 208 | 208 | 0 | 0 | 0.7538 | 0.995 (207) | 0.885 (184) |
| HC02 | hc2 | 88 | 88 | 0 | 0 | 0.7710 | 1.000 (88) | 1.000 (88) |
| HC03 | hc2 | 201 | 201 | 0 | 0 | 0.8077 | 1.000 (201) | 0.990 (199) |
| HC04 | hc2 | 232 | 232 | 0 | 0 | 0.4871 | 0.560 (130) | 0.500 (116) |
| V01 | vidstg | 110 | 110 | 0 | 0 | 0.9170 | 1.000 (110) | 1.000 (110) |
| V02 | vidstg | 1591 | 1591 | 0 | 0 | 0.9299 | 1.000 (1591) | 1.000 (1591) |
| V03 | vidstg | 123 | 123 | 0 | 0 | 0.8698 | 1.000 (123) | 1.000 (123) |
| V04 | vidstg | 36 | 36 | 0 | 0 | 0.7718 | 1.000 (36) | 1.000 (36) |

## Cohort summary

| Group | Cases | GT frames | Mean max IoU | ≥0.25 | ≥0.50 |
|---|---:|---:|---:|---:|---:|
| All | 8 | 2589 | 0.8556 | 0.960 (2486) | 0.945 (2447) |
| hc2 | 4 | 729 | 0.6859 | 0.859 (626) | 0.805 (587) |
| vidstg | 4 | 1860 | 0.9221 | 1.000 (1860) | 1.000 (1860) |

Machine-readable anonymous rows: `POOL_CANDIDATE_COVERAGE_POSTSEAL.json` (SHA-256 `dde82ad2696d6391e21bb0928e2d2b2b16a29dcca3f931b30a0e7528469d37e1`).
