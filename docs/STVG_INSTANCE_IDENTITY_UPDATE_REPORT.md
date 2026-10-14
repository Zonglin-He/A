# WHO/WHEN separation and validated Spatial OPD: eight exposed development sources

Measured finite experiment, 2026-10-14. Both branches retained. No CURRENT promotion or independent/full-data expansion.

All eight method outputs were sealed before this round's annotation evaluation. Historical development exposure is disclosed. No new YOLO, SAM2, R1 or DINO execution; no GT Oracle fit. Existing teacher selections, masks, QA intervals, source checkpoints, four centers and the original five-step/lr/sigma/tau configuration were fixed.

## Matched full-tube results

Equal source means, four sources per direction. Values are percentages. Teacher geometry uses the identical QA interval and full Student grid; absent masks fall back to the exact Frozen box without interpolation.

| Method | HC vIoU | VidSTG vIoU | HC dense sIoU | VidSTG dense sIoU |
|---|---:|---:|---:|---:|
| Frozen | 22.210 | 14.490 | 47.895 | 48.458 |
| Temporal QA | 31.286 | 22.442 | 47.895 | 48.458 |
| QA + strict-old IoU OPD | 32.436 | 25.678 | 44.358 | 60.679 |
| QA + decoupled IoU OPD | 32.436 | 31.899 | 44.358 | 69.942 |
| QA + presence admission | 32.436 | 31.899 | 44.358 | 69.942 |
| QA + loss-weighted OPD | 32.436 | 31.899 | 44.358 | 69.942 |
| QA + validated OPD | 33.601 | 31.899 | 47.363 | 69.942 |
| QA + strict geometry | 36.410 | 26.187 | 63.631 | 62.210 |
| QA + decoupled geometry | 36.410 | 34.788 | 63.631 | 76.713 |

## Controlled findings

WHO/WHEN separation restored four anchors each for V01 and V02, despite their preserved out-of-clock native time status. The selected IDs were present in the sealed mask inventory and QA interval. This is structural admissibility, not an independent semantic-correctness guarantee. V03 still has no mask at its fixed centers and remains exact no-update. Total anchors changed from 14 to 22.

The saved HC01 ordinary OPD trajectory fit teacher anchors better for three steps (mean IoU 0.394574 -> 0.467715 -> 0.505522 -> 0.551928), then collapsed to 0.152761 at step four. Three anchors became zero-IoU. The new no-label rule rejected that fourth proposal and restored parameters AND Adam, leaving three accepted steps. Final full-model re-decoding exactly matched the accepted cached decoder.

The rejection rule tests weighted teacher IoU non-decrease (tolerance 1e-7), stops on first rejected proposal, and never reads GT or retries a different learning rate. It uses the same imperfect supervision and cannot ensure referent correctness or unseen-frame improvement. HC04 still improves reference fit while harming task performance.

The loss-level weight is now mean(r * per-anchor original KL); reward remains IoU with the original softmax temperature. Here 21 of 22 anchors have r=1, the remaining r=0.8 anchor has flat/no-information IoU reward. Presence admission excludes no additional anchors. Therefore this panel cannot establish an advantage for continuous reliability weighting; its results are exactly equal to ordinary decoupled OPD.

| Case | Strict OPD - QA (pp) | Decoupled OPD - QA (pp) | Validated OPD - QA (pp) | Geometry - QA (pp) |
|---|---:|---:|---:|---:|
| HC01 | -3.930 | -3.930 | 0.730 | 3.491 |
| HC02 | 10.061 | 10.061 | 10.061 | 11.809 |
| HC03 | 0.000 | 0.000 | 0.000 | 0.000 |
| HC04 | -1.531 | -1.531 | -1.531 | 5.197 |
| V01 | 0.000 | 18.618 | 18.618 | 20.176 |
| V02 | 0.000 | 6.265 | 6.265 | 14.232 |
| V03 | 0.000 | 0.000 | 0.000 | 0.000 |
| V04 | 12.945 | 12.945 | 12.945 | 14.979 |

## Unsupervised support and uncertainty

Supervised positions are the fixed raw-decoupled reference anchors for every comparison; unsupervised support includes all other known GT physical frames, including off-grid frames. Missing GT support stays unknown rather than zero. These source-mean metrics are distinct from sparse anchor agreement.

- hc2: validated OPD - QA +2.315 pp; 2 positive / 1 negative / 1 neutral. Paired source bootstrap 95% [-0.965, +7.546] pp (n=4). Unsupervised dense sIoU: QA 47.851%, validated 47.205%.
- vidstg: validated OPD - QA +9.457 pp; 3 positive / 0 negative / 1 neutral. Paired source bootstrap 95% [+3.133, +15.782] pp (n=4). Unsupervised dense sIoU: QA 48.464%, validated 69.892%.

Geometry-only remains ahead of validated Student in both domains: this round does not establish adaptation superiority over direct expert geometry. Dense masks are used at available grid frames by the geometry-only arm, whereas Student updates use only the fixed sparse anchors; the comparison measures what sparse adaptation retains, not equal spatial-supervision density.

## Execution and decision

Actual new Student fit paths: 16; Adam proposals: 69; accepted: 68; rejected: 1; full-model final re-decodes: 16; source captures: 8. Two paths have flat reward and zero optimizer updates. Identical old/control arms were reused exactly and do not count as new fits. All eight workers exited normally under the own-unit resource guard.

No new tracking/teacher calls does not imply zero deployment expert cost. Prior video-shared YOLO/SAM2 and per-query R1 costs remain real; this round reports only incremental source capture and Student fit/re-decode cost.

Decision: the finite diagnostic is complete. Identity/time separation and one reference-agreement rejection rule have useful development evidence. HC04 negative transfer, HC unsupervised degradation, and the Teacher-geometry gap remain. Keep both branches as research candidates; do not promote CURRENT, claim stable generalization, or automatically launch independent32/full-data from these eight exposed sources. No further tuning was performed after evaluation.

Full anonymous per-case metrics, neutral/unknown support, all declared arm summaries, paired contrasts, saved-step diagnostics, actual proposal/acceptance records and source bindings accompany this report.
