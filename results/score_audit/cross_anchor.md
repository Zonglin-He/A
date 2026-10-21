# Strict proposal-holdout cross-anchor scoring probe

All 64 exposed HC development parents × eight leave-one-observation-out folds are complete. No adaptation was executed. Each fold uses raw Source, seven-row query selection and seven teacher masks/boxes for construction. The held teacher mask enters only after every proposal and predicted support map is sealed; GT enters only after all scoring readouts are sealed. The held node itself always remains Source to exclude self-matching.

This is cached teacher-output withholding, not independent seven-frame teacher inference: original ReferDINO outputs contain eight-frame tracking and temporal context. This is a scoring probe using seven observations for construction plus the eighth as feedback, not an unseen-teacher test or an OPD result. The new bank uses Source rather than ERef; compare its two matched scores to each other, not to old ERef-based v0 as a single-factor method gain. No memory or link term is used in either matched probe.

| Score-only readout | vIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|
| Source | 29.1530 | 49.4198 | 47.6990 |
| Projection | 28.9530 | 47.9896 | 47.4083 |
| CrossAnchor | 29.0158 | 50.8822 | 47.2140 |

| Compared with Projection | Delta pp | 95% parent interval pp |
|---|---:|---:|
| v | +0.0627 | [-0.6749234152697069, 0.8779085839305132] |
| unobserved_actual_sIoU | -0.1943 | [-1.2382931920740963, 0.8360800276962385] |

| Score | Accepted fold-node replacements with GT | Harm fraction | Mean replacement IoU delta pp |
|---|---:|---:|---:|
| Projection | 6579 | 54.902% | -2.1852 |
| CrossAnchor | 4629 | 55.325% | -2.4813 |

Metrics are averaged within parent over all eight folds before parent bootstrap (10,000 draws). Repeated fold-nodes are explicitly counted as such; no claim of 512 independent videos is made. All positive/negative/unchanged anonymous query outcomes, fixed original-eight observation/unobserved partitions, near/middle/far bins, candidate ranking and fingerprints are in cross_anchor.json. GT never chooses a query-specific parameter, stopping step or output. Full-frame repeated-feature aliasing can still reward an oversized box; synthetic counterexamples are retained.
