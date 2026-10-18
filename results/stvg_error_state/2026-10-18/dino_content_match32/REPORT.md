# Same student reference, different content initialization: original32

Only the appended learned content vector changes. Same references, native900 context and geometry-nearest attention exclusion, target tokens, .35 output gate, original four frames and frozen detector. Native target-high is the first argmax of the SAME native forward target-token score. This is not an ROI content initializer or a calibrated identity posterior.

Original exposed HC16/Vid16 development. Exact previous fixed24 actions are reused; invalid whole-trajectory references stay no-op. Four unsupported Vid queries remain in the denominator. All32 outputs sealed before new GT. No OPD/Student update in this validation.

| Domain | Content | Admitted / returned anchors | Queries with feedback /16 | Known-anchor gated correction pp [95% query CI] | Raw correction pp |
|---|---|---:|---:|---:|---:|
| hc2 | nearest | 9/1224 | 5/16 | +0.030505 [+0.007572, +0.060493] | -3.629758 |
| hc2 | target_high | 9/1224 | 5/16 | +0.030505 [+0.007572, +0.060493] | +0.425133 |
| vidstg | nearest | 121/1104 | 12/16 | +0.494660 [+0.087727, +1.077227] | -7.360113 |
| vidstg | target_high | 121/1104 | 12/16 | +0.494660 [+0.087727, +1.077227] | +1.639392 |

Correction quality is mean conditional output IoU minus mean identical input-reference IoU at known GT planned positions. Below-threshold or invalid candidates contribute zero correction; raw correction additionally shows returned boxes without threshold. Missing GT is unknown and does not count as failure. Each query contributes one mean, then domain means/paired query bootstrap; thousands of reference pairs are not thousands of independent samples. Unsupported queries contribute zero movement. This is a sparse-anchor interface test, not complete Student tube performance or an OPD score.

| Domain | Target-high minus nearest gated quality pp [95% CI] | Prelocked gate |
|---|---:|---|
| hc2 | +0.000000 [+0.000000, +0.000000] | False |
| vidstg | +0.000000 [+0.000000, +0.000000] | False |

Decision: `stop_content_variant_before_OPD`. Coverage must improve in each domain AND admitted-policy quality must improve over unchanged input and nearest in each domain before bounded repaired OPD. Confidence intervals are reported, not used as an arbitrary significance gate. No new content rule, threshold tuning or expansion is authorized by a negative result.

The nearest arm is checked against old fixed24 on exactly the same actions, scores, boxes and admission. All non-content decoder tensors, reference and attention-mask hashes match within each pair; cloned native-query box and target-score qualifications cover one original query per domain. Copying a high-score native content can still read a reference that points at a wrong object; a score gain alone is not a localization gain. Geometric IoU does not prove identity.

Costs: see ACTUAL_COSTS.json. Existing Source and E generation excluded. Source captures0, pretrained updates0, new training0. Both conditions share one image/text encoding per anchor. Raw media, Query/parent IDs, coordinates, GT and weights stay private. Fixed actual-pixel panels only verify alignment, with no new semantic labels. CURRENT and paused historical queues unchanged.
