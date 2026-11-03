# Pure eight-expert interpolation: corrected baseline definition

The user requested pure expert interpolation without additions. The historical implementation initialized every frame with shared-corrected E, overwriting only between the first/last valid expert anchors. Therefore the old full-domain gaps HC+2.134753pp and Vid-.106955pp concern an interpolation/E-tail HYBRID. They must not be renamed pure-expert results. Same code in both domains does not imply equal amounts of E-tail use. All old code/predictions/proofs remain immutable.

The corrected interpolator receives ONLY frame times, the original eight observation positions and their eight expert boxes. It linearly interpolates centers and log sizes between consecutive anchors, holds the nearest endpoint expert box outside anchor coverage, and restores every expert anchor box exactly. No Source/E correction, feature, confidence, temporal update, extra expert, model or spatial DP enters this spatial estimator. There is no invalid-box fallback; all256 cached cases have eight valid boxes.

All methods are scored under the SAME previously audited Current event interval, with exactly equal per-query tIoU. This common evaluation condition is not a temporal-refinement module inside interpolation and is not a standalone native-Source-interval deployment comparison.

Fixed original128 queries/128distinctparents each domain, previously exposed panel, no outcome-based sample/endpoint choice. No full-dataset expansion, no GPU/model/Source/expert/optimizer/media decode or manuscript change. Only the pure baseline is newly officially scored after a whole128 prediction seal; Source-matched-time/old-hybrid/Current scores are exactly reused. Separate cold-array/source SHA/means audits passed.

|Domain|Strategy|tIoU %|vIoU %|
|---|---|---:|---:|
|hc2|OriginalHybrid|54.968548|29.385144|
|hc2|PureExpert|54.968548|29.346216|
|hc2|Current|54.968548|31.791298|
|vidstg|OriginalHybrid|35.202435|14.865927|
|vidstg|PureExpert|35.202435|14.912114|
|vidstg|Current|35.202435|15.339259|

|Domain|Comparison|Delta v pp [95% CI]|Improve/harm/tie|Severe loss>5pp /rescue>5pp|
|---|---|---|---|---|
|hc2|Current-PureExpert|+2.445082 [+1.434673, +3.499023]|71/44/13|6/32|
|hc2|Current-OriginalHybrid|+2.406154 [+1.378931, +3.477400]|71/44/13|7/32|
|hc2|PureExpert-OriginalHybrid|-0.038928 [-0.200367, +0.062282]|9/5/114|1/0|
|vidstg|Current-PureExpert|+0.427145 [-0.420869, +1.374636]|50/47/31|6/10|
|vidstg|Current-OriginalHybrid|+0.473332 [-0.345606, +1.407154]|50/46/32|5/9|
|vidstg|PureExpert-OriginalHybrid|+0.046187 [-0.032305, +0.169036]|9/7/112|0/1|

|Domain|Duration|Queries|Current minus pure interpolation v pp [95% CI]|
|---|---|---:|---|
|hc2|short_le5s|42|+1.881121 [+0.228022, +3.824590]|
|hc2|long_gt5s|86|+2.720505 [+1.519033, +4.026621]|
|vidstg|short_le5s|78|+0.185613 [-0.475009, +0.782640]|
|vidstg|long_gt5s|50|+0.803935 [-1.119644, +2.988197]|

All per-query signed outcomes and recall/observed/unobserved/boundary metrics remain in report/TSVs. Query-equal10000pairedparent bootstrap pointwise95% CIs, not familywise or unknown validation. Domain sample-size differences in old full reports were not themselves a within-query fairness violation; the actual issue was the added E-tail baseline dependency. Amount of E-tail coverage alone does not establish its score effect. An improvement or decline after removing E must be measured, not presumed. Pure endpoint holding is one fixed transparent convention, no endpoint-rule sweep. Small interior differences from restoring raw anchor boxes exactly are floating log-exp endpoint roundoff and disclosed separately.

Only interpolation-dependent conclusions require qualification/replacement: matched-time interpolation gaps, interpolation+SourceTR combined baselines, and distance/quality/dense-attribution comparisons against the historical hybrid. The Source-only temporal experiment, fixed-E temporal ROI comparison, same-candidate finalDP contrast, TTA diagnostics and non-interpolation gamma/corruption analyses are not numerically recomputed or invalidated by this baseline-definition correction.

|Domain|Queries with historical E-tail inside common event|E-tail sampled nodes inside common event|
|---|---:|---:|
|hc2|32|126|
|vidstg|35|296|

Original all-paid outputs and older positive/negative results are preserved. Cached CPU stage wall/IO/hash costs are not deployment latency; original expert/Source acquisition costs are not zero. Media/captions/GTcoordinates/features/masks/weights stay private.
