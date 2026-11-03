# Reviewer controls: full strict simple combination and random spatial ROI

All four whole-domain tasks sealed, separately officially scored (new arms only), independently cold-array/input/receipt/reused-metric/means audited and ROOT interpreted. Existing results were not rerun. Full HC1902queries/227parents, Vid10303/732parents; region128queries/128distinctparents per domain. No new Source/expert/model/GPU/optimizer, no method or paper change. All12actual stages and actual controller exited0.

## Full-target simple-combination comparison

|Dataset|Configuration|Expert|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---|---|---:|---:|---:|---:|---:|
|hc2|Source|no|40.1415|20.7047|29.0221|8.9380|51.0497|
|hc2|SourceTemporal|no|49.6896|25.5583|39.2219|14.6162|51.0497|
|hc2|SimpleCombination|yes|49.6896|25.1409|37.8023|13.8275|47.2628|
|hc2|InitializedTemporal|yes|49.7645|26.4897|40.0631|15.7729|51.9662|
|hc2|Full|yes|49.7645|27.3490|41.2198|16.8770|52.9428|
|vidstg|Source|no|31.7285|11.1385|12.8312|3.3485|33.0254|
|vidstg|SourceTemporal|no|34.9085|12.2249|15.0053|4.3871|33.0254|
|vidstg|SimpleCombination|yes|34.9085|14.7096|19.8195|8.6965|35.5421|
|vidstg|InitializedTemporal|yes|34.9182|13.9487|18.7227|8.2306|35.0008|
|vidstg|Full|yes|34.9182|14.5582|19.4118|8.4636|35.7858|

SourceTemporal uses native Source boxes/SourceROI appearance with its own refined interval and no expert. SimpleCombination uses the exact same interval with strict raw8expert center/logsize interpolation and nearest-expert tails, no initialized-E tails/transport/DP. InitializedTemporal outputs original corrected E boxes and its own E-ROI refined interval, no spatial propagation/DP. Full uses that same E-ROI interval and the unchanged finalDP path. All expert configurations use the same eight physical observations; SourceTemporal does not consume them. Full-versus-Simple changes both pooling evidence and spatial trajectory; the following fixed-time comparison isolates the latter. Historical hybrid interpolation outputs remain separate.

|Dataset|Paired comparison|Delta tIoU pp [95% CI]|Delta vIoU pp [95% CI]|
|---|---|---|---|
|hc2|Full-SimpleCombination|+0.0750 [-0.0944, +0.2360]|+2.2082 [+1.8782, +2.5532]|
|hc2|SimpleCombination-SourceTemporal|+0.0000 [+0.0000, +0.0000]|-0.4174 [-1.0322, +0.2070]|
|hc2|Full-InitializedTemporal|+0.0000 [+0.0000, +0.0000]|+0.8593 [+0.5024, +1.2267]|
|hc2|InitializedTemporal-SourceTemporal|+0.0750 [-0.0944, +0.2360]|+0.9314 [+0.6214, +1.2433]|
|hc2|SourceTemporal-Source|+9.5481 [+8.5142, +10.6210]|+4.8536 [+4.2750, +5.4555]|
|vidstg|Full-SimpleCombination|+0.0097 [-0.1162, +0.1329]|-0.1514 [-0.3057, -0.0007]|
|vidstg|SimpleCombination-SourceTemporal|+0.0000 [+0.0000, +0.0000]|+2.4847 [+2.0838, +2.9095]|
|vidstg|Full-InitializedTemporal|+0.0000 [+0.0000, +0.0000]|+0.6094 [+0.4717, +0.7552]|
|vidstg|InitializedTemporal-SourceTemporal|+0.0097 [-0.1162, +0.1329]|+1.7238 [+1.3922, +2.0646]|
|vidstg|SourceTemporal-Source|+3.1801 [+2.7308, +3.6508]|+1.0864 [+0.9030, +1.2788]|

## Spatial trajectories at the identical Full interval

|Dataset|Trajectory|tIoU %|vIoU %|
|---|---|---:|---:|

|Dataset|Trajectory|tIoU %|vIoU %|
|---|---|---:|---:|
|hc2|Source|49.7645|25.6042|
|hc2|PureExpertFixed|49.7645|25.1774|
|hc2|InitializedFixed|49.7645|26.4897|
|hc2|Full|49.7645|27.3490|

hc2 Full minus strict pure interpolation at the same event: +2.1717 [+1.8399, +2.5197] pp. All per-query tIoUs are identical. Fixed-time Source is original Source boxes with the expert-derived Full interval, not an expert-free system.


|Dataset|Trajectory|tIoU %|vIoU %|
|---|---|---:|---:|
|vidstg|Source|34.9182|12.2274|
|vidstg|PureExpertFixed|34.9182|14.7044|
|vidstg|InitializedFixed|34.9182|13.9487|
|vidstg|Full|34.9182|14.5582|

vidstg Full minus strict pure interpolation at the same event: -0.1462 [-0.2947, -0.0005] pp. All per-query tIoUs are identical. Fixed-time Source is original Source boxes with the expert-derived Full interval, not an expert-free system.

## Region evidence on unchanged 128-parent panels

All region arms output the same corrected E trajectory, use the same q0/grid/update/readout, and recompute their own event interval. This is an evidence-pooling diagnostic; SourceROI here is not an expert-free complete system.

|Dataset|Pooling region|tIoU %|vIoU %|
|---|---|---:|---:|
|hc2|SourceROI|54.8331|30.4708|
|hc2|CorrectedROI|54.9685|30.4538|
|hc2|GlobalAppearance|54.4798|30.0459|
|hc2|BackgroundAppearance|53.4993|29.1794|
|hc2|ShuffledBoxes|53.5636|29.9100|
|hc2|RandomRegion|52.6581|29.0986|
|vidstg|SourceROI|35.5286|14.9123|
|vidstg|CorrectedROI|35.2024|14.8152|
|vidstg|GlobalAppearance|35.5610|14.9955|
|vidstg|BackgroundAppearance|35.0706|14.6665|
|vidstg|ShuffledBoxes|35.6753|14.9056|
|vidstg|RandomRegion|35.5152|14.8365|

|Dataset|Corrected ROI minus|Delta tIoU pp [95% CI]|Delta vIoU pp [95% CI]|
|---|---|---|---|
|hc2|SourceROI|+0.1355 [-0.4793, +0.7709]|-0.0170 [-0.2328, +0.2176]|
|hc2|GlobalAppearance|+0.4888 [-0.7051, +1.6720]|+0.4079 [-0.2232, +1.0842]|
|hc2|BackgroundAppearance|+1.4693 [-0.1682, +3.2338]|+1.2744 [+0.1910, +2.5230]|
|hc2|ShuffledBoxes|+1.4050 [+0.1741, +2.6608]|+0.5438 [-0.1255, +1.2308]|
|hc2|RandomRegion|+2.3104 [+0.6933, +4.1903]|+1.3552 [+0.2904, +2.6977]|
|vidstg|SourceROI|-0.3262 [-1.0259, +0.4617]|-0.0971 [-0.3551, +0.1496]|
|vidstg|GlobalAppearance|-0.3586 [-1.3341, +0.6852]|-0.1803 [-0.5289, +0.1213]|
|vidstg|BackgroundAppearance|+0.1319 [-1.0729, +1.3856]|+0.1487 [-0.3567, +0.6177]|
|vidstg|ShuffledBoxes|-0.4728 [-1.1485, +0.1941]|-0.0904 [-0.3816, +0.1973]|
|vidstg|RandomRegion|-0.3127 [-1.4473, +0.8897]|-0.0213 [-0.5088, +0.4830]|

RandomRegion translates the exact expanded1.5ROI fractional-weight support template to a SHA-selected feasible valid-grid placement per physical frame (seed20261104). Area, shape and weight multiset remain bitwise identical; overlaps, identity and fallback cases remain in the denominator. Manifest committed before labels. This controls region support size without asserting that random regions never contain the object.

hc2: 8320frames, 1576identity placements, 0unchanged-feature queries, 22whole-valid target fallback frames; mean support overlap 0.524866.

vidstg: 13502frames, 3507identity placements, 7unchanged-feature queries, 386whole-valid target fallback frames; mean support overlap 0.523905.

## Interpretation and limits

HC corrected ROI improves tIoU over random regions, while its global comparison crosses zero; Vid corrected versus every alternative regional evidence t/v CI crosses zero. The intervention does not establish two-domain target-region necessity or language/action semantic recognition. A CI crossing zero does not prove equivalence. Existing AUROC is descriptive evidence, not causal proof of object specificity. No new parameter/seed choice follows these results.

The unchanged Full still permits corrected E versus rawexpert candidates at observed frames; it does not satisfy the intended hard-anchor contract. These results describe the existing soft-anchor implementation, not a repair or validation of a hard-anchor method. All observed-frame metrics include GT-supported frames outside predicted intervals; they are not all tube-inside damage. Source-to-SourceTemporal native gains combine legal jointMAP readout change and visual evidence; existing matched SourceJointMAP controls are exported separately.

Pointwise10000paired-parent CIs, query-equal means, exposed cohorts, no familywise/unknown-test/equivalence claims. GT only after whole seals. Short<=5s/long groups, all signed harms, severe t/v union once strictly over5pp, rescue counts, observed/unobserved and boundary diagnostics are retained in each ROOT report and all-signed TSV. Inherited Source/expert acquisition remains nonzero. New prediction/score/audit walls are cached CPU experimental processing, not deployment latency; stage windows and nested seal clocks must not be added as separate costs. Vid construction revalidates the full manifest repeatedly in its frozen entry; this affects CPU experiment wall, not predictions.
