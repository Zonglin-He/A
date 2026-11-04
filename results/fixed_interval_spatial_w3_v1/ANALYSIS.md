# Full-list fixed-interval spatial comparison

Every arm uses the original unchanged Full interval and the exact same eight raw expert observations, physical native grid and query list inside each benchmark. HC:1902 queries/227 parent sources; VidSTG:10303/732. All per-query tIoU values are exactly equal across seven arms. Existing five-arm predictions/official scores are reused unchanged; only PastOnlyFixed and OneWayMatchFixed are new cached-geometry final-DP predictions.

|Dataset|Spatial strategy|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---|---:|---:|---:|---:|---:|
|hc2|PureLog|49.764548|25.177364|38.012618|13.932702|47.262804|
|hc2|PureLinear|49.764548|25.592737|38.433228|14.511041|48.294726|
|hc2|SharedCorrection|49.764548|26.489749|40.063091|15.772871|51.966189|
|hc2|Independent|49.764548|26.281532|39.484753|15.036803|50.790366|
|hc2|PastOnlyFixed|49.764548|26.872766|40.588854|15.562566|52.082326|
|hc2|OneWayMatchFixed|49.764548|26.252758|40.063091|14.353312|51.106223|
|hc2|Full|49.764548|27.349044|41.219769|16.876972|52.942787|
|vidstg|PureLog|34.918192|14.704394|19.722411|8.638261|35.542132|
|vidstg|PureLinear|34.918192|14.804402|19.858294|8.686790|36.006945|
|vidstg|SharedCorrection|34.918192|13.948734|18.722702|8.230612|35.000813|
|vidstg|Independent|34.918192|13.997841|18.615937|7.551199|34.533203|
|vidstg|PastOnlyFixed|34.918192|14.345138|18.945938|8.317966|35.366688|
|vidstg|OneWayMatchFixed|34.918192|14.066686|18.431525|7.920023|34.784561|
|vidstg|Full|34.918192|14.558151|19.411822|8.463554|35.785762|

PureLog retains the original center/log-size interpolation; PureLinear is linear xyxy-corners. Both use only raw8boxes, keep raw expert boxes exactly at observations and nearest endpoint boxes outside the anchor span. Both were already audited before this task; no target-dependent choice or relabeling. SharedCorrection directly returns the exact shared-corrected Source trajectory E, without candidate transport or finalDP. Its corrected-ROI temporal interval is exactly original Full.

|Dataset|Controlled difference|Delta vIoU pp [95% CI]|Delta unobserved IoU pp [95% CI]|
|---|---|---|---|
|hc2|Full-PureLog|+2.171679 [+1.839911, +2.519693]|+5.679982 [+4.841201, +6.569601]|
|hc2|Full-PureLinear|+1.756307 [+1.445449, +2.071917]|+4.648060 [+3.841902, +5.493081]|
|hc2|Full-SharedCorrection|+0.859295 [+0.502377, +1.226670]|+0.976598 [+0.475310, +1.493149]|
|hc2|Full-Independent|+1.067512 [+0.958161, +1.183652]|+2.152421 [+1.956221, +2.359637]|
|hc2|Full-PastOnlyFixed|+0.476278 [+0.323659, +0.629222]|+0.860461 [+0.603687, +1.127587]|
|hc2|Full-OneWayMatchFixed|+1.096286 [+0.899990, +1.293941]|+1.836564 [+1.509616, +2.173054]|
|vidstg|Full-PureLog|-0.146243 [-0.294717, -0.000487]|+0.243630 [-0.305862, +0.790250]|
|vidstg|Full-PureLinear|-0.246251 [-0.392809, -0.101615]|-0.221184 [-0.753063, +0.305061]|
|vidstg|Full-SharedCorrection|+0.609417 [+0.471700, +0.755210]|+0.784949 [+0.541043, +1.041068]|
|vidstg|Full-Independent|+0.560311 [+0.512204, +0.612344]|+1.252558 [+1.143205, +1.366358]|
|vidstg|Full-PastOnlyFixed|+0.213013 [+0.166350, +0.260819]|+0.419073 [+0.314757, +0.529399]|
|vidstg|Full-OneWayMatchFixed|+0.491465 [+0.421118, +0.566518]|+1.001200 [+0.852979, +1.160664]|

Full−SharedCorrection measures the added candidate propagation and final path selection together. Full−Independent isolates final global path selection using the same candidate bank/validity/unary, with no temporal or candidate-generation difference. Independent uses unary argmax and no adjacent links; the difference includes adjacent-link scoring plus global selection, not a solver-only comparison at an identical objective.

PastOnlyFixed removes all future-anchor support and recomputes unary scores in the saved geometry, then re-solves DP at Full interval. It retains the past/own-observed direction and the same E initialization computed from all eight observations (including future ones); only future transport candidates/support are removed. This is an offline propagation-anchor availability control, not a causal online method or removal of all future information. OneWayMatchFixed keeps both anchors but uses saved one-way feature-correspondence fits instead of reciprocal products, preserving original source features/seeds/fit constraints, with DP at the identical interval. These distinguish two different meanings of bidirectional inference. They are ablations, not new competitive learned models or a broad TTA ranking.

|Dataset|Full minus control|v gains|ties|harms|severe v >5pp|rescues >5pp|loss ≥10pp|
|---|---|---:|---:|---:|---:|---:|---:|
|hc2|Full-PureLog|1071|187|644|129|424|33|
|hc2|Full-PureLinear|1045|187|670|135|382|36|
|hc2|Full-SharedCorrection|822|305|775|198|260|96|
|hc2|Full-Independent|1140|332|430|9|106|0|
|hc2|Full-PastOnlyFixed|820|451|631|62|133|19|
|hc2|Full-OneWayMatchFixed|996|330|576|61|196|13|
|vidstg|Full-PureLog|3421|2838|4044|714|676|252|
|vidstg|Full-PureLinear|3348|2832|4123|734|625|265|
|vidstg|Full-SharedCorrection|3636|3566|3101|375|809|143|
|vidstg|Full-Independent|4129|4620|1554|19|303|2|
|vidstg|Full-PastOnlyFixed|2832|5025|2446|83|242|10|
|vidstg|Full-OneWayMatchFixed|3483|4275|2545|126|518|32|

## Evidence boundaries

Same expert INPUT does not mean fixed identical observed OUTPUT. Unchanged Full/E/Independent/directional ablations permit soft selection of E versus the raw expert, while pure interpolation hard-retains raw expert boxes. The previously identified intended hard-anchor contract remains unsatisfied and is not repaired here. Thus path-quality differences include observed-node selection; these tables do not isolate only unobserved propagation or certify the intended hard-anchor method. Both positive and negative full-dataset results are retained.

All GT support/clipping/half-open interval rules are the original official benchmark rules. Unobserved IoU is queryequal over native GT-supported non-observed frames, not dense vIoU and not independent-frame confidence. Empty support remains null. No source, expert, grid acquisition, RGB decoding, optimization or temporal posterior update is executed. Full method/profile/predictions and prior scores unchanged; no manuscript edits, target tuning or old queued jobs.

Tent/SAR remain the previous temporal-decoder LayerNorm adapters with very few interval changes; this spatial experiment does not prove superiority over broad test-time refinement strategies. Queryequal 10000 paired-parent bootstrap seed20261022; short<=5s/long and all signed severe/rescue tails retained. CIs are pointwise on already exposed full benchmarks, not familywise/equivalence/unseen confirmation. Cached CPU replay/scoring walls include IO/audit and are not deployment latency; earlier Source/expert costs remain nonzero.

New official checks:24410; max official discrepancy:2.88657986402541e-15. Independent cold NumPy DP array fields:61025; normalized dense scalar checks:216048; max scalar discrepancy:1.4432899320127e-15. ROOT independently recomputed 180paired/group metric-CIs. All6required successful childstages and independent runtime controller exited0; original controller exited1 on scoring import only, predictions retained; both failed score attempts/costs and original code/logs preserved. Exact-original-coordinate-function import compatibility only, no scientific or score formula change. Private source geometry/feature/GT/interval/box/caption/media/weights remain private; public only source/protocol/scalars/hashes.
