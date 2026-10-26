# Complete full-tube controls: vidstg

10303 complete queries, 732 parent videos. Predictions sealed before independent official CPU scoring; independent simple-fusion cold rederivation, all cached-reference array/SHA/provenance and mean audit passed. No new model/Source/expert/optimizer/GPU. Original current method and all older frozen studies remain unchanged.

| Configuration | tIoU % | vIoU % | sIoU % | R.3 % | R.5 % |
|---|---:|---:|---:|---:|---:|
|Original Source|31.728454|11.138484|32.729055|12.831214|3.348539|
|Eight-frame replacement + Source time|31.728454|11.714150|33.882320|13.559158|3.717364|
|Shared correction + Source time|31.728454|12.631858|34.897866|16.286518|6.638843|
|Same-eight interpolation + Source time|31.728454|13.507401|36.072277|17.082403|7.240610|
|Full spatial DP + Source time; temporal refinement entirely disabled|31.728454|13.261107|36.045165|16.995050|6.978550|
|Current full method|34.918192|14.558151|36.090153|19.411822|8.463554|

The first ablation bypasses all temporal refinement/readout replacement and uses original Source native per-offset MAP/hull time, uses the actual final interval-conditioned DP from the previously whole-scored/all-array-audited strict control, with the same MAIN candidate geometry, appearance/link scores and coefficients. Event-dependent eligibility and original normalization recompute under that interval. HC1902 and the preserved first2754 Vid queries additionally reexecuted fresh DP; remaining Vid fixed-interval DP and Full arrays are byte-exact audited-cache reuse, not newly rerun DP. All queries independently recompose simple fusions and all FOUR controls are newly officially scored. NoTemporalInference arrays and scalar metrics match the previous strict control exactly. It is not the NoVisualUpdate / q0 joint-MAP row.

The three simple fusions use original Source time independently of Ours. Replacement changes exactly eight observed boxes and keeps every unobserved Source box. Shared correction uses the saved E trajectory. Interpolation uses the original center/logsize rule and E outside anchor coverage. No simple method receives our posterior or trajectory DP. All four controls must therefore have exactly the same tIoU as original Source. Their vIoU/R metrics are independently scored from the newly composed full tubes, not transplanted component-table values.

Historical cache naming: the old parent Source_interval and OursS use the historical added MBR readout. These interval fields are deliberately excluded; the original Source interval is verified against the corrected current/structural audits and actual saved native logits in the four-case bridge. No legacy output is silently relabeled.

## Complete paired outcomes

| Comparison | delta tIoU pp [95% CI] | delta vIoU pp [95% CI] | severe t/v/union | rescues |
|---|---:|---:|---:|---:|
|FullMAP-Source|+3.189738 [+2.753594, +3.658697]|+3.419668 [+3.009582, +3.851146]|1562/1222/2294|4460|
|FullMAP-EightReplacementNative|+3.189738 [+2.753594, +3.658697]|+2.844001 [+2.476402, +3.230608]|1562/1151/2199|4193|
|FullMAP-SharedCorrectionNative|+3.189738 [+2.753594, +3.658697]|+1.926293 [+1.677028, +2.190595]|1562/764/1750|3637|
|FullMAP-ExpertInterpolationNative|+3.189738 [+2.753594, +3.658697]|+1.050750 [+0.802237, +1.310876]|1562/1029/1963|3489|
|FullMAP-NoTemporalRefinement|+3.189738 [+2.753594, +3.658697]|+1.297044 [+1.075346, +1.534715]|1562/599/1572|3182|
|EightReplacementNative-Source|+0.000000 [+0.000000, +0.000000]|+0.575667 [+0.498797, +0.657317]|0/205/205|465|
|SharedCorrectionNative-Source|+0.000000 [+0.000000, +0.000000]|+1.493374 [+1.201543, +1.794367]|0/1141/1141|2232|
|ExpertInterpolationNative-Source|+0.000000 [+0.000000, +0.000000]|+2.368917 [+2.008753, +2.746894]|0/1206/1206|2587|
|SharedCorrectionNative-EightReplacementNative|+0.000000 [+0.000000, +0.000000]|+0.917708 [+0.674694, +1.168005]|0/1137/1137|1899|
|ExpertInterpolationNative-EightReplacementNative|+0.000000 [+0.000000, +0.000000]|+1.793251 [+1.497273, +2.106943]|0/1119/1119|2225|

Short (GT duration <=5 seconds) and long paired CIs, all signed positive/negative query outcomes, all mean metrics and all severe losses/rescues are retained in the accompanying JSON/TSVs. CIs crossing zero do not establish equivalence. These historically exposed full sets do not constitute new blind validation and do not select parameters.

Costs are cached derivation, label scoring and cold-audit elapsed wall. V3 per-query counters cover full input/derivation/commit, while v4 saved per-query counters start after input/recomposition and cover commit only; v4 whole-worker elapsed includes all input and recomposition. Counter scopes differ and are explicitly separated by actual worker PID/start in cost_lineage.json. These are not process CPU time or deployment latency. Source and expert observations were acquired in earlier workers and remain nonzero inherited costs; the original Vid Source producer was shared with TTA and must not be counted twice. No claim of newly measured full-pipeline latency is made.
