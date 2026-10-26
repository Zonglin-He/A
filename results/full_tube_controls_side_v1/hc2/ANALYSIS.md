# Complete full-tube controls: hc2

1902 complete queries, 227 parent videos. Predictions sealed before independent official CPU scoring; independent all-array cold replay, SHA/provenance and mean audit passed. No new model/Source/expert/optimizer/GPU. Original current method and all older frozen studies remain unchanged.

| Configuration | tIoU % | vIoU % | sIoU % | R.3 % | R.5 % |
|---|---:|---:|---:|---:|---:|
|Original Source|40.141475|20.704678|50.685778|29.022082|8.937960|
|Eight-frame replacement + Source time|40.141475|21.430378|51.744875|29.390116|9.411146|
|Shared correction + Source time|40.141475|21.439119|51.911198|29.810726|10.199790|
|Same-eight interpolation + Source time|40.141475|21.886351|48.980489|31.177708|10.830705|
|Full spatial DP + Source time; temporal refinement entirely disabled|40.141475|22.383628|53.232712|31.861199|11.041009|
|Current full method|49.764548|27.349044|53.283819|41.219769|16.876972|

The first ablation bypasses all temporal refinement/readout replacement and uses original Source native per-offset MAP/hull time, then actually reruns the final interval-conditioned DP with the same MAIN candidate geometry, appearance/link scores and coefficients. Event-dependent eligibility and original normalization recompute under that interval. It reproduces the previously independently scored NoTemporalInference arrays and scalar metrics exactly. It is not the NoVisualUpdate / q0 joint-MAP row.

The three simple fusions use original Source time independently of Ours. Replacement changes exactly eight observed boxes and keeps every unobserved Source box. Shared correction uses the saved E trajectory. Interpolation uses the original center/logsize rule and E outside anchor coverage. No simple method receives our posterior or trajectory DP. All four controls must therefore have exactly the same tIoU as original Source. Their vIoU/R metrics are independently scored from the newly composed full tubes, not transplanted component-table values.

Historical cache naming: the old parent Source_interval and OursS use the historical added MBR readout. These interval fields are deliberately excluded; the original Source interval is verified against the corrected current/structural audits and actual saved native logits in the four-case bridge. No legacy output is silently relabeled.

## Complete paired outcomes

| Comparison | delta tIoU pp [95% CI] | delta vIoU pp [95% CI] | severe t/v/union | rescues |
|---|---:|---:|---:|---:|
|FullMAP-Source|+9.623073 [+8.594747, +10.683117]|+6.644365 [+5.902588, +7.408451]|346/305/458|1129|
|FullMAP-EightReplacementNative|+9.623073 [+8.594747, +10.683117]|+5.918666 [+5.236296, +6.620222]|346/315/452|1093|
|FullMAP-SharedCorrectionNative|+9.623073 [+8.594747, +10.683117]|+5.909925 [+5.250360, +6.580120]|346/306/439|1081|
|FullMAP-ExpertInterpolationNative|+9.623073 [+8.594747, +10.683117]|+5.462692 [+4.842616, +6.094744]|346/260/396|1061|
|FullMAP-NoTemporalRefinement|+9.623073 [+8.594747, +10.683117]|+4.965416 [+4.374584, +5.568715]|346/233/361|988|
|EightReplacementNative-Source|+0.000000 [+0.000000, +0.000000]|+0.725699 [+0.570970, +0.881499]|0/74/74|101|
|SharedCorrectionNative-Source|+0.000000 [+0.000000, +0.000000]|+0.734440 [+0.471785, +0.999963]|0/131/131|214|
|ExpertInterpolationNative-Source|+0.000000 [+0.000000, +0.000000]|+1.181673 [+0.656306, +1.716009]|0/272/272|437|
|SharedCorrectionNative-EightReplacementNative|+0.000000 [+0.000000, +0.000000]|+0.008741 [-0.194647, +0.215524]|0/138/138|110|
|ExpertInterpolationNative-EightReplacementNative|+0.000000 [+0.000000, +0.000000]|+0.455974 [+0.039838, +0.877583]|0/256/256|293|

Short (GT duration <=5 seconds) and long paired CIs, all signed positive/negative query outcomes, all mean metrics and all severe losses/rescues are retained in the accompanying JSON/TSVs. CIs crossing zero do not establish equivalence. These historically exposed full sets do not constitute new blind validation and do not select parameters.

Costs are cached derivation, label scoring and cold-audit elapsed wall, not process CPU time or deployment latency. Source and expert observations were acquired in earlier workers and remain nonzero inherited costs; the original Vid Source producer was shared with TTA and must not be counted twice. No claim of newly measured full-pipeline latency is made.
