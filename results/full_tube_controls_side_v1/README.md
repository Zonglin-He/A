# Complete temporal-refinement deletion and same-eight simple-fusion full tubes

The primary backbone is the current TA-STVG Source with the same original eight ReferDINO observations; this is not a CG-STVG or alternative-expert experiment. Both complete targets have independently scored outputs, separate array/provenance/SHA/mean audits, root analysis, and independent public-TSV t/v parent-CI checks. No new model/Source/expert/optimizer/GPU. These previously exposed full targets do not constitute new unseen validation and no parameter is selected.

## Entire temporal-refinement branch

| Dataset | Configuration | tIoU % | vIoU % | R.3 % | R.5 % |
|---|---|---:|---:|---:|---:|
|hc2|Full spatial DP + original Source time|40.141475|22.383628|31.861199|11.041009|
|hc2|Current full method|49.764548|27.349044|41.219769|16.876972|
|vidstg|Full spatial DP + original Source time|31.728454|13.261107|16.995050|6.978550|
|vidstg|Current full method|34.918192|14.558151|19.411822|8.463554|

The disabled branch uses original native per-offset MAP/hull time and the full MAIN spatial candidate/score/link construction with final interval-conditioned DP. It is the strict audited NoTemporalInference control, not NoVisualUpdate (q0 joint-MAP time), whose older25.85/13.87 vIoU cannot be renamed. Interval-dependent eligibility/normalization follows the unchanged DP code. Full-versus-disabled includes the temporal readout change and appearance posterior refinement; it does not isolate corrected-ROI feature causality.

| Dataset | Full minus disabled tIoU pp [95% CI] | vIoU pp [95% CI] |
|---|---:|---:|
|hc2|+9.623073 [+8.594747, +10.683117]|+4.965416 [+4.374584, +5.568715]|
|vidstg|+3.189738 [+2.753594, +3.658697]|+1.297044 [+1.075346, +1.534715]|

## Same-eight complete-tube simple fusions

| Dataset | Configuration | Time output | tIoU % | vIoU % | R.3 % | R.5 % |
|---|---|---|---:|---:|---:|---:|
|hc2|Original Source|Original Source native MAP/hull|40.141475|20.704678|29.022082|8.937960|
|hc2|Eight observed replacements, other boxes Source|Original Source native MAP/hull|40.141475|21.430378|29.390116|9.411146|
|hc2|Shared-corrected E trajectory|Original Source native MAP/hull|40.141475|21.439119|29.810726|10.199790|
|hc2|Original same-eight center/logsize interpolation|Original Source native MAP/hull|40.141475|21.886351|31.177708|10.830705|
|hc2|Current full method|Own refined joint MAP|49.764548|27.349044|41.219769|16.876972|
|vidstg|Original Source|Original Source native MAP/hull|31.728454|11.138484|12.831214|3.348539|
|vidstg|Eight observed replacements, other boxes Source|Original Source native MAP/hull|31.728454|11.714150|13.559158|3.717364|
|vidstg|Shared-corrected E trajectory|Original Source native MAP/hull|31.728454|12.631858|16.286518|6.638843|
|vidstg|Original same-eight center/logsize interpolation|Original Source native MAP/hull|31.728454|13.507401|17.082403|7.240610|
|vidstg|Current full method|Own refined joint MAP|34.918192|14.558151|19.411822|8.463554|

The simple fusions share the same original eight expert observation cache and Source-selected indices. Direct replacement and interpolation use the expert boxes; they do not consume every cached mask feature as the full structured spatial branch does. They receive neither Ours posterior/event nor trajectory DP. Replacement, shared correction and interpolation are newly composed as complete tubes and independently scored, rather than copied from a component table. All simple tIoUs equal raw Source per query by construction; differences from Full are complete-system effects, not isolated spatial effects. Interpolation uses the exact original center/logsize rule and E beyond the anchor span.

## Evidence and compute scope

HC1902 and2754 committed Vid queries preserve actual fresh full/native DP replays. Remaining fixed-interval native-DP/Full arrays reuse independently whole-scored/all-array-audited exact cached predictions. Four-case cached/fresh array-and-dtype bridges pass, all simple tubes are cold-recomposed, and every output, input lineage and mean is verified. All four controls have new official scores (48820 total metric/dense consistency checks); raw Source/current Full exact scores are inherited from prior complete audits. No old compact, feature, media or source root is changed or deleted.

All signed positive/negative outcomes, >5pp severe t/v union-once losses/rescues, short<=5s/long, sIoU/fixed-support metrics and actual per-worker cost scopes are retained in each domain folder. CIs use equal-query means and10000 parent-video cluster resamples, pointwise without familywise correction. Cached CPU work is not deployment latency; original Source/expert costs remain nonzero and shared Vid acquisition is never double-counted. V3 and v4/v6 saved per-query counters have different scopes, disclosed in cost_lineage.json. The independent import/legacy-MBR-interval/resume-registration failures and intentional owned execution switches are preserved, not skipped or used to waive hashes.

Code: [study entry and protocols](../../studies/full_tube_controls_side_v1/README.md). This completes only these two requested controls; it does not declare the parent research scope complete or change any main experiment scheduling.
