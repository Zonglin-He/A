# Adjacent-Motion Spatial OPD: No vIoU Gain Over T on the Fixed Eight-Source Cohort

## Scope and reading rules

This report uses the sealed eight-source cohort only. It reports the locked candidate tracks, 48 teacher-control calls, and postseal metrics. The cases are historically exposed; this is a finite diagnostic, not a fresh-source confirmation or a deployment promotion. No new temporal fit was run. Old T is reused exactly, and V03 reference controls are prior evidence reused with zero new fits.

The postseal scorer reports 72 metric rows, 43 tracked-geometry rows, 43 display-geometry rows, 9 actual fits, 45 actual optimizer steps, and 9 full final-model readouts. The fixed five-round protocol implies 45 declared fit rounds; declared rounds and optimizer steps are different counts.

All case names are aliases: HC01-HC04 and V01-V04. The anonymous data files retain per-case metric, comparison, candidate, fit, motion-round, teacher-control and Oracle rows while excluding source paths and hashes, frame IDs, GT coordinates, query prose, raw responses, model weights and optimizer tensors.

## Complete-tube metrics by domain

Values are percent. T is the old sealed temporal output. T + spatial arms reuse its interval and use the current spatial readout. Each domain contains four cases.

|Arm|HC2 vIoU|HC2 tIoU|HC2 dense sIoU|VidSTG vIoU|VidSTG tIoU|VidSTG dense sIoU|
|---|---:|---:|---:|---:|---:|---:|
|Frozen|22.2105|46.4149|47.8950|14.4901|27.7780|48.4576|
|Projection|27.8513|58.4342|47.8950|22.0165|40.8065|48.4576|
|Temporal QA (reused)|31.2860|60.4860|47.8950|22.4419|40.9049|48.4576|
|Spatial Direct|21.8013|46.4149|46.6776|14.2152|27.7780|47.9876|
|Spatial IoU-OPD|21.9557|46.4149|47.1380|14.4677|27.7780|48.4338|
|Spatial Motion-OPD|21.9558|46.4149|47.1382|14.4673|27.7780|48.4332|
|QA + Direct|30.4007|60.4860|46.6776|22.0589|40.9049|47.9876|
|QA + IoU-OPD|30.7073|60.4860|47.1380|22.4225|40.9049|48.4338|
|QA + Motion-OPD|30.7075|60.4860|47.1382|22.4220|40.9049|48.4332|

Observed result: adding Motion-OPD to T changed mean vIoU from 31.2860% to 30.7075% in HC2 and from 22.4419% to 22.4220% in VidSTG. Against F, the spatial IoU-OPD arm was positive in 1 case(s), negative in 2; the other 5 cases had exact-F no-update outputs. This is a negative aggregate complementarity result on these fixed historical sources; it does not establish route-wide failure or fresh-source performance.


## Per-case arm readout

Values are complete-tube vIoU percent. Positive, negative, neutral and unknown classifications use the scorer's descriptive tolerance; they are not a promotion gate.

|Case|Domain|Frozen|Projection|T|Direct|IoU-OPD|Motion-OPD|T+Direct|T+IoU|T+Motion|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|HC01|hc2|16.9582|15.4226|6.9161|16.9582|16.9582|16.9582|6.9161|6.9161|6.9161|
|HC02|hc2|35.8185|39.9419|52.2645|34.1398|34.4745|34.4749|48.1592|48.8675|48.8681|
|HC03|hc2|12.6768|23.7797|23.7797|12.6768|12.6768|12.6768|23.7797|23.7797|23.7797|
|HC04|hc2|23.3883|32.2609|42.1838|23.4303|23.7132|23.7133|42.7476|43.2659|43.2660|
|V01|vidstg|36.1814|49.2818|51.7333|35.0816|36.0916|36.0901|50.2014|51.6557|51.6537|
|V02|vidstg|9.4699|24.9128|24.4765|9.4699|9.4699|9.4699|24.4765|24.4765|24.4765|
|V03|vidstg|6.0066|6.8916|6.8916|6.0066|6.0066|6.0066|6.8916|6.8916|6.8916|
|V04|vidstg|6.3026|6.9797|6.6662|6.3026|6.3026|6.3026|6.6662|6.6662|6.6662|

## Case-level comparison counts

Each cell counts cases. F is the baseline row; T + spatial rows compare against T; spatial rows compare against F. Unknown comparisons are retained when a baseline or arm was not created.

|Arm|Positive|Negative|Neutral|Unknown|Baseline|
|---|---:|---:|---:|---:|---:|
|Frozen|0|0|0|0|8|
|Projection|7|1|0|0|0|
|Temporal QA (reused)|7|1|0|0|0|
|Spatial Direct|1|2|5|0|0|
|Spatial IoU-OPD|1|2|5|0|0|
|Spatial Motion-OPD|1|2|5|0|0|
|QA + Direct|1|2|5|0|0|
|QA + IoU-OPD|1|2|5|0|0|
|QA + Motion-OPD|1|2|5|0|0|

## P0: candidate motion geometry and Source display

Candidate support is counted from geometry_boxes_xyxy, which is the independently tracked trajectory. The Source boxes_xyxy used for display are reported separately because the Source display preserves the original four source boxes. Source display coverage is not motion-track support.

|Candidate geometry source|Count|At least 2 slots|All 4 slots|Anchor only|Missing slots|
|---|---:|---:|---:|---:|---:|
|New independent track: all candidates|43|9|7|34|106|
|New independent track: Source|8|5|4|3|11|
|New independent track: nonSource|35|4|3|31|95|
|Historical LK/display: all candidates|43|18|11|25|88|
|Historical: Source display boxes|8|8|8|0|0|
|Historical: nonSource LK track|35|10|3|25|88|

|Case|Source display valid slots|Independent Source-track valid slots|Paired valid slots|Paired slots changed from display|
|---|---:|---:|---:|---:|
|HC01|4|1|1|0|
|HC02|4|4|4|3|
|HC03|4|1|1|0|
|HC04|4|4|4|3|
|V01|4|4|4|3|
|V02|4|4|4|3|
|V03|4|1|1|0|
|V04|4|2|2|1|

New adjacent-track work totals by candidate group:

|Group|Attempted pairs|Accepted pairs|Rejected pairs|Not attempted after loss|Detected corners|FB-valid corners|RANSAC inliers|
|---|---:|---:|---:|---:|---:|---:|---:|
|all_candidates|906|836|70|5560|67699|66854|62350|
|Source|451|444|7|808|36080|35617|33351|
|nonSource|455|392|63|4752|31619|31237|28999|

The historical LK support summary treats Source as the original displayed four boxes and nonSource as the old LK geometry. It is not a Source-motion estimate. The new rows track Source and nonSource independently from the selected seed. Full normalized candidate boxes, per-slot rho, ordered trajectory, accepted/rejected adjacent-pair diagnostics, per-direction work counts, failure reasons, and display-ID/color/order mappings are in `candidate_geometry_and_display.json`; physical frame IDs and absolute timestamps are removed. The new preparation elapsed CPU time is 10.474033 seconds.

## P1: teacher control matrix

Each control denominator is eight cases, not the 48 total requests. Same-geometry counts require valid outputs in both the base and control response. The wrong-query condition is reported as choice changed, with abstentions also counted separately.

|Condition|Same geometry / 8|Changed choice / 8|Format valid / 8|Truncated / 8|Abstain or invalid / 8|
|---|---:|---:|---:|---:|---:|
|base|NA|NA|8|0|0|
|id_swap|6|NA|8|0|0|
|color_swap|7|NA|8|0|0|
|order_swap|6|NA|8|0|0|
|joint_swap|4|NA|8|0|2|
|wrong_query|NA|8|8|0|7|

Base selections by reference type: Source 6/8, nonSource 2/8. Teacher control consistency is sensitivity evidence, not proof of instance identity.

## Candidate geometry quality and postseal Oracle diagnostic

The scorer keeps candidate-present IoU, same-anchor paired Frozen IoU, GT-known common-support scores and postseal pool/display Oracle rows separate. Missing candidate geometry is zero-filled only for the stated GT-known coverage diagnostic; missing GT remains unknown and is excluded from that denominator.

|Case|Candidates|Displayed|Oracle on known common support|Displayed Oracle|Chosen candidate|Source baseline|
|---|---:|---:|---:|---:|---:|---:|
|HC01|7|5|0.196423|0.196423|0.196423|0.588250|
|HC02|4|4|0.615816|0.615816|0.615816|0.609936|
|HC03|5|5|0.000000|0.000000|0.000000|0.589733|
|HC04|7|5|0.465802|0.465802|0.406160|0.402772|
|V01|5|5|0.701516|0.701516|0.701516|0.667700|
|V02|4|4|0.623032|0.623032|0.086872|0.589977|
|V03|5|5|0.000000|0.000000|0.000000|0.445717|
|V04|6|5|0.194859|0.194859|0.000000|0.202740|

## P2: fits, actual optimizer steps and readouts

The locked spatial scope uses 1,792 parameters, the original per-domain Direct/IoU-OPD settings and five declared rounds. Motion-OPD adds the fixed adjacent-displacement loss with lambda_motion = 1.0. The report distinguishes 9 actual fits, 15 exact-F no-update placeholders, 45 actual Adam steps, and 9 full final-model readouts. Actual optimizer steps can be below declared rounds when the locked objective has no informative gradient.

|Case|Eligible reference|Direct steps|IoU steps|Motion steps|Direct parameter delta|IoU parameter delta|Motion parameter delta|
|---|---|---:|---:|---:|---:|---:|---:|
|HC01|no - exact F reuse|0|0|0|NA|NA|NA|
|HC02|yes|5|5|5|1.573542|1.329292|1.329279|
|HC03|no - exact F reuse|0|0|0|NA|NA|NA|
|HC04|yes|5|5|5|0.789829|0.850435|0.850388|
|V01|yes|5|5|5|1.588412|1.750772|1.751629|
|V02|no - exact F reuse|0|0|0|NA|NA|NA|
|V03|no - exact F reuse|0|0|0|NA|NA|NA|
|V04|no - exact F reuse|0|0|0|NA|NA|NA|

Motion-OPD round losses and pair-support counts are retained per case and fit. Old T was reused exactly; no new temporal model fit was made. The V03 Source/old-ID reference control is exact prior-evidence reuse with zero new fits.

|Motion round|Fits|Mean OPD loss|Mean motion loss|Mean total loss|Updated fits|Valid pairs summed|
|---:|---:|---:|---:|---:|---:|---:|
|1|3|-0.233971|0.000754|-0.233217|3|212|
|2|3|-0.157788|0.000749|-0.157039|3|212|
|3|3|-0.140943|0.000753|-0.140191|3|212|
|4|3|-0.180770|0.000758|-0.180012|3|212|
|5|3|-0.186479|0.000762|-0.185718|3|212|

V03 prior control status: exact_prior_evidence_reuse; new fits executed: 0. Its previously scored fixed-reference metrics are included only as exact reuse, not as motion-selected candidate evidence.

|V03 prior arm|Domain|Cases|vIoU %|tIoU %|dense sIoU %|
|---|---|---:|---:|---:|---:|
|F|vidstg|1|6.006629|14.042553|48.259917|
|Projection|vidstg|1|6.891628|15.670103|48.259917|
|Source_iou|vidstg|1|5.761919|14.042553|46.289589|
|T|vidstg|1|6.891628|15.670103|48.259917|
|oldAlternativeID3_iou|vidstg|1|1.360450|14.042553|10.345365|
|oldR1ID2_iou|vidstg|1|8.246636|14.042553|58.040965|

|V03 locked reference role|Candidate IoU on GT-known anchors|Paired Source IoU|Difference|Known labels / unknown labels|Missing candidate where GT known|
|---|---:|---:|---:|---:|---:|
|Source|0.445717|0.445717|0.000000|1 / 3|0|
|oldR1ID2|0.158103|0.445717|-0.287614|1 / 3|0|
|oldAlternativeID3|0.053349|0.445717|-0.392369|1 / 3|0|

## Boundaries

All case-level metric rows and every positive, negative, neutral and unknown comparison are preserved in `metric_and_case_rows.json`. Candidate geometry, display mappings, support states, candidate IoUs and Oracle diagnostics are in separate anonymous files. Teacher outputs retain parsed choices, control validity, call costs and format status, but omit raw answer/query prose. This finite historical-source diagnostic does not qualify stable performance on fresh sources or change the deployed method.
