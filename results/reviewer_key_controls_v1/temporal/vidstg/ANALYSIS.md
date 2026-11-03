# Reviewer targeted temporal controls: vidstg

10303 queries / 732 distinct parents. Complete task-domain seal preceded independent official scoring; all-array cold rederivation, input/prediction/receipt SHA and means audit passed. Method and parameters unchanged. Previously exposed development targets; not unseen confirmation.

|Configuration|tIoU %|vIoU %|
|---|---:|---:|
|Source|31.728454|11.138484|
|SourceJointMAP|33.450806|11.641627|
|SourceTR|34.908528|12.224898|
|ExpertInitTR|34.918192|13.948734|
|Ours|34.918192|14.558151|
|FixedE_SourceTR|34.908528|13.925847|
|Interpolation_SourceTR|34.908528|14.665409|
|Interpolation_ExpertTR|34.918192|14.665107|

|Paired comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union-once|Rescue union-once|
|---|---|---|---|---:|
|SourceTR-Source|+3.180073 [+2.730764, +3.650768]|+1.086414 [+0.902995, +1.278769]|1531/526/1535|3152|
|SourceTR-SourceJointMAP|+1.457722 [+1.024020, +1.888639]|+0.583270 [+0.416503, +0.754430]|1947/629/1949|2546|
|ExpertInitTR-SourceTR|+0.009665 [-0.116164, +0.132882]|+1.723836 [+1.392222, +2.064571]|748/1259/1758|2851|
|ExpertInitTR-FixedE_SourceTR|+0.009665 [-0.116164, +0.132882]|+0.022887 [-0.018801, +0.065669]|748/142/749|756|
|Ours-ExpertInitTR|+0.000000 [+0.000000, +0.000000]|+0.609417 [+0.471700, +0.755210]|0/375/375|809|
|Ours-SourceTR|+0.009665 [-0.116164, +0.132882]|+2.333254 [+1.985213, +2.688407]|748/1170/1694|3041|
|Ours-Interpolation_SourceTR|+0.009665 [-0.116164, +0.132882]|-0.107258 [-0.256960, +0.040337]|748/821/1320|1361|
|Interpolation_ExpertTR-Interpolation_SourceTR|+0.009665 [-0.116164, +0.132882]|-0.000302 [-0.049269, +0.047250]|748/181/749|753|
|Ours-Interpolation_ExpertTR|+0.000000 [+0.000000, +0.000000]|-0.106955 [-0.250579, +0.036605]|0/691/691|677|

The actual no-expert SourceTR uses original Source boxes and Source-box pre-text features, q0, kappa0/beta1, and its own legal jointMAP interval. It requires no ReferDINO input, geometry correction, box propagation or spatial DP. SourceJointMAP isolates the native MAP-hull versus joint-MAP readout difference. FixedE_SourceTR is a diagnostic using expert-derived E boxes with the no-expert temporal interval, so it is NOT a no-expert deployment pipeline. ExpertInitTR uses the same E trajectory with corrected-ROI temporal evidence. Their t/v differences isolate temporal intervals at fixed space. Interpolation_SourceTR uses the unchanged original eight-expert center/log-size interpolation and its OWN SourceTR interval; no current DP is granted. Interpolation_ExpertTR reproduces the old matched-time interpolation.

At fixed E space, expert-ROI minus Source-ROI temporal evidence: t +0.009665 [-0.116164, +0.132882], v +0.022887 [-0.018801, +0.065669]. Full versus interpolation plus Source-only temporal update: v -0.107258 [-0.256960, +0.040337]. A CI crossing zero does not establish equivalence; positive whole-system gains do not establish expert-ROI necessity. No default changes follow these exposed-target results.

## Short and long events

|Group|Queries|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---|---|---|
|short_le5s|6060|SourceTR-Source|-0.102081 [-0.417235, +0.217802]|-0.091194 [-0.221030, +0.037713]|
|short_le5s|6060|SourceTR-SourceJointMAP|+0.373603 [+0.014686, +0.728821]|+0.180058 [+0.034006, +0.329589]|
|short_le5s|6060|ExpertInitTR-SourceTR|-0.027566 [-0.154578, +0.102475]|+0.540288 [+0.233157, +0.866944]|
|short_le5s|6060|ExpertInitTR-FixedE_SourceTR|-0.027566 [-0.154578, +0.102475]|+0.001614 [-0.041194, +0.045425]|
|short_le5s|6060|Ours-ExpertInitTR|+0.000000 [+0.000000, +0.000000]|+0.686711 [+0.525217, +0.849912]|
|short_le5s|6060|Ours-SourceTR|-0.027566 [-0.154578, +0.102475]|+1.226999 [+0.901190, +1.575365]|
|short_le5s|6060|Ours-Interpolation_SourceTR|-0.027566 [-0.154578, +0.102475]|-0.125794 [-0.269326, +0.008091]|
|short_le5s|6060|Interpolation_ExpertTR-Interpolation_SourceTR|-0.027566 [-0.154578, +0.102475]|-0.011438 [-0.064729, +0.039275]|
|short_le5s|6060|Ours-Interpolation_ExpertTR|+0.000000 [+0.000000, +0.000000]|-0.114356 [-0.250874, +0.011892]|
|long_gt5s|4243|SourceTR-Source|+7.867759 [+7.030749, +8.734391]|+2.768314 [+2.390654, +3.160431]|
|long_gt5s|4243|SourceTR-SourceJointMAP|+3.006098 [+2.143300, +3.894830]|+1.159153 [+0.815673, +1.515100]|
|long_gt5s|4243|ExpertInitTR-SourceTR|+0.062838 [-0.180383, +0.310264]|+3.414221 [+2.863053, +3.982845]|
|long_gt5s|4243|ExpertInitTR-FixedE_SourceTR|+0.062838 [-0.180383, +0.310264]|+0.053270 [-0.032638, +0.140204]|
|long_gt5s|4243|Ours-ExpertInitTR|+0.000000 [+0.000000, +0.000000]|+0.499025 [+0.288674, +0.715895]|
|long_gt5s|4243|Ours-SourceTR|+0.062838 [-0.180383, +0.310264]|+3.913245 [+3.345484, +4.496369]|
|long_gt5s|4243|Ours-Interpolation_SourceTR|+0.062838 [-0.180383, +0.310264]|-0.080783 [-0.353173, +0.195892]|
|long_gt5s|4243|Interpolation_ExpertTR-Interpolation_SourceTR|+0.062838 [-0.180383, +0.310264]|+0.015602 [-0.081066, +0.111039]|
|long_gt5s|4243|Ours-Interpolation_ExpertTR|+0.000000 [+0.000000, +0.000000]|-0.096385 [-0.358340, +0.164588]|

All positive/negative per-query outcomes, severe losses and rescues retained. Query equal means and 10000 parent-cluster percentile pointwise95% CIs; no independent-frame or familywise claim. Cached elapsed wall includes validation/IO and is not deployment latency. Original Source and expert acquisition costs remain nonzero. No new model, Source, expert, optimizer or video decode. Private identities, captions, media, features and GT coordinates are excluded from public output.
