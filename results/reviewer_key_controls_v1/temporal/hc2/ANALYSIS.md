# Reviewer targeted temporal controls: hc2

1902 queries / 227 distinct parents. Complete task-domain seal preceded independent official scoring; all-array cold rederivation, input/prediction/receipt SHA and means audit passed. Method and parameters unchanged. Previously exposed development targets; not unseen confirmation.

|Configuration|tIoU %|vIoU %|
|---|---:|---:|
|Source|40.141475|20.704678|
|SourceJointMAP|46.693293|24.039097|
|SourceTR|49.689566|25.558305|
|ExpertInitTR|49.764548|26.489749|
|Ours|49.764548|27.349044|
|FixedE_SourceTR|49.689566|26.459871|
|Interpolation_SourceTR|49.689566|25.177966|
|Interpolation_ExpertTR|49.764548|25.214291|

|Paired comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union-once|Rescue union-once|
|---|---|---|---|---:|
|SourceTR-Source|+9.548090 [+8.514222, +10.620953]|+4.853626 [+4.274975, +5.455478]|345/218/353|989|
|SourceTR-SourceJointMAP|+2.996272 [+2.250315, +3.739577]|+1.519208 [+1.097718, +1.948427]|453/270/456|705|
|ExpertInitTR-SourceTR|+0.074982 [-0.094384, +0.236031]|+0.931444 [+0.621421, +1.243332]|88/160/222|331|
|ExpertInitTR-FixedE_SourceTR|+0.074982 [-0.094384, +0.236031]|+0.029878 [-0.040535, +0.098940]|88/21/88|87|
|Ours-ExpertInitTR|+0.000000 [+0.000000, +0.000000]|+0.859295 [+0.502377, +1.226670]|0/198/198|260|
|Ours-SourceTR|+0.074982 [-0.094384, +0.236031]|+1.790739 [+1.269914, +2.319285]|88/225/286|498|
|Ours-Interpolation_SourceTR|+0.074982 [-0.094384, +0.236031]|+2.171078 [+1.854831, +2.507748]|88/141/202|468|
|Interpolation_ExpertTR-Interpolation_SourceTR|+0.074982 [-0.094384, +0.236031]|+0.036325 [-0.040629, +0.114270]|88/20/88|86|
|Ours-Interpolation_ExpertTR|+0.000000 [+0.000000, +0.000000]|+2.134753 [+1.814042, +2.473029]|0/131/131|416|

The actual no-expert SourceTR uses original Source boxes and Source-box pre-text features, q0, kappa0/beta1, and its own legal jointMAP interval. It requires no ReferDINO input, geometry correction, box propagation or spatial DP. SourceJointMAP isolates the native MAP-hull versus joint-MAP readout difference. FixedE_SourceTR is a diagnostic using expert-derived E boxes with the no-expert temporal interval, so it is NOT a no-expert deployment pipeline. ExpertInitTR uses the same E trajectory with corrected-ROI temporal evidence. Their t/v differences isolate temporal intervals at fixed space. Interpolation_SourceTR uses the unchanged original eight-expert center/log-size interpolation and its OWN SourceTR interval; no current DP is granted. Interpolation_ExpertTR reproduces the old matched-time interpolation.

At fixed E space, expert-ROI minus Source-ROI temporal evidence: t +0.074982 [-0.094384, +0.236031], v +0.029878 [-0.040535, +0.098940]. Full versus interpolation plus Source-only temporal update: v +2.171078 [+1.854831, +2.507748]. A CI crossing zero does not establish equivalence; positive whole-system gains do not establish expert-ROI necessity. No default changes follow these exposed-target results.

## Short and long events

|Group|Queries|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---|---|---|
|short_le5s|544|SourceTR-Source|+3.452380 [+1.613816, +5.304164]|+1.617630 [+0.581645, +2.687702]|
|short_le5s|544|SourceTR-SourceJointMAP|-0.644946 [-1.871228, +0.579089]|-0.277614 [-0.989520, +0.446920]|
|short_le5s|544|ExpertInitTR-SourceTR|+0.086879 [-0.227539, +0.394597]|+0.641924 [+0.221969, +1.045163]|
|short_le5s|544|ExpertInitTR-FixedE_SourceTR|+0.086879 [-0.227539, +0.394597]|+0.029439 [-0.090925, +0.155953]|
|short_le5s|544|Ours-ExpertInitTR|+0.000000 [+0.000000, +0.000000]|+1.174988 [+0.627267, +1.719856]|
|short_le5s|544|Ours-SourceTR|+0.086879 [-0.227539, +0.394597]|+1.816912 [+1.064374, +2.559240]|
|short_le5s|544|Ours-Interpolation_SourceTR|+0.086879 [-0.227539, +0.394597]|+1.945304 [+1.417692, +2.504046]|
|short_le5s|544|Interpolation_ExpertTR-Interpolation_SourceTR|+0.086879 [-0.227539, +0.394597]|-0.032755 [-0.169651, +0.107035]|
|short_le5s|544|Ours-Interpolation_ExpertTR|+0.000000 [+0.000000, +0.000000]|+1.978059 [+1.473458, +2.525647]|
|long_gt5s|1358|SourceTR-Source|+11.989966 [+10.792441, +13.217353]|+6.149931 [+5.450730, +6.857661]|
|long_gt5s|1358|SourceTR-SourceJointMAP|+4.454905 [+3.589313, +5.350269]|+2.238995 [+1.740023, +2.753673]|
|long_gt5s|1358|ExpertInitTR-SourceTR|+0.070216 [-0.123939, +0.268090]|+1.047423 [+0.635229, +1.441094]|
|long_gt5s|1358|ExpertInitTR-FixedE_SourceTR|+0.070216 [-0.123939, +0.268090]|+0.030054 [-0.058348, +0.116481]|
|long_gt5s|1358|Ours-ExpertInitTR|+0.000000 [+0.000000, +0.000000]|+0.732831 [+0.298838, +1.192886]|
|long_gt5s|1358|Ours-SourceTR|+0.070216 [-0.123939, +0.268090]|+1.780254 [+1.127100, +2.434350]|
|long_gt5s|1358|Ours-Interpolation_SourceTR|+0.070216 [-0.123939, +0.268090]|+2.261520 [+1.877859, +2.664826]|
|long_gt5s|1358|Interpolation_ExpertTR-Interpolation_SourceTR|+0.070216 [-0.123939, +0.268090]|+0.063997 [-0.036826, +0.170055]|
|long_gt5s|1358|Ours-Interpolation_ExpertTR|+0.000000 [+0.000000, +0.000000]|+2.197523 [+1.810813, +2.605119]|

All positive/negative per-query outcomes, severe losses and rescues retained. Query equal means and 10000 parent-cluster percentile pointwise95% CIs; no independent-frame or familywise claim. Cached elapsed wall includes validation/IO and is not deployment latency. Original Source and expert acquisition costs remain nonzero. No new model, Source, expert, optimizer or video decode. Private identities, captions, media, features and GT coordinates are excluded from public output.
