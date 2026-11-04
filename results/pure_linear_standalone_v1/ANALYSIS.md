# Plain expert linear interpolation: independent implementation and corrected protocol

One estimator only: `pure.py:interpolate(anchor_times,anchor_xyxy,frame_ids)`. Eight original expert pixel-corner boxes, physical times, every frame of the input video segment; four-coordinate ordinary np.interp, exact observed boxes, nearest endpoint hold. NumPy only. No Source boxes, E/shared correction, appearance features, masks, prototypes, temporal prior/posterior/update, transport/candidates/DP/logsize. Prediction does not read Source or Full time intervals; evaluator supplies them only after whole-domain trajectory seal.

## Independent baseline, original native Source interval

This is Source + raw expert linear interpolation; expert-only boxes do not themselves predict an event interval. Use actual native Source event without any proposed temporal refinement.

|Dataset|Queries/parents|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---|---:|---:|---:|---:|
|hc2|1902/227|40.141475|22.080603|31.703470|11.041009|
|vidstg|10303/732|31.728454|13.631224|17.218286|7.376492|

## Separate controlled spatial evaluation

This is the same generated pure trajectory evaluated with Full interval, not a standalone expert temporal method. It answers only the previously requested same-time spatial question.

|Dataset|Plain expert linear vIoU %|Unchanged Full vIoU %|Full-pure pp [95% CI]|
|---|---:|---:|---|
|hc2|25.592718|27.349044|+1.756325 [+1.445464,+2.071932]|
|vidstg|14.804402|14.558151|-0.246251 [-0.392809,-0.101615]|

## Confirmed coupling and measured effects

Old PureLinearFixed generated only Source sample-grid frames; official dense interpolation does not extend beyond first/last supplied frame. This violated full-video endpoint-hold coverage. New direct expert-only trajectory covers the original full input segment: HC[0,frame_count), Vid[start_frame,end_frame).

hc2:old uncovered input frames=1902,affected queries=1902,uncovered frames within Full interval=0,within native Source interval=0. Corrected FullTime pure minus old sparse pure vIoU=-1.86337070787e-05pp.

vidstg:old uncovered input frames=41472,affected queries=8899,uncovered frames within Full interval=0,within native Source interval=0. Corrected FullTime pure minus old sparse pure vIoU=-2.14005475628e-17pp.

A boundary bug does not establish its magnitude on vIoU. Missing frames outside a predicted event do not change that event vIoU; HC clipping/interpolation order and numeric roundoff can cause small remaining differences. Retain results even when the complete expert-only rewrite leaves the earlier spatial conclusion unchanged.

Previous chat called FullTime score a standalone pure baseline; that label mixed in the proposed temporal update. FullTime is explicitly a controlled evaluation now, never an interpolation input. Additionally the initial new wrapper used a historical production `Source_interval` alias; independent v2 corrects only the source-time reference to actual native per-offset MAP hull, checks two independent reference producers and everyquery nativeSource tIoU. All initial arrays/scalars/receipts/code and actual costs are preserved; only native-time scoring added, no re-prediction or FullTime rescoring. Public verified/ reports are authoritative; raw v1 SourceTime names are historical failed-binding diagnostics, not the native baseline.

Unobserved raw diagnostics have different dense versus native support across old/new studies, so cross-support unobserved CIs are excluded from verified public comparison. All reported vIoU/tIoU/R use identical official dense labels. No claim that original interpolation corner formula used E/features is established: confirmed issues are full-frame coverage and temporal evaluation/readout binding. No method/production/paper/oldhash changes; Full observed-anchor mismatch remains unresolved.

Allnewraw8trajectories are sealed before separate CPU official scoring and independent rawinputSHA/physicaltime/parent/completeframes/anchors/tails/manualpiecewise/normalizedIoU/meansaudit. 36615officialcalls include initial legacy-time attempt12205+FullTime12205+correct nativeSource12205; no cost hidden or mislabeled as0. Queryequal10000parentbootstrap seed20261022, all signed severe/rescues/short<=5s-long, pointwise on exposed cohorts/no target tuning/equivalence claim. Cached CPU IO/scoring walls notdeployment; inherited Source/expert acquisition remains nonzero. Private media/caption/GT/box/interval/feature/mask/weights remain private.
