# R10 · Independent RGB/point-cloud geometric increment at fixed q0

**The actual direct-pseudo experiment completed and failed the prelocked progression gate.** RGB coordinate constraints provide a small matched positive increment over their removal, but the full geometry module degrades the original source in both domains. Do not feed these pseudo boxes into another parameter-adaptation run. No R10 neural TTA method is adopted; the regression-adaptation route remains unresolved rather than rejected.

## Executed scope and measurement

The model is the unchanged, audited Waymo-only epoch100 checkpoint, SHA256 `0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd`. No source retraining, teacher or EMA is used. Fresh strict-FP32 source inference and same-forward features/observations were sealed before G. The source state, query and all model buffers were restored and verified.

Input-only hash selection fixes32 distinct frames per each of five scenes:64Drone/96Quad, excluding all earlier60smoke frames and R9's160frames. TargetVAL has prior R5/R7/R8 research exposure. This is a scene-balanced exploratory diagnostic, not a fresh confirmation set, full-population evaluation, or160independent scenes. R9/R10 raw means use different cohorts and cannot establish a paired improvement over the previous method.

Actual source inputs use16384 replacement-sampled rows. Geometry deduplicates physical raw indices and coincident XYZ; source model inputs remain unchanged. Unique raw indices range2525–7807 onDrone and6094–10201 onQuad. Duplicate rows never inflate normal estimation or face support.

The160×4direct predictions, originalq0, source features, actual point observations, masks, statuses and constraint traces are retained. All direct-G predictions sealed before the new CPU score release. GT is used only by the postseal CPU evaluator, never G, segmentation, optimization, input selection or acceptance. Evaluation retains nativeyaw0 and uses the existing general3D IoU against rotatedGT; there is noPCA, GTyaw intervention, oracle selection or newGTcapacity experiment.

## What the new G measures

Language determines the original frozen grounding query. There is no new text-to-pixel localizer, color-phrase rule, query reselection or averaging of predicted candidate centers. RGB support comes from originalq0-projected bounds and point-backed foreground seeds, followed by fixed5iteration GrabCut and a seed-connected component. GrabCut is a fallible appearance segmentation method; the [official OpenCV tutorial](https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html) also documents segmentation errors and foreground/background seed assumptions. R10 has no manual pixel correction.

Actual local point normals and planar support constrain camera-facing sourcebox faces. A single measured plane constrains `center_j + sign*size_j/2`; center recovery therefore still depends on the original size prior. Partial-point means or extents are never treated as complete-object centers or dimensions. Size can change only with a stricter pair of opposing observed planes. This condition is a heuristic object hypothesis, not verified amodal completeness.

The RGB coordinate term penalizes foreground pixels outside the projected box's convex hull. It is one-sided containment, not equality between a partial foreground silhouette and the complete projected3D box. Source shape/center priors and fixed physical bounds limit correction.

The four delivered arms are:

1. `source` — originalq0 box.
2. `full_rgb` — actual point surfaces plus RGB containment.
3. `no_rgb_coordinate` — identical sourceq0, point support, face assignments, per-coordinate acceptance, prior and bounds; only the RGB coordinate loss is disabled.
4. `rgb_gated_pointmask_ablation` — remove RGB point filtering while retaining the shared RGB-derived acceptance and RGB coordinate term. This changes observations and is explicitly not a deploymentPC-only method.

Finite bounded nonconvergence is retained and flagged. Numerical invalidity falls back to source with a distinct status. Unknown implementation errors stop execution. No target-label threshold selection or parameter sweep occurred.

## All-sample task results

All entries below are mIoU percentages, on exactly the same source forward and inputs.

| Arm | Drone64 | Quad96 |
|---|---:|---:|
| Source |17.249655|29.142463|
| Full RGB geometry G |16.263056|28.875606|
| Remove RGB coordinate constraint |16.233232|28.872190|
| RGB-gated pointmask ablation |16.278545|28.876867|

Full−source is **−0.986599pp/−0.266857pp**. Acc25 falls23/64→22/64 onDrone and stays44/96 onQuad; Acc50 falls9/64→6/64 and34/96→33/96. No failed or zero-change sample was removed. IoU signs are1gain/4harms/59ties and0/3/93 at1e-8 tolerance. The worst individual losses are29.687504pp/19.359431pp; the positiveDrone tail is retained rather than averaged away.

Matched RGB coordinate increments, full minus removed-coordinate control, are **+0.029824pp/+0.003416pp**. This answers the incremental-information question more narrowly: the actual RGB coordinate term has a small positive mean effect in the fixed support/gate comparison, but it does not recover the geometry module's harm relative to source. Two andtwo samples contribute positive task increments; remaining cases are ties within tolerance. These2/3scene results do not establish population significance or a useful newTTA mechanism.

RGB point filtering itself does not improve task performance here. Full minus the RGB-gated pointmask ablation is **−0.015489pp/−0.001262pp**. Sharing accepted samples cannot be used to claim that selected points are more correct.

## Coordinates and the actual correction subset

All12changed samples are retained,6perdomain. Full/off-coordinate comparison uses one common union-of-changed set; own-changed subsets are only descriptive. Here the accepted and actual-changed sets coincide, so the same6+6cases support the following paired comparison.

| Metric | Drone source→full | Quad source→full |
|---|---:|---:|
| All-sample center L2,m |8.002379→8.005625|4.019235→4.015269|
| All-sample size L2,m |0.331419→0.331419|0.238324→0.238324|
| Changed-subset center L2,m |1.520960→1.555587|2.317740→2.254280|
| Changed-subset size L2,m |0.170058→0.170058|0.401937→0.401937|
| Changed-subset mIoU,% |40.624363→30.100642|21.431278→17.161566|

Quad's smaller center error coexists with worseIoU. Center L2 does not determine overlap when size, yaw and error direction interact. GTsize errors are local dimension-parameter errors, not globalyaw0enclosure errors.

The12center changes average0.301407m in magnitude. Larger correction is not the objective: these changes have worse net task accuracy. This is independent observed-surface fitting rather than amplified old candidate-center differences, and it still fails to provide an acceptable pseudo target.

## Mechanism activity and limitations

Segmentation returns a nonempty usable mask for157/160,61/64Drone and96/96Quad. ThreeDrone cases lack enough unique point-backed foreground pixels. Of157mask-available cases,145still have no accepted geometric coordinate. Only12cases/7.5% change: accepted center axes are11x,1y,0z. All160sizes remain exactlysource. Threeaxis pairs satisfy the basic per-face tests but none satisfy stricter opposing-face size qualification. This run therefore does not measure size correction capability and cannot reject sizeTTA.

Full and no-RGB-coordinate raw boxes are exactly equal in153cases and different in7,3Drone/4Quad. Maximum extra coordinate difference from RGB is0.016935m; the mean maximum coordinate difference among nonzero cases is0.003379m. An available RGB loss is not equivalent to a changed prediction.

All12full fits and12pointmask-ablation fits converge. The noRGBcoordinate control has11converged fits and1finite nonconverged bounded result, retained as registered. Numerical invalidity iszero. The largest FP32 excess over the prescribed center bound is2.98e-8m and is retained rather than reported as exactzero.

For12full fits, mean observed-surface loss decreases9.50408→1.24324 and total objective decreases10.61822→4.64786, while task accuracy degrades. The RGB containment term itself increases1.11413→1.34492 and rises in10cases. The solver is trading soft constraints rather than certifying RGB satisfaction or correct localization. A lower unsupervised objective cannot substitute for the measured task result.

The single-plane center conversion remains conditional on source dimensions; an observed planar patch is not necessarily the object's outer bounding face. Source identity mistakes, background contamination, source-size error, nativeyaw0 misfit and source-conditioned acceptance remain competing explanations. Current results do not separately identify their causal contributions. Low coverage describes this fixed rule, not an absence of geometric information in the dataset.

## Real visual review

Root actually viewed all10figures selected by input-only hashes before scoring. All10have zero coordinate change and were preserved. They show partial foreground, source anchors on road/trees in some cases, and foreground components connected to background. Their unchanged status cannot explain changed-case losses.

A separate postseal mechanism gallery includes **all12actual full-G changed cases**, selected without GT or gain/harm filtering. Root also actually viewed every one. Masks join adjacent vehicles in several cases, and person masks include walls, trees or other background; several objects are image-truncated or partly occluded. These are visible reliability failures of the supplied RGB support. They do not prove which fitted plane belongs to which annotated instance, and no GTidentity label was inferred from these images. Both galleries, original captions and coordinates remain private. No figure was replaced or used to change G.

## Audits, cost and decision

The source audit checks323boundfiles,160source caches and160observation caches. The unlabelled G audit verifies every source/query binding, support/raw-index deduplication, full/off-coordinate matched controls, masks, provenance, bounds and statuses. Independent score reduction checks2622summary/paired fields and1280coordinate-norm fields. The additional mechanism/gate review checks2293fields with maximum floating discrepancy1.42e-14; it reads sealed derived outputs rather than computing newIoUs or reading freshGT.

Source capture sealed pipeline time is17.318s; actual four-arm CPU geometry is145.241s; selected-box CPU scoring is1.476s. Times include diagnostic loading, hashes, IO and multiple controls, not matched single-arm deployment latency. No source orG run failed. The initial gallery-plan alias-count error was corrected before rendering with selected IDs/order unchanged and the draft retained.

The prelocked resource gate requires positive task gain and a relevant coordinate-error reduction in bothplatforms, plus a positive matched RGB-coordinate increment. It fails onDrone task+center andQuad task. **The conditional unused-observation adaptation stage is not executed.** This is a deliberate prerequisite stop, not an environment failure or a claim that same-input fitting should outperform its own target.

If adaptation is investigated later, its target must be a prediction not directly fitted: e.g.,FrozenA prior and A-only geometry, then fresh point-disjointB with frozenBq0 held fixed, plus the same-information cross-feature-kernel transfer control. Shared RGB/sameframe would still limit that to point-sampling transfer, not temporal or new-scene generalization. This later experiment remains unexecuted.

**Do not adopt this G or launch more parameter updates on its targets.** Retain the small matched RGB-coordinate signal as descriptive evidence and the full negative geometry result. More reliable target foreground/outer-surface correspondence and a justified center/size identification mechanism are prerequisites for another regressionTTA attempt. No model sweep, online continuation or further GPU work is queued.
