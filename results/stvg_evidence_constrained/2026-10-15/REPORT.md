# Evidence-constrained STVG: completed Round1, continuing matched spatial study

Status: partial. Only the first temporal comparison is complete. Round2, new32 confirmation and official13785-query Table1 are not complete. This report does not promote CURRENT.

## Controlled temporal comparison

Same32 historically exposed development parents (16 per direction), frozen TA-STVG source features, original sealed Caption–Query feedback and fixed spatial references. All predictions were sealed before development GT scoring; there are no GT-derived references or GT Oracle fits. No Bootstrap.

|Temporal arm|HC T-only vIoU (%)|Vid T-only vIoU (%)|HC T+fixed-S (%)|Vid T+fixed-S (%)|
|---|---:|---:|---:|---:|
|projection|33.172911|9.742885|33.485816|10.331769|
|original|30.564737|9.465615|30.843203|10.067195|
|stable|30.529070|9.917721|30.785915|10.575237|
|offset_trust|34.066991|9.966696|34.402504|10.572476|

Original is the old fixed five-step Adam. Stable fits the same averaged candidate KL+L2 with positive-step L-BFGS/Armijo. Offset-trust fits per-offset candidate KL plus full-grid KL(p0||p_theta), lambda1 and L2 .001, tau1, max50 accepted iterations. This is one locked comparison, not a temperature/learning-rate search.

28 inputs were usable and4 explicitly fell back for unavailable temporal input. Original final objective decreased in9/28 fits; both revised objectives decreased in28/28 with nonzero parameter updates. Stable changed24 complete decoded intervals and offset-trust changed22. All28 offset-trust fits hit the50-iteration cap: monotonic descent is established, convergence is not. The independent NumPy reader verified6012 objective/grid-decode checks with max error below2.14e-15. These proxy losses do not establish that MLLM preferences are correct.

The prelocked global equal-domain T+fixed-S endpoint selects offset_trust over stable. Development selection is disclosed; these scores are not an untouched test estimate. Projection and original remain reported controls.

## Next bounded stages

Round2 fixes temporal and spatial update parameters and compares old/new event clock × crop/full-scene IDs. Same-center cached proposals and masks are reused exactly; new centers use the unchanged SelectiveSearch256/GrabCut24/DIS, up to8new+Source mechanism. Full-scene and crop within each clock share candidate geometry and IDs. Original algorithms can be non-deterministic on fresh replay, so regenerated candidates are never silently called exact cache reuse.

An old-cache replay mismatch stopped CPU preparation before new R1/Student/GT. The preserved old partial input and a separate engineering revision keep the original frozen sources unchanged. The student entry imports and temporal interval-to-seal bindings were checked before any new Student fit.

Further32 sources exclude all64 previous coordinate parents and the route original8, plus enumerated old teacher/development registries; this is a recorded-registry exclusion, not a universal never-seen claim. Source capture is complete, adaptation/confirmation scoring has not run. The algorithm will freeze before confirmation, and confirmation cannot choose another recipe. Full official Table1 uses HC validation3482 + Vid test10303 queries, episodic F/T/S/TS with all predictions sealed before GT. Old Native YOLO/SAM queues remain stopped. No new detector, segmenter, DINO or CURRENT promotion.

Only one prelocked output-preservation ablation is allowed if the chosen spatial arm meets the declared drift condition; no new algorithm or parameter grid follows. External teacher training overlap remains unresolved.

All anonymous rows, including negative, neutral and null outcomes, are exported. Raw queries, frames, masks, coordinates, annotation files, weights and cached hidden tensors are excluded.
