# Context-support single-factor oracle

16 exposed Vid training parents, one query each; source GT context oracle; no target or optimizer; descriptive uncorrected paired bootstrap, not TTA/generalization

Only mask support changes. Same B1/frozen PTD4B/late/alpha.25, fixed event .087687 and spatial .170316 stock-merger norm, no optimizer. Correct and old wrong intervals each gain one observed neighbour on either side; correct and old translated box occupancy each gain one per-frame fractional3x3 neighbourhood. Border truncation is reported, not compensated by reselecting controls.

|arm|tIoU%|sIoU%|vIoU%|
|---|---:|---:|---:|
|original|54.003027|61.929737|41.113359|
|event_context_correct|50.703896|60.341448|38.701525|
|event_context_wrong|49.372813|59.441176|38.489139|
|spatial_context_correct|52.173759|60.354072|40.563151|
|spatial_context_wrong|52.173759|59.782234|40.273172|

Primary effects in percentage points:
- event: correct-Base -3.299131, CI[-11.709993326790565, 2.6975385362986053]; correct-wrong +1.638257, CI[-3.091342675398488, 7.876682587928789].
  Correct-Base positive/negative/zero: 3/3/10; eligible correct-wrong: 2/1/10.
- spatial: correct-Base -1.575665, CI[-7.72800664483486, 2.177499153148269]; correct-wrong +0.571838, CI[-0.08127626134076078, 1.2599810796034379].
  Correct-Base positive/negative/zero: 9/6/1; eligible correct-wrong: 9/5/2.

All16 and prespecified13 temporal/16 spatial eligible cases retained; old and context eligibility coincide. Expanded correct/wrong weights differ for7 temporal and16 spatial cases because boundary clipping differs. Independent norm matching fixes merger magnitude, not mask weight or postcast changes.
Format valid 78/80. Spatial event distributions/reference/interval remain exactly Base in32 checks; direct endpoint-only t mean stays54.003027%. Whole-output grammar failures, if present, retain the original zero scoring. No token/sample repair.

Context vs sealed prior large mask (no old intervention rerun):
- event -0.767131pp, CI[-2.616877669310555, 0.31548523664785066].
- spatial +0.136982pp, CI[-0.7741846972423599, 0.9647776151873104].

12 CPU controls; independent64 mask construction exact,240 scalar/tensor geometry max2.220e-16, 4080margin values max0.000e+00, 90root reductions max1.776e-15; 256full BF16 before/after hashes reconstructed exactly. Actual scope frozen/0updates; all16 baseline physical/native outputs replay exactly.
A draft grammar-audit helper initially guessed coordinate token names on CPU; official `<0>`..`<1000>` membership was corrected before the helper ran. Original draft/probe note retained. No GPU/inference/scoring configuration changed.

Decision: event: teacher advantage not established; stop scalar mask iteration; spatial: teacher advantage not established; stop scalar mask iteration.
Stop additional scalar-mask variants under the preregistered resource decision. Preserve sharedTHW/dual-reader as the mainline; tested configurations do not rule out all latent mechanisms. Local positive scores alone do not measure vector alignment with the optimized direction. Directional evidence-conditioned residual is a proposed next parameterization, not implemented/trained or validated here. Actual-reader learning and OPD remain gated. No new target, external qualification, production promotion or hyperparameter grid.
GPU allocation 153.088996550s including load/inference/seal; cumulative42493.15652965409s cap=null. No GPU failure or optimizer in this run. All historical failures/corrections/results retained; public release contains no videos,labels,captions,weights or raw predictions.
