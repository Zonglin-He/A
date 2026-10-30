# Proposed round2: partial-surface hypotheses and a zero-source residual adapter

Status: draft for methodological review, not a final locked algorithm or result. No new labels accessed. One mechanism, one small development cohort, one held-out confirmation cohort; no unbounded grids. Before model execution all numerical constants, code hashes and arms must be frozen in an executable plan.

## Claim and competing explanations

Observed failures: a narrow visible face can have very high point anisotropy without revealing vehicle long-axis orientation; repeated point sampling can preserve a wrong neighboring instance. Hypothesis: explicitly retaining alternative face assignments and checking held-out point evidence can identify some ambiguous corrections and supply a more useful adaptation target than prediction consistency. A competing explanation is that all useful information is already captured by direct geometry processing. Another is that single-view partial geometry cannot identify pose or same-class identity at all.

Falsification: if normal/face-coverage diagnostics do not separate later harmful pose changes from beneficial ones, do not call the reliability signal validated. If a direct latent readout matches/exceeds adapted output under the same inputs and budget, there is no demonstrated added value from parameter updates. If identity switches improve surface fit but worsen GT overlap, geometric plausibility is insufficient for linguistic identity. Failure of one draft does not establish that all geometric TTA is impossible.

## Allowed evidence and partial-observation model

Use exactly the sampled XYZ presented to the frozen source, the original caption, frozen query features and source query/token embeddings. No extra RGB model, GT class/bbox, fabricated temporal matching, or assumed sensor ray origin. More complete raw pointcloud access is not silently granted to one arm.

Keep8source candidates ranked by unchanged puretext headnoun score. Source score remains the prior, so a geometrically clean irrelevant object is not treated as semantically correct. Each candidate has three initial yaw hypotheses{0,PCA,PCA+pi/2}, existing sourcecenter/fullsize, and a bounded center offset. Local crop is fixed from source, never recomputed using GT. Initial gates8unique points and sourceL/W1.5 remain explicit comparators, not newly tuned by confirmation.

Geometry teacher replaces a single average box-SDF score by a robust partial-face likelihood. Estimate unoriented local normals using kNN covariance inside the predicted crop; normal uncertainty downweights poorly planar neighborhoods. For each point, marginalize over the six box faces and an outlier component. A face contributes normalized point-to-plane distance, disagreement between measured and face normal, and a soft tangential support constraint. Missing/unseen faces carry no occupancy penalty. This models partial observations rather than fitting a complete solid to every point. Fit center/yaw with dimensions anchored to the source; trim/robustify outliers to avoid allowing adjacent objects to dominate. Candidate,face assignments and yaw mode remain distinct latent variables.

Split unique crop points by a deterministic hash into fit/check halves. Fit modes on onehalf, compute the same energy and normal/coverage diagnostics on the otherhalf. Retain mode entropy and check-half evidence, do not collapse one high-anisotropy face into artificial certainty. Multiple plausible face assignments must remain ambiguous when the data cannot distinguish them. This does not guarantee uncertainty calibration; that is a measured hypothesis.

Identity energy combines source contextual-token prior with normalized partial-face evidence. Explicit textual relational constraints may be added only if their coordinate meaning and parser coverage are validated before labels. Default draft does NOT assume every caption has usable orientation words, that image-left equals LiDAR-y, or that same-class instance identity can be recovered from geometry. Until that audit passes, no extra relation term is enabled; report candidate switches as geometric reweighting, not proven language reasoning.

## Actual parameter adaptation and zero-init bridge

Freeze the complete released model. Capture the288-dimensional finaldecoder query feature using an audited read-only hook. Add a zero-initialized shared affine residual adapter producing5outputs perquery:3boundedcenter offsets,1yaw residual and1semantic logscore correction. W shape5×288 plus5bias=1445trainableparameters. Atzero weights, center/size/readout are bitexactsource and yaw=0. Extents stay source-fixed in this first attempt; the original6Dhead is never misrepresented as a yaw predictor.

The latent posterior supplies candidate and pose supervision. The executable candidate objective uses Huber distance between normalized log-probabilities, plus robust pose regression and a source residual trust-region penalty. A pre-data engineering check showed tiny KL roundoff amplified by Adam even for mathematically identical source targets; the same Huber logprob loss now gives a bitexact zero-update source control. This change is recorded in `CPU_ENGINEERING_CHECK.json`, before any development scores. This is a real parameter update to an added test-time interface; finalprediction must be recomputed from adapted parameters, not read from a cached teacher. Source neuralweights remain immutable. Record zero-step bridge equality, nonzero gradient count, parameter delta, teacherfit/check energy, yaw/posterior entropy, chosenidentity, and exact restoration of adapter/model state and RNG after each episode.

The main question is whether this bottleneck and held-out evidence improve predictions over directly using the latent posterior. Teacher distillation loss reduction alone is insufficient. A new affine adapter trained on the current sample is not evidence of pretrained-backbone adaptation or a novel contribution.

## Fixed comparison matrix proposed

1. Officialweights + unchanged text-only source6D/yaw0.
2. Original fixed PCA7D.
3. Original fixed surface-candidate selection7D.
4. Same frozen features/teacher information, zero-update multi-hypothesis latent readout.
5. Zero-initialized residual adapter,0steps: exactsource bridge engineering control.
6. Structuredteacher residual adapter,1and3fixedsteps, samegeometry/inputbudget.
7. A matched loss control retaining source teacher targets without partial-face evidence,3steps. This separates geometry signal from generic output fitting or added capacity.

No class-assisted primary arm. Sealed source scores permit a separate post-seal diagnostic of official GTclass-assisted span on the same examples. No sample-specific GTrollback. A post-seal allproposal oracle may diagnose representation/selection ceilings, never enter online selection.

## Development and confirmation boundary

Paths already locked in `COHORT_PATH_LOCK.json`, hash `cd61a4bcf0e3f98ee841d786d91d8ffea836c59638d089d5105d81acf6872149`. Confirmation36is fixed before development scores. Numerical constants must be set from units, training scale and engineering checks before dev execution; at most one documented dev-driven correction of implementation or mechanism, not a grid. The final mechanism is frozen before any confirmation model evaluation or labels. All confirmation arms are sealed before a separate CPUscorer reads numericboxes. Log preparer access to mixedJSON containers accurately.

Report overall/platform results and dev-confirm scene disjointness, plus historical scene exposure. Pair outcomes and negative tails are mandatory; scene-cluster intervals are descriptive with only1Drone and2Quad confirmation clusters. Report optimizer cost separately from model inference, CPUgeometry, setup and downloads. GPU remains parent-scheduled.

## Executable pre-data refinements

`surface_teacher.py` and `residual_adapter.py` define one fixed configuration. Teacher uses16unique crop points to permit independently estimated8-neighbor normal graphs in two partitions. It does not inherit the oldL/W≥1.5gate: that stays in PCA/surface baselines only. Nonpositive source extents are explicitly ineligible for teacher fitting, never abs/clipped; invalid predictions remain in coverage/metric accounting. All8candidates remain in prior/rank accounting.

Each center/yaw mode is optimized with fixed-size boxes, center bounds±.15ofsourceextent,20L-BFGS-Biterations. Outlier mixture.2; normal term.2; likelihoodtemperature.02; centerprior.5. Fit/check energies select retained modes. Equivalent modes are merged with no restart-multiplicity reward; yaw is always moduloπ. There is no angle or crossmodecenter averaging. A mode is accepted only with posterior≥.65, fit/check angle agreement≤15°, and two distinct horizontal-normal face families supported in BOTH partitions(≥3points/family, planarity≥.1, XYnormalnorm≥.8, tangential observedspan≥.25ofsourceextent). These are unvalidated identifiability heuristics, not a guarantee of true coverage.

Accepted geometry supplies at most±1 logscore bonus; rejected hypotheses leave prior and pose unchanged. Directlatent readout uses exactly that posterior and retained mode. The adapter has1445parameters, RMS-normalized frozen queryfeatures, bounded center residual.15extent, bounded yaw±π/2, fixedsourceextents, Adam lr.01,1/3steps, unitgradientclip; rank/pose weights1 and residualL2weight.1. Selectedmode angle regression uses wrapped moduloπresidual; unresolved multimodal poses are not averaged. A synthetic two-plane versus single-plane check and zero-step/three-step source-control check passed; they are engineering evidence, not task scores.

The rejection statement applies to teacher targets and directlatent output only. Shared adapter weights can move the center/yaw of rejected queries through updates from accepted ones. There is no strict network-output fallback in this frozen version. `ADAPTER_DIAGNOSTIC_LOCK.json` pre-registers a deterministic, bitexact-checked optimizer replay that records finalchosen acceptance, rejected top8/all256 center/yaw drift and changed-query counts. That replay reads noGT and adds only explicitly separated diagnostic CPUcost. If spillover explains harms, the gate must not be credited with source protection.
