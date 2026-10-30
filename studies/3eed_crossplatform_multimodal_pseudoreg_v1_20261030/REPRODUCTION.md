# Round7 reproduction and finite public-export contract

This document and `tools/build_round7_export.py` are preparation artifacts. Preparing them does not mean the full experiment, independent scoring, engineering positive control, root review, export or publication has completed. The builder is not executed by this preparation task and has no Git/network action. Root owns final result interpretation, publication readiness and the eventual build/push.

## Scientific version and source provenance

The experiment uses one3EED model, initialized from the genuinely Waymo-only supervised epoch100 checkpoint. The source model completed100epochs/15300optimizerupdates; generic frozen RoBERTa pretraining is declared, and mixed-task checkpoint initialization was forbidden. Expected source model checkpoint SHA256 is:

`0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd`.

The source training plan SHA256 is`66fabf36898f1bc92124b680435e598e7bca0809e48aeb2da35e272331afe7ce`, and model state fingerprint is`0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9`. Source training/audit code and the full source validation history are provided in the separately published round6bundle; no model/optimizer/RNG tensors are redistributed in this round7bundle. The R7runner verifies epoch/platform/provenance, safeloads weights only, strictly loads all model state and checks the fingerprint. Do not substitute the released mixed-domain model and call it a Waymo-only source.

The official upstream3EED commit and compatibility setup are documented in the round6reproduction artifact. Restore that audited checkout at the relative path expected by `crossplatform/round6/factory.py`, including the local generic RoBERTa assets and CUDAoperators. The public round7code includes the source constructor and pure geometry dependencies; it is not a duplicate full upstream checkout, data release, compiledbinary package or weight archive. Dataset/tokenizer/model assets must be locally obtained under their terms. The historical hashes bind the actual audited run; a new source training on different software/hardware can require a new registered model fingerprint and plan. Never rewrite the historical hashes to pretend a new run is byte-identical.

## Input generation and exposure

Regenerate the prior R5puretext/path input manifests from the legally accessible official data, using the prior R5preparation code and its recorded version. R7's input preparation then reads `crossplatform/round5/baseline_engineering/inputs/drone_all.json` and `quad_all.json`, and only the id/path/object-index mapping from the separately kept R5scoring index. No GTclass/box/positive_map is supplied to the model or G.

The complete available targetVAL is5912expressions: Drone2984/2scenes and Quad2928/3scenes. All unsupported/negative cases remain in full evaluation. Twenty engineering-smoke inputs are fixed before model outputs/GT scoring: four different frames per scene, aiming for three supported opening colors with available color diversity plus one unsupportednegative; fixed saltedSHAfallback fills unavailable supported slots. The frozen parser supports1442/5912input openings. The two zero-supportQuadscenes retain explicit fallback cases, so the20smoke inputs contain9supported and11unsupported. All5912inputs have a firstlexiconnoun regexmatch; no GTfallback is required.

All targetVAL data were previously researched/scored in R5. They are source-model-unseen target platforms, but not a fresh research holdout. The source model's supervised data remain Waymo-only. Smoke enrichment is for engineering feedback eligibility and does not estimate full-population accuracy.

The strict GPUmanifest contains exactly nine fields: `id,platform,scene,pointpath,metapath,imagepath,caption,suffix,seed`. Original scene/frame/caption/path/index tables and manifests are private and are not copied into the public bundle. Public protocol projections preserve their original SHA256 and aggregate counts. Regenerate private inputs locally; there is no claim that a privacy-projected plan has the original bytehash.

Sampling is fixed per expression: `default_rng(seed).choice(rawN,16384,replace=rawN<16384)`. Preserve rawfloat32XYZ copying and Drone's in-place float32`z+=1.8`; QuadXYZ is unchanged. Avoid a float64shift/cast substitute when checking oldR5pointbyte equality. The model receives the cleaned fullcaption plus exactly one` . not mentioned` suffix. It uses puretext firstnounBBF scoring and fixed source argmaxquery; the nativeTop10tie order remains separately recorded. A queryslot is not a stable physical-instance ID across different inputs.

## RGB/camera and G

RGB is additional information to the nonlearningG, not an extra learned grounding teacher. Use only the camera whitelist K/E/D from metadata; annotation-containing byte streams must not materialize GTsubtrees into a consumer. Drone sourcecoordinates undo thezshift beforeE, Quad usesE. Actual imageW/H are read fromRGB. MissingD is accepted only under the registered`assume_zero_declared` policy; zero distortion is an explicit assumption, not a physical fact.

Root reviewed five frozen point-only projections. Coarse geometry was not grossly inverted, but exact pixel accuracy, timing synchronization and complete occlusion were not established. The observed photos can have green casts/uneven illumination, so frozenHSV support is not calibrated color truth. The review must not tune Gthresholds. No originalRGB, projectedpixels, camera matrices or pointarrays are publicly copied.

G uses the unchanged finite targetcolor+noun opening parser, RGBHSV support, observed-cloud pixelzbuffer, source fixedquery neighborhood/components, duplicate/competitor checks and two-sided geometric support. It can return zero reliability. The first version preserves source sizes; all size weights arezero. Only supported center axes may move, with the frozen Gdelta rejection rule. A parsable color is not an acceptedpseudoaxis, a correctpseudobox or a usefulTTA signal.

The FOVhistory is retained explicitly: the previous implementation could round a negative floating ray into an image pixel; the repaired implementation checks continuous half-openFOV first, thenfloorpixelcells. `history/before_FOV_fix` and `CPU_FOV_REPAIR_FIXTURE.json` retain the exact counterexample, old/fixedsourceSHA and synthetic results. The historical code is not the current scientific algorithm. No photometric or geometrythreshold search is performed after viewing targetGT or scores.

## Adaptation and all16controls

Only the four finalConv tensors in `prediction_heads.5.{center_residual_head,size_pred_head}.net.8.{weight,bias}` are eligible for updates (1734parameters). Because currentGsizeweights arezero, only867centerparameters can actually receive nonzero updates. Preserve fullmodel eval, frozen BN/dropout and episodic restoration of parameters/buffers/gradient references/modes/RNG. There is no teacher, EMA, source retraining, heading estimator, PCA or query reranking.

The experiment registers B1, oneandthreesteps, SmoothL1`beta=.1metres`, plainSGD without momentum/weightdecay/clipping, and`lr=.25*beta/max(center_self_kernel,size_self_kernel)`, where each selfkernel is the cachedfeature squarednorm plusbias. Do not use a bare arbitrary LR while claiming identical normalized dynamics.

Each of RGB-textG, PC-onlyG and shuffled-colorG contributes five selected-box arms: directG, parameterSGD1/3, exactoutputspaceSGD1/3. With source, there are16arms, all retaining the same source query. Gzero/unsupported/invalidsource conditions must retain identity and explicit reason/status. Count acceptedreliability, actualpseudo displacement, executedsteps, nonzerogradients, parameterchanges and actualoutputchanges separately. Status`updated` is not enough to prove a nonzero parameter update.

For frozenfeatures and a lastlinear layer, the selectedquery update is exactly a feature-kernel-scaled outputspace step in real arithmetic; the control also replays crossqueryspillover. Numerical bridge tolerance and traces are registered. A parameterSGD gain over source alone is not proof of information or mechanism beyond directG/outputspace controls.

## Execution stages

Use the registered stage-specific plans and their original hashes, without editing a running plan. Supply regenerated private manifests and the actual local sourcecheckpoint location; the public plan has that private location removed. The original commands have this shape:

```
flock -n <workspace>/research/round2/gpu.lock <python> <workspace>/crossplatform/round7/experiment/run_crossdomain.py --phase smoke --manifest <private-smoke-manifest> --plan <registered-smoke-plan> --plan-sha256 <original-plan-sha256> --output <new-smoke-directory>
flock -n <workspace>/research/round2/gpu.lock <python> <workspace>/crossplatform/round7/experiment/run_crossdomain.py --phase full --manifest <private-full-manifest> --plan <registered-full-plan> --plan-sha256 <original-plan-sha256> --output <new-full-directory>
```

Root coordinates one physicalGPU. Seal every requested prediction/trace/repeat-restoration receipt before granting stageCPU scoring. The label-free auditor verifies program/input/prediction hashes, repeat/realConv bridge and restore receipts, sizeweightszero, normalizedsteps and parameter/outputspace agreement. It does not load an annotation index.

Only after both stage prediction transactions are sealed may root issue actual targetGTscore releases bound to allprediction/code/input/checkpoint/index hashes. The CPUscorer uses puretextfixedq and nativeyaw0, preserving raw/clamped sizes, requested/finite counts, allnegativecase deltas and GT-assistedsourcepoolTop5/10/256 diagnostics separately. General3Dconvex independently covers unique selectedarm boxes; full256coverage uses auditedbatchedpolygon×height and is not described as all256independent3D. Scenecluster bootstrap is descriptive with only2Drone/3Quadscenes. TargetGT must not select color/thresholds/query/lr/stepcount/alpha or arm.

The separate real-head engineeringpositivecontrol runs after the full5912predictionseal. It gives the firstfixedsmoke case an artificial+0.05m xcenter target and tests nonzerobackward/parameterchange, sourceConvbridge, outputspace equivalence and fullreset, withoutRGB/G/targetGT/taskmetrics. Attempt001 failed a cross-process source-pool bitexact assertion before backward; attempt002 reached real backward but failed the registered equivalence tolerance. Their originalcode, failure receipts, preparation/version hashes and scalar drift evidence are retained; neither attempt is relabeled passed.

Attempt003 is the two-mode v3precisiondiagnostic under `POSITIVE_CONTROL_V3_PLAN_LOCK.json`. Its actual state is `strict_FP32_only_engineering_pass_original_mode_failed_preserved`: `original_mode_passed:false`, `strict_FP32_mode_passed:true`. The originalmain flags produce maximum parameter-versus-outputspace error0.0001430511474609375m, exceeding the unchanged0.00005m gate. StrictFP32 passes at0.000003814697265625m for1step and0.00000762939453125m for3steps. Both modes record actual nonzero gradients/parameterchanges, repeatedsource/currentConvbridges and fullreset. Complete relative-delta/gradient/feature-kernel/crossqueryspillover/cross-process drift statistics are public; no actual sourcebox coordinates are included. Cross-process pool drift remains reported and is not explained solely by the strictFP32 equivalence pass.

The publication gate requires this exact failure-preserving v3schema, verifies both1/3step judgements and the unchanged tolerance, and rejects relabeling originalprecision as passed. StrictFP32 success establishes only the artificial update/equivalence interface in that precision. The frozen fulltask outputs, precision flags, G, metrics and scientific decision are not replaced or retested by the engineeringdiagnostic; informative G or TTA benefit does not follow from this positivecontrol.

## Final publication gates and source-index schema

The finite builder defaults to `research/round7/experiment/public_export` and refuses an existing destination. It creates no output until root supplies both finalcontrol files and every referenced artifact actually exists and hashes match. Do not run the builder just because this preparation document exists.

`PUBLICATION_READINESS.json` must contain alltrue booleans:

```
smoke_and_full_predictions_sealed
full5912_scored
full_unlabelled_audit_passed
real_positive_control_completed
root_final_review_passed
report_decision_privacy_reviewed
no_raw_data_media_weights_in_export
```

`EXPORT_FINAL_SOURCES.json` must map each following key to exactly`{"path":"workspace-relative-file","sha256":"actualSHA256"}`:

```
smoke_prediction_seal
full_prediction_seal
smoke_scalar_summary
full_scalar_summary
full_unlabelled_audit
positive_control_receipt
positive_control_seal
positive_control_plan
smoke_score_release
full_score_release
root_report
root_decision
root_final_review
```

The three positive-control keys must bind attempt003's actual `ENGINEERING_POSITIVE_CONTROL.json`, `SEALED.json` and `POSITIVE_CONTROL_V3_PLAN_LOCK.json`. Every sealed engineeringfile hash and programhash is verified. The7boolean readiness key `real_positive_control_completed` means the completed two-mode diagnostic with originalfailure preserved; it never means originalmain precision passed. The receipt, seal and plan must agree on the genuine Waymo sourcecheckpoint/model fingerprint, smoke/fullpredictionseals, artificialtarget, reliability, registered5e-5metre tolerance and fixed1/3steps.

Optional `component_aggregates` and `component_plan` keys use the same exact path+SHA schema and must appear together. They bind actual `component_diagnostic_attempt001/PUBLIC_AGGREGATES.json` and `COMPONENT_DIAGNOSTIC_PLAN_LOCK.json`, respectively. The builder verifies the originalfullpredictionseal/sourcecheckpoint, plan/program hashes, bounded14case diagnostic purpose and noGT/no-method-change assertions, then exports aggregatefields only, the path-stripped plan, diagnosticcode, method note and preparationreceipt. This result describes graph-support overlap for hash-selected eligiblecases;6/14 have both overlap fractions at least0.8. Observed pointgraph disconnection is not physical-instance identity, and this diagnostic does not change G or establish that a rejected alternative is correct. It is separate from the main evaluation and no selection uses GT/IoU.

Optional `cross_stage_reproducibility` binds root's actual `CROSS_STAGE_REPRODUCIBILITY_PUBLIC.json` via path+SHA. It is admitted only as a scalar aggregate tree: no originalrows, sampleIDs, physicalscene/frame names, coordinatevectors or row-object arrays. It is checked by the same privacydenylist and preserves its originalhash. This optional file may report genuine cross-stage reproducibility limits; it cannot silently repair the original predictions or broaden a passing precision claim.

The actualscorer output is`SCALAR_SUMMARY.json`, with sibling`RESULTS_PRIVATE.json`, `SCORER_LOCK.json`, `SEALED.json`. Rootfinalreview must contain`publication_ready:true`, `report_sha256`, `decision_sha256` matching the actualfinalreport/decision. This is a rootowned review gate, not a methodeffectiveness gate: a valid negative/no-adoption experiment can be published. `runner_failure_receipts` may additionally be a finite list ofpath+SHA records; only status/exceptiontype/phase/counts/hashes are projected, never rawlogs/tracebacks/messages.

When allconditions are met, root may run:

```
<python> <workspace>/tools/build_round7_export.py
```

No part of this builder pushes Git or changes models, predictions, source locks, scorer thresholds or deployment. Root must inspect the candidate, manifest and privacyprojection before publication.

## Exported versus private artifacts

The finite code list contains singlemodelG/adapter/camera/pipeline, runner/positivecontrol v1/v2/v3precisiondiagnostic with V1/V2 failure scalar/hash projections and preparationreceipts, CPUfixtures, originalFOVcounterexample/history, targetinput preparation/audit helpers, currentandhistoricalscoring-interfacefixes, the sourceconstructor and puregeometrydependencies. It does not recursively copy a directory. Scientificcode is byteexact. If an audithelper uses an absolute workspace `Path`, only thatroot anchor is replaced by the equivalent file-relative root; document paths get a workspace placeholder. Every such projection records originalsourceSHA, exportedSHA and mode in`EXPORT_PROVENANCE.json`. The algorithm is not altered, and originalhistorical lock hashes are not misrepresented as hashes of relocated JSON/code.

Plans are field-projected and retain originalplanSHA through provenance; checkpointlocations/privatepaths are omitted. Input counts use platform/sceneshash, not real scene names. RootRGBreview publishes only coarse writtenobservations/flags; realphotos, matrices and overlayfiles are excluded. Finalresultcases are explicit losslessly compressed scalarCSV (`FULL_CASE_SCALARS.csv.gz`,94592rows; `FULL_PSEUDO_DIAGNOSTIC_SCALARS.csv.gz`,17736rows) with opaquehashedcaseID and hashedscene, no originaltexts/frames/paths. Compression is `gzip.compress(csv_bytes,compresslevel=9,mtime=0)`, with no originalfilename and a zero timestamp; the builder asserts decompression is byteidentical to the original scalarCSV. `SCALAR_TABLE_SCHEMA.json` records columnorder, rowcounts, CSV/null conventions, Python/zlib versions, compressed and uncompressed SHA256. This packaging change neither rounds nor changes any scalar, drops a negativecase, nor changes sample/arm order. Use a standardgzip reader (for example `gzip.open(path,"rt",encoding="utf-8",newline="")` with `csv.DictReader`) to recover allrows. Repeated builds on the recorded compressionruntime are reproducible; hashes retain crossruntime header differences if any. Derived IoU/error/gradient/displacement scalars can be public; original GT/predictioncoordinates, pseudo boxes, rawmasks/P/embeddings/caches cannot. Privateindices are used solely for localreconstruction/auditing and never bundled.

Allcase/negative scalars, allrequested/finite/accepted/nonzero denominators and matchedcontrol summaries must be preserved. Do not publish only positiveexamples or conflate the artificialpositivecontrol with naturalGperformance. The rootfinalreport/decision supplies actual scientific conclusions after finalmeasurement; this reproduction procedure makes no currenteffectiveness claim.
