# R9 fixed-q0 geometry/regression diagnostics: reproduction and finite export

This study is `studies/3eed_fixed_q0_geometry_regression_diagnostic_20261030`. Three actual GPU stages are complete: normal160×11, GT-controlled160×13 and a post-normal-score pseudo-exactfit bridge160×3. The finite exporter is preparation only until root's final report, decision, review and readiness artifacts exist. It does not run a model, new GT scoring, network or Git, and does not create a public candidate merely by being prepared.

## Cohort, source and statistical units

There are160 distinct inputs,64Drone and96Quad,32 per each of five available scenes. `prepare_inputs.py` starts from the R8 input-only5912manifest, excludes R8smoke40/R7smoke20 IDs and pointpaths, and sorts each scene by `SHA256('R9_fixedq0_geometry_v1|'+id)` with id tie-break. Available distinct frames after exclusions are2791/169/2097/313/482; no fabricated replacement, color, score, GT or geometry selection is used. The original strict nine fields and seed/suffix remain unchanged. Private manifests and original identifiers are not distributed. This scene-balanced subset is not the full5912population microaverage, and all targetVAL was previously researched in R5/R7/R8; there is no fresh confirmation claim.

The only grounder is the genuine Waymo-only source100 model:100supervised sourceepochs/15300updates, checkpoint SHA256 `0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd`, fingerprint `0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9`. Generic frozen RoBERTa pretraining is declared; released mixed-platform task weights and target supervision were absent from source training. Normal R9 creates its own forward/cache. GT and bridge reuse that exact normal source query/features rather than refreshing slots or borrowing R8 predictions.

The4320 arm rows are related controlled outputs on160cases, not4320 independent samples. There are only2Drone/3Quad scene clusters. Scene means and any conditional subgroup/CI remain descriptive. Fixedq0 removes reselection structurally within R9; R8/R9 have different cohorts/protocols, so their score difference is not a same-cohort causal estimate of recovered association loss.

## Runtime and frozen external information

Restore pinned upstream [worldbench/3EED](https://github.com/worldbench/3EED), commit `843a901f5367a6be5cc484b97c6ecad462961dfa`, at `crossplatform/round3/p0/upstream`, apply the included R6compatibility patch and retain the expected `crossplatform/upstream` alias/genericRoBERTa assets. The patch changes only obsolete unused `cgi`/optionalOpen3D imports. Build the original pointnet2/batchpointnet/roiaware operators for the local ABI/GPU; originalRTX5090 usesCUDA12.8/sm120 and PyTorch2.7.0+cu128. See the prior R6source-training reproduction for full source setup/validation. No compiled operator, taskweight or optimizer/RNG tensor is redistributed.

All real stages use explicit FP32: cuDNN TF32=False, matmul TF32=False, cuDNN benchmark=False, cuDNN deterministic=True, no autocast. Source BN/dropout remain eval. Normal uses public frozen [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, original PyTorch SHA `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`, local safetensors derivative SHA `7a7177862f66645c5e923e31e07318fd609769378a197a21903fbbcc07e446b1`. All398 learned tensors strictly load and match the safe conversion roundtrip. Exact source/processor/tokenizer and conversion hashes are in the resource receipt; download them from the fixed official revision, never copy a private signedURL or substitute a differently pretrained checkpoint.

The included R8semantic encoder/acquisition/conversion code is a fixed dependency, not a new model choice. Canonical RGB shortedge224 bicubic/center224 preprocessing, official mean/std, literal fullcaption without grounder sentinel,77token truncation receipts,512D L2normalization and raw tight/context cosine are retained. No prompt/color/threshold search is introduced. CLIP supplies similarity, not3Dboxes, calibrated identity or proven relational grounding. It and the grounder run serially on one sharedGPU in normal; GT/bridge use saved features without another RGB/CLIP/backbone forward.

Input sampling is exact float32 XYZ from default_rng(seed)16384points (replacement only if necessary), Drone in-place z+=1.8/Quadidentity. K/E/D come from the streaming camera whitelist; missingD uses declared assume-zero. Fullimage/crop evidence and observed point support retain occlusion/calibration/crop contamination limits. Source/GT drawing uses audited yaw and camera transforms only after the separate GT release.

## Stage1: normal fixed-q0,11 arms

The original puretext argmax q0 stays fixed for every arm. No semantic group re-association can replace it. Existing RGB/fulltext and PC-only geometry evidence can propose eligible center corrections around q0; source dimensions remain and normal sizeweights arezero. The two modes each have directG, parameterSGD1/3, exactkernel-outputspace1/3, plus shared source11outputs. `fixed_q0_geometry.py` wraps the frozen R8 grouping/evidence routines while retaining q0; it does not use a GT-selected query.

Plain SmoothL1 beta=.1m uses alpha=.25 and LR `.25*.1/max(center_self_kernel,size_self_kernel)`, no momentum/weightdecay/clipping. Globalstrength shrinks the same pseudo target for allcontrols; it is heuristic, not calibrated correctness. Every episode starts/restores identical source state and records actualfloat32 targetdelta, gradient, parameter/output change, q0 and kernel error. Eligibility or executedstep is not evidence of nonzero or useful update. DirectG uses exactly the same target without optimizer, and kernel controls expose frozen-final-linear equivalence.

```text
flock -n <shared-gpu-lock> <python> crossplatform/round9/run_diagnostic.py --phase normal --manifest <private160manifest> --plan <new-normal-plan> --plan-sha256 <SHA> --output <fresh-normal-output>
```

Seal predictions and all160 sourcefeature caches before targetlabels. Root's postseal CPU release binds complete source/code/input/CLIP/prediction hashes; only then `score_normal.py` opens the private annotation join and produces derived scalar results and separately sealed PRIVATE_GT7D/GT6D controlsidecars. Those sidecars never enter normalG. The independent no-GT normal audit and separate scored reduction/coordinate audit must actually pass; CPUfixtures or readiness alone are insufficient.

## Stage2: labelled GT controls,13 arms — NOT TTA

A pre-GT capacity plan is preserved. Its runtime materialization binds the completed normalpredictionseal and the postrelease privateGT6D index, preserving every preplan key. AllGTcontrols reuse the same normal q0/pool/features. No GTyaw, RGB/CLIP/pointG or backbone call is used in this stage. Sourceplus C/S/CS each directtarget, conservativeSGD1/3 and exactfit gives13outputs.

Masks have distinct experimental roles: GT_C supplies center coordinates while retaining source dimensions; GT_S supplies size coordinates while retaining source center; GT_CS supplies both blocks. These are supervision masks in an oracle capability test, not unlabelled reliability or semantic identity masks. GTdimensions follow the dataset/localyaw parameter convention and may differ from yaw0 global enclosure dimensions. With predictionyaw0, even directGT6D is not a maximal-IoU upperbound. Center supervision can teleport a wrong source query; full-population oracle gains are not proof that deployable TTA can identify the target.

`head_exact_fit.py` uses a real `.5*sum(masked_error²)` loss, backward and one per-head inverse-selfkernel SGD step. It does not replace returned coordinates by the target. Allactive masked coordinates must fit within5e-5m, inactivecoordinates remain source, parameter changes/gradients/restore are recorded. Coordinate-fit capacity is primary; any taskIoU is explicitly oracle-only. ConservativeSGD and exactfit change loss and step normalization together; their difference does not isolate LR causality.

```text
flock -n <shared-gpu-lock> <python> crossplatform/round9/run_diagnostic.py --phase gt_fit --plan <new-postseal-GT-runtime-plan> --plan-sha256 <SHA> --output <fresh-GT-output>
```

A separate post-prediction CPU release gates GTcontrolled scoring. The independent actual auditor checks selected coordinate errors, traces, inactivecoordinates, samecache and allsummary reductions. Its first version wrongly compared a serialized float32 subtraction against a float64 subtraction at1e-12. The original auditor and FAILURE_AND_FIX receipt are retained. The fix compares the declared float32 operation exactly; scorer/predictions/scientific tolerance/results were not changed. Do not label this measurement bug a model failure or hide its lineage.

## Stage3: exploratory optimization bridge added after normal scores

This bridge is registered after seeing normal results. It is not fresh heldout confirmation, a preregistered normalmethod arm, a newG signal or a unique TTA claim. Its GPU program accepts only sealed normal predictions/sourcefeatures and the exact normal directG rawfloat32 boxes as targets; no scoredpath, GTindex, newRGB/P/CLIP/G/backbone call is allowed. Sourceplus RGB/PCcenter-exactfit gives3outputs. It preserves q0 and source dimensions, using center-only maskC. Source-equal targets naturally have zero gradient/change; no jitter forces activity.

```text
flock -n <shared-gpu-lock> <python> crossplatform/round9/run_pseudo_exactfit.py --plan <new-bridge-lock> --plan-sha256 <SHA> --output <fresh-bridge-output>
```

The same true L2/inverse-kernel update can reach an already available pseudo box. DirectG remains the same-information no-update control. The bridge distinguishes underfitting of the existing target from target correctness, but matching directG is not unique parameter-adaptation benefit or evidence of generalization to new views/pointclouds. Preserve absolute5e-5m tolerance and relative residual divided by actualnonzero targetshift; relative errors are undefined at zero, not set to0. Actual320 backward calls differ from30RGB/3PC nonzero target/gradient/update cases.

The independent actual bridgeaudit binds its prediction/result hashes and verifies samecache/targets/sourceprior, coordinate and scalar reductions. Additional optimizerfidelity reports keep denominator differences: kernel-ratio summaries use full64/96 platformcases while projection/targetfitting statistics use nonzero11/19RGB cases. FP32 microscopic overshoot of the idealized step bound remains reported; no strict zero-rounding claim is introduced. Loss and normalization changes together prevent an isolated LR-causality claim.

## Identity condition, actual private drawings and manual subset

The geometry scorer's annotated-spatial condition uses target IoU and a margin against deduplicated annotated otherboxes; the inventory may be incomplete. In this cohort its13Drone/47Quad mask exactly equals sourceAcc25 successes. It is outcome-conditioned, not independent semantic identity proof, and unconfirmed does not mean wrong object.

The prelocked overlay sample uses at most2confirmed+2unconfirmed per scene under `R9_identity_overlay_v1|id` hash, no extra success/score sorting. GT7D comes only from the separately sealed scoredsidecar. Drawings rotate GTyaw correctly, keep modelyaw0 and project through Drone E undo-z/QuadE. They preserve the full caption, point support, explicit source/direct/GT colors and partial-visibility limits. Actual private20figures are sealed; no image/caption/coordinate is exported.

Root actually viewed all20, nonblind to GT overlay and spatial status. Counts12same-visible-target-plausible/3different-visible-object/4offtarget-or-outside/1occluded are descriptive human plausibility labels on a stratified20gallery, not authoritative blindsemanticGT or population error rates. The manual12subset is6+6 across5scenes, including two spatiallyunconfirmed examples; its aggregate is supplemental and cannot replace160population or13/47conditional tables. Allpredictions/G/thresholds remain unchanged. Only scalar manualsummary, manualsubset aggregates and their independent recomputation are public, never private labels/IDs.

## Finite public-export gates and source interface

The builder refuses an existing output and performs no export until actual final files/hashchecks pass. It defaults to `research/round9/experiment/public_export`, then root may publish the reviewed tree under the study prefix. Historical programs/locks stay intact; a fresh replication regenerates its private cohort/assets/sidecars and registers new hashes/releases rather than rewriting oldlocks.

`PUBLICATION_READINESS.json` requires these seven true fields:

```text
three_actual_prediction_stages_sealed
three_actual_score_stages_sealed
normal_unlabelled_and_all_actual_independent_audits_passed
GT_ORACLE_and_postscore_bridge_roles_explicit
identity_condition_and_nonblind_manual_review_limits_explicit
root_final_report_decision_review_passed
privacy_reviewed_no_weights_features_GTindices_sidecars_media_captions_coordinates_or_privateIDs
```

`EXPORT_FINAL_SOURCES.json` maps each following name to exactly `{"path":"workspace-relative-artifact","sha256":"actualSHA256"}`:

```text
normal_prediction_seal
normal_scalar_summary
normal_score_release
GT_prediction_seal
GT_scalar_summary
GT_score_release
bridge_prediction_seal
bridge_scalar_summary
bridge_score_release
normal_unlabelled_audit
normal_reduction_audit
GT_capacity_audit
bridge_independent_audit
final_mechanism_review
manual_identity_review_summary
manual_subset_error_summary
optimization_fidelity_diagnostic
manual_optimizer_supplement_review
root_report
root_decision
root_final_review
```

Allthree actual predictionstates/counts/arms/strictprecision/source/program/file/plan hashes, scoreSEALED/RESULTS/SCALAR/SCORER_LOCK and root CPU releases are verified. GT and bridge sourcecache maps must match normal. Independent audit states bind actual prediction/result/sourceprogram hashes; finalmechanism review must retain the five mask-role distinctions and every bound evidencehash. Supplementreview must match the actual manualaggregate and optimizerfidelity hashes and retain nonblind scope. Original manualprivatefile may be SHAchecked for lineage only, never parsed/copied. Final rootreview requires publication_ready plus actual REPORT/DECISION hashes. Negative/nonadoption findings can pass publication review; readiness is not an effectiveness gate.

```text
python tools/build_round9_export.py
# optional --output <new-workspace-bounded-directory>
```

The finite whitelist includes newR9 code/CPUchecks/actualaudit code, original float32 auditorfailure history, sourceplan projections, required fixedR8semantic/geometry and R7adapter/camera/sampling dependencies, R6factory/compatibilitypatch and R5independentgeometry. There is no recursive copying, modelweight or featurecache export. The startup copied-wrong-digest attempt is exported as status/actualplan/loghash only; it failed before model/output creation and did not change the sourceplan. Draft/preparation receipts remain labelled as such and never substitute for actual audits.

`ALL4320_ARM_SCALARS.csv.gz` contains all1760normal+2080GT+480bridge arm rows, each with explicit scientificrole and originalstage fields where available. Shared source rows are correlated references, not extra cases. CaseIDs become case_000001..160 and scenes ordinalaliases; the private mapping is omitted. Coordinateerror norms, actual update counts, capacity residualnorms and bridge-relative scalars are allowed; raw/error3D or6D vectors are excluded. Fields absent in a stage stay empty, not fabricated as0. The table is losslessly `gzip.compress(level9,mtime0)`, verified by byteidentical decompression and recorded columns/counts/compressed+uncompressed hashes/runtime. Allsigned negative outcomes and zero/undefined denominators remain, without rounding or selected positivecases.

Plans/seals drop private path/assetdir fields and rawID sourcecache/predictionfile maps, retaining hashes and counts. Original/export SHA256 and any workspace-root-only portability projection are in EXPORT_PROVENANCE.json. Public scalaraggregates may reference relative evidencefilenames/hashes, but no original dataIDs/scenes/framepaths, caption, GT6/7Dindex/sidecar, coordinatearray, cachefeature/embedding, model/optimizerweight, media or signedURL can be included. The immutable-string privacy validator caches only successful identical-string checks with maxsize32768; it neither weakens validation nor caches failed assertions. Root reviews the actual candidate before publication; this tool never pushes or queues further experiments.

## Export-only coordinate projection correction

The first prepared builder (SHA256 `9ae8a64f2dea082332a4c4ca4f4247326f28a8ec204a81677434bdcff4e78f85`) reached the unchanged finalprivacy guard on a nested CPUfixture `coordinate_residual6` field and was correctly rejected before creating a public tree. The exact oldbuilder is retained as `tools/build_round9_export_pre_coordinate_projection.py`; the preparation-failure receipt records the field/cause and parent-tool traceback location without inventing a standalone log. No scientific source, plan, prediction, scorer, audit or measured result changed.

The repair is exclusively recursive field projection for CPU/history/plans and other metadata: all declared forbidden raw3D/6D/7D coordinate keys and PRIVATE keys are omitted before serialization, in addition to private paths/datafilemaps. Even a zero-valued `coordinate_residual6` remains a forbidden coordinate vector. Explicit experimental0/1 mask definitions (`target_components_mask`, C/S/CS groups and center-only maskC) remain because they specify intervention roles and contain no coordinates. Scalar residual/error norms, passflags and truegradient evidence remain. Original and exported SHA256 and the field-projection mode are recorded separately in provenance; the guard still rejects a forbidden field if projection misses it.
