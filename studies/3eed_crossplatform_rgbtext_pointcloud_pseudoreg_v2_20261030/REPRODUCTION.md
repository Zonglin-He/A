# R8 reproduction and finite public-export contract

This is the reproduction specification for `studies/3eed_crossplatform_rgbtext_pointcloud_pseudoreg_v2_20261030`. Preparing this document or its builder does not establish completed full inference, successful independent scoring, effective G feedback, or a deployment decision. Final outcomes belong to the separately hash-bound `REPORT.md` and `DECISION.json`. Root runs the builder only after all actual final gates exist; this preparation performs no export, model/GPU execution, target-GT access or publication.

## Source model and required upstream environment

There is one 3EED grounder, initialized from the audited **Waymo-only epoch100** source, plus one public frozen crop/text encoder. The grounder completed 100 supervised source epochs and 15,300 optimizer updates. Target Drone/Quad data did not enter source training, and the released mixed-platform task checkpoint was not used. Generic frozen RoBERTa pretraining is declared. Expected source checkpoint SHA256 is:

`0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd`

Expected source model-state fingerprint is `0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9`; source training plan SHA256 is `66fabf36898f1bc92124b680435e598e7bca0809e48aeb2da35e272331afe7ce`. SourceVAL monitoring never selects target hyperparameters. The primary source is fixed epoch100. The R6 source-training bundle contains its full provenance and validation history; trained weights and private source inputs remain local.

Preserve the exported relative workspace tree. The included `crossplatform/round6/factory.py` constructs the original upstream model. Clone [worldbench/3EED](https://github.com/worldbench/3EED) at commit `843a901f5367a6be5cc484b97c6ecad462961dfa` to `crossplatform/round3/p0/upstream`, apply the included `research/round6/UPSTREAM_COMPATIBILITY.patch`, and retain a `crossplatform/upstream` alias to that checkout. This is a layout name, not execution of the old cancelled P0 experiment. The compatibility patch removes obsolete unused `cgi` and optional visualization imports; it does not change the learned architecture or geometry. Obtain generic RoBERTa assets under their terms and make them available through the factory's pinned layout. Preserve upstream and third-party notices.

The actual environment uses Python3.11, PyTorch2.7.0+cu128/CUDA12.8, Transformers4.52.4, NumPy1.26.4, SciPy1.15.3, safetensors0.8.0 and Pillow12.3.0. Compile upstream `pointnet2`, `ops/teed_pointnet/pointnet2_batch` and `ops/teed_pointnet/roiaware_pool3d` for the local ABI/GPU; the original RTX5090 uses sm120. Compiled binaries are excluded. See the R6 reproduction for the pointnet2 build-lib directory convention and genuine source-training commands. Do not substitute imported binary compatibility or a mixed-domain checkpoint for the audited model.

R7 dependencies included byte-exact are `regression_adapter.py`, `camera_projection.py` and `experiment/runner_helpers.py`. Independent R5 geometry and R6 batched yaw-IoU helpers are included for offline scoring, together with `tools/obb_iou.py`. They do not import old experimental entrypoints or old caches. Source code, program hashes and any portability projection are distinguished explicitly.

## Frozen image/text encoder: provenance and preprocessing

The extra encoder is [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32), fixed HF revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. It supplies crop/text similarity only; it is not a 3D box generator, target-trained grounding teacher, EMA or adapted backbone. The original CLIP reference code is MIT licensed at [commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6). The HF card has no separate license tag; the resource receipt records that limitation instead of inventing a weights-license statement. No weights are redistributed.

The official pinned HF repository has `pytorch_model.bin` but no safetensors. Original size is605,247,071bytes and LFS SHA256 is `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. The local safetensors derivative SHA256 is `7a7177862f66645c5e923e31e07318fd609769378a197a21903fbbcc07e446b1`. All398 learned tensor keys/shapes strictly load, and conversion roundtrip is bitexact. The two legacy `position_ids` tensors are removed only after exact arange verification, because current Transformers keeps these deterministic buffers nonpersistently. No learned weight is omitted. Only `torch.load(weights_only=True,map_location='cpu')` is used for the official tensor conversion; unrestricted pickle and remote executable model code are forbidden.

Regenerate the local resource directory `crossplatform/round8/assets/clip-vit-base-patch32`. Download `config.json`, `preprocessor_config.json`, `merges.txt`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and optional model-card reference from `https://huggingface.co/openai/clip-vit-base-patch32/resolve/<fixed-revision>/<filename>`. Verify every file against `resources/SEMANTIC_RESOURCE_RECEIPT.json`'s `safe_loading.runtime_files_sha256`; the derivative weight entry is verified after conversion. The finite downloader `fetch_semantic_assets.py` downloads/resumes only the original public checkpoint and checks the official LFS hash. Do not print or save redirected signed URLs or access tokens.

```text
python crossplatform/round8/fetch_semantic_assets.py
python crossplatform/round8/prepare_safe_semantic_weights.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python crossplatform/round8/check_semantic_encoder_cpu.py
```

The runtime loads local safetensors only, freezes every parameter, uses eval/FP32, and verifies resource hashes. The official processor converts RGB, resizes shortest edge224 with bicubic interpolation, center-crops224, rescales1/255 and normalizes with mean `[.48145466,.4578275,.40821073]`, std `[.26862954,.26130258,.27577711]`. The synthetic CPU fixture matched the original transform structure with maximum difference0, and checked inference repeat/frozen state. These are engineering checks, not target-quality evidence.

Text is the literal full description without the grounder's ` . not mentioned` sentinel. There are77 total tokens including start/end. Right-truncation lengths/counts are saved; long relation clauses can be lost. Features are512D L2-normalized. `match` emits raw cosines, with no softmax, probability calibration or learned logit-scale multiplication. The caller supplies group-major `[G0tight,G0context,G1tight,G1context,...]` crops, at most32; tight/context information stays separate and is fused by G with fixed .5/.5 weights. Canonical center-cropping can remove ends of elongated ROIs; do not silently substitute aspect stretching or a new prompt template.

## Inputs, camera and research exposure

Full targetVAL retains all5912 available inputs: Drone2984 from2 scenes and Quad2928 from3 scenes. These platforms were absent from source supervision but were previously researched in R5/R7; this is not fresh research confirmation. Only five scene clusters exist, so expression/crop/query-level replication does not create independent scenes.

`prepare_inputs.py` regenerates the fixed40case engineering smoke from the R7 strict input-only whitelist: for each sorted platform/scene, rank by `SHA256('R8_G_prereq_v1|'+id)`, exclude R7smoke20 IDs, take8 distinct pointpaths. The live source paths are all unique, and selected IDs and frames have no old-smoke overlap. No color, old metric, GT, source prediction or geometry condition enters selection. Color counts are descriptive only. Full5912 bytes match the R7 strict input manifest, but **every R8 source forward is newly computed under the registered strictFP32 flags**; old R7 source pools/features/scores are not the baseline.

GPU manifests contain exactly `id,platform,scene,pointpath,metapath,imagepath,caption,suffix,seed`. Regenerate those private manifests locally from authorized data; they are not in the public bundle. Keep native point sampling: `default_rng(seed).choice(rawN,16384,replace=rawN<16384)`, float32 XYZ, Drone in-place float32 z+=1.8, Quad unchanged. Seed/suffix and source text readout remain locked. The grounder receives fullcaption+sentinel and puretext firstlexiconnoun BBF; GTclass/positive_map never enters G. All5912 inputs had a noun-lexicon match; no GT fallback is needed.

Read metadata only through the streaming K/E/D camera whitelist; do not materialize `ground_info` for online consumers. Drone projection undoes the source zshift before E; Quad uses E. Missing D uses `assume_zero_declared`, an assumption rather than recovered distortion. Continuous half-open FOV is checked before floor pixel cells and observed-cloud zbuffer. Visibility is limited to the sampled observed point cloud, not a complete scene visibility oracle.

## Actual G and separate mechanisms

G prepares finite source proposals from the top32 text pool plus16 PC-support complement, groups them with complete-link support/box overlap constraints, and retains at most16 physical-proposal hypotheses for image crops. The word physical describes intended spatial grouping, not a verified instance identity. q0 remains the original source argmax. q1 is the original text-score-max proposal within the selected group, never a relabeled q0 or a newly regressed box.

The fullcaption/crop cosine selects a group with a small fixed PC-support term. PC-only and a fixed one-position group-cosine permutation are matched controls, not new independently trained models. Canonical group members and near-duplicate geometry representatives are defined by the frozen config. Center consensus combines correlated predictions from the same model/point sample; it is not independent geometric truth. The directG control has access to the exact same consensus information.

Only supported center axes can change. Sizes remain q1's pretrained source prior, all size weights arezero, and native yaw remainszero without PCA. The registered trust-scale correction is:

`final_pseudo_center = anchor_q1 + r_global * (raw_consensus_center - anchor_q1)`.

Globalstrength combines margin/support/multiple-representative factors heuristically; it is not a calibrated probability. It shrinks the **shared pseudo target**, not the learning rate a second time. Per-axis weights indicate eligible/relative dispersion support. A common scalar inside `sum(w*loss)/sum(w)` would cancel; the original protocol draft records that problem and the final scale decision supersedes its optional LR-gate suggestion.

Root viewed all10 fixed-hash, two-per-scene private images before the full run. Same visible cars/people can remain split across complete-link groups. Some tight/context crops include neighboring cars, background or only person fragments. A high similarity or consensus can therefore refer to the wrong physical object, and the current encoder is not validated for precise relational reasoning. Dedup is not perfect, and group size does not count independent observations. These negative supporting conditions are retained; viewing the images did not change G thresholds, prompts or method arms. Public root observations contain no original images or full captions.

## Parameter updates and all19 same-information controls

Only the four lastConv tensors of `prediction_heads.5.{center_residual_head,size_pred_head}.net.8.{weight,bias}` are eligible (1734 parameters); zero size loss leaves867 center parameters capable of nonzero updates. Frozen source features and original q1 are held fixed during each episodic update. BN/dropout/CLIP remain eval/frozen, with no backbone training, EMA or source retraining. One-step and three-step episodes start from the same source state and restore parameters/buffers/grad references/modes/RNG.

For each mode `rgb_fulltext`, `pc_only`, `semantic_shuffle`, preserve six arms: association-only q1; directG; parameterSGD1/3; exact output-space/kernel1/3. Shared source q0 gives19 arms. Use SmoothL1 beta=.1m, plainSGD without momentum/weight decay/clipping, LR `.25*.1/max(||h_center(q1)||²+1,||h_size(q1)||²+1)`. No extra globalstrength LR multiplier is applied. Nonlearning association, aggregation and limited box updates must not be attributed wholesale to parameter TTA.

Frozen final-linear features imply a matched Gram-kernel output-space update in real arithmetic. Registered strictFP32 numerical equivalence tolerance is5e-5m; preserve actual numerical errors rather than assuming exact finite-precision identity. All real source/CLIP/head calls run sequentially on one GPU, without autocast: cuDNN TF32=False, matmul TF32=False, cuDNN benchmark=False, cuDNN deterministic=True. These are fresh matched source flags. The earlier R7 strict-only positivecontrol does not prove cross-process pool identity or task benefit.

Record q0→q1 changes, raw consensus disagreement, finalfloat64 pseudo displacement, actualfloat32 displacement, accepted axes, executedsteps, nonzerogradients, parameterchanges and selectedoutputchanges separately. Merely setting `w>0` or a status to updated is not a nonzero update. Unsupported/no-group cases fail closed; preserve them and every negative task delta.

## Real execution, sealing and independent scoring

Historical plans bind exact local code, assets, sourcecheckpoint and manifests. A fresh replication must register its own local paths/hashes and authorization under a new namespace. Never edit a historical lock to make a changed replication appear identical. The original command shape is:

```text
flock -n <workspace>/research/round2/gpu.lock <python> <workspace>/crossplatform/round8/run_association.py --phase smoke --manifest <private-smoke40> --plan <new-smoke-lock> --plan-sha256 <hash> --output <new-smoke-directory>
flock -n <workspace>/research/round2/gpu.lock <python> <workspace>/crossplatform/round8/run_association.py --phase full --manifest <private-full5912> --plan <new-full-lock> --plan-sha256 <hash> --output <new-full-directory>
```

Both phases require real strict source repeat/lastConv bridges, full grounder restoration and unchanged frozen CLIP. Seal all requested predictions/traces before target scoring. The independent unlabeled auditor verifies code/input/asset/prediction hashes, actual q1 anchors and size priors, scale, quantization, nonzero-update receipts, output-space agreement and reset. It reads no target annotation index and does not establish pseudo quality. Synthetic fixtures alone cannot pass this real gate.

Only after **both smoke and full predictions are sealed** does root grant target GT scoring. The scorer checks the release against predictions, code, sourcecheckpoint, CLIP resources and private join; target labels never choose query/prompt/threshold/LR/alpha/steps. Source q0, association q1, directG, parameter and kernel outcomes are separately compared. Offline K>1/group-best IoU are GT-assisted coverage diagnostics, not online selectors or physical-instance purity.

Every unique selected arm is independently checked by general3D convex geometry against audited batched polygon×height IoU with maxerror<1e-8 and no strict-threshold disagreements. Full256 candidate coverage uses the batched implementation and is not represented as all256 independent3D. Native yaw0, float32 output coordinates promoted to float64, strict IoU>.25/.5 and minimum1e-6 dimension clamp are retained. Requested/finite/failed/zero denominators and all negative cases remain visible; undefined denominators are null. Report platform micro/equal-scene and paired scene-level results, with the explicit2/3-cluster limitation.

## Finite public builder and final artifacts

The builder `tools/build_round8_export.py` is only prepared at this stage. It has no model/GPU/network/Git action and refuses an existing destination. It creates no output until all hash-bound actual artifacts and privacy gates pass. Default output is `research/round8/experiment/public_export`; root will later place the reviewed candidate under the study directory.

`PUBLICATION_READINESS.json` requires alltrue fields:

```text
smoke40_and_full5912_predictions_sealed
smoke_and_full_independent_scored
smoke_and_full_unlabelled_audits_passed
actual_fixed10_unlabelled_visual_review_completed
root_final_review_passed
report_decision_privacy_reviewed
no_private_data_weights_identifiers_or_coordinates_in_export
```

`EXPORT_FINAL_SOURCES.json` must bind each key to exactly `{"path":"workspace-relative-artifact","sha256":"actualSHA256"}`:

```text
smoke_prediction_seal
full_prediction_seal
smoke_scalar_summary
full_scalar_summary
smoke_unlabelled_audit
full_unlabelled_audit
smoke_score_release
full_score_release
root_report
root_decision
root_final_review
full_postseal_diagnostic
smoke_score_reduction_audit
full_score_reduction_audit
```

`smoke_score_reduction_audit` and `full_score_reduction_audit` bind the actual independent `SMOKE40_SCORE_REDUCTION_AUDIT.json` / `FULL5912_SCORE_REDUCTION_AUDIT.json` once generated. The builder requires state `independent_R8_score_reduction_audit_pass`,40/5912counts, the actual auditor/scorer hashes, and a `results_sha256` matching both the respective score-seal fileentry and actual `RESULTS_PRIVATE.json` hash. It also requires positive checked-cell count, max reduction error≤1e-12, all19 query/delta arithmetic verification and source256coverage aggregation verification. Readiness booleans alone cannot substitute for those actual audit receipts. Public projections contain only explicit aggregate/hash/signed-count fields, not the private scored rows. The included `audit_scored_results.py`, `check_reduction_synthetic.py`, syntheticfixture, current/previous evaluationprogramlocks, postsealdiagnostic independent/synthetic reviews and `analyze_smoke_G_mechanism.py` provide reproducible code and review lineage. Program-root/privatepath fields are removed from projected locks; original/export hashes remain recorded. The reduction audit independently recomputes scalar reductions; it does not claim new fullpool3D geometry or new GT scoring.

`full_postseal_diagnostic` must bind the actual `evaluation/full_diagnostic_attempt001/SCALAR_DIAGNOSTIC.json`, once generated. At builder preparation this file is absent and is not represented as completed. Its sibling diagnostic seal must verify the publicfile SHA and5912count, and the scalarfile must bind the fullscore seal, pre-GT `POSTSEAL_DIAGNOSTIC_PLAN_v1.json`, and exact `diagnose_scored_results.py` programhash. The builder explicitly admits aggregate platform/subgroup/coverage fields only. `DIAGNOSTIC_PRIVATE.json`, originalscene rows and `SIGNED_TAILS_PRIVATE.json` are never copied or opened for export. Signed descriptive tails use `delta < -1e-8` for negative and `delta > 1e-8` for positive; ties within ±1e-8 never fill either tail. Each side retains at most10 actual signed examples per registered platform/contrast and separately records its availablecount; a side with no qualifying cases is empty. The originalID tie-break stays private. Public subgroup statistics include the number of defined sceneclusters, micro delta, equal-scene macro delta, gain/harm/tie counts and gained/lost/retained-success/retained-failure strictAcc25/50 transitions. Coverage categories describe the G-path staircase, not instance identity or necessarily actual failure: a q0fallback can already succeed despite a prepared/ROI-path miss. No separate publictails are generated: all positive/negative deltas remain in the complete aliased caseCSV. Its `negative_vs_source` flag denotes the raw sign (<0), whereas descriptive harm/tail counts use the declared1e-8 tolerance; full unrounded deltas allow either definition to be reconstructed. The pre-GT diagnosticplan, actual unlabeled `SMOKE_G_MECHANISM_DIAGNOSTIC.json` and its explanatory note are retained as separate evidence; the latter is a40case mechanism audit, not a GTquality result or a retuned G. The original v1prepared builder is retained as a nonexecuted historical snapshot with originalSHA.

Scalars are the scorer's real `SCALAR_SUMMARY.json` with sibling `RESULTS_PRIVATE.json`, `SCORER_LOCK.json` and `SEALED.json`. Root review requires `publication_ready:true`, `report_sha256` and `decision_sha256` matching actual final files. A valid negative/nonadoption result can pass publication review; this is not a performance gate. Optional `failure_receipts` is a finite path+SHA list, exporting only status/classification/count/programhash fields and original receipt hash, never original logs/messages/tracebacks. PreGPU FOV/floatfloor G histories, fixtures, locks and original/export code hashes remain explicitly historical; obsolete versions were not the real-data method. Scorer draft/fixture lineage is likewise retained without relabeling it current.

Root's eventual command is:

```text
python tools/build_round8_export.py
# or --output <new-workspace-bounded-candidate-directory>
```

The finite allowlist includes new G/encoder/runner and CPUfixtures, resource acquisition/conversion code, input/review tools, independent scorer/auditor and historical drafts, necessary fixed R7 dependencies, R6 factory/upstream compatibility patch, pure geometry helpers, protocols, source/CLIP provenance, independent audits, actual root visual observations, actual final summaries/report/decision and complete scalar tables. It never recursively copies a directory.

Case tables are `FULL_CASE_SCALARS.csv.gz` (5912×19=112,328 rows) and `FULL_G_SCALARS.csv.gz` (5912×3=17,736 rows). Compression is `gzip.compress(...,compresslevel=9,mtime=0)` with decompression byteidentity asserted; `SCALAR_TABLE_SCHEMA.json` records columns, rowcounts, encoding/None conventions, Python/zlib and compressed/uncompressed hashes. Original private output IDs are replaced by `case_000001...` in sealed manifest order; five private scene names become ordinal `scene_01...scene_05`. No ID mapping is exported. Every scalar is retained without rounding or positive-only selection. Uncompressed table representation and order remain reproducible on the recorded runtime.

No weights, rawimages/points/media, captions, GT/prediction/pseudobox coordinates, privateIDs/scene/frame names, embeddings/cosine distributions, annotation indices or signedURLs may enter the public bundle. Raw per-sample group-oracle distributions are excluded; labeled aggregate/scalar error metrics are allowed. Scientific code is byteexact except any explicitly recorded workspace-root-only portability projection. Plans drop private `_path`/`_dir` fields; JSON provenance keeps original and exported SHA256 so the projected file is never misrepresented as the original lock. Private rendered images remain local. Root must inspect the final manifest and privacy projections before publication; the builder never pushes.

The initially inspected diagnostic program had SHA256 `76fd8a65efa2595af5800df3d4e03d5aacbf3b196d8163a67f181f7b1c83789e`. The pre-GT corrected program has SHA256 `deaae13f57b6c8bc578a661c681e093adbc879c048043d9e95a3d8d39cca1a35`, adding signed-tail filtering, explicit scene macros/retained failures and coverage-scope language. The initial export preparation did not retain its own originalscript snapshot. Root subsequently reverse-applied the two recorded review patches into a separate draft and verified that its SHA256 exactly matches the previously recorded original76fd8a65 hash. That byteexact recovered original is retained at `research/round8/evaluation/history/diagnose_scored_results_pre_review.py`, with the recovery provenance distinguished from an original-time snapshot. It was never executed on scored results or GT. The pre-GT independent review corrected signed filtering, scene macros and coverage wording; only the corrected program is the actual final diagnostic. The final diagnostic binds that current corrected hash. G/inference/predictions are unchanged by this descriptive-program correction.
