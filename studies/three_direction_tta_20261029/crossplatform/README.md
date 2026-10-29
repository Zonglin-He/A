# 3EED local research export (review only; not pushed)

This bundle preserves two network-LN TTA pilots, a fixed point-PCA yaw developmental check, and a frozen-rule36new-frame replication. The released official checkpoint was mixed-platform trained. **All main source/PCA results use official weights with a modified text-only first-noun readout, not the original label-assisted officialspan readout.** This is target-only adaptation/refinement, not source-only cross-platform transfer. The posthocGT-class-assisted source diagnostic agrees on35/36queries but produces a differentbaseline; see results/replication36/readout_diagnostic.json. It is not an unlabeledmethod and does not replace the lockedPCA mainresults.

The point-PCA method adds yaw to a6D source box while fixing its center and extent. It is explicitly a test-time geometry/latentpose refinement; it does not update network weights. Source and PCA predictions were separately sealed before GT scoring. The36frame rule was fixed from the initial9development cases. Twenty-four newframes overlap development scenes; only the12Waymo frames use11newscenes. Do not conflate this with independent cross-platform generalization.

## Correct metrics and preserved failure history

Use corrected `scored_native.json` for the two LN pilots and `results/geometry_pca_development9/scored.json` plus `results/replication36/scored.json` for geometric refinement. The `INVALID_full9d_history.json` files are deliberately retained invalid measurements. Their schema assumption is withdrawn in `MEASUREMENT_CORRECTION.json`. Original rawprediction hashes and planhashes provide lineage; no rawprediction file is in this export.

## Minimal reproduction

1. Clone https://github.com/worldbench/3EED at commit843a901f5367a6be5cc484b97c6ecad462961dfa into `upstream/`, then review/apply `compat/upstream.patch`. Completevendor is intentionally not included.
2. Use Python3.11, PyTorch2.7/cu128, CUDA12.8 compiler with sm120 support and packages listed in `environment.json`. Compile bothPointNet extensions andROI operator. `code/build.py` documents the measured local build workaround; adapt its baseenvironment path for a differentmachine. The PointNet inplace build may require copying its resulting extension into the package directory.
3. Obtain official dataset and checkpoint from https://huggingface.co/datasets/RRRong/3EED. Expected assetSHA256 values are in `evidence/assets_verified.json`. Retrieve tokenizer/config from officialRoBERTa; strict fullcheckpoint restoration covers the frozen textencoder and forbids missing-key/random fallback.
4. Place the measured scripts under a working project root as in `code/`. `download_assets.py`/`prepare.py` reconstruct the original deterministic9frame cohort. `replication36/download.py` reconstructs first12hash-ordered validation frames/platform excluding those9. Captions are loaded locally from the officialdataset and are never supplied by thisbundle. Exact hashedframe IDs appear in metric records. Rebuild local manifests/locks because filesystempaths are intentionally private/hostspecific; retain the frozen selection, seed20261029+i and fixedPCA rules. Publicplan views preserve originalprivate hashes and redact localpaths/relative datasetpaths; they are audit records, not files to substitute as raw privateplan bytes.
5. For LN pilots use `run.py` (v1) and `run_v2.py` (single query-normalization correction). Never interpret archived full9D scorer output as valid; corrected protocol is first7geometryfields. For geometry run `geometry_refine.py` or `replication36/source.py` then `replication36/pca.py`, seal outputs, then the corresponding7D OBB scorer. New36 grouping/bootstrap implementation is `replication36/score_groups.py` and its groupdefinition was locked before GT scoring.

All source files are included byte-for-byte as measured; sourcehashes in plans refer to thosefiles. Sourceinitialization, localpath registration, compilation and publicdata acquisition require local setup. This is a minimal inspectable reproduction package, not a turnkey container image.

## Privacy and export boundary

No dataset, RGB, pointcloud, rawcaption, GTcoordinate, weight, rawprediction/featurecache, compiledbinary, completevendor checkout or credential is included. Local scientificcase figures remain private. Per-scene statistics use stable pseudonym IDs. All positive/negative scalar tails and the invalid-measurement correction are retained. This staging directory has not been pushed.

## Added bounded surface-selection phase

`results/surface_development45` is exposeddevelopment, not heldout validation. The fixed singlebox-surfaceSDF loss selects among yaw0/PCA/PCA+pi/2 with sourcecenter/extent/crop fixed. Its preregistered continuationcondition passed on45exposedframes; exactlyoneextra36new-frame source/PCA/surfacevalidation followed under unchangedrules. The study thenended. `results/surface_validation36` retains allthree arms, adversecases, prescoringgroups, scene-descriptive bootstrap and33-of36officialspan readoutdiagnostic. Newsource results remain officialweights plus modifiedtext-onlyreadout. Read `DECISION_SURFACE_UPDATE.json`: no reliablegate/deployment adoption andno claim ofnetworkTTAcontribution.

The portable core `code/reference_surface.py` takespoints and fixedsource/PCAboxes, uses no GT/image/model, and matches all81local sealedoutputs bitexact. To reproduce the core with modeloutputs, first call `reference_pca.refine(points, sourcebox)`, then `reference_surface.refine(points, source7d, pca7d)`. Fullmeasured workflows are preserved byte-for-byte under `code/surface_dev45` and `code/replication36_surface`; localmanifests/locks must be rebuilt asdocumented above because privatepaths andrawinputcaptions areexcluded.

The earlier79-file package manifest is preserved in `lineage/manifest_initial79.json`. Existing79same-path files are unchanged except thisREADME andSUMMARY, which are explicitlyupdated toreflect finalscope. Newplans/metrics/source are appended.
