# Frozen ReferDINO-supervised spatial OPD research implementation

This is the HC research recipe. All TA-STVG, GroundingDINO-T and ReferDINO pretrained parameters remain frozen. Per query, eight independent 256-dimensional content vectors (2048 temporary variables) and a fresh Adam optimizer start at zero. No LayerNorm state is carried across queries. Time is Source finalMBR. Source is TA-STVG trained on VidSTG. The two original temporal offsets remain separate.

The full chain is RGB video + query → Source → original four GroundingDINO-T observations → E supported global correction + finalMBR → native ReferDINO eight-frame trajectory → output-box mixture OPD → frozen decoder full trajectory. E is an initializer, not the same thing as the ReferDINO teacher. Both expert stages must be counted. No new weights are trained or downloaded. Native ReferDINO is isolated in a subprocess because the upstream packages both use a top-level `models` namespace.

The single configuration is [config.json](config.json). Final temperature is written only after the predeclared HC128 comparison. The `seeds` field specifies experimental repeats, not a three-model deployment ensemble; `default_seed` is the deterministic deployment repeat. Final output is the actual updated student trajectory, not a candidate chosen by the teacher.

## Run

From the repository root, with the local frozen weights/runtimes and Python dependencies installed:

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_referdino_opd_v1.run \
  --input request.json --output prediction.pt --cache-out optional_cache.pt --seed 41
```

`request.json` contains `metadata` with `video_path`, `caption`, `index`, original `width`/`height`, and strictly increasing original physical `frame_ids`; optional `video_sha256` verifies media integrity. HC decoding uses the original FFmpeg output timing. Supply the same physical grid as the Source inference protocol for matched benchmark replication. No benchmark roster or annotation is needed. The optional top-level `key` fixes the per-query random-number stream; exact cached regression must retain this key as well as the seed. An optional `parses` field may contain the original query-only parse; when absent, the existing local Stanza parser is used. No target label or category is read. `--cache-in` accepts a cache with identical input signature; this is an explicit acceleration mode and does not make the original evidence free.

Offline scoring is separate:

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_referdino_opd_v1.score \
  --barrier prediction_manifest.json --labels offline_labels.json --output scores.json
```

The manifest seals prediction paths and SHA256 before labels are read. Labels are explicitly offline: per alias `interval`, `metadata` with width/height, and `boxes_xyxy_pixels` keyed by original physical frame. HC official clipping/interpolation and zero-overlap semantics are retained. Experiments use the same scoring functions and verify exact agreement with the established metric bridge.

## Policy and controls

Means are `(cx,cy,logw,logh)` of actual E-retaining student outputs. The equal-weight two-scale Gaussian mixture uses `D=diag(w,h,1,1)`, stopped within each round. Thirty-two candidates include eight antithetic pairs at each scale. Weights are constructed from the same fixed ReferDINO pseudo-boxes using `softmax(-(5 L1 + 2(1-GIoU))/tau)`; these are not native or calibrated teacher probabilities. Exact joint mixture NLL, including all normalizers, differentiates only the student mean. Positive-size out-of-image candidates are retained as in the executed policy.

Live samples around the current decoded mean and current size; fresh Fixed samples around initial mean and initial size using new paired noise each round, while fitting the current student. Thus the comparison refreshes both center and size context. Direct minimizes the same pseudo-box cost on actual decoder predictions. All controls rerun the complete frozen spatial decoder after each temporary state update.

The predictor, state and policy modules do not import an experiment roster, scorer or report script. Source memory-management wrappers preserve the prior exact arithmetic; the shared source-capture primitives are not an additional adaptation method. Historical protocols, predictions, negative tails and the older P7-A deployment remain separate evidence.
