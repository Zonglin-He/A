# R8 real-run interface — root owns GPU lease and smoke/full release

Scientific G implementation: `association_geometry.py`; final pixel convention is
continuous FOV then floor pixel cells, matching R7 exactly. Earlier57cf and7dec
versions/fixtures/locks are retained under `history/` and were not used on real data.
`r_global` and each factor are heuristic strengths, not calibrated probabilities.

```
crossplatform/env/bin/python -B crossplatform/round8/run_association.py \
  --phase smoke --manifest <absolute-path-GPU-manifest.json> \
  --plan <absolute-new-plan.json> --plan-sha256 <sha256> --output <absolute-fresh-output>
```

Use the same code for full after root's real smoke/nonzero-gradient/physical-overlay
checks. Parent must hold shared `research/round2/gpu.lock`. No independent background
GPU worker or CLIP job is spawned: source forward, CPU group preparation, CLIP image
and text encoding, then tiny regression updates run sequentially in one process.
This preparation does not authorize execution or substitute for the real smoke.

Required plan fields are the previous source/data contracts (`phase`, authorization,
B1,expected_samples,manifest SHA, source epoch100 path/SHA/model fingerprint/source
training plan, exact R6factory/upstream/genericRoBERTa hashes, seed,sentinel,
noun_fallback,missing_distortion_policy) plus:

- `program_sha256` paths relative to workspace root. Include R8driver/G/semantic
  encoder and imported R7adapter/camera/experiment runner_helpers. Base is explicitly
  recorded as workspace_root in seal. Historical source files are never edited.
- `strict_FP32_flags` exactly cudnn_allow_tf32=False,matmul_allow_tf32=False,
  cudnn_benchmark=False,cudnn_deterministic=True. Precision highest; no autocast.
- `alpha=.25`, `beta_metres=.1`, `steps=[1,3]`.
- `G_config_sha256=992b6a565cd22a2eedf494f0a26815de95bafe4da38af2214aa748e587c584a7`.
- `r_global_policy=shrink_shared_pseudo_target_no_extra_LR_multiplier`.
- `CLIP_device=cuda`, `CLIP_batch_size=32`, `sequential_single_GPU_only=True`,
  absolute `CLIP_asset_dir`, `CLIP_conversion_receipt_sha256`.

Manifest is the exact nine-key GT-free input allowlist from R7 (id/platform/scene/
pointpath/metapath/imagepath/caption/suffix/seed); paths absolute, stable id joins
independent private scoring index. Points sampling is exact R5 float32/seed16384
with Drone in-place z+1.8/Quad identity, without GT filtering. Grounder sees literal
caption plus registered sentinel, puretext firstnoun BBF readout. CLIP sees only the
literal original caption without that sentinel, and originalRGB groupmajor crops
[g0tight,g0context,g1tight,g1context,...]. CLIP77token truncation/crop/resource receipt
is saved. No color gate, target phrase shortcut or spatial-relation understanding
claim is added. MissingD is declared assume-zero, not physically verified zero.

The unchanged source cache contains q0 and all256 candidate regressor features.
Associated arms use a dataclasses.replace(cache,query_index=q1) copy; q1 is an
actual selected source proposal, and q0 is never retagged. LR at q1 is
`.25*.1/max(||h_center(q1)||²+1,||h_size(q1)||²+1)`, with NO second global trust
multiplier. Global strength already shrinks the shared direct/TTA pseudo target.
Each1/3step call starts from identical source weights and restores them. No teacher,
EMA, backbone/BN/CLIP update or source retraining occurs. Size remains q1's prior,
w_size0,yaw0. Unsupported camera/no eligible support/zero r may preserve source.

## NPZ private schema

Per`<id>.npz` allraw source pool256 and19selected arms:

- source_boxes_raw6/eval6[256,6], source_text_scores256[256], source_native_top10[10],
  q_source scalar. Native sorting ties do not change q0's fixedargmax definition.
- modes=['rgb_fulltext','pc_only','semantic_shuffle'], q_association[3].
- raw_consensus_center float64[3,3], r_global float64[3],
  pseudo_target6_float64[3,6], actual pseudo_target6 float32[3,6],
  pseudo_reliability6 float32[3,6]. Quantization to a source-equal target is not
  counted as effective adaptation just because the original heuristic w was>0.
- arm_names/arm_modes/arm_query_index/arm_status[19], arm_boxes_raw6/eval6[19,6].
  Order is source(q0) + eachmode×association_only,direct_G,param1,param3,
  outputspace1,outputspace3 (all q1). Eval clamp identical min1e-6 on dimensions.
- dedup_membership[256] (-1 for proposals excluded from finite candidatepool),
  semantic_cosines_tight_context[G,2],roi_group_ids[G],roi_xyxy[G,2,4].
- uint8[32] point_sample_sha256,raw_indices_sha256,image_file_sha256.

Trace retains group membership/anchor/supportcount/complete-link pair minima,
source/associated group relation, full-caption semantic components and gap, raw
consensus delta/global strength/final64delta/actual32delta, genuine gradient/parameter
delta/spillover/equivalent-kernel error and exactLR, source/camera/image hashes,
CLIP freeze and truncation/crop/resource receipts, timing and allocatedpeak. A gate
acceptance is named eligibility, not actualparameter change; both counts reported.
No GT object identity is implied by a dedup group, visibility or CLIP cosine.

Sample0 real source repeat/last-Conv bridge bitexact, grounder restore and CLIP state
unchanged checks precede final sealing. At end grounder full fingerprint/restored
modes/gradNone and CLIP full fingerprint/parameter versions/buffers are checked.
SEALED binds allNPZ+trace/plan/manifest/repeat receipts, source model and frozenCLIP
hashes, strictprecision and n_samples. Independent CPU scorer receives only a root
release bound to this seal and private source-of-GT join after all prediction arms
are sealed. GT/scoring is not imported or invoked by this driver.

## Required next stage

AST and synthetic geometry/schema fixtures pass but establish no real task effect.
Root should run the bounded real smoke, count actual nonzero gradients/parameter
changes (separate rawproposal disagreement, r shrink, float32targetdelta), and view
physicalq1/ROI overlays. Only then freeze any full5912 evaluation. Keep source,
association-only,directG,parameter/kernel and PC/shuffle controls and negative tails;
consensus is correlated source-model aggregation, not independent geometric truth.
