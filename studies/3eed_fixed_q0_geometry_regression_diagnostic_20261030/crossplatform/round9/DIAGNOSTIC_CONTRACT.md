# R9 identity-preserving geometry and isolated GT capacity control

Run state: CPU implementation prepared; actual160normal/GT execution is owned by
root after cohort/code review and explicit GPU release. No R6-R8 files modified.
No target GT/index/media/model inference was opened/executed in preparation. Toy
CPU geometry/head/cache fixtures are engineering gates, not task results.

## Registered population and common conditions

Root160inputs:64Drone/96Quad, five available scenes×32, newly hash-selected frames
excluding R8smoke40 and R7smoke20. They are research-exposed targetVAL, not new scenes
or untouched confirmation. Normal GPUmanifest has only nine GT-free keys. Use the
new strictFP32 normal source pass; neither R8 prediction arrays nor cross-process
query slot assumptions are reused. Same source100 weights / RoBERTa / frozenCLIP /
pointcoords and calibration as locked R8. MissingD is declared assumedzero.

CLI phase `normal`:

```
crossplatform/env/bin/python -B crossplatform/round9/run_diagnostic.py \
 --phase normal --manifest <absolute-diagnostic160_gpu_inputs.json> \
 --plan <new-normal-plan.json> --plan-sha256 <sha> --output <fresh-normal-output>
```

Plan keeps R8 source/asset hashes, strictFP32 flags, B1,alpha.25,beta.1,steps1/3,
shared pseudo-target shrink policy and sequential oneGPU. New fields are
`phase=normal`, `expected_samples=160`, R9authorizationSHA, and
`selection_policy=fixed_original_q0_group_no_reselect`. Program hashes are relative
to workspace root: R9runner/fixedG/exactfit plus imported R8association/semantic,
R7adapter/camera/helpers and sourcefactory/upstream dependencies. No dataset loader
that reads target boxes is used. Root owns the one physicalGPU flock and <=30min
initial budget; no automatic fullVAL expansion is authorized by this runner.

## Fixed-q0 G, exactly what changed from R8

`fixed_q0_geometry.py` reuses the frozen R8 pool/groups/ROI/robust weighted consensus,
sizeprior and target-confidence shrink arithmetic, but removes the reselect step.
The only permitted group is q0's original dedup group, and the actual anchor is
q0. If q0 has nonpositive raw dimensions, is absent from preparedpool, or its group
is absent from selected semanticROIs, G returns q0/source identity with zero w.
A new query is never substituted on these paths.

RGBmargin compares the forced q0group full-caption CLIP cosine with all other
encoded groups; zero/negative margin remains conservative zero-confidence. PC-only
uses this same fixed q0group and support gap, never its largest-support alternative.
RGB ranking mismatch is saved as a diagnostic, not a new loss, fallback switch,
query edit or identity-correctness label. No color gate or arbitrary jitter added.
Confidence still shrinks the shared target, not an ineffective global factor in
normalizedw; normalizedw gives axis-relative reliability/eligibility only. Size
stays rawsource/w0, yaw0, beta=.1 and alpha=.25/max-featurekernel LR unchanged.
Single-member/no-margin/low-coverage may give no updates; no threshold is relaxed
just to increase acceptance.

Normal11arms are source + eachmode (`rgb_fulltext`,`pc_only`)×
`direct_G,param1,param3,outputspace1,outputspace3`. Every arm_query_index=q0;
there is no associated query or new source-canonical identity. DirectG receives
exactly the float32 pseudo target used by SGD/kernel. Raw/Eval clamp is uniform.
Store q_source,forced_q_source[2],rawconsensus64[2,3],r[2],pseudo64/actual32[2,6],
w[2,6],groupmembership256,ROIs/semanticcosines and allstep traces as in R8. Keep
machinezero targets distinct from merely positive confidence.

## Same-forward private cache seal

Normal immediately saves `source_cache/<id>.npz` BEFORE any G/update. Fields are
`base_xyz[1,256,3]`, `center_features,size_features[1,288,256]`,
`boxes_raw6[1,256,6]`, `text_scores[1,256]`, `query_index`,
`selected_parameter_sha256`, `sample_id`, checkpointSHA, uint8[32] point sample,
indices, actual-model-text and canonical-manifestrow hashes. These feature/token-
linked raw prediction caches are private and excluded from public export.

Normalseal contains all160prediction and160cache files,source_cache_files map,
n_samples160,n_arms11,strictflags,program/model/input hashes and source/CLIP freeze
checks. Normal exposes no GTtargetindex key/path. All caches and normal arms must
be sealed before controlled labels are readable. Independent identity confirmation
is a later CPU evaluator using actual available target/other annotations; G never
reads or claims identity labels.

## Isolated GT phase, no fresh backbone or query slots

```
crossplatform/env/bin/python -B crossplatform/round9/run_diagnostic.py \
 --phase gt_fit --plan <new-GT-plan.json> --plan-sha256 <sha> --output <fresh-GT-output>
```

GTphase rejects `--manifest`, never reads RGB/P/camera/G, and does not construct
CLIP. It loads the exact source100model and normal sealed sourcecache. All normal
file hashes and source/model fingerprint are checked; normal cached finalConv
replay must be bitexact. A cache diagnostic is serialized before that bridge gate.
This avoids generating a new query slot/identity after labels become visible.

Only after those normal-seal checks and explicit `GT_index_read_release_path/SHA`
verification may `target_index_path` be read. Release requires
`GT_index_read_authorized=true`,normal prediction sealSHA,target indexSHA,source
checkpointSHA,current programSHA map and source_cache_count160. GTindex is a list
of exactly160 unique rows with ONLY `id,target6_modelcoords`; set must equal the
normalcache IDs. Target6 must be finite and size positive; invalid labels fail
explicitly, never dropped. GTyaw is not in GPUindex or update loss. Root's separate
CPU scorer retains real GTorientation for task IoU.

GTplan binds normal_prediction_seal_path/SHA,target_index_path/SHA,read release
path/SHA,GT_groups={'C':111000,'S':000111,'CS':111111}, same source/assets/strictflags,
alpha.25,beta.1,SGDsteps1/3 and exactfit_coordinate_atol_metres=5e-5.

Controlled13arms: source + each`GT_C,GT_S,GT_CS`×
`direct_target,param1,param3,exactfit`. Each sees the same legitimate GT6 with the
explicit coordinate mask. Inactive coordinates never contribute loss/step and
output must remain bitexact rawsource; even a rawnegative source size is not
silently clamped for C. Directtarget is coordinate-mask replacement. Validation
of the fullGTsize does not make its inactive coordinates affect C optimization.

SGD1/3 use unchanged R7weightedSmoothL1/step rule and carry kernel numeric proof in
trace (not extra arms). Exactfit is a REAL backward/optimizerstep on the same final
linear heads: each activated block uses `.5*sum((pred3-GT3)^2)` and its own
`lr=1/(||h_block||²+1)`; no sharedmaxnorm and no outputoverride. C activates center,
S activates size, CS both. Persist per-coordinate residual, gradients/parameter
changes, inactive-coordinate equality,spillover and fullstate recovery. Fixed
5e-5m failure is preserved in trace and seal's all_exactfit_coordinate_checks_pass;
never loosen it after seeing real values.

GTNPZ includes sourcepool, q0,13arm raw/eval/query/status,fulltarget6,group masks,
constructed maskedtarget/sourceprior perarm and coordinate_residual6_by_arm.
These supervised caches are PRIVATE; public results may retain only aggregate
residual/capability/task scalars and code/protocol hashes. GTIoU is NOT a guaranteed
upperbound: yaw remains0 and exact rawGTsize need not maximize axis-aligned IoU
against a rotated GT box. Coordinate capacity, pseudo-label correctness, proposal
identity and actual unlabeled adaptation utility are separate questions.

All modes are episodic; source parameters/buffers/grad/modes/RNG restored. No
teacher/EMA or trained target model is retained. Source final architecture alone
can fit fixed-query coordinates under oracle supervision; that would not prove
an unlabeled signal can infer the right target or fix an incorrect q0 identity.
