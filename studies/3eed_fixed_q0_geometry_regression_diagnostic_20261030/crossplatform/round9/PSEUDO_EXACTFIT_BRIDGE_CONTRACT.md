# R9 post-normal-score exploratory optimizer bridge

This is a bounded diagnostic added after the normal160 score showed directG had
small gains while conservativeSGD3 recovered little of the same target's change.
It is NOT heldout confirmation or a distinct parameter-TTA contribution. It does
not alter the normal/GT experiments, their plans, G, target boxes or step settings.

Inputs: the complete normal160_attempt002 prediction seal, its source_cache map,
and its160top-level normal predictionNPZ map. Only normal raw directG selected
boxes are supervision targets; no GTindex, scored result path, new P/RGB/camera/G
or CLIP call is accepted. The root execution plan/release binds all maps, source
checkpoint and current program closure. Bridge rejects label/scored input fields.

Runtime:

```
crossplatform/env/bin/python -B crossplatform/round9/run_pseudo_exactfit.py \
 --plan <new-private-bridge-plan.json> --plan-sha256 <sha> --output <fresh-output>
```

Root alone schedules one sharedGPUlock, after prelockedGT13. Plan requires:

- purpose=`R9_post_normal_score_exploratory_pseudo_exactfit_optimizer_bridge`,
  GPU_execution_authorized=true,expected_samples160,
  post_normal_score_exploratory_optimizer_bridge=true.
- exact same strictFP32 flags/source100 source/assets/authorization hashes as R9.
  program_sha256 paths are workspace-root relative and include this script,
  frozen run_diagnostic.py,head_exact_fit.py,R7adapter/helpers and sourceclosure.
- mask_C=[1,1,1,0,0,0], exactfit_coordinate_atol_metres=5e-5,
  head_step_rule=`one_real_half_sum_L2_SGD_center_lr_inverse_feature_kernel_no_search`.
- normal_prediction_seal_path/SHA,source_cache_files=entire normalcache map,
  normal_prediction_files=normal top-level160NPZ map,
  bridge_read_release_path/SHA.

Release additionally requires unlabelled_pseudo_exactfit_execution_authorized=true,
normal sealSHA,source checkpointSHA,same cache/prediction/program maps,
post_normal_score_exploratory_optimizer_bridge=true and
no_GT_scored_media_or_new_G_inputs=true. All normal files are hash-verified before
model construction; selectedcache/prediction files are rechecked at read time.

For each sample, load exact same sourcecache/q0 and strictsource model, verify
same pool/scores/point/index hashes and finalConv replay. No new sourceforward or
queryslot. TargetRGB and targetPC are the EXACT normal direct_G raw6D selected
outputs, already float32 with unsupported coordinates equal rawsource. Both must
preserve source sizes bitexact. Nonpositive rawsource size explicitly returns
sourcefallback, applicable=false; no hiddenabs/clamp. Uniformeval clamp follows
normal convention for every arm.

Three arms only: source, rgb_fulltext__pseudo_exactfit, pc_only__pseudo_exactfit.
Fit center with the frozen head_exact_fit function: real loss=.5sum(center−target)²,
one backward/SGDstep at1/(||h_center(q0)||²+1); sizehead is unactivated,center
parameters/buffers/grad/modes/RNG restored episodically. This LR is derived from
the final linear feature kernel, not searched. A source-equal target may naturally
have zero gradient/parameter change; no jitter or forcedupdate is introduced.

Persist all actual residuals/deltas, normalized feature kernel, gradients/parameter
changes, inactive-coordinate equality,spillover and fullmodel restoration. Keep
5e-5 absolute coordinate result AND relative residual vs actual target displacement
because the target may be microscopic; absolute tolerance alone does not prove a
large fraction of a tiny target was fitted. No taskIoU upperbound is implied.

OutputNPZ: sourcepool raw/eval6,scores,q0; arm_names/modes/query/status3,armraw/eval
[3,6],normal_directG_targets_raw6[2,6],coordinate_residual_to_normal_directG6[2,6],
mode coordinate-passflags/applicability,maskC,normalcache/prediction/point/index
hashes. Trace is private; no coordinates/weights/features/rawpredictions become
public just because this run contains no labels. Seal binds normal/source maps,
release,code,source checkpoint,strictprecision and explicit exploratory scope.
A separate CPU evaluation release may compare this run with the ALREADY sealed
normaldirectG/SGD/kernel results. This GPU program does not read those task scores.

Scientific interpretation: if one-step realparameter fitting matches directG while
SGD1/3 stays near source, that supports conservative optimization underfitting the
same current-input pseudo target. It does not improve pseudo-label quality, resolve
wrong q0 identity, establish transfer to new P/views, or show unique parameter
adaptation benefit: exact final-linear fitting at one cached query is itself
functionally equivalent to directly using the pseudo box. Retain negative tails and
restrict any gains to this postscore160cohort; no automatic fullVAL or threshold/
alpha/loss scan is authorized.
