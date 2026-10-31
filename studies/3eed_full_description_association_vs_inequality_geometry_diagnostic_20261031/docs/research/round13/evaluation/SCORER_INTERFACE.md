# R13 nine-arm direct scorer

This script is ready for root's actual complete-prediction release. It has only been exercised on synthetic arrays/temporary synthetic files; no actual GT/index payload, actual predictions, raw sensor data or model was read during development.

```
CUDA_VISIBLE_DEVICES='' crossplatform/env/bin/python -B research/round13/evaluation/score_direct.py \
 --predictions /absolute/composite_output \
 --manifest /absolute/strict9key_manifest.json \
 --cases /absolute/CASE116_PRIVATE.json \
 --scoring-index /absolute/postseal_R5_join.json \
 --release /absolute/CPU_SCORE_RELEASE.json --release-sha256 ROOT_HASH \
 --output /absolute/new_scored_output
```

## Inputs and before-GT gate

The model manifest is the original list of strict nine-key rows. Separate input-only cases rows use id/platform/scene_PRIVATE/source_dir/role, with `new_content` mapped to `new_content108` and `exposed_R12_all_changed_regression` mapped to `regression8`. Counts are checked against a root release, not hardcoded to old146. No derived manifest is written over source inputs.

Composite flat `<id>.npz` requires sample_id, arm_names, arm_query_index, arm_boxes_raw6/eval6, arm_status, source_boxes_raw6[256,6], source_text_scores256[256], q_source, case_role. Raw selected/pool boxes must be finite float32. Eval payload preserves the existing runner's size clamp dtype; float64 is used only inside the IoU geometry core. Nine arm names and order are fixed:

source / A_full_consensus / A_full_order0 / A_noun_consensus / A_deranged_consensus / B_inequality_full / B_inequality_noRGB / B_inequality_half / B_old_surface_equality.

All A boxes must equal the corresponding unchanged source-pool query. All B queries must be q0. B half must exactly equal float32 source+.5*(Bfull−source); positive-only B full/noRGB/half cannot shrink source dimensions. Every source directory must be among the verified sealed artifacts; selected composite sourcepool/scores/q0 are compared bitexact to the immutable source cache fields boxes_raw6[1,256,6]/text_scores[1,256]/query_index. This bridge does not decode query features or source sample points.

Required root release keys:

- CPU_SCORE_RELEASE=true; stage=`R13_association_vs_inequality_geometry_direct`; all_A_B_predictions_and_helper_outputs_sealed_before_GT=true.
- prediction_seal_sha256, manifest_sha256, cases_sha256, scoring_index_sha256, scorer_sha256.
- source_checkpoint_path and source_checkpoint_sha256 (byte integrity only; no model load).
- expected_case_role_counts, e.g. new_content108:108 / regression8:8.
- sealed_artifacts list of strict `{role,directory,seal_sha256}` records; `required_seal_roles` must exactly match actual unique roles. Include composite, every reused/new source, and every helper/artifact needed by root's complete closure. Each nonempty seal.files allowlist is verified. A seal.program_sha256 dictionary is checked path by path; a single producer SHA string (native mask-capture schema) is accepted only if it equals a value in the explicit, already byte-verified release.program_bindings map. Unknown producer strings/types are rejected. Source seals without a producer field retain their explicit release/capture-plan program closure. Checkpoint fields, where present, must match.
- Nonempty program_bindings/input_bindings/prompt_bindings path→SHA256 mappings. Paths may be absolute or workspace-relative.
- geometry_dependency_sha256 exactly as SCORER_PROGRAM_LOCK.json (R12 metrics, R5 transform/core, general3D OBB implementation).

The script verifies all seals/member hashes/bindings, complete NPZ IDs, source-cache bridges, arm/clamp/q/half/cohort contracts before writing SCORER_LOCK.json. Only then is the private scoring-index JSON payload opened. Byte hashing that index beforehand does not parse its labels. Original metadata containers are SHA-verified before target `ground_info[object_index].bbox_3d` is loaded. No other-object/semantic identity diagnosis is run.

Private index rows retain R12/R5 fields: id/platform/scene/meta_path/metadata_container_sha256/object_index and optional waymo_view_suffix. R5 transform truncates Drone/Quad raw9D to official yaw7D and performs actual float32 output convention; Drone z+=1.8. Prediction yaw remains0.

## Metrics and interpretation

New content108 and exposed regression8 are never pooled. Each cohort reports platform micro and scene-equal mIoU, strict IoU>.25/.5, center/size L2, per-coordinate absolute6 errors, changed/query-change counts and solver/status histograms. Cross-platform summary is additional; scene keys are platform-namespaced. Coordinate errors describe raw center/size parameters, not a GT-yaw axis-aligned enclosing box.

All arms vs source plus preregistered A full/order/noun/deranged and B full/noRGB/half/old-surface contrasts report signed positive/negative/tie sums/counts, scene-cluster descriptive intervals (reused R12 seed1201/draws2000) and fixed largest/smallest five private tails. Few scenes do not establish significance or population generalization.

All256 source-pool IoUs are a separate offline GT-assisted upper bound. MaxIoU/coverage and rank-recoverable25/50 are not method outputs or instance-identity labels. Additional score-retrieval top1/5/10/256 is explicitly CPU descending-score/ascending-index tie diagnostic, **not native GPU tie order**. First qualifying candidate source-score rank intervals retain tie ambiguity. Primary selected q0 is the actual original argmax.

Outputs are RESULTS_PRIVATE.json (contains IDs/percase values/private tails), SCALAR_SUMMARY.json (no private rows/tails), SCORER_LOCK.json and SEALED.json. No automatic publication or dataset/model writes occur.

## Pre-run producer-schema repair lineage

The initial scorer bd9410d53ce47fe6aebf574cfb1dc5b303e03873cd6eea6b0c6f250f599a422f assumed every program_sha256 was a dictionary. Before actual scoring, root identified native mask seals use a single SHA string. Source, original lock, fixtures and interface are retained in history/preflight_scorer_v1_before_string_producer_guard/. The current version adds the explicit release-bound string check above; no metric, transform, arm or label boundary changed. Synthetic checks include a valid mask-style producer and a freshly re-sealed but unbound producer digest rejected before the scoring lock/label read.
